import os
import sys
import cv2
import argparse
import time
import threading
import numpy as np
from fastapi import FastAPI, Response, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from typing import Optional

from py_utils.imagenet_labels import IMAGENET_LABELS, short_label

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

OBJ_THRESH = 0.30          # live confidence cut (slider)
TOPK_DEFAULT = 5           # live top-k (slider)
IMG_SIZE = (384, 384)      # (width, height) — overridden at runtime from the .hef
EMA_ALPHA = 0.6            # smoothing: 0.6 * previous + 0.4 * current

# The HEF output order is the standard ImageNet-1k label order used by the
# Hailo classifier (py_utils/imagenet_labels.py).
CLASSES = IMAGENET_LABELS

_ema_probs = None
_OUTPUT_LOGGED = False


def load_classes(path):
    """Optional override of the label list (one label per line)."""
    global CLASSES
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = [line.strip() for line in f if line.strip()]
        if len(items) == len(IMAGENET_LABELS):
            CLASSES = tuple(items)
            print(f"Loaded {len(items)} labels from {path}", flush=True)
        else:
            print(f"Warning: {path} has {len(items)} labels, expected "
                  f"{len(IMAGENET_LABELS)}; keeping the built-in list", flush=True)
    except Exception as e:
        print(f"Error loading labels from {path}: {e}", flush=True)


class ClassificationConfig:
    def __init__(self):
        self.obj_thresh = OBJ_THRESH
        self.top_k = TOPK_DEFAULT
        self.lock = threading.Lock()

    def update(self, obj_thresh, top_k):
        with self.lock:
            if obj_thresh is not None:
                self.obj_thresh = float(obj_thresh)
            if top_k is not None:
                self.top_k = int(top_k)

    def get(self):
        with self.lock:
            return self.obj_thresh, self.top_k


det_config = ClassificationConfig()

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_global_model = None


class FrameBuffer:
    def __init__(self):
        self.raw = None
        self.annotated = None
        self.annotated_version = 0
        self.jpeg = None
        self.jpeg_version = 0
        self.cond = threading.Condition()

    def push_annotated(self, frame):
        with self.cond:
            self.raw = frame
            self.annotated = frame
            self.annotated_version += 1
            self.cond.notify_all()

    def wait_annotated(self, version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.annotated_version > version, timeout=timeout)
            return self.annotated, self.annotated_version

    def push_jpeg(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.jpeg_version += 1
            self.cond.notify_all()

    def wait_jpeg(self, version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg_version > version, timeout=timeout)
            return self.jpeg, self.jpeg_version

    def get_raw_frame(self):
        with self.cond:
            return self.raw.copy() if self.raw is not None else None


frame_buffer = FrameBuffer()


class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.version = 0
        self.last_read = 0
        self.stopped = False
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while not stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame = frame
                self.version += 1
                self.cond.notify_all()
        with self.cond:
            self.stopped = True
            self.cond.notify_all()

    def read(self, timeout=1.0):
        with self.cond:
            self.cond.wait_for(
                lambda: self.version > self.last_read or self.stopped, timeout=timeout
            )
            if self.frame is None:
                return False, None
            self.last_read = self.version
            return True, self.frame.copy()

    def stop(self):
        with self.cond:
            self.stopped = True
            self.cond.notify_all()
        self.thread.join(timeout=2)


class VideoAnalyzer:
    def __init__(self):
        self.model = None
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model):
        self.model = model

    def start_analysis(self, input_path, output_path):
        if self.is_processing:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._process_video, args=(input_path, output_path)
        )
        self._thread.daemon = True
        self._thread.start()
        return True

    def _process_video(self, input_path, output_path):
        self.is_processing = True
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = "Cannot open video"
            self.is_processing = False
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        if not writer.isOpened():
            self.error_msg = "Cannot open output writer"
            self.is_processing = False
            cap.release()
            return
        idx = 0
        global _ema_probs
        _ema_probs = None
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if self.model is not None:
                    class_ids, scores = classify_frame(frame, self.model)
                    draw_predictions(frame, class_ids, scores)
                writer.write(frame)
                idx += 1
                if total > 0:
                    self.progress = int(idx / total * 100)
        except Exception as e:
            self.error_msg = str(e)
        finally:
            cap.release()
            writer.release()
            self.is_processing = False
            if not self.error_msg:
                self.progress = 100

    def stop(self):
        self._stop_event.set()


video_analyzer = VideoAnalyzer()

app = FastAPI(title="reComputer CAS-ViT-M ImageNet Classification (Hailo-8)")


def _primary_output(hailo_output):
    if isinstance(hailo_output, dict):
        return next(iter(hailo_output.values()))
    if isinstance(hailo_output, (list, tuple)):
        return hailo_output[0]
    return hailo_output


def _softmax(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _logits_to_probs(hailo_output):
    """Return a 1000-way probability vector (host softmax when the HEF emits
    logits; the cas_vit_m compile has device softmax disabled)."""
    global _OUTPUT_LOGGED
    if hailo_output is None:
        return None
    arr = np.asarray(_primary_output(hailo_output), dtype=np.float32).squeeze()
    if arr.ndim == 0:
        return None
    arr = arr.reshape(-1)
    if arr.size != len(CLASSES):
        if not _OUTPUT_LOGGED:
            print(f"[CAS-ViT-M] unsupported output shape: {np.asarray(hailo_output).shape}",
                  flush=True)
            _OUTPUT_LOGGED = True
        return None
    finite = arr[np.isfinite(arr)]
    looks_like_probs = (
        finite.size > 0
        and float(finite.min()) >= 0.0
        and float(finite.max()) <= 1.0
        and 0.98 <= float(finite.sum()) <= 1.02
    )
    probs = arr if looks_like_probs else _softmax(arr)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not _OUTPUT_LOGGED:
        top = np.argsort(probs)[::-1][:3]
        print(
            "[CAS-ViT-M] output " + ", ".join(
                f"{short_label(CLASSES[i])}={float(probs[i]):.3f}" for i in top
            ),
            flush=True,
        )
        _OUTPUT_LOGGED = True
    return probs


def _topk(probs, conf_thresh, top_k):
    if probs is None:
        return None, None
    order = np.argsort(probs)[::-1][: max(1, int(top_k))]
    keep = [i for i in order if float(probs[i]) >= float(conf_thresh)]
    if not keep:
        return None, None
    return np.array(keep, dtype=np.int64), probs[keep]


def preprocess_frame(frame):
    """Classification preprocessing: plain resize to the HEF input; keep BGR.

    The CAS-ViT ALLS recipe performs bgr_to_rgb and ImageNet normalization
    inside the HEF, so feed raw uint8 BGR.
    """
    return cv2.resize(frame, (IMG_SIZE[0], IMG_SIZE[1]), interpolation=cv2.INTER_LINEAR)


def classify_frame(frame, model, smooth=True):
    """Run one frame and return (class_ids, scores) after optional EMA smoothing."""
    global _ema_probs
    outputs = model.run(preprocess_frame(frame))
    probs = _logits_to_probs(outputs)
    if probs is None:
        return None, None
    if smooth:
        _ema_probs = probs if _ema_probs is None else (
            EMA_ALPHA * _ema_probs + (1.0 - EMA_ALPHA) * probs
        )
        probs = _ema_probs
    conf, top_k = det_config.get()
    return _topk(probs, conf, top_k)


def draw_predictions(image, class_ids, scores):
    """Overlay a compact Top-K panel in the top-left corner."""
    if class_ids is None or len(class_ids) == 0:
        return
    h, w = image.shape[:2]
    scale = max(0.45, min(0.8, w / 1400.0))
    thickness = 1 if scale < 0.7 else 2
    line_h = int(26 * scale) + 8
    panel_w = min(w - 16, int(420 * scale))
    panel_h = line_h * (len(class_ids) + 1) + 14
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, dst=image)

    title = "CAS-ViT-M - ImageNet Top-K"
    cv2.putText(image, title, (16, 8 + line_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 230, 118), thickness)
    for rank, (cid, score) in enumerate(zip(class_ids, scores)):
        label = short_label(CLASSES[int(cid)])
        text = f"#{rank + 1} {label} {float(score) * 100:4.1f}%"
        y = 8 + line_h * (rank + 1) + line_h - 6
        cv2.putText(image, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thickness)
        bar_y = y + 3
        bar_w = int((panel_w - 32) * min(1.0, float(score)))
        cv2.rectangle(image, (16, bar_y), (16 + panel_w - 32, bar_y + 3),
                      (70, 70, 70), -1)
        cv2.rectangle(image, (16, bar_y), (16 + bar_w, bar_y + 3),
                      (0, 230, 118), -1)


def inference_loop(cap, model, is_video_file, target_fps):
    fps_counter = 0.0
    target_period = 1.0 / target_fps if target_fps > 0 else 0.0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            if video_analyzer.is_processing:
                time.sleep(1.0)
                next_time = time.time()
                continue
            ok, frame = cap.read()
            if not ok:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            start = time.time()
            class_ids, scores = classify_frame(frame, model)
            inference_time = time.time() - start
            draw_predictions(frame, class_ids, scores)
            inst_fps = 1.0 / inference_time if inference_time > 0 else 0.0
            fps_counter = 0.9 * fps_counter + 0.1 * inst_fps if fps_counter > 0 else inst_fps
            cv2.putText(frame, f"Hailo FPS: {fps_counter:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_buffer.push_annotated(frame)
            if target_period > 0:
                now = time.time()
                next_time += target_period
                sleep_for = next_time - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -target_period:
                    next_time = now + target_period
    finally:
        stop_event.set()


def encode_loop(preview_w, preview_h, quality):
    version = -1
    while not stop_event.is_set():
        frame, version = frame_buffer.wait_annotated(version, timeout=1.0)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if preview_w > 0 and preview_h > 0 and (w, h) != (preview_w, preview_h):
            frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            frame_buffer.push_jpeg(buf.tobytes())


def run_fastapi(host, port):
    print("Registered routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:45} | Methods: {route.methods}", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


@app.get("/api/config")
async def get_config():
    conf, top_k = det_config.get()
    return {"obj_thresh": conf, "top_k": top_k}


@app.post("/api/config")
async def set_config(config: dict):
    det_config.update(config.get("obj_thresh"), config.get("top_k"))
    return {"success": True}


@app.get("/api/video_feed")
async def video_feed():
    def gen():
        version = -1
        while True:
            jpeg, version = frame_buffer.wait_jpeg(version, timeout=1.0)
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        return {"success": False, "message": "No file"}
    safe = os.path.basename(file.filename)
    path = os.path.join(UPLOAD_DIR, safe)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"success": True, "filename": safe}


@app.get("/api/video/list")
async def list_videos():
    names = sorted(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else []
    return {"success": True, "files": names}


@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    src = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    if not os.path.isfile(src):
        return {"success": False, "message": "File not found"}
    base = os.path.splitext(os.path.basename(filename))[0]
    dst = os.path.join(OUTPUT_DIR, f"{base}_annotated.mp4")
    ok = video_analyzer.start_analysis(src, dst)
    return {"success": ok, "output": os.path.basename(dst) if ok else None}


@app.get("/api/video/status")
async def video_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg,
    }


@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if os.path.isfile(path):
        return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))
    return {"success": False, "message": "File not found"}


@app.post("/api/models/cas_vit_m/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
):
    if _global_model is None:
        return {"success": False, "message": "Model not initialized"}
    try:
        img = None
        source_info = ""
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_info = "uploaded image"
        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                if timestamp is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if ok:
                    img = frame
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)
        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"
        if img is None:
            return {"success": False, "message": "No valid input source found"}
        h, w = img.shape[:2]

        probs = _logits_to_probs(_global_model.run(preprocess_frame(img)))
        conf_thresh = float(conf) if conf is not None else det_config.get()[0]
        top_k = int(iou) if iou is not None else det_config.get()[1]
        class_ids, scores = _topk(probs, conf_thresh, top_k)
        predictions = []
        if class_ids is not None:
            predictions = [
                {
                    "class": CLASSES[int(cid)],
                    "label": short_label(CLASSES[int(cid)]),
                    "confidence": float(score),
                }
                for cid, score in zip(class_ids, scores)
            ]
        return {
            "success": True,
            "source": source_info,
            "predictions": predictions,
            "image": {"width": w, "height": h},
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


@app.get("/")
async def index():
    return Response(content=HTML_PAGE, media_type="text/html")


HTML_PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CAS-ViT-M ImageNet Classification · Hailo-8</title>
  <style>
    body { background:#111; color:#eee; font-family:system-ui,sans-serif; text-align:center; }
    h1 { color:#00e676; }
    .video-box { margin:20px auto; max-width:960px; }
    .controls { display:flex; justify-content:center; gap:40px; margin:12px; }
    .control-group { text-align:left; }
    input[type=range] { width:220px; }
    .value-display { margin-left:8px; color:#00e676; font-weight:bold; }
    .panel { background:#1e1e1e; padding:16px; border-radius:8px; max-width:720px; margin:16px auto; text-align:left; }
    .btn { background:#00e676; color:#000; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; font-weight:bold; }
    .progress { width:100%; background:#333; border-radius:8px; height:16px; overflow:hidden; margin-top:10px; }
    .progress > div { height:100%; width:0%; background:#00e676; }
  </style>
</head>
<body>
  <h1>CAS-ViT-M · ImageNet Classification · Hailo-8</h1>
  <div class="video-box"><img id="stream" src="/api/video_feed" style="max-width:100%;"></div>
  <div class="controls">
    <div class="control-group">
      <label>Confidence Threshold</label><br>
      <input type="range" id="conf" min="0.01" max="1.0" step="0.01" value="0.30">
      <span id="confValue" class="value-display">0.30</span>
    </div>
    <div class="control-group">
      <label>Top-K</label><br>
      <input type="range" id="topk" min="1" max="10" step="1" value="5">
      <span id="topkValue" class="value-display">5</span>
    </div>
  </div>
  <div class="panel">
    <label>Analyze a local video (.mp4)</label><br>
    <input type="file" id="videoFile" accept=".mp4">
    <button class="btn" onclick="uploadAndAnalyze()">Upload &amp; Analyze</button>
    <div class="progress"><div id="bar"></div></div>
    <div id="status" style="margin-top:8px;color:#aaa;"></div>
  </div>
  <script>
    const conf = document.getElementById('conf');
    const topk = document.getElementById('topk');
    const confValue = document.getElementById('confValue');
    const topkValue = document.getElementById('topkValue');
    function pushConfig() {
      confValue.innerText = parseFloat(conf.value).toFixed(2);
      topkValue.innerText = topk.value;
      fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ obj_thresh: parseFloat(conf.value), top_k: parseInt(topk.value) }) });
    }
    conf.oninput = pushConfig;
    topk.oninput = pushConfig;
    fetch('/api/config').then(r => r.json()).then(d => {
      conf.value = d.obj_thresh; topk.value = d.top_k; pushConfig();
    });
    async function uploadAndAnalyze() {
      const f = document.getElementById('videoFile').files[0];
      if (!f) { alert('Choose an .mp4 first'); return; }
      const form = new FormData(); form.append('file', f);
      const up = await (await fetch('/api/video/upload', { method: 'POST', body: form })).json();
      if (!up.success) { alert('upload failed'); return; }
      const fd = new FormData(); fd.append('filename', up.filename);
      await fetch('/api/video/analyze', { method: 'POST', body: fd });
      poll();
    }
    function poll() {
      fetch('/api/video/status').then(r => r.json()).then(s => {
        document.getElementById('bar').style.width = (s.progress || 0) + '%';
        document.getElementById('status').innerText = s.is_processing
          ? ('Processing ' + s.current_video + ' … ' + (s.progress || 0) + '%')
          : (s.error ? ('Error: ' + s.error) : 'Done. Download the annotated video from /api/video/download/<name>');
        if (s.is_processing) setTimeout(poll, 1000);
      });
    }
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="CAS-ViT-M ImageNet classification on CM5 + Hailo-8 (Web Preview Mode)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--class_path", type=str,
                        help="Optional file with 1000 ImageNet labels (one per line)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview_width", type=int, default=1280)
    parser.add_argument("--preview_height", type=int, default=720)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--cam_width", type=int, default=1280)
    parser.add_argument("--cam_height", type=int, default=720)
    parser.add_argument("--target_fps", type=float, default=30.0)
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available. Install the matching hailort wheel.")
        return

    load_classes(args.class_path)

    global _global_model, IMG_SIZE
    model = HailoInfer(args.model_path)
    IMG_SIZE = (model.input_w, model.input_h)
    print(f"Model input size: {model.input_w}x{model.input_h}", flush=True)
    _global_model = model
    video_analyzer.set_engine(model)

    web_thread = threading.Thread(
        target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web preview: http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        print("Web-only mode. Use the predict API or upload a video.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        is_video_file = True
        cap_source = cap
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        is_video_file = False
        cap_source = LatestFrameReader(cap).start()

    if not cap.isOpened():
        print("Error: cannot open the video source")
        return

    infer_thread = threading.Thread(
        target=inference_loop,
        args=(cap_source, model, is_video_file, args.target_fps),
        daemon=True)
    encode_thread = threading.Thread(
        target=encode_loop,
        args=(args.preview_width, args.preview_height, args.jpeg_quality),
        daemon=True)
    infer_thread.start()
    encode_thread.start()
    try:
        while infer_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        stop_event.set()
        if not is_video_file:
            cap_source.stop()
        infer_thread.join(timeout=2)
        encode_thread.join(timeout=2)
        cap.release()
        model.release()


if __name__ == "__main__":
    main()