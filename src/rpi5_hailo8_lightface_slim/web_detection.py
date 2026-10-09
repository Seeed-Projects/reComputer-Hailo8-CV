import os
import sys
import cv2
import argparse
import time
import subprocess
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from typing import Optional, List

from py_utils.coco_utils import COCO_test_helper
import sys; sys.path.insert(0, os.path.dirname(__file__))

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

# LightFace decodes on the CPU (base/lightface.yaml):
# The zoo nominates score_threshold 0.1 and nms_iou_thresh 0.3 for
# lightface_slim; the live preview uses a higher confidence cut for a cleaner
# image — lower the slider to see more.
OBJ_THRESH = 0.5
NMS_THRESH = 0.3
IMG_SIZE = (320, 240)  # (width, height) — 240x320 input, overridden at runtime from the .hef

# Anchor configuration and the decode pipeline live in the
# "LightFace Slim post-processing" block below.
_SCALE_FACTORS = (10.0, 5.0)  # SSD decode variances

# LightFace Slim detects a single class: face.
DEFAULT_CLASSES = (
    "face",
)

CLASSES = DEFAULT_CLASSES

def load_classes(path):
    global CLASSES
    if not path or not os.path.exists(path):
        CLASSES = DEFAULT_CLASSES
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        import re
        items = re.findall(r'"([^"]*)"', content)
        if items:
            CLASSES = tuple(items)
            print(f"Successfully loaded {len(CLASSES)} classes from {path}")
        else:
            items = [item.strip().strip('"') for item in content.split(',') if item.strip()]
            if items:
                CLASSES = tuple(items)
                print(f"Loaded {len(CLASSES)} classes from {path} (fallback parsing)")
            else:
                print(f"Warning: No classes found in {path}, using default COCO classes")
                CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default COCO classes")
        CLASSES = DEFAULT_CLASSES

class DetectionConfig:
    def __init__(self):
        self.obj_thresh = OBJ_THRESH
        self.nms_thresh = NMS_THRESH
        self.lock = threading.Lock()

    def update(self, obj_thresh, nms_thresh):
        with self.lock:
            self.obj_thresh = obj_thresh
            self.nms_thresh = nms_thresh

    def get(self):
        with self.lock:
            return self.obj_thresh, self.nms_thresh

det_config = DetectionConfig()

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoAnalyzer:
    def __init__(self, model=None, co_helper=None):
        self.model = model
        self.co_helper = co_helper
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model, co_helper):
        self.model = model
        self.co_helper = co_helper

    def start_analysis(self, input_path, output_path):
        if self.is_processing:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))
        self._thread.daemon = True
        self._thread.start()
        return True

    @staticmethod
    def _open_writer(output_path, width, height, fps):
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', f'{fps}', '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '0',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, 'ffmpeg'
        except FileNotFoundError:
            pass
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[VideoAnalyzer] Using cv2 mp4v (slower; install ffmpeg for 5x speedup)", flush=True)
            return out, 'mp4v'
        out.release()
        return None, None

    def _process_video(self, input_path, output_path):
        self.is_processing = True
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Error: Cannot open video {input_path}"
            self.is_processing = False
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"
            self.is_processing = False
            cap.release()
            return
        out, kind = self._open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "Error: No usable video writer (ffmpeg + cv2 mp4v both failed)"
            self.is_processing = False
            cap.release()
            return
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if self.model and self.co_helper:
                    processed_img, lb_info = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(processed_img)
                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                        if boxes is not None and len(boxes) > 0:
                            real_boxes = unletterbox_boxes(boxes, lb_info)
                            h, w = frame.shape[:2]
                            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                            draw_boxes(frame, real_boxes, scores, score_thresh=obj)
                if kind == 'ffmpeg':
                    out.stdin.write(frame.tobytes())
                else:
                    out.write(frame)
                frame_idx += 1
                self.progress = int((frame_idx / total_frames) * 100)
        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            cap.release()
            if kind == 'ffmpeg':
                try:
                    out.stdin.close()
                except Exception:
                    pass
                try:
                    out.wait(timeout=30)
                except Exception:
                    out.kill()
            else:
                out.release()
            self.is_processing = False
            if not self.error_msg:
                self.progress = 100

    def stop(self):
        self._stop_event.set()

video_analyzer = VideoAnalyzer()

app = FastAPI(title="reComputer LightFace Slim Hailo-8")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", OBJ_THRESH), config.get("nms_thresh", NMS_THRESH))
    return {"status": "success"}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = os.listdir(UPLOAD_DIR)
    outputs = os.listdir(OUTPUT_DIR)
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    success = video_analyzer.start_analysis(input_path, output_path)
    if success:
        return {"status": "started", "output": output_filename}
    else:
        return {"status": "error", "message": "Already processing another video"}

@app.get("/api/video/status")
async def get_analysis_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg
    }

@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

_global_model = None
_global_co_helper = None

@app.post("/api/models/lightface_slim/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None)
):
    if _global_model is None or _global_co_helper is None:
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
                ret, frame = cap.read()
                if ret:
                    img = frame
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)
        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"
        if img is None:
            return {"success": False, "message": "No valid input source found (image, video, or camera)"}
        h, w = img.shape[:2]
        input_img, lb_info = preprocess_frame(img, _global_co_helper)
        outputs = _global_model.run(input_img)
        current_obj_thresh, current_nms_thresh = det_config.get()
        target_conf = conf if conf is not None else current_obj_thresh
        target_iou = iou if iou is not None else current_nms_thresh
        boxes, scores = post_process_hailo(outputs, target_conf, target_iou, IMG_SIZE[1], IMG_SIZE[0])
        predictions = []
        if boxes is not None and len(boxes) > 0:
            real_boxes = unletterbox_boxes(boxes, lb_info)
            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
            eff_conf = target_conf if target_conf and target_conf > 0 else _LIGHTFACE_SCORE_THRES
            predictions = [
                {
                    "box": [int(value) for value in real_boxes[index]],
                    "confidence": float(scores[index]),
                }
                for index in range(len(real_boxes))
                if float(scores[index]) >= eff_conf
            ]
        return {"success": True, "source": source_info, "predictions": predictions, "image": {"width": w, "height": h}}
    except Exception as e:
        return {"success": False, "message": str(e)}

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
    def wait_annotated(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.annotated_version > last_version, timeout=timeout)
            return self.annotated, self.annotated_version
    def push_jpeg(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.jpeg_version += 1
            self.cond.notify_all()
    def wait_jpeg(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg_version > last_version, timeout=timeout)
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
        self._last_read_version = 0
        self._stopped = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
    def start(self):
        self._thread.start()
        return self
    def _loop(self):
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._cond:
                self.frame = frame
                self.version += 1
                self._cond.notify_all()
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
    def read(self, timeout=1.0):
        with self._cond:
            self._cond.wait_for(
                lambda: self.version > self._last_read_version or self._stopped, timeout=timeout)
            if self.frame is None:
                return False, None
            self._last_read_version = self.version
            return True, self.frame.copy()
    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=2)

@app.get("/api/video_feed")
async def video_feed():
    def generate():
        last_v = -1
        while True:
            jpeg, last_v = frame_buffer.wait_jpeg(last_v, timeout=1.0)
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html>
      <head>
        <title>reComputer LightFace · Hailo-8</title>
        <style>
          body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
          .container { max-width: 1200px; margin: 0 auto; }
          .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; width: 100%; max-width: 800px; }
          .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; vertical-align: top; margin: 10px; }
          .control-group { margin-bottom: 15px; }
          .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
          .slider-container { display: flex; align-items: center; gap: 15px; }
          input[type=range] { flex-grow: 1; cursor: pointer; }
          .value-display { min-width: 50px; font-family: monospace; background: #444; padding: 2px 8px; border-radius: 4px; text-align: center; }
          h1 { color: #00e676; }
          .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #333; }
          .tab { padding: 10px 30px; cursor: pointer; border-bottom: 3px solid transparent; transition: 0.3s; font-weight: bold; }
          .tab.active { border-bottom-color: #00e676; color: #00e676; }
          .tab-content { display: none; }
          .tab-content.active { display: block; }
          .video-analysis { text-align: left; background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px; }
          .btn { background: #00e676; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 5px; }
          .btn:hover { background: #00c853; }
          .btn:disabled { background: #555; cursor: not-allowed; }
          .progress-container { width: 100%; background: #444; border-radius: 10px; margin: 15px 0; height: 20px; position: relative; overflow: hidden; }
          .progress-bar { height: 100%; background: #00e676; width: 0%; transition: 0.3s; }
          .progress-text { position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 20px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }
          table { width: 100%; border-collapse: collapse; margin-top: 15px; }
          th, td { text-align: left; padding: 10px; border-bottom: 1px solid #444; }
          th { color: #888; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>LightFace Slim · RPi5 + Hailo-8</h1>
          <div class="tabs">
            <div class="tab active" onclick="showTab('realtime')">Real-time Detection</div>
            <div class="tab" onclick="showTab('analysis')">Local Video Analysis</div>
          </div>
          <div id="realtime" class="tab-content active">
            <div class="video-box">
              <img id="streamImg" src="/api/video_feed" style="max-width: 100%; height: auto;">
            </div>
            <div class="controls">
              <div class="control-group">
                <label>Confidence Threshold</label>
                <div class="slider-container">
                  <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.5">
                  <span id="confValue" class="value-display">0.50</span>
                </div>
              </div>
              <div class="control-group">
                <label>NMS IOU Threshold (CPU)</label>
                <div class="slider-container">
                  <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                  <span id="iouValue" class="value-display">0.40</span>
                </div>
              </div>
            </div>
          </div>
          <div id="analysis" class="tab-content">
            <div class="video-analysis">
              <h3>Analyze Local Video</h3>
              <div class="control-group">
                <label>Upload New Video (.mp4)</label>
                <input type="file" id="videoUpload" accept=".mp4">
                <button class="btn" onclick="uploadVideo()">Upload</button>
              </div>
              <div id="processingArea" style="display: none;">
                <p id="statusText">Processing: <span id="currentFileName">-</span></p>
                <div class="progress-container">
                  <div id="progressBar" class="progress-bar"></div>
                  <div id="progressText" class="progress-text">0%</div>
                </div>
                <p id="errorText" style="color: #ff5252;"></p>
              </div>
              <div class="control-group">
                <label>File Management</label>
                <button class="btn" onclick="refreshFileList()">Refresh List</button>
                <table>
                  <thead><tr><th>File Name</th><th>Action</th></tr></thead>
                  <tbody id="fileTableBody"></tbody>
                </table>
              </div>
            </div>
          </div>
          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: 8000</p>
        </div>
        <script>
          function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
            if (tabId === 'realtime') { document.getElementById('streamImg').src = '/api/video_feed'; }
            else { document.getElementById('streamImg').src = ''; refreshFileList(); }
          }
          const confSlider = document.getElementById('confSlider');
          const iouSlider = document.getElementById('iouSlider');
          const confValue = document.getElementById('confValue');
          const iouValue = document.getElementById('iouValue');
          function updateConfig() {
            const obj_thresh = parseFloat(confSlider.value);
            const nms_thresh = parseFloat(iouSlider.value);
            confValue.innerText = obj_thresh.toFixed(2);
            iouValue.innerText = nms_thresh.toFixed(2);
            fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ obj_thresh, nms_thresh }) });
          }
          confSlider.oninput = updateConfig;
          iouSlider.oninput = updateConfig;
          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh; iouSlider.value = data.nms_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2); iouValue.innerText = data.nms_thresh.toFixed(2);
          });
          async function uploadVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) return alert('Please select a file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const btn = event.currentTarget;
            btn.disabled = true; btn.innerText = 'Uploading...';
            try { await fetch('/api/video/upload', { method: 'POST', body: formData }); alert('Upload successful'); refreshFileList(); }
            catch (e) { alert('Upload failed'); }
            finally { btn.disabled = false; btn.innerText = 'Upload'; }
          }
          async function refreshFileList() {
            const res = await fetch('/api/video/list');
            const data = await res.json();
            const tbody = document.getElementById('fileTableBody');
            tbody.innerHTML = '';
            data.uploads.forEach(f => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${f} (Original)</td><td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>`; tbody.appendChild(tr); });
            data.outputs.forEach(f => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${f} (Analyzed)</td><td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>`; tbody.appendChild(tr); });
          }
          async function analyzeVideo(filename) {
            const formData = new FormData(); formData.append('filename', filename);
            const res = await fetch('/api/video/analyze', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'started') { startStatusPolling(); } else { alert(data.message || 'Error starting analysis'); }
          }
          let pollInterval;
          function startStatusPolling() {
            document.getElementById('processingArea').style.display = 'block';
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(async () => {
              const res = await fetch('/api/video/status'); const data = await res.json();
              document.getElementById('currentFileName').innerText = data.current_video;
              document.getElementById('progressBar').style.width = data.progress + '%';
              document.getElementById('progressText').innerText = data.progress + '%';
              document.getElementById('errorText').innerText = data.error || '';
              if (!data.is_processing && data.progress === 100) { clearInterval(pollInterval); alert('Analysis completed!'); refreshFileList(); }
              else if (!data.is_processing && data.error) { clearInterval(pollInterval); }
            }, 1000);
          }
          fetch('/api/video/status').then(res => res.json()).then(data => { if (data.is_processing) startStatusPolling(); });
        </script>
      </body>
    </html>
    """, media_type="text/html")

def run_fastapi(host, port):
    print("\n" + "="*50, flush=True)
    print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


# ---------------------------------------------------------------------------
# LightFace Slim post-processing (SSD-style heads, CPU decode + NMS)
#
# Ported from hailo_model_zoo core/postprocessing (meta_arch "lightface",
# base/lightface_slim_base.yaml -> base/widerface.yaml -> base/base.yaml).
# The compiled HEF exposes 8 raw heads and no on-chip NMS (verified on the
# artifact: output_layer1..8, and the alls script only bakes in the
# normalization):
#   stride 8  (30x40): bbox 12ch (3 anchors x 4 deltas) / conf 6ch (3 x bg+face)
#   stride 16 (15x20): bbox  8ch (2 anchors)            / conf 4ch
#   stride 32 ( 8x10): bbox  8ch (2 anchors)            / conf 4ch
#   stride 64 ( 4x5):  bbox 12ch (3 anchors)            / conf 6ch
# steps and min_sizes come from base/lightface_slim_base.yaml:
#   steps 8/16/32/64, min_sizes [10,16,24] [32,48] [64,96] [128,192,256]
#   -> 4,420 anchors.
# Pipeline: anchors -> SSD decode (variances 10/5) -> softmax over the two
# confidence channels -> threshold -> greedy NMS (single "face" class).
#
# Preprocessing: the HEF bakes normalize_in_net (mean 127 / std 128) and this
# compile has no in-graph bgr_to_rgb conversion, so the app feeds raw uint8
# BGR — the zoo's face-detection pipeline reads BGR and the model's original
# pipeline (Ultra-Light-Fast-Generic-Face-Detector) uses cv2 BGR input with
# BGR-ordered means. The frame is resized with the aspect ratio preserved and
# padded bottom/right with 0, matching the zoo's face-detection preprocessing.
# ---------------------------------------------------------------------------

from itertools import product as _product

LIGHTFACE_ANCHORS = {
    "steps": (8, 16, 32, 64),
    "min_sizes": (
        (10, 16, 24),
        (32, 48),
        (64, 96),
        (128, 192, 256),
    ),
}

_LIGHTFACE_SCORE_THRES = 0.5   # live preview cut when the slider is 0
_LIGHTFACE_NMS_THRES = 0.3     # base/lightface_slim_base.yaml

_anchors_cache = None
_LIGHTFACE_LOGGED = False


def _extract_anchors(min_sizes, steps):
    """Anchors in normalized center-size (cx, cy, w, h) - port of
    FaceDetectionPostProc.extract_anchors. Feature maps are ceil(dim/step);
    each cell holds one anchor per min_size."""
    img_h, img_w = IMG_SIZE[1], IMG_SIZE[0]
    anchors = []
    for idx, step in enumerate(steps):
        fh = int(np.ceil(img_h / step))
        fw = int(np.ceil(img_w / step))
        for i, j in _product(range(fh), range(fw)):
            for min_size in min_sizes[idx]:
                s_kx = min_size / img_w
                s_ky = min_size / img_h
                cx = (j + 0.5) / fw
                cy = (i + 0.5) / fh
                anchors.append([cx, cy, s_kx, s_ky])
    return np.clip(np.array(anchors, dtype=np.float32), 0.0, 1.0)


def _get_anchors():
    global _anchors_cache
    if _anchors_cache is None:
        _anchors_cache = _extract_anchors(
            LIGHTFACE_ANCHORS["min_sizes"], LIGHTFACE_ANCHORS["steps"])
    return _anchors_cache


def _head_layout(input_h, input_w):
    """Expected feature map -> (bbox channels, conf channels). The same channel
    count appears at two different scales, so the classification is derived
    from the declared anchor set instead of guessing from channels alone."""
    layout = {}
    for step, min_sizes in zip(LIGHTFACE_ANCHORS["steps"],
                               LIGHTFACE_ANCHORS["min_sizes"]):
        fh = int(np.ceil(input_h / step))
        fw = int(np.ceil(input_w / step))
        layout[(fh, fw)] = (4 * len(min_sizes), 2 * len(min_sizes))
    return layout


def _decode_boxes(box_deltas, anchors):
    """SSD decode with variances (10, 5) -> xyxy normalized. Port of
    FaceDetectionPostProc._decode_boxes."""
    scale_xy, scale_wh = _SCALE_FACTORS
    cx = anchors[:, 0] + box_deltas[:, 0] / scale_xy * anchors[:, 2]
    cy = anchors[:, 1] + box_deltas[:, 1] / scale_xy * anchors[:, 3]
    w = anchors[:, 2] * np.exp(box_deltas[:, 2] / scale_wh)
    h = anchors[:, 3] * np.exp(box_deltas[:, 3] / scale_wh)
    x1 = cx - w / 2
    y1 = cy - h / 2
    return np.stack([x1, y1, x1 + w, y1 + h], axis=1)


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _nms(boxes, scores, iou_thresh, max_det=200):
    """Plain greedy NMS on xyxy boxes (single face class)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = np.argsort(scores)[::-1][:max_det]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Full LightFace Slim decode. Returns (boxes, scores):
      boxes  (N, 4) xyxy in input-pixel space
      scores (N,)   softmaxed face confidence (background sliced off)
    obj_thresh / nms_thresh of 0 fall back to the preview defaults.
    """
    global _LIGHTFACE_LOGGED
    if hailo_output is None:
        return None, None
    if isinstance(hailo_output, dict):
        named = list(hailo_output.items())
    elif isinstance(hailo_output, (list, tuple)):
        named = [(str(i), t) for i, t in enumerate(hailo_output)]
    else:
        named = [("output", hailo_output)]

    layout = _head_layout(input_h, input_w)
    boxes_raw, confs_raw, shapes = [], [], []
    for name, tensor in named:
        shapes.append(f"{name}{tuple(np.asarray(tensor).shape)}")
        arr = np.asarray(tensor)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            continue
        fh, fw, c = arr.shape
        want = layout.get((fh, fw))
        if not want:
            continue
        if c == want[0]:
            boxes_raw.append(arr)
        elif c == want[1]:
            confs_raw.append(arr)

    if not _LIGHTFACE_LOGGED:
        # Log every vstream name/shape and the value ranges once, so a wrong
        # layout or a missing activation is visible on the first inference.
        def _rng(a):
            a = np.asarray(a, dtype=np.float32)
            return f"{float(a.min()):.3g}..{float(a.max()):.3g}"

        print("[LightFace] vstreams: " + "; ".join(shapes), flush=True)
        print(f"[LightFace] bbox_range={[_rng(b) for b in boxes_raw]} "
              f"conf_range={[_rng(c) for c in confs_raw]}", flush=True)
        _LIGHTFACE_LOGGED = True

    if len(boxes_raw) != len(layout) or len(confs_raw) != len(layout):
        print(f"[LightFace] unexpected output layout: {shapes}", flush=True)
        return None, None

    # Spatial order must match the anchor generation order (stride 8 first).
    def spatial_key(arr):
        return arr.shape[0] * arr.shape[1]
    boxes_raw.sort(key=spatial_key, reverse=True)
    confs_raw.sort(key=spatial_key, reverse=True)

    loc = np.concatenate([b.reshape(-1, 4) for b in boxes_raw], axis=0)
    conf = np.concatenate([c.reshape(-1, 2) for c in confs_raw], axis=0)
    anchors = _get_anchors()
    if not (loc.shape[0] == conf.shape[0] == anchors.shape[0]):
        print(f"[LightFace] anchor mismatch: loc {loc.shape} conf {conf.shape} "
              f"anchors {anchors.shape}", flush=True)
        return None, None

    scores = _softmax(conf)[:, 1]          # face class (slice off background)
    thres = obj_thresh if obj_thresh and obj_thresh > 0 else _LIGHTFACE_SCORE_THRES
    keep = scores > thres
    if not np.any(keep):
        return None, None

    boxes = _decode_boxes(loc[keep], anchors[keep])
    kept_scores = scores[keep]

    # Drop degenerate boxes before NMS.
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid):
        return None, None
    boxes, kept_scores = boxes[valid], kept_scores[valid]

    iou = nms_thresh if nms_thresh and nms_thresh > 0 else _LIGHTFACE_NMS_THRES
    order = _nms(boxes, kept_scores, iou)
    boxes, kept_scores = boxes[order], kept_scores[order]

    # Normalized -> input-pixel space, clipped to the input frame.
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * input_w, 0, input_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * input_h, 0, input_h)
    return boxes.astype(np.float32), kept_scores.astype(np.float32)


def unletterbox_boxes(boxes, lb_info):
    """Map xyxy boxes from the 240x320 aspect-preserving input back to the
    original frame. lb_info = (scale, pad_h, pad_w) from preprocess_frame:
    the frame was resized by `scale` (both axes) and padded bottom/right, so
    the inverse transform is a pure division by the resize scale."""
    if boxes is None or len(boxes) == 0:
        return boxes
    scale, _pad_h, _pad_w = lb_info
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] /= scale
    out[:, [1, 3]] /= scale
    return out


def draw_boxes(image, boxes, scores, lb_info=None, score_thresh=_LIGHTFACE_SCORE_THRES):
    """Draw face boxes (already mapped to frame pixels) + confidence labels."""
    if boxes is None or len(boxes) == 0:
        return
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"face {float(scores[i]):.2f}", (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def preprocess_frame(frame, co_helper):
    """Aspect-ratio-preserving resize + bottom/right pad to 240x320
    (matching the Model Zoo's face-detection preprocessing:
    _ar_preserving_resize_and_crop, padding_color=0), keeping the frame in
    BGR order - this compile has no in-graph bgr_to_rgb conversion and the
    model's original pipeline reads BGR.

    The HEF uses normalize_in_net (mean 127 / std 128), so the app feeds raw
    uint8 BGR. Returns (img, (scale, pad_h, pad_w)) for un-letterboxing.
    """
    ih, iw = IMG_SIZE[1], IMG_SIZE[0]
    fh, fw = frame.shape[:2]
    # Match tf.cond: scale by the limiting axis, then pad the other to the
    # target (bottom/right padding, color 0).
    scale = min(iw / fw, ih / fh)
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    img[:new_h, :new_w] = resized
    return img, (scale, ih - new_h, iw - new_w)


def inference_loop(cap, model, co_helper, is_video_file, target_fps):
    fps_counter = 0
    target_period = 1.0 / target_fps if target_fps > 0 else 0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            if video_analyzer.is_processing:
                time.sleep(1.0)
                next_time = time.time()
                continue
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            processed_img, lb_info = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time
            if outputs is not None:
                obj, nms = det_config.get()
                boxes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                if boxes is not None and len(boxes) > 0:
                    real_boxes = unletterbox_boxes(boxes, lb_info)
                    h, w = frame.shape[:2]
                    real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                    real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                    draw_boxes(frame, real_boxes, scores, score_thresh=obj)
            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'Hailo FPS: {fps_counter:.1f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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


def encode_loop(preview_w, preview_h, jpeg_quality):
    last_v = -1
    while not stop_event.is_set():
        frame, last_v = frame_buffer.wait_annotated(last_v, timeout=1.0)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if preview_w > 0 and preview_h > 0 and (w, h) != (preview_w, preview_h):
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        else:
            preview = frame
        ok, buf = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            frame_buffer.push_jpeg(buf.tobytes())


def main():
    parser = argparse.ArgumentParser(description='LightFace Slim on RPi5 + Hailo-8 (Web Preview Mode)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .hef model (Hailo Executable Format)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (default: 0). Use -1 to disable camera and run web-only mode.')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt file for dynamic category loading')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    parser.add_argument('--preview_width', type=int, default=1280, help='MJPEG preview width (0 to disable resize). Default 1280.')
    parser.add_argument('--preview_height', type=int, default=720, help='MJPEG preview height (0 to disable resize). Default 720.')
    parser.add_argument('--jpeg_quality', type=int, default=80, help='MJPEG preview JPEG quality 1-100. Default 80.')
    parser.add_argument('--cam_width', type=int, default=1280, help='Requested USB camera width. Default 1280.')
    parser.add_argument('--cam_height', type=int, default=720, help='Requested USB camera height. Default 720.')
    parser.add_argument('--target_fps', type=float, default=30.0, help='Cap live preview inference rate (fps). 0 = uncapped. Default 30.')
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available. Install the hailort wheel matching your driver version.")
        return

    if args.class_path:
        load_classes(args.class_path)

    global _global_model, _global_co_helper, IMG_SIZE
    model = HailoInfer(args.model_path)
    IMG_SIZE = (model.input_w, model.input_h)
    print(f"Model input size: {model.input_w}x{model.input_h}", flush=True)
    co_helper = COCO_test_helper(enable_letter_box=True)

    _global_model = model
    _global_co_helper = co_helper
    video_analyzer.set_engine(model, co_helper)

    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    print(f"Preview: {args.preview_width}x{args.preview_height} JPEG q={args.jpeg_quality} | target_fps={args.target_fps}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        print("Running in Video Analysis Mode. Access Web UI to process local videos.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        capture_source = cap
        is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        capture_source = None
        is_video_file = False

    if not cap.isOpened():
        print(f"Error: Cannot open video source (ID: {args.camera_id if not args.video_path else args.video_path})")
        return

    if not is_video_file:
        capture_source = LatestFrameReader(cap).start()

    inf_thread = threading.Thread(target=inference_loop,
                                  args=(capture_source, model, co_helper, is_video_file, args.target_fps),
                                  daemon=True)
    enc_thread = threading.Thread(target=encode_loop,
                                  args=(args.preview_width, args.preview_height, args.jpeg_quality),
                                  daemon=True)
    inf_thread.start()
    enc_thread.start()

    try:
        while inf_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        if not is_video_file:
            capture_source.stop()
        inf_thread.join(timeout=2)
        enc_thread.join(timeout=2)
        cap.release()
        model.release()

if __name__ == '__main__':
    main()
