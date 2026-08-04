import os
import sys
import cv2
import argparse
import time
import queue
import subprocess
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from typing import Optional, List

from py_utils.coco_utils import COCO_test_helper

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (512, 512)  # (width, height) must match the CenterPose HEF input

DEFAULT_CLASSES = ("person",)
NUM_KEYPOINTS = 17
KEYPOINT_SCORE_THRESH = 0.30

COCO_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
)

KEYPOINT_COLORS = np.array([
    [0, 255, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255],
    [0, 255, 0], [0, 255, 0], [0, 200, 255], [0, 200, 255],
    [255, 0, 255], [255, 0, 255], [255, 128, 0], [255, 128, 0],
    [255, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 0],
], dtype=np.uint8)

CLASSES = DEFAULT_CLASSES
_POSE_OUTPUT_LOGGED = False

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
                    print(f"Warning: No classes found in {path}, using default person class")
                    CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default person class")
        CLASSES = DEFAULT_CLASSES

class DetectionConfig:
    def __init__(self):
        self.obj_thresh = 0.25
        self.nms_thresh = 0.45
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
        """Try ffmpeg subprocess with libx264 ultrafast first — on Pi 5 this
        is ~5x faster than cv2's mp4v at 4K (40ms vs 150ms per frame).
        Falls back to cv2 mp4v if ffmpeg isn't installed (e.g., non-Docker run).

        Returns (writer, kind) where kind is 'ffmpeg' or 'mp4v'. The caller
        uses kind to decide whether to write via proc.stdin.write or out.write."""
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{fps}',
            '-i', '-',
            # ultrafast = simplest motion search; threads=0 lets x264 use all 4 cores.
            # NOT using -tune zerolatency: it forces single-thread sliced threading,
            # which on Pi 5 cuts throughput in half. For offline encoding we want
            # frame-level threading instead.
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '0',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
            print(f"[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, 'ffmpeg'
        except FileNotFoundError:
            pass

        # Fallback: cv2 mp4v (works without ffmpeg binary, but slow at 4K)
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

        # Single-threaded loop — empirical: on Pi 5 with 4K frames, the
        # producer/consumer pipeline experiment was 2-3x SLOWER than the
        # straight loop. Likely because 4K BGR frames (~24MB each) in a
        # Python queue trigger heavy GC, and libavcodec's mp4v encoder
        # doesn't play nicely with concurrent libav decoders on the same
        # process. The straight loop is the boring-but-fast choice here.
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.model and self.co_helper:
                    processed_img = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(processed_img)

                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes, classes, scores, keypoints = post_process_hailo(
                            outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0]
                        )
                        if boxes is not None:
                            real_boxes = self.co_helper.get_real_box(boxes)
                            real_keypoints = unletterbox_keypoints(keypoints, self.co_helper)
                            draw(frame, real_boxes, scores, classes, real_keypoints)

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

app = FastAPI(title="reComputer CenterPose RegNetX-800MF (RPi5 + Hailo-8)")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", 0.25), config.get("nms_thresh", 0.45))
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

@app.post("/api/models/centerpose_regnetx_800mf/predict")
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

        input_img = preprocess_frame(img, _global_co_helper)
        outputs = _global_model.run(input_img)

        current_obj_thresh, current_nms_thresh = det_config.get()
        target_conf = conf if conf is not None else current_obj_thresh
        target_iou = iou if iou is not None else current_nms_thresh

        boxes, classes, scores, keypoints = post_process_hailo(
            outputs, target_conf, target_iou, IMG_SIZE[1], IMG_SIZE[0]
        )

        predictions = []
        if boxes is not None and len(boxes) > 0:
            real_boxes = _global_co_helper.get_real_box(boxes)
            real_keypoints = unletterbox_keypoints(keypoints, _global_co_helper)
            for idx, (box, score, cl) in enumerate(zip(real_boxes, scores, classes)):
                points = []
                if real_keypoints is not None and idx < len(real_keypoints):
                    points = [
                        {"x": int(x), "y": int(y), "score": float(kp_score)}
                        for x, y, kp_score in real_keypoints[idx]
                        if np.isfinite(x) and np.isfinite(y) and kp_score >= KEYPOINT_SCORE_THRESH
                    ]
                predictions.append({
                    "class": CLASSES[cl],
                    "confidence": float(score),
                    "box": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3])
                    },
                    "keypoints": points
                })

        return {
            "success": True,
            "source": source_info,
            "predictions": predictions,
            "image": {
                "width": w,
                "height": h
            }
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    """Latest-frame buffer with version tracking.

    Two stages separated by version counters:
      - annotated: 4K BGR frame from the inference thread, waiting to be encoded.
                   Encode thread blocks on wait_annotated().
      - jpeg:      downscaled preview JPEG bytes, ready to ship to browsers.
                   MJPEG stream consumers block on wait_jpeg().

    `raw` mirrors the latest annotated frame for the realtime predict API.
    Consumers wake on a condvar instead of polling, so a slow network can't
    cause stale-frame pileup.
    """
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
    """Continuously read a live camera and expose only the newest frame.

    USB/V4L2 camera buffers can queue old frames when inference is slower than
    capture. A single-threaded read/infer loop then shows stale frames even if
    the web stream itself has no queue. This reader drains the camera in the
    background and lets inference always consume the latest available frame.
    """
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
                lambda: self.version > self._last_read_version or self._stopped,
                timeout=timeout
            )
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
        <title>reComputer CenterPose RegNetX-800MF</title>
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
          .tab-container { margin-top: 30px; }
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
          <h1>CenterPose RegNetX-800MF · RPi5 + Hailo-8</h1>

          <div class="tabs">
            <div class="tab active" onclick="showTab('realtime')">Real-time Pose</div>
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
                  <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25">
                  <span id="confValue" class="value-display">0.25</span>
                </div>
              </div>

              <div class="control-group">
                <label>IOU Threshold (built-in NMS)</label>
                <div class="slider-container">
                  <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                  <span id="iouValue" class="value-display">0.45</span>
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
                  <thead>
                    <tr><th>File Name</th><th>Action</th></tr>
                  </thead>
                  <tbody id="fileTableBody">
                  </tbody>
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

            if (tabId === 'realtime') {
                document.getElementById('streamImg').src = '/api/video_feed';
            } else {
                document.getElementById('streamImg').src = '';
                refreshFileList();
            }
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

            fetch('/api/config', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ obj_thresh, nms_thresh })
            });
          }

          confSlider.oninput = updateConfig;
          iouSlider.oninput = updateConfig;

          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh;
            iouSlider.value = data.nms_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2);
            iouValue.innerText = data.nms_thresh.toFixed(2);
          });

          async function uploadVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) return alert('Please select a file');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const btn = event.currentTarget;
            btn.disabled = true;
            btn.innerText = 'Uploading...';

            try {
                await fetch('/api/video/upload', { method: 'POST', body: formData });
                alert('Upload successful');
                refreshFileList();
            } catch (e) {
                alert('Upload failed');
            } finally {
                btn.disabled = false;
                btn.innerText = 'Upload';
            }
          }

          async function refreshFileList() {
            const res = await fetch('/api/video/list');
            const data = await res.json();
            const tbody = document.getElementById('fileTableBody');
            tbody.innerHTML = '';

            data.uploads.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Original)</td>
                    <td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>
                `;
                tbody.appendChild(tr);
            });

            data.outputs.forEach(f => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${f} (Analyzed)</td>
                    <td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>
                `;
                tbody.appendChild(tr);
            });
          }

          async function analyzeVideo(filename) {
            const formData = new FormData();
            formData.append('filename', filename);
            const res = await fetch('/api/video/analyze', { method: 'POST', body: formData });
            const data = await res.json();

            if (data.status === 'started') {
                startStatusPolling();
            } else {
                alert(data.message || 'Error starting analysis');
            }
          }

          let pollInterval;
          function startStatusPolling() {
            document.getElementById('processingArea').style.display = 'block';
            if (pollInterval) clearInterval(pollInterval);

            pollInterval = setInterval(async () => {
                const res = await fetch('/api/video/status');
                const data = await res.json();

                document.getElementById('currentFileName').innerText = data.current_video;
                document.getElementById('progressBar').style.width = data.progress + '%';
                document.getElementById('progressText').innerText = data.progress + '%';
                document.getElementById('errorText').innerText = data.error || '';

                if (!data.is_processing && data.progress === 100) {
                    clearInterval(pollInterval);
                    alert('Analysis completed!');
                    refreshFileList();
                } else if (!data.is_processing && data.error) {
                    clearInterval(pollInterval);
                }
            }, 1000);
          }

          fetch('/api/video/status').then(res => res.json()).then(data => {
            if (data.is_processing) startStatusPolling();
          });
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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nms(boxes, scores, iou_thresh, max_det=100):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def _iter_output_tensors(hailo_output):
    if isinstance(hailo_output, dict):
        for name, tensor in hailo_output.items():
            yield name, tensor
    elif isinstance(hailo_output, (list, tuple)):
        for idx, tensor in enumerate(hailo_output):
            yield str(idx), tensor
    else:
        yield "output", hailo_output


def _normalize_feature_tensor(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 2, 17, 34) and arr.shape[-1] not in (1, 2, 17, 34):
        arr = np.moveaxis(arr, 0, -1)
    return arr.astype(np.float32, copy=False)


def _output_number(name):
    """Return the numeric suffix from names such as ``.../conv63``."""
    digits = ""
    for char in reversed(str(name)):
        if char.isdigit():
            digits = char + digits
        elif digits:
            break
    return int(digits) if digits else -1


def _collect_centerpose_heads(hailo_output):
    """Map the six CenterPose vstreams to their semantic output heads.

    The v2.19 HEF exposes conv60..conv65. Channel counts uniquely identify
    hm (1), hps (34), and hm_hp (17); the three 2-channel heads are ordered as
    wh (conv61), reg (conv63), and hp_offset (conv65).
    """
    heads = {}
    two_channel = []
    seen = []
    for name, tensor in _iter_output_tensors(hailo_output):
        arr = _normalize_feature_tensor(tensor)
        seen.append(f"{name}:{tuple(np.asarray(tensor).shape)}->{tuple(arr.shape)}")
        if arr.ndim != 3:
            continue
        channels = arr.shape[-1]
        if channels == 1:
            heads["hm"] = arr
        elif channels == 34:
            heads["hps"] = arr
        elif channels == 17:
            heads["hm_hp"] = arr
        elif channels == 2:
            two_channel.append((_output_number(name), name, arr))

    explicit = {61: "wh", 63: "reg", 65: "hp_offset"}
    remaining = []
    for number, name, arr in two_channel:
        role = explicit.get(number)
        if role:
            heads[role] = arr
        else:
            remaining.append((number, name, arr))

    missing_roles = [role for role in ("wh", "reg", "hp_offset") if role not in heads]
    for role, (_, _, arr) in zip(missing_roles, sorted(remaining, key=lambda item: (item[0], item[1]))):
        heads[role] = arr
    return heads, seen


def _heatmap_nms(heatmap):
    pooled = cv2.dilate(heatmap, np.ones((3, 3), dtype=np.uint8))
    return heatmap * (heatmap >= pooled - 1e-7)


def _topk_2d(heatmap, k):
    flat = heatmap.reshape(-1)
    k = min(int(k), flat.size)
    if k <= 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(flat[indices])[::-1]]
    width = heatmap.shape[1]
    return flat[indices], (indices // width).astype(np.int32), (indices % width).astype(np.int32)


def _decode_centerpose(heads, obj_thresh, nms_thresh, input_h, input_w, max_det=100):
    required = ("hm", "wh", "hps", "reg")
    if not all(role in heads for role in required):
        return None, None, None, None

    center_heat = heads["hm"][..., 0]
    if center_heat.min() < 0.0 or center_heat.max() > 1.0:
        center_heat = _sigmoid(center_heat)
    center_heat = _heatmap_nms(center_heat)
    scores, ys, xs = _topk_2d(center_heat, max_det)
    valid = scores >= obj_thresh
    scores, ys, xs = scores[valid], ys[valid], xs[valid]
    if scores.size == 0:
        return None, None, None, None

    wh = np.maximum(heads["wh"][ys, xs], 0.0)
    reg = heads["reg"][ys, xs]
    centers = np.stack((xs, ys), axis=1).astype(np.float32) + reg
    hps = heads["hps"][ys, xs].reshape(-1, NUM_KEYPOINTS, 2)
    keypoint_xy = hps + centers[:, None, :]

    boxes = np.column_stack((
        centers[:, 0] - wh[:, 0] * 0.5,
        centers[:, 1] - wh[:, 1] * 0.5,
        centers[:, 0] + wh[:, 0] * 0.5,
        centers[:, 1] + wh[:, 1] * 0.5,
    )).astype(np.float32)
    keypoint_scores = np.repeat(scores[:, None], NUM_KEYPOINTS, axis=1).astype(np.float32)

    # CenterNet/CenterPose refines regressed joints with per-joint heatmap peaks.
    if "hm_hp" in heads:
        joint_heat = heads["hm_hp"]
        if joint_heat.min() < 0.0 or joint_heat.max() > 1.0:
            joint_heat = _sigmoid(joint_heat)
        hp_offset = heads.get("hp_offset")
        for joint in range(NUM_KEYPOINTS):
            joint_map = _heatmap_nms(joint_heat[..., joint])
            joint_scores, joint_ys, joint_xs = _topk_2d(joint_map, 40)
            keep = joint_scores >= 0.10
            joint_scores = joint_scores[keep]
            joint_ys = joint_ys[keep]
            joint_xs = joint_xs[keep]
            if joint_scores.size == 0:
                continue
            candidates = np.stack((joint_xs, joint_ys), axis=1).astype(np.float32)
            if hp_offset is not None:
                candidates += hp_offset[joint_ys, joint_xs]
            else:
                candidates += 0.5
            distances = np.linalg.norm(
                keypoint_xy[:, joint, None, :] - candidates[None, :, :], axis=2
            )
            nearest = np.argmin(distances, axis=1)
            chosen = candidates[nearest]
            chosen_scores = joint_scores[nearest]
            widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
            heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
            inside = (
                (chosen[:, 0] >= boxes[:, 0]) & (chosen[:, 0] <= boxes[:, 2]) &
                (chosen[:, 1] >= boxes[:, 1]) & (chosen[:, 1] <= boxes[:, 3])
            )
            close = distances[np.arange(len(scores)), nearest] <= 0.30 * np.maximum(widths, heights)
            refine = inside & close
            keypoint_xy[refine, joint] = chosen[refine]
            keypoint_scores[refine, joint] = chosen_scores[refine]

    feat_h, feat_w = center_heat.shape
    scale_x = input_w / float(feat_w)
    scale_y = input_h / float(feat_h)
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    keypoint_xy[:, :, 0] *= scale_x
    keypoint_xy[:, :, 1] *= scale_y
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h - 1)

    keep = _nms(boxes, scores, nms_thresh, max_det=max_det)
    if not keep:
        return None, None, None, None
    keypoints = np.concatenate((keypoint_xy, keypoint_scores[..., None]), axis=2)
    return (
        boxes[keep].astype(np.float32),
        np.zeros(len(keep), dtype=np.int32),
        scores[keep].astype(np.float32),
        keypoints[keep].astype(np.float32),
    )


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Decode CenterPose's six CenterNet-style HailoRT output heads."""
    global _POSE_OUTPUT_LOGGED

    if hailo_output is None:
        return None, None, None, None

    heads, seen = _collect_centerpose_heads(hailo_output)
    if not _POSE_OUTPUT_LOGGED:
        print(f"[CenterPose] outputs: {'; '.join(seen)}", flush=True)
        _POSE_OUTPUT_LOGGED = True
    return _decode_centerpose(heads, obj_thresh, nms_thresh, input_h, input_w)


def unletterbox_keypoints(keypoints, co_helper):
    if keypoints is None or not hasattr(co_helper, "letter_box_info_list"):
        return keypoints
    info = co_helper.letter_box_info_list[-1]
    points = keypoints.copy()
    points[:, :, 0] = np.clip((points[:, :, 0] - info.dw) / info.w_ratio, 0, info.origin_shape[1])
    points[:, :, 1] = np.clip((points[:, :, 1] - info.dh) / info.h_ratio, 0, info.origin_shape[0])
    return points


def draw(image, boxes, scores, classes, keypoints=None):
    for idx, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
        top, left, right, bottom = [int(_b) for _b in box]
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {score:.2f}',
                    (top, max(20, left - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if keypoints is None or idx >= len(keypoints):
            continue

        pts = keypoints[idx]
        visible = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & (pts[:, 2] >= KEYPOINT_SCORE_THRESH)
        for a, b in COCO_SKELETON:
            if visible[a] and visible[b]:
                pa = (int(pts[a, 0]), int(pts[a, 1]))
                pb = (int(pts[b, 0]), int(pts[b, 1]))
                cv2.line(image, pa, pb, (0, 255, 0), 2)
        for kp_idx, (x, y, kp_score) in enumerate(pts):
            if visible[kp_idx]:
                color = tuple(int(v) for v in KEYPOINT_COLORS[kp_idx])
                cv2.circle(image, (int(x), int(y)), 3, color, -1)


def preprocess_frame(frame, co_helper):
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img = co_helper.letter_box(im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    # This CenterPose HEF was compiled with the original OpenCV/BGR
    # normalization ([104.04, 113.985, 119.85]); keep BGR channel order.
    return img

def inference_loop(cap, model, co_helper, is_video_file, target_fps):
    """Capture + preprocess + Hailo inference + draw on full-res frame.
    Pushes annotated frames into frame_buffer; the encode thread takes over from there.

    `target_fps` caps the loop rate so the preview path doesn't starve other
    Hailo/CPU consumers (notably VideoAnalyzer). Hailo can run 100+ fps but
    no human needs a 100 fps preview, and the leftover budget keeps offline
    video analysis healthy. Pass 0 to disable the cap.

    When VideoAnalyzer is running, the loop drops to 1 fps automatically —
    a frozen preview is fine for a few seconds, and the freed CPU/Hailo
    cycles roughly halve the analyze wall time."""
    fps_counter = 0
    target_period = 1.0 / target_fps if target_fps > 0 else 0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            # Yield to VideoAnalyzer when it's running.
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

            processed_img = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time

            if outputs is not None:
                obj, nms = det_config.get()
                boxes, classes, scores, keypoints = post_process_hailo(
                    outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0]
                )
                if boxes is not None:
                    real_boxes = co_helper.get_real_box(boxes)
                    real_keypoints = unletterbox_keypoints(keypoints, co_helper)
                    draw(frame, real_boxes, scores, classes, real_keypoints)

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
    """Take the latest annotated frame, resize for preview, JPEG-encode, publish.
    Runs at its own pace so a slow encode never back-pressures inference."""
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
    parser = argparse.ArgumentParser(description='CenterPose RegNetX-800MF on RPi5 + Hailo-8 (Web Preview Mode)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .hef model (Hailo Executable Format)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (default: 0 for /dev/video0). Use -1 to disable camera and run web-only mode.')
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

    global _global_model, _global_co_helper
    model = HailoInfer(args.model_path)
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
        # Ask the camera to deliver MJPG so USB bandwidth isn't wasted on raw
        # YUYV. Most modern USB webcams do MJPG natively; if not, V4L2 quietly
        # falls back to YUYV and we just lose this win — no error.
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
