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

# Demo defaults. The HEF runs NMS on-chip (HPP: sigmoid + nms on device), so
# nms_thresh is accepted for API parity but ignored; obj_thresh filters the
# parsed detections (the in-HEF score threshold nms_scores_th=0.2 is compiled
# into the model). 0.25 keeps the live preview readable; lower the slider to
# inspect lower-confidence boxes.
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height) — overridden at runtime from the .hef

# YOLOv7 detects the standard 80 COCO classes (nms_config: classes=80,
# meta_arch=yolov5 HPP NMS). cls_id from the NMS buffer indexes this tuple
# directly in COCO class order; the Model Zoo evaluation uses labels_offset=1
# for COCO category ids.
DEFAULT_CLASSES = (
    "person",          # 0
    "bicycle",         # 1
    "car",             # 2
    "motorcycle",      # 3
    "airplane",        # 4
    "bus",             # 5
    "train",           # 6
    "truck",           # 7
    "boat",            # 8
    "traffic light",   # 9
    "fire hydrant",    # 10
    "stop sign",       # 11
    "parking meter",   # 12
    "bench",           # 13
    "bird",            # 14
    "cat",             # 15
    "dog",             # 16
    "horse",           # 17
    "sheep",           # 18
    "cow",             # 19
    "elephant",        # 20
    "bear",            # 21
    "zebra",           # 22
    "giraffe",         # 23
    "backpack",        # 24
    "umbrella",        # 25
    "handbag",         # 26
    "tie",             # 27
    "suitcase",        # 28
    "frisbee",         # 29
    "skis",            # 30
    "snowboard",       # 31
    "sports ball",     # 32
    "kite",            # 33
    "baseball bat",    # 34
    "baseball glove",  # 35
    "skateboard",      # 36
    "surfboard",       # 37
    "tennis racket",   # 38
    "bottle",          # 39
    "wine glass",      # 40
    "cup",             # 41
    "fork",            # 42
    "knife",           # 43
    "spoon",           # 44
    "bowl",            # 45
    "banana",          # 46
    "apple",           # 47
    "sandwich",        # 48
    "orange",          # 49
    "broccoli",        # 50
    "carrot",          # 51
    "hot dog",         # 52
    "pizza",           # 53
    "donut",           # 54
    "cake",            # 55
    "chair",           # 56
    "couch",           # 57
    "potted plant",    # 58
    "bed",             # 59
    "dining table",    # 60
    "toilet",          # 61
    "tv",              # 62
    "laptop",          # 63
    "mouse",           # 64
    "remote",          # 65
    "keyboard",        # 66
    "cell phone",      # 67
    "microwave",       # 68
    "oven",            # 69
    "toaster",         # 70
    "sink",            # 71
    "refrigerator",    # 72
    "book",            # 73
    "clock",           # 74
    "vase",            # 75
    "scissors",        # 76
    "teddy bear",      # 77
    "hair drier",      # 78
    "toothbrush",      # 79
)

CLASSES = DEFAULT_CLASSES
_OUTPUT_LOGGED = True   # Disable verbose raw NMS diagnostics in production.

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
                        boxes, classes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                        if boxes is not None and len(boxes) > 0:
                            real_boxes = unletterbox_boxes(boxes, lb_info)
                            h, w = frame.shape[:2]
                            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                            draw(frame, real_boxes, scores, classes)
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

app = FastAPI(title="reComputer YOLOv7 Hailo-8")

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

@app.post("/api/models/yolov7/predict")
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
        boxes, classes, scores = post_process_hailo(outputs, target_conf, target_iou, IMG_SIZE[1], IMG_SIZE[0])
        predictions = []
        if boxes is not None and len(boxes) > 0:
            real_boxes = unletterbox_boxes(boxes, lb_info)
            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
            for box, score, cl in zip(real_boxes, scores, classes):
                cl = int(cl)
                if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
                    continue
                predictions.append({
                    "class": CLASSES[cl],
                    "confidence": float(score),
                    "box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])}
                })
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
        <title>reComputer YOLOv7 · Hailo-8</title>
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
          <h1>YOLOv7 · RPi5 + Hailo-8</h1>
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
                  <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25">
                  <span id="confValue" class="value-display">0.25</span>
                </div>
              </div>
              <div class="control-group">
                <label>IOU Threshold (NMS on-chip — slider has no effect)</label>
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
# YOLOv7 post-processing (on-chip NMS / HPP)
#
# The HEF runs NMS on-chip (HPP: sigmoid + nms on device; 640x640 input, 80
# COCO classes, max 80 boxes per class) and exposes a single output vstream.
# The parser nodes are the YOLOv7 detection heads, so the host runs no box
# decode and no NMS — it only reads the post-NMS result.
# Rows are [ymin, xmin, ymax, xmax, score], normalized to [0,1] of the
# letterboxed network input; the caller scales to input pixels and
# un-letterboxes to the original frame.
#
# ALLS recipe (cfg/alls/generic/yolov7.alls):
#   change_output_activation(sigmoid)
#   nms_postprocess(cfg/postprocess_config/yolov7_nms_config.json,
#                   meta_arch=yolov5, engine=cpu)
#   -> nms_scores_th 0.2, nms_iou_th 0.6, max_proposals_per_class 80
#
# HailoRT 4.23 (Hailo-8) returns the raw NMS-by-class buffer as a flat
# 1-D FLOAT32 array in COMPACT layout: per class one integer-valued
# detection count followed by only that many (ymin, xmin, ymax, xmax, score)
# tuples — no padding to the 80-box maximum. Fixed-stride assumptions
# (e.g. 401 floats per class) do not hold: they misalign every class after
# the first and silently drop most detections. _per_class_iterable also
# keeps handlers for the layouts other HailoRT builds can expose (ragged
# NMS-by-score object arrays, dense (1,C,5,D) / (1,C,D,5)) so the parser
# stays portable. The first-inference log prints the raw type/shape so the
# layout can be verified on hardware (SOP §10).
# ---------------------------------------------------------------------------

def _first_output(hailo_output):
    if isinstance(hailo_output, dict):
        return next(iter(hailo_output.values())) if hailo_output else None
    if isinstance(hailo_output, (list, tuple)) and hailo_output:
        return hailo_output[0]
    return hailo_output


def _is_ragged(output):
    """True when `output` is already the bare per-class list of an NMS-by-score
    result (no vstream dict / batch wrapper around it). Without this guard
    _first_output would unwrap class 0 and silently lose every detection."""
    if not isinstance(output, (list, tuple)) or not output:
        return False
    if len(output) > NUM_CLASSES:
        return False
    for element in output:
        if isinstance(element, (list, tuple)):
            continue
        try:
            arr = np.asarray(element)
        except ValueError:
            continue
        if arr.size and not (arr.ndim == 2 and arr.shape[-1] == 5):
            return False
    return True


NUM_CLASSES = len(DEFAULT_CLASSES)   # 80 — matches the HEF's on-chip NMS classes
MAX_BOXES_PER_CLASS = 80             # HEF HPP limit (max_proposals_per_class, 80x5x80)
_EMPTY_PER_CLASS = ()                # sentinel: no detections could be parsed


def _parse_compact_nms_by_class(buf):
    """Parse the flat HailoRT NMS-by-class buffer (Hailo-8, HailoRT 4.23):
    per class one integer-valued detection count followed by exactly that
    many (ymin, xmin, ymax, xmax, score) rows, classes back to back with no
    padding. The host buffer may be sized for the worst case, so parsing
    stops at the first implausible class header — the padding tail is not
    data. Returns a list of (N, 5) float arrays, one per parsed class, or
    None when nothing validates."""
    classes = []
    offset = 0
    n = buf.shape[0]
    while offset < n and len(classes) < NUM_CLASSES:
        raw = float(buf[offset])
        if not np.isfinite(raw):
            break
        count = int(round(raw))
        if count < 0 or count > MAX_BOXES_PER_CLASS:
            break
        end = offset + 1 + count * 5
        if end > n:
            break
        if count:
            dets = buf[offset + 1:end].reshape(count, 5)
            # On-chip NMS rows are normalized: coordinates and sigmoid scores
            # all fall in [0,1] (SOP §14). Values outside mean the header was
            # misaligned padding — stop before it poisons the output.
            if (not np.all(np.isfinite(dets))
                    or dets.min() < -1e-3 or dets.max() > 1.0 + 1e-3):
                break
            classes.append(dets)
        else:
            classes.append(np.zeros((0, 5), dtype=buf.dtype))
        offset = end
    return classes if classes else None


def _per_class_iterable(output):
    """Return an object indexable by cls_id, each yielding (N, 5) detection
    rows [ymin, xmin, ymax, xmax, score]. Handles the HailoRT NMS layouts:
    the flat compact NMS-by-class buffer (Hailo-8, HailoRT 4.23),
    ragged/object (NMS-by-score), and dense float32 (1,C,5,D) or (1,C,D,5)."""
    # Ragged (NMS-by-score): output is shape (1, num_classes) where each element
    # is a per-class (N, 5) array with N varying by class. np.asarray raises
    # ValueError on this inhomogeneous shape, so guard it and take output[0]
    # (the per-class iterable) directly — same path R20 yolov8 uses.
    try:
        arr = np.asarray(output)
    except ValueError:
        # HailoRT 4.23 get_buffer() returns the per-class ragged list
        # directly (80 arrays, without an outer batch dimension). Only
        # unwrap when a backend explicitly adds a single batch wrapper.
        if (isinstance(output, (list, tuple)) and len(output) == 1
                and isinstance(output[0], (list, tuple))
                and len(output[0]) == NUM_CLASSES):
            return output[0]
        return output
    if arr.dtype == object:
        if arr.ndim >= 2 and arr.shape[0] == 1:
            return arr[0]
        return output
    # Dense float32.
    if arr.ndim == 4:
        # (batch, num_classes, A, B). Ensure (batch, num_classes, max_dets, 5).
        if arr.shape[2] == 5 and arr.shape[3] != 5:
            arr = np.transpose(arr, (0, 1, 3, 2))
        return arr[0]  # (num_classes, max_dets, 5)
    if arr.ndim == 3:
        # (num_classes, A, B). Ensure (num_classes, max_dets, 5).
        if arr.shape[1] == 5 and arr.shape[2] != 5:
            arr = np.transpose(arr, (0, 2, 1))
        return arr
    # Flat 1-D/2-D float32: the raw HPP NMS-by-class buffer, compact packing.
    if arr.ndim <= 2:
        parsed = _parse_compact_nms_by_class(arr.ravel())
        if parsed is not None:
            return parsed
        print(f"[YOLOv7] flat output shape {arr.shape} did not validate as "
              f"compact NMS-by-class; no detections parsed", flush=True)
        return _EMPTY_PER_CLASS
    # Fallback: assume output[0] is already the per-class iterable.
    return output[0]


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Parse the on-chip NMS output into boxes/classes/scores.

    Returns (boxes, classes, scores) where boxes are xyxy in input-pixel space
    (the letterboxed network input); the caller un-letterboxes to the original
    frame. nms_thresh is accepted for API parity but ignored (NMS is on-chip).
    """
    global _OUTPUT_LOGGED

    if hailo_output is None:
        return None, None, None

    output = hailo_output if _is_ragged(hailo_output) else _first_output(hailo_output)
    per_class = _per_class_iterable(output)

    if not _OUTPUT_LOGGED:
        # SOP §14: log the output range once — NMS boxes must be in [0,1]
        # (normalized) and scores in [0,1]. Tens-of-thousands values mean the
        # quantized encoding leaked (executor must request FLOAT32).
        try:
            raw = np.asarray(output)
            shape_str, dtype_str = str(raw.shape), str(raw.dtype)
            try:
                rng_s = f", range=[{np.nanmin(raw):.2f}, {np.nanmax(raw):.2f}]"
            except (TypeError, ValueError):
                rng_s = ""
        except ValueError:
            shape_str, dtype_str, rng_s = "ragged (NMS-by-score)", "object", ""
        print(
            f"[YOLOv7] raw output type={type(output).__name__}, "
            f"shape={shape_str}, dtype={dtype_str}{rng_s}",
            flush=True,
        )
        # Per-class detection dump (SOP §14): first valid rows per layout.
        try:
            per_class_probe = _per_class_iterable(output)
            n_shown = 0
            for cls_id, dets in enumerate(per_class_probe):
                if dets is None:
                    continue
                d = np.asarray(dets)
                if d.size == 0 or d.ndim == 0:
                    continue
                if d.ndim == 1:
                    d = d.reshape(-1, 5)
                valid = d[d[..., 4] > 0.1] if d.shape[-1] == 5 else d
                if valid.shape[0]:
                    print(f"[YOLOv7] cls{cls_id}: {valid.shape[0]} rows, "
                          f"first={np.round(valid[0], 3).tolist()}", flush=True)
                    n_shown += 1
                if n_shown >= 5:
                    break
            if n_shown == 0:
                # Nothing above 0.1 — dump raw rows regardless of score so
                # the log shows whether scores are pre-sigmoid logits or
                # normalized probabilities, and whether boxes look sane.
                shown = 0
                for cls_id, dets in enumerate(per_class_probe):
                    if dets is None:
                        continue
                    d = np.asarray(dets)
                    if d.size == 0 or d.ndim == 0:
                        continue
                    if d.ndim == 1:
                        d = d.reshape(-1, 5)
                    for r in range(min(2, d.shape[0])):
                        print(f"[YOLOv7] RAW cls{cls_id}[{r}]: "
                              f"{np.round(d[r], 4).tolist()}", flush=True)
                        shown += 1
                        if shown >= 8:
                            break
                    if shown >= 8:
                        break
                if shown == 0:
                    print("[YOLOv7] every class returned empty detection "
                          "lists from HailoRT", flush=True)
        except Exception as exc:
            print(f"[YOLOv7] probe error: {exc}", flush=True)
        _OUTPUT_LOGGED = True

    boxes, classes, scores = [], [], []
    for cls_id, dets in enumerate(per_class):
        if dets is None:
            continue
        dets = np.asarray(dets)
        if dets.size == 0 or dets.ndim == 0:
            continue
        if dets.ndim == 1:
            dets = dets.reshape(-1, 5)
        if dets.shape[-1] != 5:
            continue
        for det in dets:
            score = float(det[4])
            if score < obj_thresh:
                continue
            ymin, xmin, ymax, xmax = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
            boxes.append([xmin * input_w, ymin * input_h, xmax * input_w, ymax * input_h])
            classes.append(int(cls_id))
            scores.append(score)

    if not boxes:
        return None, None, None
    return (np.array(boxes, dtype=np.float32),
            np.array(classes, dtype=np.int32),
            np.array(scores, dtype=np.float32))


def unletterbox_boxes(boxes, lb_info):
    """Map xyxy boxes from the letterboxed network input back to the original
    frame. `lb_info` is the (ratio, dw, dh) captured for this exact frame by
    preprocess_frame — independent of the shared co_helper list, which races
    across threads (live preview + VideoAnalyzer) per the STDC1 note."""
    if boxes is None or len(boxes) == 0:
        return boxes
    ratio, dw, dh = lb_info
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    return out


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        cl = int(cl)
        if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'{CLASSES[cl]} {float(score):.2f}',
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def preprocess_frame(frame, co_helper):
    """Letterbox + BGR to RGB. Returns (img, lb_info) where lb_info captures the
    exact ratio + padding for un-letterboxing independent of shared co_helper
    state.

    The HEF uses normalize_in_net with ImageNet RGB mean/std and
    padding_color=114, so letterbox pads with gray (114) and the app feeds raw
    uint8 RGB pixels — no manual normalization.
    """
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img, ratio, (dw, dh) = co_helper.letter_box(
        im=frame.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(114, 114, 114),
        info_need=True,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (ratio, dw, dh)

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
                boxes, classes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                if boxes is not None and len(boxes) > 0:
                    real_boxes = unletterbox_boxes(boxes, lb_info)
                    h, w = frame.shape[:2]
                    real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                    real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                    draw(frame, real_boxes, scores, classes)
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
    parser = argparse.ArgumentParser(description='YOLOv7 on RPi5 + Hailo-8 (Web Preview Mode)')
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
