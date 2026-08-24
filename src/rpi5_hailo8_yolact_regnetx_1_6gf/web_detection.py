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

# YOLACT performs its own decode + Fast NMS + mask assembly on the CPU
# (base/yolact.yaml): score_threshold=0.05 (Detect pre-filter),
# nms_iou_thresh=0.5. OBJ_THRESH feeds the Detect pre-filter and the final
# visualization cut; NMS_THRESH feeds the Fast NMS step.
OBJ_THRESH = 0.25
NMS_THRESH = 0.5
IMG_SIZE = (512, 512)  # (width, height) — overridden at runtime from the .hef

# Anchor configuration — verbatim from base/yolact.yaml.
# 5 FPN feature maps (stride 8..128), each with 3 scales x 3 aspect ratios:
# 9 priors/cell x (64^2+32^2+16^2+8^2+4^2) = 49,104 priors total, in
# normalized center-size notation over the 512x512 input.
YOLACT_ANCHORS = {
    "feature_map": [64, 32, 16, 8, 4],
    "scales": [
        [24.0, 30.238105197476955, 38.097625247236785],
        [48.0, 60.47621039495391, 76.19525049447357],
        [96.0, 120.95242078990782, 152.39050098894714],
        [192.0, 241.90484157981564, 304.7810019778943],
        [384.0, 483.8096831596313, 609.5620039557886],
    ],
    "aspect_ratios": [1.0, 0.5, 2.0],
}
NUM_CLASSES = 81  # 80 COCO + background (class 0)
NUM_MASK_COEFFS = 32
_PROTO_SIZE = 128
_TOP_K = 200
_MAX_NUM_DETECTIONS = 100

# YOLACT conf heads cover classes 0..79 (model class index, background
# excluded after the softmax). The official visualization indexes
# CLASS_NAMES_COCO directly by class index — verbatim order below
# (hailo_model_zoo/core/datasets/datasets_info.py, 80 entries).
DEFAULT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
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
                        boxes, classes, scores, masks = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                        if boxes is not None and len(boxes) > 0:
                            real_boxes = unletterbox_boxes(boxes, lb_info)
                            h, w = frame.shape[:2]
                            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                            draw(frame, real_boxes, scores, classes, masks)
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

app = FastAPI(title="reComputer YOLACT RegNetX-1.6GF Hailo-8")

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

@app.post("/api/models/yolact_regnetx_1_6gf/predict")
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
        boxes, classes, scores, masks = post_process_hailo(outputs, target_conf, target_iou, IMG_SIZE[1], IMG_SIZE[0])
        predictions = []
        if boxes is not None and len(boxes) > 0:
            real_boxes = unletterbox_boxes(boxes, lb_info)
            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
            mask_areas = []
            if masks is not None and len(masks):
                mask_areas = [float(np.count_nonzero(m > 0.5)) for m in masks]
            for i, (box, score, cl) in enumerate(zip(real_boxes, scores, classes)):
                cl = int(cl)
                if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
                    continue
                mask_area = None
                if mask_areas and i < len(mask_areas):
                    mask_area = int(mask_areas[i])
                predictions.append({
                    "class": CLASSES[cl],
                    "confidence": float(score),
                    "box": {"x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3])},
                    "mask_area_px": mask_area,
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
        <title>reComputer YOLACT · Hailo-8</title>
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
          <h1>YOLACT RegNetX-1.6GF · RPi5 + Hailo-8</h1>
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
                <label>NMS IOU Threshold (CPU Fast NMS)</label>
                <div class="slider-container">
                  <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                  <span id="iouValue" class="value-display">0.50</span>
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
# YOLACT post-processing (CPU decode + Fast NMS + mask assembly)
#
# Ported 1:1 from hailo_model_zoo v2.19.0
# (core/postprocessing/instance_segmentation_postprocessing.py, meta_arch
# "yolact"). The HEF exposes 16 raw heads; there is no on-chip NMS:
#
#   proto      (1, 128, 128, 32)  prototype masks
#   5 scales (feature maps 64/32/16/8/4), each with:
#     bbox  (1, F, F, 36)         4 coords x 9 priors
#     mask  (1, F, F, 288)        32 coefficients x 9 priors
#     conf  (1, F, F, 729)        81 classes x 9 priors
#
# All outputs are in normalized [0,1] coordinates over the 512x512 input,
# except proto/coeffs which live on the 128x128 proto grid. The pipeline:
# _make_priors (49,104 anchors) -> SSD _decode -> per-instance argmax class
# -> Fast NMS (per-class top-200, class-aware overlap) -> finalize with
# score_thresh -> masks = sigmoid(proto @ coeffs^T) -> crop to box.
# ---------------------------------------------------------------------------

_priors_cache = None


def _make_priors(anchors, img_size):
    """Generate all anchors in normalized center-size notation.
    Order must match the reshape convention of the HEF heads: for each feature
    map cell (y-major, x-minor) then scale then aspect ratio."""
    from itertools import product
    from math import sqrt

    priors = []
    square_anchors = True if len(anchors["scales"][0]) == 1 else False
    for conv_size, pred_scale in zip(anchors["feature_map"], anchors["scales"]):
        prior_data = []
        for j, i in product(range(conv_size), range(conv_size)):
            # +0.5 because priors are in center-size notation
            x = (i + 0.5) / conv_size
            y = (j + 0.5) / conv_size
            for scale in pred_scale:
                for ar in anchors["aspect_ratios"]:
                    ar = sqrt(ar)
                    w = scale * ar / img_size
                    h = w if square_anchors else scale / ar / img_size
                    prior_data += [x, y, w, h]
        prior_data = np.reshape(prior_data, (-1, 4))
        priors.append(prior_data)
    return np.concatenate(priors, axis=-2)


def _get_priors():
    global _priors_cache
    if _priors_cache is None:
        _priors_cache = _make_priors(YOLACT_ANCHORS, IMG_SIZE[0])
    return _priors_cache


def _decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
         priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _intersect(box_a, box_b):
    max_xy = np.minimum(np.expand_dims(box_a[:, :, 2:], axis=2),
                        np.expand_dims(box_b[:, :, 2:], axis=1))
    min_xy = np.maximum(np.expand_dims(box_a[:, :, :2], axis=2),
                        np.expand_dims(box_b[:, :, :2], axis=1))
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def _jaccard(box_a, box_b, iscrowd=False):
    if len(box_a.shape) == 2:
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        use_batch = False
    else:
        use_batch = True

    inter = _intersect(box_a, box_b)
    area_a = np.expand_dims(
        (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), axis=2)
    area_b = np.expand_dims(
        (box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), axis=1)
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else np.squeeze(out, axis=0)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _softmax(x):
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def _sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = np.array(_x1, np.float64)
        _x2 = np.array(_x2, np.float64)
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=None)
    x2 = np.clip(x2 + padding, a_max=img_size, a_min=None)
    return x1, x2


def _crop(masks, boxes, padding=1):
    h, w, n = masks.shape
    x1, x2 = _sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = _sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = np.reshape(np.arange(w), (1, -1, 1))
    cols = np.reshape(np.arange(h), (-1, 1, 1))

    masks_left = rows >= np.reshape(x1, (1, 1, -1))
    masks_right = rows < np.reshape(x2, (1, 1, -1))
    masks_up = cols >= np.reshape(y1, (1, 1, -1))
    masks_down = cols < np.reshape(y2, (1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * (1.0 * crop_mask)


class YolactDetect:
    """Fast-NMS based detection, ported from the Model Zoo Detect class."""

    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh):
        self._num_classes = num_classes
        self._top_k = top_k
        self._nms_thresh = nms_thresh
        self._conf_thresh = conf_thresh

    def _fast_nms(self, boxes, masks, scores, iou_threshold=0.5, top_k=200):
        max_num_detections = _MAX_NUM_DETECTIONS
        idx = np.flip(np.argsort(scores, axis=1), axis=1)
        scores = np.flip(np.sort(scores, axis=1), axis=1)

        idx = idx[:, :top_k]
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.shape

        boxes = np.reshape(boxes[idx[:]], (num_classes, num_dets, 4))
        masks = np.reshape(masks[idx[:]], (num_classes, num_dets, -1))

        iou = _jaccard(boxes, boxes)
        iou = np.triu(iou, 1)
        iou_max = np.amax(iou, axis=1)

        # Keep detections with no higher-IoU competitor within the same class,
        # then merge across classes and keep the top max_num_detections.
        keep = iou_max <= iou_threshold

        classes = np.arange(num_classes)[:, None]
        classes = np.reshape(np.repeat(classes, keep.shape[1]), (num_classes, keep.shape[1]))
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        idx = np.flip(np.argsort(scores, axis=0), axis=0)
        scores = np.flip(np.sort(scores, axis=0), axis=0)
        idx = idx[:max_num_detections]
        scores = scores[:max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def __call__(self, loc_data, proto_data, conf_data, mask_data, prior_data):
        """loc/conf/mask_data: (num_priors, ...) per head; returns dict with
        detection_boxes/mask/detection_classes/detection_scores(/proto)."""
        num_priors = prior_data.shape[0]
        conf_preds = np.transpose(
            np.reshape(conf_data, (num_priors, self._num_classes)), (1, 0))

        # Per-instance argmax class, excluding background (class 0).
        cur_scores = conf_preds[1:, :]
        conf_scores = np.amax(cur_scores, axis=0)

        keep = conf_scores > self._conf_thresh
        scores = cur_scores[:, keep]
        boxes = _decode(loc_data[keep, :], prior_data[keep, :])
        masks = mask_data[keep, :]

        if scores.shape[1] == 0:
            return None

        boxes, masks, classes, scores = self._fast_nms(
            boxes, masks, scores, self._nms_thresh, self._top_k)

        result = {"detection_boxes": boxes, "mask": masks,
                  "detection_classes": classes, "detection_scores": scores}
        if proto_data is not None and len(classes):
            result["proto"] = proto_data
        return result


class YolactPostprocess:
    """End-to-end port of yolact_postprocessing: head reassembly from the 16
    HEF outputs, Detect, and mask finalization (sigmoid + crop)."""

    # Output-tensor classification table: keyed on the per-tensor channel
    # count and spatial size, matched against the YAML output_shape list.
    _HEAD_SPECS = {
        (128, 32): "proto",
        (64, 36): "bbox", (64, 288): "mask", (64, 729): "conf",
        (32, 36): "bbox", (32, 288): "mask", (32, 729): "conf",
        (16, 36): "bbox", (16, 288): "mask", (16, 729): "conf",
        (8, 36): "bbox", (8, 288): "mask", (8, 729): "conf",
        (4, 36): "bbox", (4, 288): "mask", (4, 729): "conf",
    }
    # Fast-NMS needs boxes/masks/conf per (descending) feature-map size.
    _HEAD_ORDER = [(64, "bbox"), (64, "mask"), (64, "conf"),
                   (32, "bbox"), (32, "mask"), (32, "conf"),
                   (16, "bbox"), (16, "mask"), (16, "conf"),
                   (8, "bbox"), (8, "mask"), (8, "conf"),
                   (4, "bbox"), (4, "mask"), (4, "conf")]

    def __init__(self, score_threshold, nms_iou_thresh):
        self._score_threshold = score_threshold
        self._detect = YolactDetect(
            num_classes=NUM_CLASSES, top_k=_TOP_K,
            conf_thresh=score_threshold, nms_thresh=nms_iou_thresh)

    def __call__(self, outputs):
        """outputs: dict {vstream_name: ndarray} or list from HailoRT.
        Returns (boxes, classes, scores, masks) with:
          boxes   (N, 4) xyxy in input-pixel space
          classes (N,)  0..79 COCO class indices
          scores  (N,)  softmaxed confidences
          masks   (N, 128, 128) float masks in the 512x512 input space
        """
        if isinstance(outputs, dict):
            endnodes = list(outputs.values())
        elif isinstance(outputs, (list, tuple)):
            endnodes = list(outputs)
        else:
            endnodes = [outputs]

        if not endnodes:
            return None, None, None, None

        # Classify heads by (spatial, channels) — independent of dict order.
        heads = {}
        for e in endnodes:
            arr = np.asarray(e)
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim != 3:
                continue
            h, w, c = arr.shape
            key = (h, c)
            role = self._HEAD_SPECS.get(key)
            if role is None:
                continue
            heads.setdefault((h, role), []).append(arr)
        try:
            proto = heads[(128, "proto")][0]
            locs = [heads[(f, "bbox")][0] for f, r in self._HEAD_ORDER if r == "bbox"]
            masks = [heads[(f, "mask")][0] for f, r in self._HEAD_ORDER if r == "mask"]
            confs = [heads[(f, "conf")][0] for f, r in self._HEAD_ORDER if r == "conf"]
        except (KeyError, IndexError):
            print(f"[YOLACT] unexpected output layout: {[a.shape for a in endnodes]}", flush=True)
            return None, None, None, None

        priors = _get_priors()
        num_priors = priors.shape[0]

        loc = np.concatenate([np.reshape(a, (-1, 4)) for a in locs], axis=0)
        conf = np.concatenate([np.reshape(a, (-1, NUM_CLASSES)) for a in confs], axis=0)
        mask_coeff = np.concatenate([np.reshape(a, (-1, NUM_MASK_COEFFS)) for a in masks], axis=0)
        assert loc.shape[0] == conf.shape[0] == mask_coeff.shape[0] == num_priors, (
            loc.shape, conf.shape, mask_coeff.shape, num_priors)
        det_output = self._detect(loc, proto, _softmax(conf), mask_coeff, priors)

        if det_output is None:
            return None, None, None, None

        boxes = det_output["detection_boxes"]
        classes = det_output["detection_classes"]
        scores = det_output["detection_scores"]
        if boxes is None or len(boxes) == 0:
            return None, None, None, None

        # Final score filter + mask assembly (port of _finalize_detections_yolact).
        keep = scores > self._score_threshold
        boxes, classes, scores = boxes[keep].astype(np.float64), classes[keep], scores[keep]
        mask_coeff = det_output["mask"][keep]
        if len(boxes) == 0:
            return None, None, None, None

        proto_data = det_output["proto"]
        masks_out = np.matmul(proto_data, mask_coeff.transpose())
        masks_out = _sigmoid(masks_out)
        masks_out = _crop(masks_out, boxes)

        # boxes are normalized over the 512x512 input -> input-pixel space.
        boxes_px = boxes.copy()
        boxes_px[:, [0, 2]] *= IMG_SIZE[0]
        boxes_px[:, [1, 3]] *= IMG_SIZE[1]
        boxes_px = np.clip(boxes_px, 0, [IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[1]])
        # (128, 128, N) -> (N, 128, 128)
        masks_out = np.transpose(masks_out, (2, 0, 1))

        return (boxes_px.astype(np.float32), classes.astype(np.int32),
                scores.astype(np.float32), masks_out.astype(np.float32))


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Wraps YolactPostprocess for the app's common interface.

    Returns (boxes, classes, scores, masks):
      boxes   (N, 4) xyxy in 512x512 input-pixel space
      classes (N,)  0..79 COCO indices
      scores  (N,)  softmax confidences
      masks   (N, 128, 128) float masks over the 512x512 input
    """
    pp = YolactPostprocess(score_threshold=obj_thresh, nms_iou_thresh=nms_thresh)
    try:
        return pp(hailo_output)
    except Exception as e:
        print(f"[YOLACT] postprocess error: {e}", flush=True)
        return None, None, None, None


def unletterbox_boxes(boxes, lb_info):
    """Scale xyxy boxes from the 512x512 input back to the original frame.

    The official pipeline resizes the raw frame to 512x512 with plain
    bilinear interpolation (no letterbox), so un-scaling is a per-axis
    division by the exact resize ratio captured in lb_info=(sx, sy) —
    independent of the shared co_helper state (races across threads)."""
    if boxes is None or len(boxes) == 0:
        return boxes
    sx, sy = lb_info
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] /= sx
    out[:, [1, 3]] /= sy
    return out


# Pick visually distinct BGR colors per class (same palette as the Model Zoo
# visualization, which cycles hues per class index).
def _mask_color(cls_id):
    palette = [
        (54, 67, 244), (99, 30, 233), (176, 39, 156), (183, 58, 103),
        (181, 81, 63), (243, 150, 33), (244, 169, 3), (212, 188, 0),
        (136, 150, 0), (80, 175, 76), (74, 195, 139), (57, 220, 205),
        (59, 235, 255), (7, 193, 255), (0, 152, 255), (34, 87, 255),
        (72, 85, 121), (158, 158, 158), (139, 125, 96),
    ]
    return palette[int(cls_id) % len(palette)]


def draw(image, boxes, scores, classes, masks=None, mask_thresh=0.5, mask_alpha=0.45):
    """Draw boxes + labels, and overlay instance masks when masks is not None
    (N, 128, 128) float masks over the 512x512 input space, resized here to
    the frame and composited with per-class colors (port of prep_display)."""
    h, w = image.shape[:2]
    if masks is not None and len(masks):
        resized = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                            for m in masks], axis=0)  # (N, h, w)
        binary = resized > mask_thresh
        overlay = image.copy()
        for i, cl in enumerate(classes):
            cl = int(cl)
            if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
                continue
            color = _mask_color(cl)
            overlay[binary[i]] = (
                (overlay[binary[i]].astype(np.float32) * (1 - mask_alpha)
                 + np.array(color, dtype=np.float32) * mask_alpha)
            ).astype(np.uint8)
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
        combined_mask = np.any(binary, axis=0)  # (h, w)
        image[combined_mask] = overlay[combined_mask]
    for box, score, cl in zip(boxes, scores, classes):
        cl = int(cl)
        if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), _mask_color(cl), 2)
        cv2.putText(image, f'{CLASSES[cl]} {float(score):.2f}',
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def preprocess_frame(frame, co_helper):
    """Resize the raw frame to 512x512 with bilinear interpolation (matching
    the Model Zoo's mobilenet_ssd preprocessing — no letterbox, no padding)
    and convert BGR -> RGB.

    The HEF uses normalize_in_net with ImageNet RGB mean/std
    ([123.68, 116.78, 103.94] / [58.4, 57.12, 57.38]), so the app feeds raw
    uint8 RGB pixels. Returns (img, (sx, sy)) where sx/sy are the exact
    input->frame scale factors for un-scaling boxes/masks.
    """
    ih, iw = IMG_SIZE[1], IMG_SIZE[0]
    scale_x = float(iw) / frame.shape[1]
    scale_y = float(ih) / frame.shape[0]
    img = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (scale_x, scale_y)

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
                boxes, classes, scores, masks = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                if boxes is not None and len(boxes) > 0:
                    real_boxes = unletterbox_boxes(boxes, lb_info)
                    h, w = frame.shape[:2]
                    real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
                    real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
                    draw(frame, real_boxes, scores, classes, masks)
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
    parser = argparse.ArgumentParser(description='YOLACT RegNetX-1.6GF on RPi5 + Hailo-8 (Web Preview Mode)')
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
