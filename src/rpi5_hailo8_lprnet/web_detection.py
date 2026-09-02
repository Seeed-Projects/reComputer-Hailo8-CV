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

# Demo defaults. Model Zoo v2.17 evaluates at score_threshold=0.1 and the
# official TAPPAS LPR demo uses 0.3. This HEF exposes raw YOLO heads, so decode
# and NMS run on the CPU.
OBJ_THRESH = 0.10
NMS_THRESH = 0.30
IMG_SIZE = (416, 416)  # (width, height) — overridden at runtime from the .hef

# Single-class label order from the official Model Zoo network YAML.
DEFAULT_CLASSES = ("license_plate",)

CLASSES = DEFAULT_CLASSES
_DET_OUTPUT_LOGGED = False

# tiny_yolov4_license_plates anchors from Model Zoo v2.17.
#   stride 32 (13x13 grid): [[81,82],[135,169],[344,319]]
#   stride 16 (26x26 grid): [[10,14],[23,27],[37,58]]
TINY_YOLOV4_ANCHORS = (
    np.array([[81.0, 82.0], [135.0, 169.0], [344.0, 319.0]], dtype=np.float32),  # stride 32
    np.array([[10.0, 14.0], [23.0, 27.0], [37.0, 58.0]], dtype=np.float32),     # stride 16
)
TINY_YOLOV4_STRIDES = (32, 16)
NUM_CLASSES = 1
OCR_CHARS = tuple("0123456789-")
OCR_THRESHOLD = 0.90
OCR_MIN_CHARS = 7
LPR_SIZE = (300, 75)
_OCR_OUTPUT_LOGGED = False

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
                print(f"Warning: No classes found in {path}, using the license-plate label")
                CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using the license-plate label")
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
    def __init__(self, detector_model=None, lpr_model=None, co_helper=None):
        self.detector_model = detector_model; self.lpr_model = lpr_model
        self.co_helper = co_helper
        self.is_processing = False; self.progress = 0
        self.current_video = ""; self.error_msg = ""
        self._stop_event = threading.Event(); self._thread = None
    def set_engine(self, detector_model, lpr_model, co_helper):
        self.detector_model = detector_model; self.lpr_model = lpr_model
        self.co_helper = co_helper
    def start_analysis(self, input_path, output_path):
        if self.is_processing: return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))
        self._thread.daemon = True; self._thread.start(); return True
    @staticmethod
    def _open_writer(output_path, width, height, fps):
        cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','bgr24','-s',f'{width}x{height}','-r',f'{fps}','-i','-',
               '-c:v','libx264','-preset','ultrafast','-threads','0',
               '-pix_fmt','yuv420p','-movflags','+faststart',output_path]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, 'ffmpeg'
        except FileNotFoundError: pass
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[VideoAnalyzer] Using cv2 mp4v", flush=True)
            return out, 'mp4v'
        out.release(); return None, None
    def _process_video(self, input_path, output_path):
        self.is_processing = True; self.progress = 0; self.error_msg = ""
        self.current_video = os.path.basename(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Error: Cannot open video {input_path}"; self.is_processing = False; return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"; self.is_processing = False; cap.release(); return
        out, kind = self._open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "Error: No usable video writer"; self.is_processing = False; cap.release(); return
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                if self.detector_model and self.lpr_model and self.co_helper:
                    obj, nms = det_config.get()
                    process_lpr_frame(
                        frame, self.detector_model, self.lpr_model,
                        self.co_helper, obj, nms, annotate=True,
                    )
                if kind == 'ffmpeg': out.stdin.write(frame.tobytes())
                else: out.write(frame)
                frame_idx += 1; self.progress = int((frame_idx / total_frames) * 100)
        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            cap.release()
            if kind == 'ffmpeg':
                try: out.stdin.close()
                except: pass
                try: out.wait(timeout=30)
                except: out.kill()
            else: out.release()
            self.is_processing = False
            if not self.error_msg: self.progress = 100
    def stop(self): self._stop_event.set()

video_analyzer = VideoAnalyzer()
app = FastAPI(title="reComputer LPRNet Pipeline Hailo-8")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get(); return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", OBJ_THRESH), config.get("nms_thresh", NMS_THRESH))
    return {"status": "success"}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    return {"uploads": os.listdir(UPLOAD_DIR), "outputs": os.listdir(OUTPUT_DIR)}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path): raise HTTPException(status_code=404, detail="Video not found")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise HTTPException(status_code=400, detail="Cannot open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"; output_path = os.path.join(OUTPUT_DIR, output_filename)
    success = video_analyzer.start_analysis(input_path, output_path)
    return {"status": "started", "output": output_filename} if success else {"status": "error", "message": "Already processing"}

@app.get("/api/video/status")
async def get_analysis_status():
    return {"is_processing": video_analyzer.is_processing, "progress": video_analyzer.progress,
            "current_video": video_analyzer.current_video, "error": video_analyzer.error_msg}

@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path): raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

_global_detector_model = None; _global_lpr_model = None; _global_co_helper = None

@app.post("/api/models/lprnet/predict")
async def predict(file: Optional[UploadFile] = File(None), video: Optional[UploadFile] = File(None),
                  timestamp: Optional[float] = Form(None), realtime: Optional[bool] = Form(False),
                  conf: Optional[float] = Form(None), iou: Optional[float] = Form(None)):
    if _global_detector_model is None or _global_lpr_model is None or _global_co_helper is None:
        return {"success": False, "message": "Model not initialized"}
    try:
        img = None; source_info = ""
        if file:
            contents = await file.read(); nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR); source_info = "uploaded image"
        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read()); tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                if timestamp is not None: cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if ret: img = frame; source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)
        if img is None: img = frame_buffer.get_raw_frame(); source_info = "realtime camera frame"
        if img is None: return {"success": False, "message": "No valid input source found"}
        current_obj, current_nms = det_config.get()
        h, w = img.shape[:2]
        predictions = process_lpr_frame(
            img, _global_detector_model, _global_lpr_model, _global_co_helper,
            conf if conf is not None else current_obj,
            iou if iou is not None else current_nms,
            annotate=False,
        )
        return {"success": True, "source": source_info, "predictions": predictions, "image": {"width": w, "height": h}}
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    def __init__(self):
        self.raw = None; self.annotated = None; self.annotated_version = 0
        self.jpeg = None; self.jpeg_version = 0; self.cond = threading.Condition()
    def push_annotated(self, frame):
        with self.cond: self.raw = frame; self.annotated = frame; self.annotated_version += 1; self.cond.notify_all()
    def wait_annotated(self, last_version, timeout=1.0):
        with self.cond: self.cond.wait_for(lambda: self.annotated_version > last_version, timeout=timeout)
        return self.annotated, self.annotated_version
    def push_jpeg(self, jpeg_bytes):
        with self.cond: self.jpeg = jpeg_bytes; self.jpeg_version += 1; self.cond.notify_all()
    def wait_jpeg(self, last_version, timeout=1.0):
        with self.cond: self.cond.wait_for(lambda: self.jpeg_version > last_version, timeout=timeout)
        return self.jpeg, self.jpeg_version
    def get_raw_frame(self):
        with self.cond: return self.raw.copy() if self.raw is not None else None

frame_buffer = FrameBuffer()

class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap; self.frame = None; self.version = 0
        self._last_read_version = 0; self._stopped = False
        self._cond = threading.Condition(); self._thread = threading.Thread(target=self._loop, daemon=True)
    def start(self): self._thread.start(); return self
    def _loop(self):
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret: time.sleep(0.01); continue
            with self._cond: self.frame = frame; self.version += 1; self._cond.notify_all()
        with self._cond: self._stopped = True; self._cond.notify_all()
    def read(self, timeout=1.0):
        with self._cond:
            self._cond.wait_for(lambda: self.version > self._last_read_version or self._stopped, timeout=timeout)
            if self.frame is None: return False, None
            self._last_read_version = self.version; return True, self.frame.copy()
    def stop(self):
        with self._cond: self._stopped = True; self._cond.notify_all()
        self._thread.join(timeout=2)

@app.get("/api/video_feed")
async def video_feed():
    def generate():
        last_v = -1
        while True:
            jpeg, last_v = frame_buffer.wait_jpeg(last_v, timeout=1.0)
            if jpeg is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html><head><title>reComputer LPRNet Pipeline · Hailo-8</title>
    <style>
      body{background:#1a1a1a;color:white;text-align:center;font-family:sans-serif;margin:0;padding:20px}
      .container{max-width:1200px;margin:0 auto}
      .video-box{margin:20px auto;display:inline-block;border:5px solid #333;border-radius:10px;overflow:hidden;background:#000;width:100%;max-width:800px}
      .controls{background:#2a2a2a;padding:20px;border-radius:10px;display:inline-block;text-align:left;min-width:400px;vertical-align:top;margin:10px}
      .control-group{margin-bottom:15px}.control-group label{display:block;margin-bottom:5px;font-weight:bold}
      .slider-container{display:flex;align-items:center;gap:15px}input[type=range]{flex-grow:1;cursor:pointer}
      .value-display{min-width:50px;font-family:monospace;background:#444;padding:2px 8px;border-radius:4px;text-align:center}
      h1{color:#00e676}.tabs{display:flex;justify-content:center;margin-bottom:20px;border-bottom:2px solid #333}
      .tab{padding:10px 30px;cursor:pointer;border-bottom:3px solid transparent;transition:.3s;font-weight:bold}
      .tab.active{border-bottom-color:#00e676;color:#00e676}.tab-content{display:none}.tab-content.active{display:block}
      .video-analysis{text-align:left;background:#2a2a2a;padding:20px;border-radius:10px;margin:10px}
      .btn{background:#00e676;color:#000;border:none;padding:8px 16px;border-radius:4px;cursor:pointer;font-weight:bold;margin:5px}
      .btn:hover{background:#00c853}.btn:disabled{background:#555;cursor:not-allowed}
      .progress-container{width:100%;background:#444;border-radius:10px;margin:15px 0;height:20px;position:relative;overflow:hidden}
      .progress-bar{height:100%;background:#00e676;width:0%;transition:.3s}
      .progress-text{position:absolute;width:100%;text-align:center;top:0;left:0;line-height:20px;font-size:12px;font-weight:bold;color:#fff;text-shadow:1px 1px 2px #000}
      table{width:100%;border-collapse:collapse;margin-top:15px}th,td{text-align:left;padding:10px;border-bottom:1px solid #444}th{color:#888}
    </style></head><body>
    <div class="container"><h1>LPRNet License Plate Pipeline · RPi5 + Hailo-8</h1>
    <div class="tabs"><div class="tab active" onclick="showTab('realtime')">Real-time Detection</div><div class="tab" onclick="showTab('analysis')">Local Video Analysis</div></div>
    <div id="realtime" class="tab-content active">
      <div class="video-box"><img id="streamImg" src="/api/video_feed" style="max-width:100%;height:auto;"></div>
      <div class="controls">
        <div class="control-group"><label>Confidence Threshold</label><div class="slider-container"><input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25"><span id="confValue" class="value-display">0.25</span></div></div>
        <div class="control-group"><label>IOU Threshold (NMS)</label><div class="slider-container"><input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.30"><span id="iouValue" class="value-display">0.30</span></div></div>
      </div>
    </div>
    <div id="analysis" class="tab-content"><div class="video-analysis"><h3>Analyze Local Video</h3>
      <div class="control-group"><label>Upload New Video (.mp4)</label><input type="file" id="videoUpload" accept=".mp4"><button class="btn" onclick="uploadVideo()">Upload</button></div>
      <div id="processingArea" style="display:none;"><p>Processing: <span id="currentFileName">-</span></p><div class="progress-container"><div id="progressBar" class="progress-bar"></div><div id="progressText" class="progress-text">0%</div></div><p id="errorText" style="color:#ff5252;"></p></div>
      <div class="control-group"><label>File Management</label><button class="btn" onclick="refreshFileList()">Refresh List</button><table><thead><tr><th>File Name</th><th>Action</th></tr></thead><tbody id="fileTableBody"></tbody></table></div>
    </div></div>
    <p style="color:#888;margin-top:20px;">Streaming via FastAPI + MJPEG | Port: 8000</p></div>
    <script>
      function showTab(t){document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(t).classList.add('active');event.currentTarget.classList.add('active');if(t==='realtime'){document.getElementById('streamImg').src='/api/video_feed'}else{document.getElementById('streamImg').src='';refreshFileList()}}
      const cs=document.getElementById('confSlider'),is=document.getElementById('iouSlider'),cv=document.getElementById('confValue'),iv=document.getElementById('iouValue');
      function uc(){const o=parseFloat(cs.value),n=parseFloat(is.value);cv.innerText=o.toFixed(2);iv.innerText=n.toFixed(2);fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({obj_thresh:o,nms_thresh:n})})}
      cs.oninput=uc;is.oninput=uc;fetch('/api/config').then(r=>r.json()).then(d=>{cs.value=d.obj_thresh;is.value=d.nms_thresh;cv.innerText=d.obj_thresh.toFixed(2);iv.innerText=d.nms_thresh.toFixed(2)});
      async function uploadVideo(){const f=document.getElementById('videoUpload');if(!f.files[0])return alert('Select a file');const fd=new FormData();fd.append('file',f.files[0]);const b=event.currentTarget;b.disabled=true;b.innerText='Uploading...';try{await fetch('/api/video/upload',{method:'POST',body:fd});alert('Uploaded');refreshFileList()}catch(e){alert('Failed')}finally{b.disabled=false;b.innerText='Upload'}}
      async function refreshFileList(){const r=await fetch('/api/video/list');const d=await r.json();const t=document.getElementById('fileTableBody');t.innerHTML='';d.uploads.forEach(f=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${f} (Original)</td><td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>`;t.appendChild(tr)});d.outputs.forEach(f=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${f} (Analyzed)</td><td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>`;t.appendChild(tr)})}
      async function analyzeVideo(f){const fd=new FormData();fd.append('filename',f);const r=await fetch('/api/video/analyze',{method:'POST',body:fd});const d=await r.json();if(d.status==='started'){startStatusPolling()}else{alert(d.message||'Error')}}
      let pi;function startStatusPolling(){document.getElementById('processingArea').style.display='block';if(pi)clearInterval(pi);pi=setInterval(async()=>{const r=await fetch('/api/video/status');const d=await r.json();document.getElementById('currentFileName').innerText=d.current_video;document.getElementById('progressBar').style.width=d.progress+'%';document.getElementById('progressText').innerText=d.progress+'%';document.getElementById('errorText').innerText=d.error||'';if(!d.is_processing&&d.progress===100){clearInterval(pi);alert('Done!');refreshFileList()}else if(!d.is_processing&&d.error){clearInterval(pi)}},1000)}
      fetch('/api/video/status').then(r=>r.json()).then(d=>{if(d.is_processing)startStatusPolling()})
    </script></body></html>
    """, media_type="text/html")

def run_fastapi(host, port):
    print("\n"+"="*50, flush=True); print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"): print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50+"\n", flush=True); sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


# ---------------------------------------------------------------------------
# Tiny-YOLOv4 post-processing (CPU decode — NO on-chip NMS)
#
# The HEF exposes two raw YOLOv3 heads (no on-chip NMS, no HPP):
#   13x13x18 (stride 32)  — 3 anchors × (4 box + 1 obj + 1 cls) = 18
#   26x26x18 (stride 16)
#
# Decode follows the official Hailo Model Zoo yolo.py _yolo3_decode:
#   box_center = (sigmoid(raw_xy) + grid_offset) * stride   [pixels]
#   box_scale  = exp(raw_wh) * anchor                        [pixels]
#   obj_score  = sigmoid(objness)
#   cls_score  = sigmoid(class_pred)
#   final      = obj_score * cls_score   (YOLOv3: obj × class)
#   box        = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]            [xyxy pixels]
# Then NMS on CPU.
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def _nms(boxes, scores, iou_thresh, max_det=100):
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep

def _decode_yolov3_head(head, anchors, stride, obj_thresh):
    """Decode a single YOLOv3 head.
    head: (grid_h, grid_w, 18) = (H, W, 3*(5+1))
    anchors: (3, 2) array of (w, h) in pixel units
    stride: int (32 or 16)
    Returns: (boxes_xyxy, classes, scores) or None if no detections.
    """
    H, W, _ = head.shape
    # Channel layout is anchor-major (per the Model Zoo yolo decode:
    # transpose to (N, C, H*W) then reshape (BS, num_anchors*num_pred, H*W)):
    # (H, W, 18) = [a0: xy wh obj cls | a1: ... | a2: ...].
    # Reshape to (H, W, 3, 6) grouping each anchor's 6 predictions.
    head = head.reshape(H, W, 3, 5 + NUM_CLASSES)

    raw_xy = head[..., 0:2]    # (H, W, 3, 2)
    raw_wh = head[..., 2:4]   # (H, W, 3, 2)
    raw_obj = head[..., 4]    # (H, W, 3)
    raw_cls = head[..., 5:]   # (H, W, 3, 1)

    # Sigmoid obj + cls
    sig_obj = _sigmoid(raw_obj)         # (H, W, 3)
    sig_cls = _sigmoid(raw_cls)         # (H, W, 3, 1)
    scores = sig_obj[..., None] * sig_cls  # YOLOv3: objectness × class score

    # Grid offsets
    grid_y, grid_x = np.meshgrid(np.arange(H, dtype=np.float32),
                                 np.arange(W, dtype=np.float32), indexing='ij')
    grid_x = grid_x[..., None]  # (H, W, 1)
    grid_y = grid_y[..., None]  # (H, W, 1)

    # Centers: (sigmoid(tx) + gx) * stride
    cx = (_sigmoid(raw_xy[..., 0]) + grid_x) * stride  # (H, W, 3)
    cy = (_sigmoid(raw_xy[..., 1]) + grid_y) * stride

    # Scales: exp(tw) * anchor_w
    bw = np.exp(np.clip(raw_wh[..., 0], -50, 50)) * anchors[:, 0]  # (H, W, 3)
    bh = np.exp(np.clip(raw_wh[..., 1], -50, 50)) * anchors[:, 1]

    # Boxes xyxy in pixel coords
    x1 = cx - bw / 2; y1 = cy - bh / 2
    x2 = cx + bw / 2; y2 = cy + bh / 2

    # Flatten and filter
    x1 = x1.ravel(); y1 = y1.ravel(); x2 = x2.ravel(); y2 = y2.ravel()
    flat_scores = scores.reshape(-1, NUM_CLASSES)
    max_score = flat_scores.max(axis=1)            # (H*W*3,)
    mask = max_score >= obj_thresh

    if not mask.any():
        return None

    boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    classes = flat_scores[mask].argmax(axis=1)
    scores_out = flat_scores[mask].max(axis=1)
    return boxes, classes, scores_out

def _as_hwc(tensor):
    arr = np.asarray(tensor)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 6, 18) and arr.shape[-1] not in (3, 6, 18):
        arr = np.moveaxis(arr, 0, -1)
    return arr


def _collect_yolo_heads(hailo_output):
    """Accept both combined (18-channel) and legacy split HEF outputs."""
    if isinstance(hailo_output, dict):
        named = {str(name): _as_hwc(value) for name, value in hailo_output.items()}
    elif isinstance(hailo_output, (list, tuple)):
        named = {f"output_{idx}": _as_hwc(value) for idx, value in enumerate(hailo_output)}
    else:
        named = {"output_0": _as_hwc(hailo_output)}

    heads = [arr for arr in named.values() if arr.ndim == 3 and arr.shape[-1] == 18]
    if len(heads) >= 2:
        return heads

    # Older Hailo-8 HEFs can expose centers/scales/objectness/probabilities as
    # separate tensors. Reassemble them into the raw 3 x (xy, wh, obj, cls)
    # layout consumed by _decode_yolov3_head().
    split_heads = []
    spatial_shapes = sorted({arr.shape[:2] for arr in named.values() if arr.ndim == 3})
    for height, width in spatial_shapes:
        group = {name: arr for name, arr in named.items() if arr.ndim == 3 and arr.shape[:2] == (height, width)}
        centers = next((arr for name, arr in group.items() if "center" in name.lower()), None)
        scales = next((arr for name, arr in group.items() if "scale" in name.lower()), None)
        obj = next((arr for name, arr in group.items() if "obj" in name.lower()), None)
        probs = next((arr for name, arr in group.items() if "prob" in name.lower()), None)
        if centers is None or scales is None or obj is None or probs is None:
            continue
        centers = centers.reshape(height, width, 3, 2)
        scales = scales.reshape(height, width, 3, 2)
        obj = obj.reshape(height, width, 3, 1)
        probs = probs.reshape(height, width, 3, NUM_CLASSES)
        split_heads.append(np.concatenate((centers, scales, obj, probs), axis=-1).reshape(height, width, 18))
    return split_heads


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Decode the two YOLOv3 raw heads into boxes/classes/scores.

    Returns (boxes, classes, scores) where boxes are xyxy in input-pixel space
    (the 416x416 letterboxed input). The caller un-letterboxes to the original
    frame via unletterbox_boxes().
    """
    global _DET_OUTPUT_LOGGED

    if hailo_output is None:
        return None, None, None

    heads = _collect_yolo_heads(hailo_output)

    if not _DET_OUTPUT_LOGGED:
        for i, h in enumerate(heads):
            print(f"[Tiny-YOLOv4-License-Plates] head{i}: shape={h.shape}", flush=True)
        _DET_OUTPUT_LOGGED = True

    if len(heads) < 2:
        return None, None, None

    # Smaller grid is stride 32; larger grid is stride 16.
    heads.sort(key=lambda h: h.shape[0] * h.shape[1], reverse=False)

    all_boxes, all_classes, all_scores = [], [], []
    for i, (head, stride, anchors) in enumerate(zip(heads, TINY_YOLOV4_STRIDES, TINY_YOLOV4_ANCHORS)):
        result = _decode_yolov3_head(head, anchors, stride, obj_thresh)
        if result is not None:
            boxes, classes, scores = result
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)

    if not all_boxes:
        return None, None, None

    boxes = np.concatenate(all_boxes).astype(np.float32)
    classes = np.concatenate(all_classes).astype(np.int32)
    scores = np.concatenate(all_scores).astype(np.float32)

    keep = _nms(boxes, scores, nms_thresh)
    if not keep:
        return None, None, None
    return boxes[keep], classes[keep], scores[keep]

def unletterbox_boxes(boxes, lb_info):
    """Map xyxy boxes from the 416x416 ar-preserving input (content at the
    origin, bottom/right pad) back to the original frame: pure division by
    the resize scale — padding is not centered, so no offset to subtract."""
    if boxes is None or len(boxes) == 0:
        return boxes
    scale, _pad_h, _pad_w = lb_info
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] /= scale
    out[:, [1, 3]] /= scale
    return out

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        cl = int(cl)
        if cl < 0 or cl >= len(CLASSES): continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, f'{CLASSES[cl]} {float(score):.2f}',
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def preprocess_frame(frame, co_helper):
    """Aspect-preserving resize + pad 0, BGR passthrough. Returns (img, lb_info).

    Official pipeline (network YAML): meta_arch=mobilenet_ssd_ar, i.e.
    _ar_preserving_resize_and_crop with padding_color=0. The alls script has
    NO input_conversion(bgr_to_rgb) and normalization(mean=0, std=255), so
    the HEF expects raw uint8 BGR frames — no RGB swap, no pad-114
    letterbox. Earlier revisions fed RGB with gray padding, which produced
    zero detections on hardware.
    """
    ih, iw = IMG_SIZE[1], IMG_SIZE[0]
    fh, fw = frame.shape[:2]
    scale = min(iw / fw, ih / fh)
    new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((ih, iw, 3), dtype=np.uint8)
    img[:new_h, :new_w] = resized
    return img, (scale, ih - new_h, iw - new_w)


def preprocess_lpr(crop, lpr_model):
    """Resize a BGR plate crop; the HEF contains BGR-to-RGB and normalization."""
    return cv2.resize(
        crop, (lpr_model.input_w, lpr_model.input_h), interpolation=cv2.INTER_AREA
    ).astype(np.uint8)


def decode_lpr(hailo_output):
    """Apply the official mean + CTC-greedy LPRNet post-processing."""
    global _OCR_OUTPUT_LOGGED
    if hailo_output is None:
        return "", 0.0, ""
    if isinstance(hailo_output, dict):
        named = {str(name): np.asarray(value) for name, value in hailo_output.items()}
    elif isinstance(hailo_output, (list, tuple)):
        named = {f"output_{idx}": np.asarray(value) for idx, value in enumerate(hailo_output)}
    else:
        named = {"output_0": np.asarray(hailo_output)}

    if not _OCR_OUTPUT_LOGGED:
        for name, arr in named.items():
            print(f"[LPRNet] {name}: shape={arr.shape} dtype={arr.dtype}", flush=True)
        _OCR_OUTPUT_LOGGED = True

    tensor = None
    for arr in named.values():
        squeezed = np.squeeze(arr)
        if squeezed.ndim in (2, 3) and 11 in squeezed.shape and 19 in squeezed.shape:
            tensor = squeezed.astype(np.float32)
            break
    if tensor is None:
        return "", 0.0, ""

    if tensor.ndim == 2:
        if tensor.shape != (19, 11):
            tensor = tensor.T
        tensor = tensor[None, ...]
    else:
        class_axis = tensor.shape.index(11)
        time_axis = tensor.shape.index(19)
        ensemble_axis = next(axis for axis in range(3) if axis not in (class_axis, time_axis))
        tensor = np.transpose(tensor, (ensemble_axis, time_axis, class_axis))

    logits = tensor.mean(axis=0)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    labels = logits.argmax(axis=1)
    confidences = probs.max(axis=1)

    decoded = []
    decoded_conf = []
    previous = None
    blank = len(OCR_CHARS) - 1
    for label, confidence in zip(labels.tolist(), confidences.tolist()):
        if label == blank:
            previous = label
            continue
        if label == previous:
            continue
        decoded.append(OCR_CHARS[label])
        decoded_conf.append(float(confidence))
        previous = label

    candidate = "".join(decoded)
    mean_conf = float(np.mean(decoded_conf)) if decoded_conf else 0.0
    accepted = candidate if mean_conf >= OCR_THRESHOLD and len(candidate) >= OCR_MIN_CHARS else ""
    return accepted, mean_conf, candidate


def plate_quality(crop):
    """Match TAPPAS' Laplacian-variance gate on the central 80% of a plate."""
    if crop is None or crop.size == 0 or crop.shape[0] <= 10 or crop.shape[1] <= 10:
        return -1.0
    height, width = crop.shape[:2]
    x_pad = max(1, int(width * 0.1)); y_pad = max(1, int(height * 0.1))
    center = crop[y_pad:height - y_pad, x_pad:width - x_pad]
    if center.size == 0:
        return -1.0
    resized = cv2.resize(center, (200, 40), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 255, 0, cv2.NORM_INF)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def process_lpr_frame(frame, detector_model, lpr_model, co_helper, obj_thresh, nms_thresh, annotate=True):
    detector_input, lb_info = preprocess_frame(frame, co_helper)
    detector_output = detector_model.run(detector_input)
    boxes, classes, scores = post_process_hailo(
        detector_output, obj_thresh, nms_thresh, IMG_SIZE[1], IMG_SIZE[0]
    )
    predictions = []
    if boxes is None or len(boxes) == 0:
        return predictions

    height, width = frame.shape[:2]
    real_boxes = unletterbox_boxes(boxes, lb_info)
    real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, width - 1)
    real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, height - 1)

    for box, score in zip(real_boxes, scores):
        x1, y1, x2, y2 = [int(value) for value in box]
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        quality = plate_quality(crop)
        plate_text = ""; text_conf = 0.0; candidate = ""
        if quality >= 100.0:
            lpr_output = lpr_model.run(preprocess_lpr(crop, lpr_model))
            plate_text, text_conf, candidate = decode_lpr(lpr_output)
        prediction = {
            "class": "license_plate",
            "confidence": float(score),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "plate": plate_text or None,
            "ocr_candidate": candidate or None,
            "ocr_confidence": text_conf,
            "quality": quality,
        }
        predictions.append(prediction)
        if annotate:
            color = (0, 255, 0) if plate_text else (0, 255, 255)
            label = plate_text if plate_text else f"plate {float(score):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return predictions


def inference_loop(cap, detector_model, lpr_model, co_helper, is_video_file, target_fps):
    fps_counter = 0
    target_period = 1.0 / target_fps if target_fps > 0 else 0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            if video_analyzer.is_processing: time.sleep(1.0); next_time = time.time(); continue
            ret, frame = cap.read()
            if not ret:
                if is_video_file: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break
            start_time = time.time()
            obj, nms = det_config.get()
            process_lpr_frame(frame, detector_model, lpr_model, co_helper, obj, nms, annotate=True)
            inference_time = time.time() - start_time
            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'Pipeline FPS: {fps_counter:.1f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_buffer.push_annotated(frame)
            if target_period > 0:
                now = time.time(); next_time += target_period
                sleep_for = next_time - now
                if sleep_for > 0: time.sleep(sleep_for)
                elif sleep_for < -target_period: next_time = now + target_period
    finally:
        stop_event.set()

def encode_loop(preview_w, preview_h, jpeg_quality):
    last_v = -1
    while not stop_event.is_set():
        frame, last_v = frame_buffer.wait_annotated(last_v, timeout=1.0)
        if frame is None: continue
        h, w = frame.shape[:2]
        if preview_w > 0 and preview_h > 0 and (w, h) != (preview_w, preview_h):
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        else: preview = frame
        ok, buf = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok: frame_buffer.push_jpeg(buf.tobytes())

def main():
    parser = argparse.ArgumentParser(description='Tiny-YOLOv4 + LPRNet pipeline on RPi5 + Hailo-8')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the LPRNet .hef model')
    parser.add_argument('--detector_model_path', type=str,
                        default='model/tiny_yolov4_license_plates.hef',
                        help='Path to the license-plate detector .hef model')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (-1 to disable)')
    parser.add_argument('--video_path', type=str, help='Path to video file')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--preview_width', type=int, default=1280)
    parser.add_argument('--preview_height', type=int, default=720)
    parser.add_argument('--jpeg_quality', type=int, default=80)
    parser.add_argument('--cam_width', type=int, default=1280)
    parser.add_argument('--cam_height', type=int, default=720)
    parser.add_argument('--target_fps', type=float, default=30.0)
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available."); return
    if args.class_path: load_classes(args.class_path)

    global _global_detector_model, _global_lpr_model, _global_co_helper, IMG_SIZE, LPR_SIZE
    detector_model = HailoInfer(args.detector_model_path)
    lpr_model = HailoInfer(args.model_path)
    IMG_SIZE = (detector_model.input_w, detector_model.input_h)
    LPR_SIZE = (lpr_model.input_w, lpr_model.input_h)
    print(f"Detector input size: {detector_model.input_w}x{detector_model.input_h}", flush=True)
    print(f"LPRNet input size: {lpr_model.input_w}x{lpr_model.input_h}", flush=True)
    co_helper = COCO_test_helper(enable_letter_box=True)
    _global_detector_model = detector_model; _global_lpr_model = lpr_model
    _global_co_helper = co_helper
    video_analyzer.set_engine(detector_model, lpr_model, co_helper)

    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        print("Running in Video Analysis Mode.", flush=True)
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: print("Interrupted")
        finally:
            lpr_model.release(); detector_model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path); capture_source = cap; is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        capture_source = None; is_video_file = False
    if not cap.isOpened():
        print(f"Error: Cannot open video source"); return
    if not is_video_file: capture_source = LatestFrameReader(cap).start()

    inf_thread = threading.Thread(
        target=inference_loop,
        args=(capture_source, detector_model, lpr_model, co_helper, is_video_file, args.target_fps),
        daemon=True,
    )
    enc_thread = threading.Thread(target=encode_loop, args=(args.preview_width, args.preview_height, args.jpeg_quality), daemon=True)
    inf_thread.start(); enc_thread.start()
    try:
        while inf_thread.is_alive(): time.sleep(0.5)
    except KeyboardInterrupt: print("Interrupted by user")
    finally:
        stop_event.set()
        if not is_video_file: capture_source.stop()
        inf_thread.join(timeout=2); enc_thread.join(timeout=2)
        cap.release(); lpr_model.release(); detector_model.release()

if __name__ == '__main__':
    main()
