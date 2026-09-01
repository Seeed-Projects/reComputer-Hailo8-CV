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
import sys; sys.path.insert(0, os.path.dirname(__file__))

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height) must match the .hef input shape

DEFAULT_CLASSES = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
)

# Cityscapes trainId palette in BGR order because OpenCV images are BGR.
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],   # road
    [232, 35, 244],   # sidewalk
    [70, 70, 70],     # building
    [156, 102, 102],  # wall
    [153, 153, 190],  # fence
    [153, 153, 153],  # pole
    [30, 170, 250],   # traffic light
    [0, 220, 220],    # traffic sign
    [35, 142, 107],   # vegetation
    [152, 251, 152],  # terrain
    [180, 130, 70],   # sky
    [60, 20, 220],    # person
    [0, 0, 255],      # rider
    [142, 0, 0],      # car
    [70, 0, 0],       # truck
    [100, 60, 0],     # bus
    [100, 80, 0],     # train
    [230, 0, 0],      # motorcycle
    [32, 11, 119],    # bicycle
], dtype=np.uint8)

CLASSES = DEFAULT_CLASSES
_SEG_OUTPUT_LOGGED = False

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
                    print(f"Warning: No classes found in {path}, using default Cityscapes classes")
                    CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default Cityscapes classes")
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
        """Try ffmpeg subprocess with libx264 ultrafast first 鈥?on Pi 5 this
        is ~5x faster than cv2's mp4v at 4K (40ms vs 150ms per frame).
        Falls back to cv2 mp4v if ffmpeg is not installed (e.g., non-Docker run).

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

        # Single-threaded loop 鈥?empirical: on Pi 5 with 4K frames, the
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
                    processed_img, lb_info = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(processed_img)

                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes, scores, class_ids, masks = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                        if boxes is not None:
                            draw_boxes(frame, boxes, scores, class_ids, masks, lb_info)
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

app = FastAPI(title="reComputer Hailo-CV Web Preview (CM5 + Hailo-8)")

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

@app.post("/api/models/yolo26m_seg/predict")
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

        boxes, scores, class_ids, masks = post_process_hailo(outputs, OBJ_THRESH, 0, IMG_SIZE[1], IMG_SIZE[0])

        predictions = []
        if boxes is not None:
            real_boxes = unletterbox_boxes(boxes, lb_info)
            real_boxes[:, [0, 2]] = np.clip(real_boxes[:, [0, 2]], 0, w - 1)
            real_boxes[:, [1, 3]] = np.clip(real_boxes[:, [1, 3]], 0, h - 1)
            frame_masks = unletterbox_masks(masks, lb_info, img.shape) if masks is not None else None
            draw_boxes(img, real_boxes, scores, class_ids, frame_masks, None)
            predictions = [
                {
                    "class": COCO_CLASSES[int(class_ids[index]) % len(COCO_CLASSES)],
                    "confidence": float(scores[index]),
                    "box": {"x1": int(real_boxes[index][0]), "y1": int(real_boxes[index][1]),
                            "x2": int(real_boxes[index][2]), "y2": int(real_boxes[index][3])},
                }
                for index, box in enumerate(boxes)
                if float(scores[index]) >= OBJ_THRESH
            ]

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
        <title>reComputer Hailo-CV Web Preview</title>
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
          <h1>CM5 + Hailo-8 Real-time Detection</h1>

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
                <label>IOU Threshold (applied on top of built-in NMS)</label>
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

def _primary_output_tensor(hailo_output):
    if isinstance(hailo_output, dict):
        return next(iter(hailo_output.values()))
    if isinstance(hailo_output, (list, tuple)):
        return hailo_output[0]
    return hailo_output


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# YOLO26-seg post-processing (one2one heads, CPU decode)
#
# Ported from hailo_model_zoo core/postprocessing (meta_arch "yolo26_seg",
# base/yolo26_seg.yaml). The HEF exposes 10 heads, no on-chip NMS:
#   3 strides (32/16/8), feature maps 20x20 / 40x40 / 80x80, each with:
#     bbox   (BS, F, F, 64)   one2one distances: l, t, r, b x 16 bins
#     score  (BS, F, F, 80)   per-class logits (sigmoid on CPU)
#     mask   (BS, F, F, 32)   mask coefficients
#   plus proto (BS, 160, 160, 32).
#
# YOLO26 is "one2one": ONE prediction per grid cell (no anchors, no NMS).
# Decode follows YoloPostProc._yolo6_decode:
#   x1y1 = offset + 0.5 - ltr; x2y2 = offset + 0.5 + ltr (in stride units)
# then ultralytics-style top-k selection (yolo26_filter) picks the final
# (anchor, class) pairs, and process_mask assembles sigmoid(coeffs @ proto)
# cropped to each box.
# ---------------------------------------------------------------------------

_Y26_STRIDES = (32, 16, 8)          # head order: 20x20, 40x40, 80x80
_Y26_REG_BINS = 16                  # 64 channels / 4 sides
_Y26_PROTO = 160                    # proto grid
_Y26_CLASSES = 80
_Y26_TOP_K = 100                    # post_nms_topk (Model Zoo default)
_Y26_SCORE_THRES = 0.25             # live preview cut


def _classify_heads(endnodes):
    """Sort the 10 output vstreams into (bbox, score, mask) per stride + proto.
    Heads are identified by channel count and spatial size; HailoRT dict
    order is not relied upon."""
    bboxes, scores, masks, proto = [], [], [], None
    for e in endnodes:
        arr = np.asarray(e)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            continue
        h, w, c = arr.shape
        if h == _Y26_PROTO and c == 32:
            proto = arr
        elif c == 64:
            bboxes.append(arr)
        elif c == 80:
            scores.append(arr)
        elif c == 32:
            masks.append(arr)

    def spatial(a):
        return a.shape[0] * a.shape[1]
    # _Y26_STRIDES is (32, 16, 8): ascending spatial order (20x20 -> 80x80).
    bboxes.sort(key=spatial)
    scores.sort(key=spatial)
    masks.sort(key=spatial)
    return bboxes, scores, masks, proto


def _decode_yolo26_boxes(bbox_head, stride):
    """One2one distance decode (port of _yolo6_decode + DFL softmax).

    bbox_head: (F, F, 64) raw; returns xyxy in input pixels, shape
    (F*F, 4) with row-major grid order (y-major, x-minor).
    """
    fh, fw, _ = bbox_head.shape
    # DFL: softmax over 16 bins, expectation -> distance in stride units.
    d = bbox_head.reshape(fh, fw, 4, _Y26_REG_BINS)
    d = _softmax_last(d)                      # (F, F, 4, 16)
    bins = np.arange(_Y26_REG_BINS, dtype=np.float32)
    dist = d @ bins                           # (F, F, 4) stride units

    # Grid offsets: cell center in stride units (broadcast-safe: keep (F, F)).
    gx, gy = np.meshgrid(np.arange(fw), np.arange(fh))  # (F, F)
    cx = gx.astype(np.float32) + 0.5
    cy = gy.astype(np.float32) + 0.5

    # one2one: distances from the cell center (offset + 0.5).
    l, t, r, b = dist[..., 0], dist[..., 1], dist[..., 2], dist[..., 3]
    x1 = (cx - l) * stride
    y1 = (cy - t) * stride
    x2 = (cx + r) * stride
    y2 = (cy + b) * stride

    boxes = np.stack([x1, y1, x2, y2], axis=-1).reshape(-1, 4)
    return boxes


def _softmax_last(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _crop_mask(masks, boxes):
    """Zero mask pixels outside each box (port of crop_mask). masks (N, h, w),
    boxes (N, 4) in mask-pixel coords."""
    n = masks.shape[0]
    integer_boxes = np.ceil(boxes).astype(int)
    x1, y1, x2, y2 = np.array_split(
        np.where(integer_boxes > 0, integer_boxes, 0), 4, axis=1)
    for k in range(n):
        masks[k, :y1[k, 0], :] = 0
        masks[k, :, :x1[k, 0]] = 0
        if y2[k, 0] < masks.shape[1]:
            masks[k, y2[k, 0]:, :] = 0
        if x2[k, 0] < masks.shape[2]:
            masks[k, :, x2[k, 0]:] = 0
    return masks


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Full YOLO26-seg decode. Returns (boxes, scores, class_ids, masks):
      boxes     (N, 4) xyxy in input-pixel space
      scores    (N,)   sigmoided class scores
      class_ids (N,)   COCO class indices
      masks     (N, h, w) float masks over the input frame, or None
    obj_thresh overrides the default score cut; nms_thresh unused (one2one).
    """
    global _SEG_OUTPUT_LOGGED
    if hailo_output is None:
        return None, None, None, None
    if isinstance(hailo_output, dict):
        endnodes = list(hailo_output.values())
    elif isinstance(hailo_output, (list, tuple)):
        endnodes = list(hailo_output)
    else:
        endnodes = [hailo_output]

    bboxes, scores_raw, masks_raw, proto = _classify_heads(endnodes)
    if not (len(bboxes) == len(scores_raw) == len(masks_raw) == 3) or proto is None:
        if not _SEG_OUTPUT_LOGGED:
            print(f"[YOLO26-seg] unexpected output layout: "
                  f"{[np.asarray(e).shape for e in endnodes]}", flush=True)
            _SEG_OUTPUT_LOGGED = True
        return None, None, None, None

    # Per-stride decode, concatenated in stride 32 -> 16 -> 8 order.
    all_boxes, all_scores, all_coeffs = [], [], []
    for i, stride in enumerate(_Y26_STRIDES):
        fh, fw, _ = bboxes[i].shape
        boxes = _decode_yolo26_boxes(bboxes[i], stride)          # (F*F, 4)
        sc = _sigmoid(scores_raw[i].reshape(-1, _Y26_CLASSES))   # (F*F, 80)
        cf = masks_raw[i].reshape(-1, 32)                        # (F*F, 32)
        all_boxes.append(boxes)
        all_scores.append(sc)
        all_coeffs.append(cf)
    boxes = np.concatenate(all_boxes, axis=0)        # (A, 4) A=8400
    scores = np.concatenate(all_scores, axis=0)      # (A, 80)
    coeffs = np.concatenate(all_coeffs, axis=0)      # (A, 32)

    if not _SEG_OUTPUT_LOGGED:
        print(f"[YOLO26-seg] anchors={len(boxes)}, proto={proto.shape}", flush=True)
        _SEG_OUTPUT_LOGGED = True

    # ultralytics get_topk_index (port of yolo26_filter): two-stage top-k
    # allowing multiple classes per anchor, no NMS.
    k = _Y26_TOP_K
    max_scores = scores.max(axis=1)                  # (A,)
    A = len(max_scores)
    k1 = min(k, A)
    ori_idx = np.argpartition(max_scores, -k1)[-k1:]  # top-k anchors
    topk_scores = scores[ori_idx]                     # (k1, 80)
    flat = topk_scores.reshape(-1)                    # (k1*80,)
    k2 = min(k, flat.size)
    top = np.argsort(flat)[::-1][:k2]                # top-k (anchor, class)
    sel_scores = flat[top]
    anchor_in_topk = top // _Y26_CLASSES
    class_ids = (top % _Y26_CLASSES).astype(np.int32)
    orig_anchor = ori_idx[anchor_in_topk]

    keep = sel_scores > (obj_thresh if obj_thresh > 0 else _Y26_SCORE_THRES)
    boxes = boxes[orig_anchor][keep]
    scores = sel_scores[keep]
    class_ids = class_ids[keep]
    coeffs = coeffs[orig_anchor][keep]
    if len(boxes) == 0:
        return None, None, None, None

    # Clip to the input frame.
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h)

    # Mask assembly: sigmoid(coeffs @ proto^T) upsampled to input size, then
    # cropped to each box (port of process_mask).
    ph, pw, pc = proto.shape
    masks = _sigmoid(coeffs @ proto.reshape(-1, pc).T)   # (N, ph*pw)
    masks = masks.reshape(-1, ph, pw)
    masks = np.stack([
        cv2.resize(m, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        for m in masks
    ], axis=0)                                           # (N, ih, iw)
    # Scale boxes to mask coords (mask is input-sized already after resize).
    masks = _crop_mask(masks, boxes)
    return boxes, scores, class_ids, masks


def _mask_color(cls_id):
    palette = [
        (54, 67, 244), (99, 30, 233), (176, 39, 156), (183, 58, 103),
        (181, 81, 63), (243, 150, 33), (244, 169, 3), (212, 188, 0),
        (136, 150, 0), (80, 175, 76), (74, 195, 139), (57, 220, 205),
        (59, 235, 255), (7, 193, 255), (0, 152, 255), (34, 87, 255),
        (72, 85, 121), (158, 158, 158), (139, 125, 96),
    ]
    return palette[int(cls_id) % len(palette)]


COCO_CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def unletterbox_boxes(boxes, lb_info):
    """Map xyxy boxes from the letterboxed input back to the original frame.
    lb_info = (ratio, dw, dh) captured by preprocess_frame."""
    if boxes is None or len(boxes) == 0:
        return boxes
    ratio, dw, dh = lb_info
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    return out


def unletterbox_masks(masks, lb_info, frame_shape):
    """Map (N, ih, iw) masks back to the original frame: crop the letterbox
    padding first, then resize to the frame. lb_info = (ratio, dw, dh)."""
    if masks is None or len(masks) == 0:
        return masks
    ratio, dw, dh = lb_info
    ih, iw = masks.shape[1], masks.shape[2]
    content_w = int(round(iw - 2 * dw)) if dw * 2 < iw else iw
    content_h = int(round(ih - 2 * dh)) if dh * 2 < ih else ih
    # Letterbox pads symmetrically (dw may be split left/right by the helper);
    # be conservative and use the helper convention: pad is centered.
    x0 = int(round(dw)) if dw > 0 else 0
    y0 = int(round(dh)) if dh > 0 else 0
    x1 = iw - x0 if dw > 0 else iw
    y1 = ih - y0 if dh > 0 else ih
    crops = masks[:, y0:y1, x0:x1]
    fh, fw = frame_shape[:2]
    out = np.stack([
        cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)
        for m in crops
    ], axis=0)
    return out


def draw_boxes(image, boxes, scores, class_ids, masks=None, lb_info=None,
               mask_thresh=0.5, mask_alpha=0.45):
    """Draw instance masks (overlaid, per-class color) + boxes + labels."""
    h, w = image.shape[:2]
    if masks is not None and len(masks):
        frame_masks = unletterbox_masks(masks, lb_info, image.shape) \
            if lb_info is not None else masks
        binary = frame_masks > mask_thresh
        overlay = image.copy()
        for i, cl in enumerate(class_ids):
            color = _mask_color(cl)
            overlay[binary[i]] = (
                (overlay[binary[i]].astype(np.float32) * (1 - mask_alpha)
                 + np.array(color, dtype=np.float32) * mask_alpha)
            ).astype(np.uint8)
        image[binary] = overlay[binary]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        cl = int(class_ids[i]) % len(COCO_CLASSES)
        color = _mask_color(cl)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{COCO_CLASSES[cl]} {float(scores[i]):.2f}',
                    (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1)


def preprocess_frame(frame, co_helper):
    """Letterbox + BGR to RGB. Returns (img, lb_info) where lb_info captures the
    exact ratio + padding used for this frame so the mask can be un-letterboxed
    independent of any shared co_helper state (the helper appends to its own
    list across threads, so relying on its last entry is racy)."""
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img, ratio, (dw, dh) = co_helper.letter_box(
        im=frame.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(0, 0, 0),
        info_need=True,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (ratio, dw, dh)

def inference_loop(cap, model, co_helper, is_video_file, target_fps):
    """Capture + preprocess + Hailo inference + draw on full-res frame.
    Pushes annotated frames into frame_buffer; the encode thread takes over from there.

    `target_fps` caps the loop rate so the preview path doesn't starve other
    Hailo/CPU consumers (notably VideoAnalyzer). Hailo can run 100+ fps but
    no human needs a 100 fps preview, and the leftover budget keeps offline
    video analysis healthy. Pass 0 to disable the cap.

    When VideoAnalyzer is running, the loop drops to 1 fps automatically; a frozen preview is fine for a few seconds, and the freed CPU/Hailo
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

            processed_img, lb_info = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time

            if outputs is not None:
                obj, nms = det_config.get()
                boxes, scores, class_ids, masks = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                if boxes is not None:
                    draw_boxes(frame, boxes, scores, class_ids, masks, lb_info)

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
    parser = argparse.ArgumentParser(description='YOLO26m-seg on CM5 + Hailo-8 (Web Preview Mode)')
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

    global _global_model, _global_co_helper, IMG_SIZE
    model = HailoInfer(args.model_path)
    # Pull the real (H, W) out of the .hef so any segmentation model (513, 640,
    # 1024, non-square, etc.) works without editing constants. Stored as (W, H)
    # to match the rest of the file.
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
        # Ask the camera to deliver MJPG so USB bandwidth isn't wasted on raw
        # YUYV. Most modern USB webcams do MJPG natively; if not, V4L2 quietly
        # falls back to YUYV and we just lose this win 鈥?no error.
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

