import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse

from py_utils.landmark_utils import (
    FACE_LANDMARK_COUNT,
    extract_landmarks,
    expand_face_box,
    map_landmarks_to_frame,
)

sys.path.insert(0, os.path.dirname(__file__))

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer

    HAILO_AVAILABLE = True
except ImportError as exc:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({exc}), inference will fail")

MODEL_SLUG = "face_landmarks_lite"
MODEL_INPUT_SIZE = (192, 192)  # (width, height)
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "model", "face_landmarks_lite.hef"
)
DEFAULT_YUNET = os.path.join(
    os.path.dirname(__file__),
    "model",
    "face_detection_yunet_2023mar.onnx",
)

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Config:
    def __init__(self, score_threshold=0.6, nms_threshold=0.3, max_faces=8):
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_faces = int(max_faces)
        self.lock = threading.Lock()

    def update(self, score_threshold=None, nms_threshold=None, max_faces=None):
        with self.lock:
            if score_threshold is not None:
                self.score_threshold = max(0.0, min(1.0, float(score_threshold)))
            if nms_threshold is not None:
                self.nms_threshold = max(0.0, min(1.0, float(nms_threshold)))
            if max_faces is not None:
                self.max_faces = max(1, min(32, int(max_faces)))

    def get(self):
        with self.lock:
            return (
                self.score_threshold,
                self.nms_threshold,
                self.max_faces,
            )


config = Config()


class FaceDetector:
    """YuNet face detector running on the CPU; it supplies landmark ROIs."""

    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.3):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "OpenCV FaceDetectorYN is required (opencv-python >= 4.5.4)"
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet model not found: {model_path}")

        self._lock = threading.Lock()
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (640, 360),
            float(score_threshold),
            float(nms_threshold),
            5000,
        )
        self.model_path = model_path
        print(f"YuNet face detector loaded: {model_path}")

    def detect(self, frame, score_threshold, nms_threshold):
        h, w = frame.shape[:2]
        if h < 2 or w < 2:
            return np.empty((0, 15), dtype=np.float32)

        scale = min(1.0, 960.0 / float(max(h, w)))
        dw = max(2, int(round(w * scale)))
        dh = max(2, int(round(h * scale)))
        small = cv2.resize(
            frame,
            (dw, dh),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
        )

        with self._lock:
            # OpenCV exposes these setters on FaceDetectorYN.  If a build does
            # not, the thresholds supplied at create() remain in effect.
            try:
                self._detector.setScoreThreshold(float(score_threshold))
                self._detector.setNMSThreshold(float(nms_threshold))
            except AttributeError:
                pass
            self._detector.setInputSize((dw, dh))
            _, faces = self._detector.detect(small)

        if faces is None:
            return np.empty((0, 15), dtype=np.float32)

        faces = np.asarray(faces, dtype=np.float32).copy()
        # FaceDetectorYN returns boxes and five 2-D landmarks in the resized
        # image; column 14 is the confidence and must not be scaled.
        inv_scale = 1.0 / scale
        faces[:, 0:14] *= inv_scale
        return faces


def draw_face(image, box, score, points):
    x, y, w, h = [int(round(v)) for v in box[:4]]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"face {score:.2f}",
        (x, max(18, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    for px, py, _ in points:
        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
            cv2.circle(image, (int(px), int(py)), 1, (0, 255, 0), -1)


def process_frame(frame, model, detector, score_threshold, nms_threshold, max_faces):
    """Detect faces, run the Hailo landmark HEF on each crop, draw results."""
    detections = detector.detect(frame, score_threshold, nms_threshold)
    results = []
    if detections.size == 0:
        return frame, results

    h, w = frame.shape[:2]
    for face in detections[:max_faces]:
        face_score = float(face[14])
        if face_score < score_threshold:
            continue

        crop_box = expand_face_box(face[:4], w, h, margin=0.2)
        if crop_box is None:
            continue

        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(
            crop, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        outputs = model.run(rgb)
        points, landmark_score, normalized = extract_landmarks(outputs)
        if points is None:
            continue

        points = map_landmarks_to_frame(points, crop_box)
        confidence = (
            landmark_score
            if landmark_score is not None
            else face_score
        )
        if not 0.0 <= confidence <= 1.0:
            confidence = face_score

        draw_face(frame, face[:4], face_score, points)
        results.append(
            {
                "box": {
                    "x": float(face[0]),
                    "y": float(face[1]),
                    "width": float(face[2]),
                    "height": float(face[3]),
                },
                "score": face_score,
                "landmark_score": float(confidence),
                "normalized_coordinates": bool(normalized),
                "points": points,
            }
        )

    return frame, results


class VideoAnalyzer:
    def __init__(self):
        self.model = None
        self.detector = None
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model, detector):
        self.model = model
        self.detector = detector

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

    @staticmethod
    def _open_writer(output_path, width, height, fps):
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-threads",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print("[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, "ffmpeg"
        except FileNotFoundError:
            pass

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(
                "[VideoAnalyzer] Using cv2 mp4v (install ffmpeg for speed)",
                flush=True,
            )
            return out, "mp4v"
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"
            self.is_processing = False
            cap.release()
            return

        out, kind = self._open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "Error: No usable video writer"
            self.is_processing = False
            cap.release()
            return

        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if self.model is not None and self.detector is not None:
                    conf, iou, max_faces = config.get()
                    process_frame(
                        frame, self.model, self.detector, conf, iou, max_faces
                    )
                if kind == "ffmpeg":
                    out.stdin.write(frame.tobytes())
                else:
                    out.write(frame)
                frame_idx += 1
                self.progress = int((frame_idx / total_frames) * 100)
        except Exception as exc:
            self.error_msg = f"Process error: {exc}"
        finally:
            cap.release()
            if kind == "ffmpeg":
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

app = FastAPI(title="reComputer Face Landmarks Lite Hailo-8")


@app.get("/api/config")
async def get_config():
    conf, iou, max_faces = config.get()
    return {
        "score_threshold": conf,
        "nms_threshold": iou,
        "max_faces": max_faces,
    }


@app.post("/api/config")
async def update_config(payload: dict):
    config.update(
        score_threshold=payload.get(
            "score_threshold",
            payload.get("conf", payload.get("obj_thresh")),
        ),
        nms_threshold=payload.get(
            "nms_threshold",
            payload.get("iou", payload.get("nms_thresh")),
        ),
        max_faces=payload.get("max_faces"),
    )
    return await get_config()


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename or "upload.mp4")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"success": True, "filename": filename, "path": path}


@app.get("/api/video/list")
async def list_videos():
    uploads = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        if os.path.isfile(os.path.join(UPLOAD_DIR, name)):
            uploads.append(name)
    outputs = []
    for name in sorted(os.listdir(OUTPUT_DIR)):
        if os.path.isfile(os.path.join(OUTPUT_DIR, name)):
            outputs.append(name)
    return {"uploads": uploads, "outputs": outputs}


@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    safe_name = os.path.basename(filename)
    input_path = os.path.join(UPLOAD_DIR, safe_name)
    if not os.path.isfile(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    stem, _ = os.path.splitext(safe_name)
    output_name = f"{stem}_face_landmarks.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    started = video_analyzer.start_analysis(input_path, output_path)
    if not started:
        raise HTTPException(status_code=409, detail="Another video is processing")
    return {"success": True, "status": "started", "output": output_name}


@app.get("/api/video/status")
async def get_analysis_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg,
    }


@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


_global_model = None
_global_detector = None


@app.post(f"/api/models/{MODEL_SLUG}/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
):
    if _global_model is None or _global_detector is None:
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
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ) as tmp:
                tmp.write(await video.read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            try:
                if cap.isOpened():
                    if timestamp is not None:
                        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                    ret, frame = cap.read()
                    if ret:
                        img = frame
                        source_info = (
                            f"video frame at {timestamp if timestamp else 0}s"
                        )
            finally:
                cap.release()
                os.unlink(tmp_path)
        else:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"

        if img is None:
            return {
                "success": False,
                "message": "No valid input source found",
            }

        score_threshold, nms_threshold, max_faces = config.get()
        if conf is not None:
            score_threshold = max(0.0, min(1.0, float(conf)))
        if iou is not None:
            nms_threshold = max(0.0, min(1.0, float(iou)))

        annotated = img.copy()
        _, results = process_frame(
            annotated,
            _global_model,
            _global_detector,
            score_threshold,
            nms_threshold,
            max_faces,
        )

        height, width = img.shape[:2]
        predictions = []
        for result in results:
            predictions.append(
                {
                    "box": result["box"],
                    "score": result["score"],
                    "landmark_score": result["landmark_score"],
                    "landmarks": [
                        {
                            "x": float(point[0]),
                            "y": float(point[1]),
                            "z": float(point[2]),
                        }
                        for point in result["points"]
                    ],
                }
            )

        return {
            "success": True,
            "source": source_info,
            "model": MODEL_SLUG,
            "count": len(predictions),
            "predictions": predictions,
            "image": {"width": width, "height": height},
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


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
            self.cond.wait_for(
                lambda: self.annotated_version > last_version,
                timeout=timeout,
            )
            return self.annotated, self.annotated_version

    def push_jpeg(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.jpeg_version += 1
            self.cond.notify_all()

    def wait_jpeg(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(
                lambda: self.jpeg_version > last_version,
                timeout=timeout,
            )
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
                lambda: self.version > self._last_read_version
                or self._stopped,
                timeout=timeout,
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
        last_version = -1
        while not stop_event.is_set():
            jpeg, last_version = frame_buffer.wait_jpeg(
                last_version, timeout=1.0
            )
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )

    return StreamingResponse(
        generate(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/")
async def index():
    return Response(
        content="""<!doctype html>
<html>
  <head>
    <title>Face Landmarks Lite · RPi5 + Hailo-8</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
      .container { max-width: 1200px; margin: 0 auto; }
      .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; width: 100%; max-width: 900px; }
      .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; vertical-align: top; margin: 10px; }
      .control-group { margin-bottom: 15px; }
      .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
      .slider-container { display: flex; align-items: center; gap: 15px; }
      input[type=range] { flex-grow: 1; cursor: pointer; }
      .value-display { min-width: 50px; font-family: monospace; background: #444; padding: 2px 8px; border-radius: 4px; text-align: center; }
      h1 { color: #00e676; margin-bottom: 4px; }
      .subtitle { color: #888; margin-top: 0; }
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
      <h1>Face Landmarks Lite · RPi5 + Hailo-8</h1>
      <p class="subtitle">468 facial landmarks per face · Hailo-8 HEF + OpenCV YuNet CPU face ROI</p>
      <div class="tabs">
        <div class="tab active" onclick="showTab('realtime', event)">Real-time Landmarks</div>
        <div class="tab" onclick="showTab('analysis', event)">Local Video Analysis</div>
      </div>

      <div id="realtime" class="tab-content active">
        <div class="video-box">
          <img id="streamImg" src="/api/video_feed" style="max-width: 100%; height: auto;" alt="live preview">
        </div>
        <div class="controls">
          <div class="control-group">
            <label>Face Confidence Threshold</label>
            <div class="slider-container">
              <input type="range" id="confSlider" min="0.10" max="0.95" step="0.05" value="0.60">
              <span id="confValue" class="value-display">0.60</span>
            </div>
          </div>
          <div class="control-group">
            <label>NMS IOU Threshold (CPU)</label>
            <div class="slider-container">
              <input type="range" id="iouSlider" min="0.10" max="0.90" step="0.05" value="0.30">
              <span id="iouValue" class="value-display">0.30</span>
            </div>
          </div>
          <p style="color:#888; margin-bottom:0;">REST: POST /api/models/face_landmarks_lite/predict</p>
        </div>
      </div>

      <div id="analysis" class="tab-content">
        <div class="video-analysis">
          <h3>Analyze Local Video</h3>
          <div class="control-group">
            <label>Upload New Video (.mp4, .avi, .mov)</label>
            <input type="file" id="videoUpload" accept=".mp4,.avi,.mov">
            <button class="btn" id="uploadBtn" onclick="uploadVideo(event)">Upload</button>
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
      function showTab(tabId, event) {
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        if (event && event.currentTarget) event.currentTarget.classList.add('active');
        if (tabId === 'realtime') { document.getElementById('streamImg').src = '/api/video_feed'; }
        else { document.getElementById('streamImg').src = ''; refreshFileList(); }
      }

      const confSlider = document.getElementById('confSlider');
      const iouSlider = document.getElementById('iouSlider');
      const confValue = document.getElementById('confValue');
      const iouValue = document.getElementById('iouValue');

      function updateConfig() {
        const score_threshold = parseFloat(confSlider.value);
        const nms_threshold = parseFloat(iouSlider.value);
        confValue.innerText = score_threshold.toFixed(2);
        iouValue.innerText = nms_threshold.toFixed(2);
        fetch('/api/config', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ score_threshold, nms_threshold })
        });
      }
      confSlider.oninput = updateConfig;
      iouSlider.oninput = updateConfig;
      fetch('/api/config').then(res => res.json()).then(data => {
        confSlider.value = data.score_threshold;
        iouSlider.value = data.nms_threshold;
        confValue.innerText = Number(data.score_threshold).toFixed(2);
        iouValue.innerText = Number(data.nms_threshold).toFixed(2);
      });

      async function uploadVideo(event) {
        const fileInput = document.getElementById('videoUpload');
        if (!fileInput.files[0]) return alert('Please select a file');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        const btn = event.currentTarget;
        btn.disabled = true;
        btn.innerText = 'Uploading...';
        try {
          const res = await fetch('/api/video/upload', { method: 'POST', body: formData });
          if (!res.ok) throw new Error('Upload failed');
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
        (data.uploads || []).forEach(f => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${f} (Original)</td><td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>`;
          tbody.appendChild(tr);
        });
        (data.outputs || []).forEach(f => {
          const tr = document.createElement('tr');
          tr.innerHTML = `<td>${f} (Analyzed)</td><td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>`;
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
          alert(data.detail || data.message || 'Error starting analysis');
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
""",
        media_type="text/html",
    )


def run_fastapi(host, port):
    uvicorn.run(app, host=host, port=port, log_level="warning")


def inference_loop(cap, model, detector, is_video_file, target_fps):
    fps_counter = 0.0
    target_period = 1.0 / target_fps if target_fps > 0 else 0.0
    next_time = time.time()

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

        conf, iou, max_faces = config.get()
        start_time = time.time()
        process_frame(frame, model, detector, conf, iou, max_faces)
        inference_time = time.time() - start_time

        inf_fps = 1.0 / inference_time if inference_time > 0 else 0.0
        fps_counter = (
            0.9 * fps_counter + 0.1 * inf_fps
            if fps_counter > 0
            else inf_fps
        )
        cv2.putText(
            frame,
            f"pipeline FPS: {fps_counter:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame_buffer.push_annotated(frame)

        if target_period > 0:
            now = time.time()
            next_time += target_period
            sleep_for = next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -target_period:
                next_time = now + target_period

    stop_event.set()


def encode_loop(preview_w, preview_h, jpeg_quality):
    last_version = -1
    while not stop_event.is_set():
        frame, last_version = frame_buffer.wait_annotated(
            last_version, timeout=1.0
        )
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if (
            preview_w > 0
            and preview_h > 0
            and (w, h) != (preview_w, preview_h)
        ):
            preview = cv2.resize(
                frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA
            )
        else:
            preview = frame
        ok, buf = cv2.imencode(
            ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if ok:
            frame_buffer.push_jpeg(buf.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description="Face Landmarks Lite on CM5 + Hailo-8 (Web Preview Mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Hailo-8 face_landmarks_lite.hef",
    )
    parser.add_argument(
        "--yunet_model",
        type=str,
        default=DEFAULT_YUNET,
        help="Path to the YuNet ONNX face detector",
    )
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera ID; -1 disables the camera (web/API only)",
    )
    parser.add_argument(
        "--video_path", type=str, help="Video file (overrides camera_id)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview_width", type=int, default=1280)
    parser.add_argument("--preview_height", type=int, default=720)
    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument("--cam_width", type=int, default=1280)
    parser.add_argument("--cam_height", type=int, default=720)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--max_faces", type=int, default=8)
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print(
            "Error: HailoRT is not available. Install the HailoRT 4.23 "
            "aarch64 wheel matching the host driver."
        )
        return

    config.update(args.conf, args.iou, args.max_faces)

    global _global_model, _global_detector
    model = HailoInfer(args.model_path)
    detector = FaceDetector(args.yunet_model, args.conf, args.iou)
    _global_model = model
    _global_detector = detector
    video_analyzer.set_engine(model, detector)

    web_thread = threading.Thread(
        target=run_fastapi, args=(args.host, args.port), daemon=True
    )
    web_thread.start()
    print(f"Web preview: http://{args.host}:{args.port}", flush=True)

    if args.camera_id == -1 and not args.video_path:
        print("Running in web/API only mode", flush=True)
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
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
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        capture_source = None
        is_video_file = False

    if not cap.isOpened():
        print("Error: cannot open video source", flush=True)
        model.release()
        return

    if not is_video_file:
        capture_source = LatestFrameReader(cap).start()

    inference_thread = threading.Thread(
        target=inference_loop,
        args=(
            capture_source,
            model,
            detector,
            is_video_file,
            args.target_fps,
        ),
        daemon=True,
    )
    encode_thread = threading.Thread(
        target=encode_loop,
        args=(args.preview_width, args.preview_height, args.jpeg_quality),
        daemon=True,
    )
    inference_thread.start()
    encode_thread.start()

    try:
        while inference_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if not is_video_file:
            capture_source.stop()
        inference_thread.join(timeout=2)
        encode_thread.join(timeout=2)
        cap.release()
        model.release()


if __name__ == "__main__":
    main()