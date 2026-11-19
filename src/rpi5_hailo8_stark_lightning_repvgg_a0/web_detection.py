import argparse
import os
import subprocess
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from py_utils.hailo_executor import HailoInfer
from py_utils.stark_tracker import StarkTracker

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

_global_model = None

app = FastAPI(title="reComputer STARK Hailo-8")


def draw_state(image, state, trail, fps):
    x, y, w, h = [int(round(v)) for v in state]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"STARK {fps:.1f} FPS",
        (x, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    if len(trail) > 1:
        pts = np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [pts], False, (0, 200, 255), 2)


def _open_writer(output_path, width, height, fps):
    """Prefer ffmpeg/libx264; fall back to cv2's mp4v."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return proc, "ffmpeg"
    except FileNotFoundError:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if out.isOpened():
        return out, "mp4v"
    out.release()
    return None, None


class TrackingJob:
    def __init__(self):
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.output_filename = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self, input_path, output_path, box):
        with self._lock:
            if self.is_processing:
                return False
            self.is_processing = True
            self.progress = 0
            self.error_msg = ""
            self.current_video = os.path.basename(input_path)
            self.output_filename = os.path.basename(output_path)
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._process,
                args=(input_path, output_path, box),
                daemon=True,
            )
            self._thread.start()
            return True

    def _process(self, input_path, output_path, box):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Cannot open video: {input_path}"
            self.is_processing = False
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out, kind = _open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "No usable video writer (ffmpeg and cv2 mp4v failed)"
            self.is_processing = False
            cap.release()
            return

        tracker = StarkTracker(_global_model)
        trail = deque(maxlen=60)
        fps_ema = 0.0
        frame_idx = 0
        try:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read the first frame")
            tracker.initialize(frame, box)
            draw_state(frame, tracker.state, trail, 0.0)
            if kind == "ffmpeg":
                out.stdin.write(frame.tobytes())
            else:
                out.write(frame)
            frame_idx = 1
            self.progress = int(100 / max(total, 1))

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                started = time.time()
                state = tracker.update(frame)
                elapsed = time.time() - started
                inst = 1.0 / elapsed if elapsed > 0 else 0.0
                fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                trail.append(
                    (int(state[0] + 0.5 * state[2]), int(state[1] + 0.5 * state[3]))
                )
                draw_state(frame, state, trail, fps_ema)
                if kind == "ffmpeg":
                    out.stdin.write(frame.tobytes())
                else:
                    out.write(frame)
                frame_idx += 1
                if total > 0:
                    self.progress = min(99, int(frame_idx / total * 100))
        except Exception as exc:
            traceback.print_exc()
            self.error_msg = f"Tracking error: {exc}"
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


job = TrackingJob()


def _safe_name(filename):
    return os.path.basename(filename or "")


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    name = _safe_name(file.filename)
    if not name:
        raise HTTPException(status_code=400, detail="Missing filename")
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)
    return {"filename": name, "status": "uploaded"}


@app.get("/api/video/list")
async def list_videos():
    return {
        "uploads": sorted(os.listdir(UPLOAD_DIR)),
        "outputs": sorted(os.listdir(OUTPUT_DIR)),
    }


@app.get("/api/video/first_frame")
async def first_frame(filename: str):
    path = os.path.join(UPLOAD_DIR, _safe_name(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=400, detail="Cannot read the first frame")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/api/video/analyze")
async def analyze_video(
    filename: str = Form(...),
    x: float = Form(...),
    y: float = Form(...),
    w: float = Form(...),
    h: float = Form(...),
):
    name = _safe_name(filename)
    input_path = os.path.join(UPLOAD_DIR, name)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    if w <= 1 or h <= 1:
        raise HTTPException(status_code=400, detail="Initial box is too small")

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    x = max(0.0, x)
    y = max(0.0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    if w <= 1 or h <= 1:
        raise HTTPException(
            status_code=400,
            detail=f"Initial box must be inside the {width}x{height} frame",
        )

    base = os.path.splitext(name)[0]
    output_name = f"{base}_stark_results_{int(time.time() * 1000)}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    if not job.start(input_path, output_path, (x, y, w, h)):
        raise HTTPException(status_code=409, detail="Another tracking job is running")
    return {"status": "started", "output": output_name}


@app.post("/api/video/analyze/stop")
async def stop_analysis():
    job.stop()
    return {"status": "stopping"}


@app.get("/api/video/status")
async def video_status():
    return {
        "is_processing": job.is_processing,
        "progress": job.progress,
        "current_video": job.current_video,
        "output": job.output_filename,
        "error": job.error_msg,
    }


@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    path = os.path.join(OUTPUT_DIR, _safe_name(filename))
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=os.path.basename(path))


HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>STARK Tracking · Hailo-8</title>
<style>
  body { background:#141414; color:#eee; font-family:system-ui,sans-serif; margin:0; padding:24px; }
  .wrap { max-width:1100px; margin:0 auto; }
  h1 { color:#00e676; font-size:24px; margin:0 0 6px; }
  .sub { color:#999; margin-bottom:22px; }
  .card { background:#1f1f1f; border:1px solid #333; border-radius:10px; padding:18px; margin-bottom:18px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
  button { background:#00e676; color:#08210f; border:0; border-radius:6px; padding:9px 16px; font-weight:700; cursor:pointer; }
  button.secondary { background:#333; color:#eee; }
  button:disabled { background:#444; color:#888; cursor:not-allowed; }
  input[type=file] { color:#ddd; }
  #canvasWrap { display:none; margin-top:14px; }
  canvas { width:100%; height:auto; background:#000; cursor:crosshair; border-radius:6px; display:block; }
  .hint { color:#9e9e9e; font-size:13px; margin-top:8px; }
  .progress { height:20px; background:#333; border-radius:10px; overflow:hidden; margin-top:12px; }
  .bar { height:100%; width:0%; background:#00e676; transition:width .25s; }
  .status { margin-top:10px; color:#bbb; font-size:14px; }
  .error { color:#ff5252; }
  #result { display:none; margin-top:14px; }
  video { width:100%; border-radius:6px; background:#000; }
  a { color:#00e676; }
  ul { padding-left:18px; }
  li { margin:4px 0; }
  .pick { background:none; border:0; color:#00e676; cursor:pointer; padding:0; font:inherit; text-decoration:underline; }
</style>
</head>
<body>
<div class="wrap">
  <h1>STARK Single-Object Tracking · Hailo-8</h1>
  <div class="sub">Upload a video, drag a box around the target in the first frame, then run tracking.</div>

  <div class="card">
    <b>1. Upload video</b>
    <div class="row" style="margin-top:10px;">
      <input type="file" id="videoFile" accept=".mp4,.avi,.mov,.mkv"/>
      <button onclick="uploadVideo()">Upload</button>
      <button class="secondary" onclick="refreshList()">Refresh list</button>
    </div>
    <div id="fileList" class="hint"></div>
  </div>

  <div class="card">
    <b>2. Select the target in the first frame</b>
    <div id="canvasWrap">
      <canvas id="canvas"></canvas>
      <div class="hint">Drag a rectangle around the target. Selected box: <span id="boxText">none</span></div>
    </div>
    <div id="noFrame" class="hint">Choose an uploaded video first.</div>
  </div>

  <div class="card">
    <b>3. Run tracking</b>
    <div class="row" style="margin-top:10px;">
      <button id="runBtn" onclick="runTracking()">Run tracking</button>
      <button class="secondary" onclick="stopTracking()">Stop</button>
    </div>
    <div class="progress"><div id="bar" class="bar"></div></div>
    <div id="status" class="status">Idle.</div>
    <div id="result">
      <video id="resultVideo" controls></video>
      <div style="margin-top:8px;"><a id="downloadLink" href="#">Download annotated MP4</a></div>
    </div>
  </div>
</div>

<script>
let currentVideo = null;
let firstImage = null;
let selection = null;
let dragStart = null;
let pollTimer = null;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function setStatus(text, isError) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = 'status' + (isError ? ' error' : '');
}

async function uploadVideo() {
  const input = document.getElementById('videoFile');
  if (!input.files[0]) { alert('Choose a video first'); return; }
  const form = new FormData();
  form.append('file', input.files[0]);
  setStatus('Uploading...');
  try {
    const res = await fetch('/api/video/upload', { method:'POST', body:form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Upload failed');
    currentVideo = data.filename;
    await loadFirstFrame(data.filename);
    refreshList();
    setStatus('Uploaded. Drag a box around the target.');
  } catch (e) {
    setStatus(e.message, true);
  }
}

async function loadFirstFrame(filename) {
  currentVideo = filename;
  selection = null;
  document.getElementById('boxText').textContent = 'none';
  const res = await fetch('/api/video/first_frame?filename=' + encodeURIComponent(filename));
  if (!res.ok) { setStatus('Cannot load first frame', true); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    firstImage = img;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    drawCanvas();
    document.getElementById('canvasWrap').style.display = 'block';
    document.getElementById('noFrame').style.display = 'none';
    URL.revokeObjectURL(url);
  };
  img.src = url;
}

function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (firstImage) ctx.drawImage(firstImage, 0, 0, canvas.width, canvas.height);
  if (selection) {
    ctx.fillStyle = 'rgba(0,230,118,0.15)';
    ctx.fillRect(selection.x, selection.y, selection.w, selection.h);
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = Math.max(2, canvas.width / 600);
    ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
  }
}

function canvasPoint(ev) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.min(Math.max(0, (ev.clientX - rect.left) * canvas.width / rect.width), canvas.width),
    y: Math.min(Math.max(0, (ev.clientY - rect.top) * canvas.height / rect.height), canvas.height),
  };
}

canvas.addEventListener('mousedown', (ev) => {
  if (!firstImage) return;
  dragStart = canvasPoint(ev);
  selection = { x: dragStart.x, y: dragStart.y, w: 0, h: 0 };
  drawCanvas();
});
canvas.addEventListener('mousemove', (ev) => {
  if (!dragStart) return;
  const p = canvasPoint(ev);
  selection = {
    x: Math.min(dragStart.x, p.x),
    y: Math.min(dragStart.y, p.y),
    w: Math.abs(p.x - dragStart.x),
    h: Math.abs(p.y - dragStart.y),
  };
  drawCanvas();
});
window.addEventListener('mouseup', () => {
  if (!dragStart) return;
  dragStart = null;
  if (!selection || selection.w < 4 || selection.h < 4) {
    selection = null;
    document.getElementById('boxText').textContent = 'none';
  } else {
    document.getElementById('boxText').textContent =
      `${Math.round(selection.x)}, ${Math.round(selection.y)}, ` +
      `${Math.round(selection.w)}, ${Math.round(selection.h)}`;
  }
  drawCanvas();
});

async function runTracking() {
  if (!currentVideo) { alert('Upload a video first'); return; }
  if (!selection || selection.w < 4 || selection.h < 4) {
    alert('Drag a box around the target in the first frame');
    return;
  }
  const form = new FormData();
  form.append('filename', currentVideo);
  form.append('x', selection.x);
  form.append('y', selection.y);
  form.append('w', selection.w);
  form.append('h', selection.h);

  document.getElementById('runBtn').disabled = true;
  document.getElementById('result').style.display = 'none';
  const oldVideo = document.getElementById('resultVideo');
  oldVideo.pause();
  oldVideo.removeAttribute('src');
  oldVideo.load();
  setStatus('Starting...');
  const res = await fetch('/api/video/analyze', { method:'POST', body:form });
  const data = await res.json();
  if (!res.ok) {
    document.getElementById('runBtn').disabled = false;
    setStatus(data.detail || 'Cannot start tracking', true);
    return;
  }
  startPolling();
}

async function stopTracking() {
  await fetch('/api/video/analyze/stop', { method:'POST' });
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    const res = await fetch('/api/video/status');
    const data = await res.json();
    document.getElementById('bar').style.width = data.progress + '%';
    setStatus(`Processing ${data.current_video || ''} ... ${data.progress}%`, !!data.error);
    if (data.error) {
      clearInterval(pollTimer);
      document.getElementById('runBtn').disabled = false;
      setStatus(data.error, true);
      return;
    }
    if (!data.is_processing) {
      clearInterval(pollTimer);
      document.getElementById('runBtn').disabled = false;
      if (data.error) {
        setStatus(data.error, true);
      } else if (data.progress >= 100) {
        setStatus('Done.');
        if (data.output) {
          const url = '/api/video/download/' + encodeURIComponent(data.output) + '?v=' + Date.now();
          const video = document.getElementById('resultVideo');
          video.src = url;
          video.load();
          document.getElementById('downloadLink').href = url;
          document.getElementById('result').style.display = 'block';
        }
        refreshList();
      } else {
        setStatus('Stopped.');
      }
    }
  }, 1000);
}

async function refreshList() {
  const res = await fetch('/api/video/list');
  const data = await res.json();
  const el = document.getElementById('fileList');
  el.innerHTML = '';
  if (!data.uploads.length) { el.textContent = 'No uploaded videos yet.'; return; }
  const title = document.createElement('div');
  title.textContent = 'Uploaded videos:';
  el.appendChild(title);
  const ul = document.createElement('ul');
  data.uploads.forEach((name) => {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.className = 'pick';
    b.textContent = name;
    b.onclick = () => loadFirstFrame(name);
    li.appendChild(b);
    ul.appendChild(li);
  });
  el.appendChild(ul);
}

refreshList();
fetch('/api/video/status').then(r => r.json()).then(d => {
  if (d.is_processing) startPolling();
});
</script>
</body>
</html>"""


@app.get("/")
async def index():
    return Response(content=HTML, media_type="text/html")


def main():
    parser = argparse.ArgumentParser(description="STARK single-object tracking on Hailo-8")
    parser.add_argument("--model_path", required=True, help="Path to the STARK HEF")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global _global_model
    _global_model = HailoInfer(args.model_path)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", log_config=None)


if __name__ == "__main__":
    main()