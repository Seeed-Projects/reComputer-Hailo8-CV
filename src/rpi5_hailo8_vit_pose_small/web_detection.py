import os, sys, cv2, argparse, time, subprocess, threading
import numpy as np
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn, shutil
from typing import Optional
from py_utils.coco_utils import COCO_test_helper

stop_event = threading.Event()
try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e})")

OBJ_THRESH = 0.3  # keypoint confidence threshold
NMS_THRESH = 0.45  # unused (no NMS in pose), kept for API parity
IMG_SIZE = (192, 256)  # (width, height) — overridden at runtime

# COCO 17 keypoints
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

# Skeleton connections (0-indexed)
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]

KPT_COLORS = np.array([
    [0,255,0],[0,255,0],[0,255,0],[0,255,0],[0,255,0],
    [51,153,255],[51,153,255],[51,153,255],[51,153,255],[51,153,255],
    [51,153,255],[255,128,0],[255,128,0],[255,128,0],[255,128,0],
    [255,128,0],[255,128,0],
], dtype=np.uint8)

LINK_COLORS = np.array([
    [255,128,0],[255,128,0],[255,128,0],[255,128,0],[255,51,255],
    [255,51,255],[255,51,255],[51,153,255],[51,153,255],[51,153,255],
    [51,153,255],[51,153,255],[0,255,0],[0,255,0],[0,255,0],
    [0,255,0],[0,255,0],[0,255,0],[0,255,0],
], dtype=np.uint8)

_DET_LOGGED = False

class DetectionConfig:
    def __init__(self):
        self.obj_thresh = OBJ_THRESH
        self.nms_thresh = NMS_THRESH
        self.lock = threading.Lock()
    def update(self, obj, nms):
        with self.lock: self.obj_thresh = obj; self.nms_thresh = nms
    def get(self):
        with self.lock: return self.obj_thresh, self.nms_thresh

det_config = DetectionConfig()
UPLOAD_DIR = "workspace/uploads"; OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True); os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoAnalyzer:
    def __init__(self):
        self.model = None; self.co_helper = None
        self.is_processing = False; self.progress = 0
        self.current_video = ""; self.error_msg = ""
        self._stop = threading.Event(); self._thread = None
    def set_engine(self, m, c): self.model = m; self.co_helper = c
    def start_analysis(self, inp, out):
        if self.is_processing: return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._proc, args=(inp, out))
        self._thread.daemon = True; self._thread.start(); return True
    @staticmethod
    def _writer(path, w, h, fps):
        cmd = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo',
               '-pix_fmt','bgr24','-s',f'{w}x{h}','-r',f'{fps}','-i','-',
               '-c:v','libx264','-preset','ultrafast','-threads','0',
               '-pix_fmt','yuv420p','-movflags','+faststart',path]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return p, 'ffmpeg'
        except FileNotFoundError: pass
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))
        return (out, 'mp4v') if out.isOpened() else (None, None)
    def _proc(self, inp, out):
        self.is_processing = True; self.progress = 0; self.error_msg = ""
        self.current_video = os.path.basename(inp)
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened(): self.error_msg = "Cannot open"; self.is_processing = False; return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: self.error_msg = "Invalid frames"; self.is_processing = False; cap.release(); return
        wr, kind = self._writer(out, w, h, fps)
        if wr is None: self.error_msg = "No writer"; self.is_processing = False; cap.release(); return
        idx = 0
        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret: break
                if self.model and self.co_helper:
                    img, lb = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(img)
                    if outputs is not None:
                        obj, _ = det_config.get()
                        kpts = post_process_hailo(outputs, obj, 0, IMG_SIZE[1], IMG_SIZE[0])
                        if kpts is not None:
                            real_kpts = unletterbox_keypoints(kpts, lb)
                            draw(frame, real_kpts, obj)
                if kind == 'ffmpeg': wr.stdin.write(frame.tobytes())
                else: wr.write(frame)
                idx += 1; self.progress = int(idx / total * 100)
        except Exception as e: self.error_msg = str(e)
        finally:
            cap.release()
            if kind == 'ffmpeg':
                try: wr.stdin.close()
                except: pass
                try: wr.wait(30)
                except: wr.kill()
            else: wr.release()
            self.is_processing = False
            if not self.error_msg: self.progress = 100
    def stop(self): self._stop.set()

video_analyzer = VideoAnalyzer()
app = FastAPI(title="reComputer ViTPose-Small Hailo-8")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get(); return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(c: dict):
    det_config.update(c.get("obj_thresh", OBJ_THRESH), c.get("nms_thresh", NMS_THRESH))
    return {"status": "success"}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    p = os.path.join(UPLOAD_DIR, file.filename)
    with open(p, "wb") as b: shutil.copyfileobj(file.file, b)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    return {"uploads": os.listdir(UPLOAD_DIR), "outputs": os.listdir(OUTPUT_DIR)}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    inp = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(inp): raise HTTPException(404, "Not found")
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened(): raise HTTPException(400, "Cannot open")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    nb = os.path.splitext(filename)[0]
    of = f"{nb}_{w}x{h}_results.mp4"; op = os.path.join(OUTPUT_DIR, of)
    s = video_analyzer.start_analysis(inp, op)
    return {"status": "started", "output": of} if s else {"status": "error", "message": "Busy"}

@app.get("/api/video/status")
async def get_status():
    return {"is_processing": video_analyzer.is_processing, "progress": video_analyzer.progress,
            "current_video": video_analyzer.current_video, "error": video_analyzer.error_msg}

@app.get("/api/video/download/{fn}")
async def download(fn: str):
    p = os.path.join(OUTPUT_DIR, fn)
    if not os.path.exists(p): raise HTTPException(404, "Not found")
    return FileResponse(p, media_type='video/mp4', filename=fn)

_g_model = None; _g_co = None

@app.post("/api/models/vit_pose_small/predict")
async def predict(file: Optional[UploadFile] = File(None), video: Optional[UploadFile] = File(None),
                  timestamp: Optional[float] = Form(None), realtime: Optional[bool] = Form(False),
                  conf: Optional[float] = Form(None), iou: Optional[float] = Form(None)):
    if _g_model is None or _g_co is None: return {"success": False, "message": "Not initialized"}
    try:
        img = None; src = ""
        if file:
            c = await file.read(); img = cv2.imdecode(np.frombuffer(c, np.uint8), cv2.IMREAD_COLOR); src = "image"
        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t: t.write(await video.read()); tp = t.name
            cap = cv2.VideoCapture(tp)
            if cap.isOpened():
                if timestamp is not None: cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                r, fr = cap.read()
                if r: img = fr; src = f"frame @ {timestamp or 0}s"
                cap.release()
            os.unlink(tp)
        if img is None: img = frame_buffer.get_raw_frame(); src = "camera"
        if img is None: return {"success": False, "message": "No input"}
        h, w = img.shape[:2]
        inp, lb = preprocess_frame(img, _g_co)
        outputs = _g_model.run(inp)
        obj, _ = det_config.get()
        kpts = post_process_hailo(outputs, conf or obj, 0, IMG_SIZE[1], IMG_SIZE[0])
        preds = []
        if kpts is not None:
            rk = unletterbox_keypoints(kpts, lb)
            for i, (x, y, s) in enumerate(rk):
                if s >= (conf or obj) and np.isfinite(x) and np.isfinite(y):
                    preds.append({"keypoint": KEYPOINT_NAMES[i], "x": int(x), "y": int(y), "score": float(s)})
        return {"success": True, "source": src, "keypoints": preds, "image": {"width": w, "height": h}}
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    def __init__(self):
        self.raw = None; self.annotated = None; self.av = 0
        self.jpeg = None; self.jv = 0; self.cond = threading.Condition()
    def push_annotated(self, f):
        with self.cond: self.raw = f; self.annotated = f; self.av += 1; self.cond.notify_all()
    def wait_annotated(self, v, timeout=1.0):
        with self.cond: self.cond.wait_for(lambda: self.av > v, timeout=timeout); return self.annotated, self.av
    def push_jpeg(self, j):
        with self.cond: self.jpeg = j; self.jv += 1; self.cond.notify_all()
    def wait_jpeg(self, v, timeout=1.0):
        with self.cond: self.cond.wait_for(lambda: self.jv > v, timeout=timeout); return self.jpeg, self.jv
    def get_raw_frame(self):
        with self.cond: return self.raw.copy() if self.raw is not None else None

frame_buffer = FrameBuffer()

class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap; self.frame = None; self.v = 0; self._lr = 0
        self._stop = False; self._c = threading.Condition()
        self._t = threading.Thread(target=self._loop, daemon=True)
    def start(self): self._t.start(); return self
    def _loop(self):
        while not stop_event.is_set():
            r, f = self.cap.read()
            if not r: time.sleep(0.01); continue
            with self._c: self.frame = f; self.v += 1; self._c.notify_all()
        with self._c: self._stop = True; self._c.notify_all()
    def read(self, t=1.0):
        with self._c:
            self._c.wait_for(lambda: self.v > self._lr or self._stop, timeout=t)
            if self.frame is None: return False, None
            self._lr = self.v; return True, self.frame.copy()
    def stop(self):
        with self._c: self._stop = True; self._c.notify_all()
        self._t.join(timeout=2)

@app.get("/api/video_feed")
async def video_feed():
    def gen():
        v = -1
        while True:
            j, v = frame_buffer.wait_jpeg(v, timeout=1.0)
            if j: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + j + b'\r\n')
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html><head><title>reComputer ViTPose-Small · Hailo-8</title>
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
    <div class="container"><h1>ViTPose-Small · RPi5 + Hailo-8</h1>
    <div class="tabs"><div class="tab active" onclick="showTab('rt')">Real-time Pose</div><div class="tab" onclick="showTab('an')">Local Video</div></div>
    <div id="rt" class="tab-content active">
      <div class="video-box"><img id="s" src="/api/video_feed" style="max-width:100%;height:auto;"></div>
      <div class="controls"><div class="control-group"><label>Keypoint Confidence</label><div class="slider-container"><input type="range" id="c" min="0.01" max="1.0" step="0.01" value="0.30"><span id="cv" class="value-display">0.30</span></div></div></div>
    </div>
    <div id="an" class="tab-content"><div class="video-analysis"><h3>Analyze Video</h3>
      <div class="control-group"><label>Upload (.mp4)</label><input type="file" id="vu" accept=".mp4"><button class="btn" onclick="uv()">Upload</button></div>
      <div id="pa" style="display:none;"><p>Processing: <span id="cf">-</span></p><div class="progress-container"><div id="pb" class="progress-bar"></div><div id="pt" class="progress-text">0%</div></div><p id="et" style="color:#ff5252;"></p></div>
      <div class="control-group"><button class="btn" onclick="rf()">Refresh</button><table><thead><tr><th>File</th><th>Action</th></tr></thead><tbody id="ft"></tbody></table></div>
    </div></div><p style="color:#888;margin-top:20px;">FastAPI + MJPEG | Port: 8000</p></div>
    <script>
      function showTab(t){document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(t).classList.add('active');event.currentTarget.classList.add('active');if(t==='rt'){document.getElementById('s').src='/api/video_feed'}else{document.getElementById('s').src='';rf()}}
      const c=document.getElementById('c'),cv=document.getElementById('cv');function uc(){const o=parseFloat(c.value);cv.innerText=o.toFixed(2);fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({obj_thresh:o,nms_thresh:0.45})})}c.oninput=uc;fetch('/api/config').then(r=>r.json()).then(d=>{c.value=d.obj_thresh;cv.innerText=d.obj_thresh.toFixed(2)});
      async function uv(){const f=document.getElementById('vu');if(!f.files[0])return alert('Select a file');const fd=new FormData();fd.append('file',f.files[0]);const b=event.currentTarget;b.disabled=true;b.innerText='Uploading...';try{await fetch('/api/video/upload',{method:'POST',body:fd});alert('Uploaded');rf()}catch(e){alert('Failed')}finally{b.disabled=false;b.innerText='Upload'}}
      async function rf(){const r=await fetch('/api/video/list');const d=await r.json();const t=document.getElementById('ft');t.innerHTML='';d.uploads.forEach(f=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${f}</td><td><button class="btn" onclick="av('${f}')">Analyze</button></td>`;t.appendChild(tr)});d.outputs.forEach(f=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${f}</td><td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>`;t.appendChild(tr)})}
      async function av(f){const fd=new FormData();fd.append('filename',f);const r=await fetch('/api/video/analyze',{method:'POST',body:fd});const d=await r.json();if(d.status==='started'){sp()}else{alert(d.message||'Error')}}
      let pi;function sp(){document.getElementById('pa').style.display='block';if(pi)clearInterval(pi);pi=setInterval(async()=>{const r=await fetch('/api/video/status');const d=await r.json();document.getElementById('cf').innerText=d.current_video;document.getElementById('pb').style.width=d.progress+'%';document.getElementById('pt').innerText=d.progress+'%';document.getElementById('et').innerText=d.error||'';if(!d.is_processing&&d.progress===100){clearInterval(pi);alert('Done!');rf()}else if(!d.is_processing&&d.error){clearInterval(pi)}},1000)}
      fetch('/api/video/status').then(r=>r.json()).then(d=>{if(d.is_processing)sp()})
    </script></body></html>
    """, media_type="text/html")

def run_fastapi(host, port):
    print("\n"+"="*50, flush=True); print("Routes:", flush=True)
    for r in app.routes:
        if hasattr(r, "methods"): print(f"  {r.path:35} {r.methods}", flush=True)
    print("="*50+"\n", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


# ---------------------------------------------------------------------------
# ViTPose-Small post-processing (heatmap argmax → keypoints → skeleton)
#
# The HEF outputs a single heatmap: 64x48x17 (H, W, K=17 COCO keypoints).
# No on-chip NMS. The decode:
#   1. For each of 17 channels: argmax → (y, x) in heatmap space
#   2. Score = max value of the channel
#   3. Scale: heatmap (64, 48) → input (256, 192) by ratio H/64, W/48
#   4. Un-letterbox to original frame
#   5. Draw keypoints + skeleton
# ---------------------------------------------------------------------------

def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """Decode ViTPose heatmap into 17 keypoints.

    Returns (keypoints, ) where keypoints is (17, 3) = [x, y, score] in
    input-pixel space, or None if no valid keypoints.
    """
    global _DET_LOGGED

    if hailo_output is None:
        return None

    # Get the single output tensor
    if isinstance(hailo_output, dict):
        output = next(iter(hailo_output.values()))
    elif isinstance(hailo_output, (list, tuple)):
        output = hailo_output[0]
    else:
        output = hailo_output

    try:
        arr = np.asarray(output)
    except (ValueError, TypeError) as e:
        if not _DET_LOGGED:
            print(f"[ViTPose] np.asarray failed: {e}", flush=True)
            _DET_LOGGED = True
        return None
    if not _DET_LOGGED:
        print(f"[ViTPose] raw output shape={arr.shape}, dtype={arr.dtype}", flush=True)
        _DET_LOGGED = True

    # Normalize to (K=17, H=64, W=48)
    if arr.ndim == 4:
        arr = arr[0]  # drop batch
    if arr.ndim == 3:
        # Could be (17, 64, 48) [K,H,W] or (64, 48, 17) [H,W,K]
        if arr.shape[-1] == 17:
            arr = np.moveaxis(arr, -1, 0)  # HWC → CHW
        elif arr.shape[0] != 17:
            # Try (64, 48, 17) → move last to first
            if arr.shape[-1] == 17:
                arr = np.moveaxis(arr, -1, 0)
    if arr.ndim != 3 or arr.shape[0] != 17:
        print(f"[ViTPose] unexpected shape after normalize: {arr.shape}", flush=True)
        return None

    K, H, W = arr.shape  # (17, 64, 48)
    keypoints = np.zeros((K, 3), dtype=np.float32)  # [x, y, score]

    for k in range(K):
        hm = arr[k]  # (H, W) = (64, 48)
        idx = np.argmax(hm)
        score = float(hm.flat[idx])
        y = int(idx // W)
        x = int(idx % W)
        # Scale to input space
        x_input = x * input_w / W
        y_input = y * input_h / H
        keypoints[k] = [x_input, y_input, score]

    return keypoints

def unletterbox_keypoints(kpts, lb_info):
    """Map keypoints from letterboxed input space to original frame."""
    if kpts is None:
        return kpts
    ratio, dw, dh = lb_info
    out = kpts.copy()
    out[:, 0] = (out[:, 0] - dw) / ratio
    out[:, 1] = (out[:, 1] - dh) / ratio
    return out

def draw(image, kpts, obj_thresh):
    if kpts is None:
        return
    for i, (x, y, s) in enumerate(kpts):
        if s >= obj_thresh and np.isfinite(x) and np.isfinite(y):
            color = tuple(int(v) for v in KPT_COLORS[i])
            cv2.circle(image, (int(x), int(y)), 4, color, -1)
    for idx, (a, b) in enumerate(SKELETON):
        if kpts[a, 2] >= obj_thresh and kpts[b, 2] >= obj_thresh:
            color = tuple(int(v) for v in LINK_COLORS[idx])
            pa = (int(kpts[a, 0]), int(kpts[a, 1]))
            pb = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(image, pa, pb, color, 2)

def preprocess_frame(frame, co_helper):
    """Letterbox + BGR to RGB. Input 256x192 (HxW).

    normalize_in_net with ImageNet RGB mean/std; no input_conversion → feed
    raw uint8 RGB after letterboxing.
    """
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img, ratio, (dw, dh) = co_helper.letter_box(
        im=frame.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(0, 0, 0), info_need=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (ratio, dw, dh)

def inference_loop(cap, model, co_helper, is_video_file, target_fps):
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
            processed_img, lb_info = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inf_time = time.time() - start_time
            if outputs is not None:
                try:
                    obj, _ = det_config.get()
                    kpts = post_process_hailo(outputs, obj, 0, IMG_SIZE[1], IMG_SIZE[0])
                    if kpts is not None:
                        real_kpts = unletterbox_keypoints(kpts, lb_info)
                        h, w = frame.shape[:2]
                        real_kpts[:, 0] = np.clip(real_kpts[:, 0], 0, w - 1)
                        real_kpts[:, 1] = np.clip(real_kpts[:, 1], 0, h - 1)
                        draw(frame, real_kpts, obj)
                except Exception as e:
                    print(f"[ViTPose] post-process error: {e}", flush=True)
            inf_fps = 1.0 / inf_time if inf_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'Hailo FPS: {fps_counter:.1f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_buffer.push_annotated(frame)
            if target_period > 0:
                now = time.time(); next_time += target_period
                sleep_for = next_time - now
                if sleep_for > 0: time.sleep(sleep_for)
                elif sleep_for < -target_period: next_time = now + target_period
    finally:
        stop_event.set()

def encode_loop(pw, ph, q):
    v = -1
    while not stop_event.is_set():
        frame, v = frame_buffer.wait_annotated(v, timeout=1.0)
        if frame is None: continue
        h, w = frame.shape[:2]
        if pw > 0 and ph > 0 and (w, h) != (pw, ph):
            frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok: frame_buffer.push_jpeg(buf.tobytes())

def main():
    parser = argparse.ArgumentParser(description='ViTPose-Small on RPi5 + Hailo-8')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--camera_id', type=int, default=0)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--class_path', type=str)
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
        print("Error: HailoRT not available."); return
    if args.class_path: pass  # vit_pose has fixed keypoints

    global _g_model, _g_co, IMG_SIZE
    model = HailoInfer(args.model_path)
    IMG_SIZE = (model.input_w, model.input_h)
    print(f"Model input: {model.input_w}x{model.input_h}", flush=True)
    co_helper = COCO_test_helper(enable_letter_box=True)
    _g_model = model; _g_co = co_helper
    video_analyzer.set_engine(model, co_helper)

    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web at http://{args.host}:{args.port}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: print("Interrupted")
        finally: model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path); cs = cap; ivf = True
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        cs = None; ivf = False
    if not cap.isOpened(): print("Error: Cannot open video"); return
    if not ivf: cs = LatestFrameReader(cap).start()

    inf = threading.Thread(target=inference_loop, args=(cs, model, co_helper, ivf, args.target_fps), daemon=True)
    enc = threading.Thread(target=encode_loop, args=(args.preview_width, args.preview_height, args.jpeg_quality), daemon=True)
    inf.start(); enc.start()
    try:
        while inf.is_alive(): time.sleep(0.5)
    except KeyboardInterrupt: print("Interrupted")
    finally:
        stop_event.set()
        if not ivf: cs.stop()
        inf.join(timeout=2); enc.join(timeout=2)
        cap.release(); model.release()

if __name__ == '__main__':
    main()
