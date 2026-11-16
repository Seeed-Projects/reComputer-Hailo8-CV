import os
import sys
import time
import argparse
import threading
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response, StreamingResponse

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as exc:
    HAILO_AVAILABLE = False
    HailoInfer = None
    print(f"Warning: HailoRT not available ({exc}), inference will fail")

from py_utils.byte_decode import decode_bytetrack
from py_utils.byte_tracker import BYTETracker, TrackArgs

MODEL_SLUG = "bytetrack_nano_mot17"
MODEL_LABEL = "ByteTrack-Nano (MOT17)"
IMG_SIZE = (1088, 608)  # width, height; overridden from the HEF
SCORE_THRESH = 0.1
NMS_THRESH = 0.7

stop_event = threading.Event()
frame_buffer = None
_global_model = None
_tracker = None
_tracker_lock = threading.Lock()


def reset_tracker():
    global _tracker
    with _tracker_lock:
        _tracker = BYTETracker(TrackArgs(track_thresh=0.5, track_buffer=30, match_thresh=0.8))


class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.version = 0
        self.cond = threading.Condition()

    def push(self, frame):
        with self.cond:
            self.frame = frame
            self.version += 1
            self.cond.notify_all()

    def wait(self, version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.version > version, timeout=timeout)
            return self.frame, self.version


def letterbox(frame, size):
    target_w, target_h = size
    h, w = frame.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w] = resized
    return canvas, (ratio, dw, dh)


def preprocess_frame(frame):
    canvas, lb = letterbox(frame, IMG_SIZE)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), lb


def unletterbox(boxes, lb):
    ratio, dw, dh = lb
    out = boxes.copy().astype(np.float32)
    out[:, [0, 2]] = (out[:, [0, 2]] - dw) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - dh) / ratio
    return out


def update_tracker(boxes, scores):
    if boxes is None or len(boxes) == 0:
        det = np.zeros((0, 5), dtype=np.float32)
    else:
        det = np.concatenate([boxes, scores[:, None]], axis=1).astype(np.float32)
    with _tracker_lock:
        return _tracker.update(det)


def draw_tracks(frame, tracks, lb):
    h, w = frame.shape[:2]
    for track in tracks:
        box = unletterbox(np.asarray([track.tlbr]), lb)[0]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)
        cv2.putText(frame, f"ID {track.track_id} person {track.score:.2f}",
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def run_frame(frame, infer=True):
    global _global_model
    if infer and _global_model is not None:
        img, lb = preprocess_frame(frame)
        outputs = _global_model.run(img)
        boxes, classes, scores = decode_bytetrack(
            outputs, IMG_SIZE[1], IMG_SIZE[0], SCORE_THRESH, NMS_THRESH
        )
        tracks = update_tracker(boxes, scores)
    else:
        lb = (1.0, 0, 0)
        tracks = []
    return tracks, lb


def inference_loop(cap, is_video_file, target_fps):
    target_period = 1.0 / target_fps if target_fps > 0 else 0.0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    reset_tracker()
                    continue
                break
            tracks, lb = run_frame(frame)
            draw_tracks(frame, tracks, lb)
            frame_buffer.push(frame)
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


def encode_loop(quality=80):
    last = -1
    while not stop_event.is_set():
        frame, last = frame_buffer.wait(last)
        if frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            with frame_buffer.cond:
                frame_buffer.jpeg = buf.tobytes()
                frame_buffer.jpeg_version = getattr(frame_buffer, "jpeg_version", 0) + 1
                frame_buffer.cond.notify_all()


app = FastAPI(title=f"reComputer {MODEL_LABEL}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_SLUG}


@app.get("/api/video_feed")
async def video_feed():
    def gen():
        version = -1
        while True:
            with frame_buffer.cond:
                frame_buffer.cond.wait_for(
                    lambda: getattr(frame_buffer, "jpeg_version", 0) > version,
                    timeout=1.0,
                )
                jpeg = getattr(frame_buffer, "jpeg", None)
                version = getattr(frame_buffer, "jpeg_version", 0)
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post(f"/api/models/{MODEL_SLUG}/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
):
    if _global_model is None:
        return {"success": False, "message": "Model not initialized"}
    try:
        img = None
        source = ""
        if file:
            data = await file.read()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            source = "uploaded image"
        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read())
                path = tmp.name
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                if timestamp is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if ok:
                    img = frame
                    source = f"video frame at {timestamp or 0}s"
                cap.release()
            os.unlink(path)
        if img is None:
            with frame_buffer.cond:
                img = frame_buffer.frame.copy() if frame_buffer.frame is not None else None
            source = "realtime camera frame"
        if img is None:
            return {"success": False, "message": "No valid input source"}

        tracks, lb = run_frame(img)
        h, w = img.shape[:2]
        predictions = []
        if tracks:
            boxes = unletterbox(np.asarray([t.tlbr for t in tracks]), lb)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
            for track, box in zip(tracks, boxes):
                predictions.append({
                    "track_id": int(track.track_id),
                    "class": "person",
                    "confidence": float(track.score),
                    "box": {"x1": float(box[0]), "y1": float(box[1]),
                            "x2": float(box[2]), "y2": float(box[3])},
                })
        return {"success": True, "source": source, "predictions": predictions,
                "image": {"width": w, "height": h}}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>%s</title><style>body{background:#111;color:#eee;text-align:center;font-family:system-ui}
h1{color:#00e676}img{max-width:100%%;margin-top:16px}</style></head>
<body><h1>%s</h1><img src="/api/video_feed"></body></html>""" % (MODEL_LABEL, MODEL_LABEL)


@app.get("/")
async def index():
    return Response(content=HTML, media_type="text/html")


def main():
    global _global_model, IMG_SIZE, frame_buffer
    parser = argparse.ArgumentParser(description=f"{MODEL_LABEL} on Hailo")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_path")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target_fps", type=float, default=30.0)
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available")
        return

    _global_model = HailoInfer(args.model_path)
    IMG_SIZE = (_global_model.input_w, _global_model.input_h)
    reset_tracker()
    frame_buffer = FrameBuffer()
    print(f"Model input: {IMG_SIZE[0]}x{IMG_SIZE[1]}", flush=True)

    web = threading.Thread(target=lambda: uvicorn.run(app, host=args.host, port=args.port, log_level="info"), daemon=True)
    web.start()
    encode = threading.Thread(target=encode_loop, daemon=True)
    encode.start()
    print(f"Web preview: http://{args.host}:{args.port}", flush=True)

    cap = cv2.VideoCapture(args.video_path if args.video_path else args.camera_id)
    if not cap.isOpened():
        print("Error: cannot open video source")
        return
    is_video_file = bool(args.video_path)
    try:
        inference_loop(cap, is_video_file, args.target_fps)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        _global_model.release()


if __name__ == "__main__":
    main()