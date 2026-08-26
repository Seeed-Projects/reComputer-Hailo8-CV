#!/usr/bin/env python3
import argparse
import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from py_utils.db_postprocess import detect_boxes
from py_utils.hailo_executor import HailoInfer

MODEL_ID = "paddle_ocr_v5_mobile_detection"
engine = {"model": None, "video": "video/test.mp4"}
lock = threading.Lock()
settings = {"binaryThreshold": 0.30, "boxThreshold": 0.60}


def predict(frame):
    model = engine["model"]
    if model is None:
        raise RuntimeError("Model has not been initialized")
    resized = cv2.resize(frame, (model.input_w, model.input_h))
    with lock:
        output = next(iter(model.run(resized).values()))
        boxes = detect_boxes(output, frame, settings["binaryThreshold"], settings["boxThreshold"])
    return [{"polygon": box.tolist(), "box": {"x": int(box[:,0].min()), "y": int(box[:,1].min()), "width": int(box[:,0].max()-box[:,0].min()), "height": int(box[:,1].max()-box[:,1].min())}} for box in boxes]


def stream():
    capture = cv2.VideoCapture(engine["video"])
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            output = frame.copy()
            for item in predict(frame):
                cv2.polylines(output, [np.asarray(item["polygon"], dtype=np.int32)], True, (58, 197, 92), 2)
            ok, jpg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1 / 8)
    finally:
        capture.release()


@asynccontextmanager
async def lifespan(app):
    yield
    if engine["model"]: engine["model"].release()


app = FastAPI(title="PaddleOCR v5 Mobile Detection on Hailo-8", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def index():
    return '<h1>PaddleOCR v5 Mobile Detection · Hailo-8</h1><img style="max-width:100%" src="/api/video_feed"><p>POST an image to /api/models/paddle_ocr_v5_mobile_detection/predict</p>'

@app.get("/api/config")
def get_config(): return settings

@app.post("/api/config")
async def update_config(payload: dict):
    for key in settings:
        if key in payload: settings[key] = float(payload[key])
    return settings

@app.post(f"/api/models/{MODEL_ID}/predict")
async def api_predict(file: UploadFile = File(...)):
    frame = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(status_code=400, detail="Upload is not a valid image")
    predictions = await asyncio.to_thread(predict, frame)
    return {"model": MODEL_ID, "predictions": predictions, "count": len(predictions)}

@app.get("/api/video_feed")
def video_feed(): return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model/paddle_ocr_v5_mobile_detection.hef")
    parser.add_argument("--video_path", default="video/test.mp4")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not os.path.isfile(args.model_path): raise SystemExit(f"Missing HEF: {args.model_path}")
    engine["model"] = HailoInfer(args.model_path); engine["video"] = args.video_path
    print(f"Detector input: {engine['model'].input_w}x{engine['model'].input_h}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
