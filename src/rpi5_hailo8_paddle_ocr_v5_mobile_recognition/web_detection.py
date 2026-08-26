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

from py_utils.ctc_decoder import decode, preprocess
from py_utils.hailo_executor import HailoInfer

MODEL_ID = "paddle_ocr_v5_mobile_recognition"
engine = {"model": None, "video": "video/test.mp4"}
lock = threading.Lock()


def predict(frame):
    model = engine["model"]
    if model is None: raise RuntimeError("Model has not been initialized")
    with lock:
        output = next(iter(model.run(preprocess(frame, model.input_w, model.input_h)).values()))
    text, confidence = decode(output)
    return {"text": text, "confidence": round(confidence, 4)}


def stream():
    capture = cv2.VideoCapture(engine["video"])
    try:
        while True:
            ok, frame = capture.read()
            if not ok: capture.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            result = predict(frame)
            output = cv2.copyMakeBorder(frame, 0, 52, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(output, result["text"] or "<unreadable>", (12, output.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (35, 90, 35), 2, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if ok: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(1 / 8)
    finally: capture.release()


@asynccontextmanager
async def lifespan(app):
    yield
    if engine["model"]: engine["model"].release()

app = FastAPI(title="PaddleOCR v5 Mobile Recognition on Hailo-8", lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def index(): return '<h1>PaddleOCR v5 Mobile Recognition · Hailo-8</h1><img style="max-width:100%" src="/api/video_feed"><p>POST one cropped text line to /api/models/paddle_ocr_v5_mobile_recognition/predict</p>'

@app.post(f"/api/models/{MODEL_ID}/predict")
async def api_predict(file: UploadFile = File(...)):
    frame = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(status_code=400, detail="Upload is not a valid image")
    result = await asyncio.to_thread(predict, frame)
    return {"model": MODEL_ID, **result}

@app.get("/api/video_feed")
def video_feed(): return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model/paddle_ocr_v5_mobile_recognition.hef")
    parser.add_argument("--video_path", default="video/test.mp4")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not os.path.isfile(args.model_path): raise SystemExit(f"Missing HEF: {args.model_path}")
    engine["model"] = HailoInfer(args.model_path); engine["video"] = args.video_path
    print(f"Recognizer input: {engine['model'].input_w}x{engine['model'].input_h}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
