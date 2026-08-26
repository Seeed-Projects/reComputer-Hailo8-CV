FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 ffmpeg && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY hailort-packages/*.whl /tmp/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential python3-dev && pip install --no-cache-dir /tmp/hailort-*.whl && apt-get purge -y --auto-remove build-essential python3-dev && rm -rf /var/lib/apt/lists/* /tmp/*.whl
COPY . .
EXPOSE 8000
CMD ["python", "web_detection.py", "--model_path", "model/paddle_ocr_v5_mobile_detection.hef", "--video_path", "video/test.mp4"]
