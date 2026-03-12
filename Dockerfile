FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake u2netp (4MB) into image — zero download on startup
ENV U2NET_HOME=/app/u2net_models
RUN python -c "\
import os; os.environ['U2NET_HOME']='/app/u2net_models'; \
os.makedirs('/app/u2net_models', exist_ok=True); \
from rembg import new_session; new_session('u2netp'); \
print('Files:', os.listdir('/app/u2net_models'))"

COPY main.py .
COPY model.onnx .

EXPOSE 8000

# ── 1 worker only ──
# RAM per worker: ~250MB (Python+CV+ONNX+rembg)
# 1 worker = ~250MB — safe on 512MB
# Concurrency handled by ThreadPoolExecutor(max_workers=3) inside main.py
# = 3 parallel predictions from 1 process
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]