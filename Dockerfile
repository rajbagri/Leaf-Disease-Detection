# Python 3.10 slim — no TensorFlow needed at runtime
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY main.py .

COPY model.onnx .

# Render maps $PORT automatically; default 8000
EXPOSE 8000

# 3 Gunicorn workers — safe for 512MB with ONNX Runtime
# Each worker is ~40-60MB, total well under 512MB
CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]