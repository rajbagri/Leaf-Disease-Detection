FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV headless + TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Bake model directly into image (no cold start download)
COPY model.onnx .

EXPOSE 8000

# 6 workers — safe for 512MB RAM with ONNX Runtime (~4.4MB model)
CMD ["gunicorn", "-w", "6", "-k", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]