FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from rembg import new_session; new_session('u2net'); print('rembg u2net model downloaded.')"

# Copy app and model
COPY main.py .
COPY model.onnx .

EXPOSE 8000

# 2 workers — safe for 512MB RAM with rembg + onnx loaded
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]