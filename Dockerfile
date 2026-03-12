FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY model.onnx .

EXPOSE 8000

# $PORT is set by Render at runtime — must use shell form CMD (not exec form)
# so the variable gets expanded correctly
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 60 --bind 0.0.0.0:${PORT:-8000} main:app