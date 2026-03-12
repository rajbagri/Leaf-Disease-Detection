FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download u2netp model and store in a known location inside the image
# U2NET_HOME tells rembg exactly where to save/look for the model
ENV U2NET_HOME=/app/u2net_models

RUN python -c "\
import os; \
os.environ['U2NET_HOME'] = '/app/u2net_models'; \
os.makedirs('/app/u2net_models', exist_ok=True); \
from rembg import new_session; \
new_session('u2netp'); \
print('u2netp downloaded to /app/u2net_models'); \
import os; files = os.listdir('/app/u2net_models'); \
print('Files:', files)"

COPY main.py .
COPY model.onnx .

EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", "--bind", "0.0.0.0:8000", "main:app"]