# Use a stable Python version that supports TensorFlow well
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port (Render will map this automatically)
EXPOSE 8000

# Start FastAPI with gunicorn + uvicorn worker
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
