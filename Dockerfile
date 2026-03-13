# Use the official Python 3.10 slim image as the base.
# 'slim' excludes unnecessary system packages, keeping the image size small.
FROM python:3.10-slim

# Set the working directory inside the container.
# All subsequent COPY, RUN, and CMD instructions operate relative to this path.
WORKDIR /app

# Install system-level dependencies required by OpenCV and other native libraries.
#   libgomp1      -- OpenMP runtime, needed by some ML/numerical libraries for parallelism
#   libglib2.0-0  -- GLib base library, required by OpenCV's highgui module
#   libsm6        -- X Session Management library, required by OpenCV
#   libxext6      -- X11 miscellaneous extensions library, required by OpenCV
#   libxrender-dev -- X Rendering Extension library, required by OpenCV
# The apt cache is deleted afterward to reduce the final image size.
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file first and install before copying application code.
# Docker caches each layer separately — by copying requirements.txt alone here,
# the pip install layer is only re-run when requirements.txt actually changes,
# not on every code change. This significantly speeds up iterative builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source and the ONNX model file into the container.
COPY main.py .
COPY model.onnx .

# Document that the application listens on port 8000 by default.
# This does not publish the port to the host — it is informational only.
# The actual port binding is controlled by the CMD below at runtime.
EXPOSE 8000

# Start the application using Gunicorn with a Uvicorn worker for ASGI support.
#
# Flags:
#   -w 1                          -- Single worker process (sufficient for most
#                                    single-container deployments; increase if needed)
#   -k uvicorn.workers.UvicornWorker -- Use Uvicorn's ASGI worker class instead of
#                                       Gunicorn's default WSGI worker, required for FastAPI
#   --timeout 60                  -- Kill and restart a worker if it does not respond
#                                    within 60 seconds (prevents hung processes)
#   --bind 0.0.0.0:${PORT:-8000}  -- Listen on all interfaces; use the $PORT environment
#                                    variable injected by Render at runtime, falling back
#                                    to 8000 for local development if $PORT is not set
#   main:app                      -- The ASGI application object: 'app' inside 'main.py'
#
# NOTE: Shell form (not exec form) is intentional here.
# Exec form (JSON array) does not invoke a shell, so environment variables like
# $PORT are not expanded. Shell form runs via /bin/sh -c, which expands $PORT correctly.
CMD gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 60 --bind 0.0.0.0:${PORT:-8000} main:app