"""
Leaf Disease Detection API
--------------------------
A FastAPI application that accepts a leaf image, runs it through a
multi-stage preprocessing pipeline, and returns a disease classification
using a pre-converted ONNX model.

Endpoints:
    POST /predict  -- Upload a leaf image; returns disease label and confidence
    GET  /ping     -- Health check
"""

import os

# Limit CPU thread usage to 1 per library to prevent thread over-subscription
# on low-resource containers (e.g. Render free tier). Without this, ONNX Runtime
# and OpenBLAS each try to spawn as many threads as there are CPU cores, which
# causes contention and can actually slow down inference on single-core hosts.
os.environ["OMP_NUM_THREADS"]      = "1"   # OpenMP -- used by ONNX Runtime internals
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # OpenBLAS -- used by NumPy linear algebra ops

from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2, io, asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# Thread pool used to run the synchronous inference pipeline (_infer) without
# blocking the async event loop. max_workers=3 allows up to 3 concurrent
# inference requests; requests beyond that queue automatically.
_executor = ThreadPoolExecutor(max_workers=3)


# ---------------------------------------------------------------------------
# Model loading (runs once at startup, not per request)
# ---------------------------------------------------------------------------

print("Loading ONNX model...")

# Configure ONNX Runtime session options before loading the model.
# These settings are applied once and reused for every inference call.
opts = ort.SessionOptions()
opts.intra_op_num_threads     = 1                                          # Threads within a single op (e.g. matrix multiply)
opts.inter_op_num_threads     = 1                                          # Threads across parallel ops in the graph
opts.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL           # Run ops sequentially; avoids thread overhead on small models
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # Apply all graph-level optimizations at load time

# Load the ONNX model into an inference session.
# CPUExecutionProvider is specified explicitly to avoid ONNX Runtime attempting
# to use CUDA or other accelerators that are not present in this environment.
session     = ort.InferenceSession("model.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
INPUT_NAME  = session.get_inputs()[0].name    # Name of the model's input tensor (e.g. "input")
OUTPUT_NAME = session.get_outputs()[0].name   # Name of the model's output tensor (e.g. "dense")

print("Model ready.")

# Ordered list of class labels corresponding to the model's output indices.
# Index 0 = Healthy Leaf, Index 1 = Powdery Mildew, etc.
# Must match the label ordering used during model training exactly.
CLASSES = ["Healthy Leaf", "Powdery Mildew", "Downy Mildew", "Rust", "Leaf Spot"]


# ---------------------------------------------------------------------------
# Stage 1: Image quality and leaf presence validation
# ---------------------------------------------------------------------------

def validate(img_array: np.ndarray) -> dict:
    """
    Checks whether the image is suitable for inference.

    Three quality checks are performed:
        - Blur:       Laplacian variance measures edge sharpness.
                      A low value means the image lacks detail (too blurry).
        - Brightness: Mean grayscale intensity. Catches images that are
                      too dark (sensor noise, no lighting) or overexposed
                      (completely washed out, no visible features).
        - Leaf presence: HSV color masks detect whether plant-colored pixels
                      exist. Rejects non-leaf uploads (blank photos, objects).

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        dict: Contains 'valid' (bool) and either 'reason' (str) on failure,
              or 'blur_score', 'brightness', 'leaf_coverage' (float) on success.
    """
    # Convert to grayscale -- blur and brightness checks operate on intensity only
    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()   # High = sharp, low = blurry
    brightness = float(gray.mean())                       # Range: 0 (black) to 255 (white)

    if blur < 5:
        return {"valid": False, "reason": "Image is too blurry. Please upload a clearer photo."}
    if brightness < 10:
        return {"valid": False, "reason": "Image is too dark. Please upload a well-lit photo."}
    if brightness > 250:
        return {"valid": False, "reason": "Image is overexposed. Please upload a better photo."}

    # Convert to HSV for color-based leaf detection.
    # HSV separates hue (color type) from saturation and brightness, making
    # color range thresholds more robust to lighting variation than RGB.
    hsv     = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total   = img_array.shape[0] * img_array.shape[1]

    # Fraction of pixels where the green channel is brighter than both red and blue.
    # A healthy leaf typically has a high proportion of such pixels.
    g_ratio = float(np.sum((g > r) & (g > b)) / total)

    # HSV masks covering the full range of leaf colors:
    #   green  -- healthy and mildly diseased tissue
    #   brown  -- aged or heavily infected tissue
    #   red1/2 -- rust disease; HSV hue wraps around at 0/180 so two ranges are needed
    masks = [
        cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255])),  # green
        cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230])),  # brown
        cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230])),  # red1
        cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230])),  # red2
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)

    # Fraction of total pixels that matched any plant-like color
    coverage = float(np.sum(combined > 0) / total)

    # Reject if neither the green-dominance check nor the color coverage check
    # finds enough plant-colored content -- likely not a leaf photo
    if g_ratio < 0.03 and coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected. Please upload a clear photo of a plant leaf."}

    return {
        "valid":         True,
        "blur_score":    round(blur, 2),
        "brightness":    round(brightness, 2),
        "leaf_coverage": round(coverage, 2),
    }


# ---------------------------------------------------------------------------
# Stage 2: Background removal
#
# Isolates the leaf from its background using HSV color masking and
# morphological operations, then replaces the background with white.
#
# Steps:
#   1. Build a leaf mask using wide HSV ranges (covers all disease states)
#   2. Morphological close -- fills holes inside the leaf (disease spots, veins)
#   3. Dilate then erode -- recovers missed leaf-edge pixels, smooths mask border
#   4. Keep only the largest contour -- drops stray background noise blobs
#   5. Composite onto a white canvas -- background becomes white, leaf unchanged
# ---------------------------------------------------------------------------

def remove_background(img_array: np.ndarray) -> np.ndarray:
    """
    Removes the image background and replaces it with white pixels.

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        np.ndarray: Image with background replaced by white (255, 255, 255).
    """
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Wide HSV ranges to capture all leaf states: healthy, diseased, aged
    masks = [
        cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),  # green/yellow
        cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),  # brown
        cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),  # red1
        cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),  # red2
    ]
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    # Morphological close: fills internal holes (disease spots, leaf veins, shadows).
    # A 20x20 elliptical kernel is large enough to bridge most gaps inside a leaf.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Dilate then erode: recovers leaf-edge pixels the HSV mask may have missed,
    # then pulls the border back in to avoid including too much background.
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask         = cv2.dilate(mask, kernel_small, iterations=2)
    mask         = cv2.erode(mask,  kernel_small, iterations=1)

    # Retain only the largest connected contour (the main leaf body).
    # This discards small isolated blobs caused by background colors that
    # fall within the leaf HSV range (e.g. green grass, brown soil patches).
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean_mask = np.zeros_like(mask)
        largest    = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    # Composite: start with an all-white canvas, then paste leaf pixels on top.
    # White background is neutral and matches the training data preprocessing.
    result           = np.full_like(img_array, 255)   # white canvas
    result[mask > 0] = img_array[mask > 0]            # overlay leaf pixels

    return result


# ---------------------------------------------------------------------------
# Stage 3: Noise reduction
# ---------------------------------------------------------------------------

def remove_noise(img_array: np.ndarray) -> np.ndarray:
    """
    Reduces image noise using a bilateral filter.

    A bilateral filter smooths regions of similar color while preserving
    sharp edges, unlike a Gaussian blur which blurs edges indiscriminately.
    It considers both spatial proximity and color similarity between pixels.

    Parameters used:
        d=5           -- Neighborhood diameter; small enough for speed, large
                         enough to catch sensor noise clusters
        sigmaColor=30 -- Low color tolerance: only blends pixels with very
                         similar colors, so leaf color detail is preserved exactly
        sigmaSpace=30 -- Spatial tolerance: how far apart two pixels can be
                         and still influence each other

    Parameters:
        img_array (np.ndarray): Input image as an RGB NumPy array.

    Returns:
        np.ndarray: Noise-reduced image as an RGB NumPy array.
    """
    return cv2.bilateralFilter(img_array, d=5, sigmaColor=30, sigmaSpace=30)


# ---------------------------------------------------------------------------
# Full inference pipeline (synchronous -- runs in thread pool)
# ---------------------------------------------------------------------------

def _infer(image_bytes: bytes) -> dict:
    """
    Runs the complete preprocessing and inference pipeline on raw image bytes.

    This function is synchronous and CPU-bound. It is called via
    asyncio's run_in_executor so it does not block the async event loop.

    Pipeline stages:
        1. Decode image bytes to an RGB NumPy array
        2. Validate image quality and leaf presence
        3. Remove background
        4. Remove noise
        5. Resize to 256x256 and normalize pixel values to [0.0, 1.0]
        6. Run ONNX model inference; return top class and confidence score

    Parameters:
        image_bytes (bytes): Raw bytes of the uploaded image file.

    Returns:
        dict: Contains 'disease' (str), 'confidence' (float), and
              'image_quality' metrics (blur_score, brightness, leaf_coverage).

    Raises:
        HTTPException 400: If the image cannot be decoded, fails validation,
                           or does not contain a detectable leaf.
    """
    # Decode the raw bytes into a PIL image and convert to RGB NumPy array.
    # io.BytesIO wraps the bytes so PIL can treat them as a file-like object.
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file. Please upload a JPG or PNG.")

    arr = np.array(img)

    # Stage 1: Validate blur, brightness, and leaf presence
    quality = validate(arr)
    if not quality["valid"]:
        raise HTTPException(400, quality["reason"])

    # Stage 2: Remove background using HSV masking and morphological cleanup
    arr = remove_background(arr)

    # Stage 3: Reduce noise with a color-preserving bilateral filter
    arr = remove_noise(arr)

    # Stage 4: Resize to the model's expected input resolution (256x256).
    # LANCZOS resampling provides high-quality downscaling with minimal aliasing.
    # Normalize pixel values from [0, 255] to [0.0, 1.0] to match training preprocessing.
    img = Image.fromarray(arr).resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Add a batch dimension: shape becomes (1, 256, 256, 3).
    # The model expects a batch of images, not a single image tensor.
    inp = np.expand_dims(arr, axis=0)

    # Stage 5: Run ONNX inference.
    # session.run returns a list of output arrays; [0] gets the first (and only) output.
    # out shape: (1, num_classes) -- one score per class per image in the batch.
    out  = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    cls  = int(np.argmax(out))    # Index of the highest-scoring class
    conf = float(np.max(out))     # Score of the top class (used as confidence)

    return {
        "disease":    CLASSES[cls],
        "confidence": round(conf, 4),
        "image_quality": {
            "blur_score":    quality["blur_score"],
            "brightness":    quality["brightness"],
            "leaf_coverage": quality["leaf_coverage"],
        },
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a leaf image upload and returns a disease classification.

    Validates file type and size before passing bytes to the inference pipeline.
    The synchronous _infer function is offloaded to a thread pool executor to
    avoid blocking the async event loop during CPU-intensive preprocessing.

    Returns:
        JSON with 'disease', 'confidence', and 'image_quality' fields.
    """
    # Reject non-image uploads before reading the full file contents
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Invalid file type. Please upload a JPG or PNG.")

    data = await file.read()

    # Enforce a 10MB size cap to prevent memory spikes from oversized uploads
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 10MB.")

    # Run the blocking inference pipeline in a background thread so the event
    # loop remains free to accept other incoming requests during processing
    return await asyncio.get_event_loop().run_in_executor(_executor, _infer, data)


@app.get("/ping")
def ping():
    """
    Health check endpoint. Returns a static response to confirm the server is running.
    Used by Render and other platforms to verify the container started successfully.
    """
    return {"message": "server is alive"}