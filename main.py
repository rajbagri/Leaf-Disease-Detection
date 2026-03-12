import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2, io, asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
_executor = ThreadPoolExecutor(max_workers=3)

# ── Load ONNX model once ──────────────────────────────────────────────────────
print("Loading ONNX model...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session     = ort.InferenceSession("model.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
print("Model ready.")

CLASSES = ["Healthy Leaf", "Powdery Mildew", "Downy Mildew", "Rust", "Leaf Spot"]


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Quality + Leaf Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate(img_array: np.ndarray) -> dict:

    gray       = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur       = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())

    if blur < 5:
        return {"valid": False, "reason": "Image is too blurry. Please upload a clearer photo."}
    if brightness < 10:
        return {"valid": False, "reason": "Image is too dark. Please upload a well-lit photo."}
    if brightness > 250:
        return {"valid": False, "reason": "Image is overexposed. Please upload a better photo."}

    hsv     = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total   = img_array.shape[0] * img_array.shape[1]
    g_ratio = float(np.sum((g > r) & (g > b)) / total)

    masks = [
        cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255])),  # green
        cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230])),  # brown
        cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230])),  # red1
        cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230])),  # red2
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    coverage = float(np.sum(combined > 0) / total)

    if g_ratio < 0.03 and coverage < 0.03:
        return {"valid": False, "reason": "No leaf detected. Please upload a clear photo of a plant leaf."}

    return {
        "valid":         True,
        "blur_score":    round(blur, 2),
        "brightness":    round(brightness, 2),
        "leaf_coverage": round(coverage, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Background Removal (pure OpenCV, zero extra RAM)
#
#  How it works:
#  1. Build leaf mask using wide HSV ranges (green + brown + red = all leaf states)
#  2. Morphological close — fills holes inside leaf (disease spots, veins)
#  3. Find largest contour — picks the main leaf, drops stray noise
#  4. White background composite — background → white, leaf pixels untouched
# ═══════════════════════════════════════════════════════════════════════════════

def remove_background(img_array: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Wide HSV mask — covers healthy green, diseased yellow/brown/rust/spot
    masks = [
        cv2.inRange(hsv, np.array([10, 15, 15]), np.array([110, 255, 255])),  # green/yellow
        cv2.inRange(hsv, np.array([3,  15, 15]), np.array([30,  255, 220])),  # brown
        cv2.inRange(hsv, np.array([0,  15, 15]), np.array([3,   255, 220])),  # red1
        cv2.inRange(hsv, np.array([170,15, 15]), np.array([180, 255, 220])),  # red2
    ]
    mask = masks[0]
    for m in masks[1:]:
        mask = cv2.bitwise_or(mask, m)

    # Morphological close — fills holes (disease spots, veins, shadows)
    # kernel 20x20 is large enough to bridge gaps inside a leaf
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Fill small holes with dilate
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask         = cv2.dilate(mask, kernel_small, iterations=2)
    mask         = cv2.erode(mask,  kernel_small, iterations=1)

    # Keep only the largest contour = main leaf body, drop background noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        clean_mask = np.zeros_like(mask)
        largest    = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
        mask = clean_mask

    # Composite: leaf pixels kept, background → white (255,255,255)
    result              = np.full_like(img_array, 255)   # white canvas
    result[mask > 0]    = img_array[mask > 0]            # paste leaf pixels

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — Noise Removal (bilateral filter — color preserving)
#
#  Removes camera noise / compression artifacts.
#  sigmaColor=30 is intentionally low — preserves leaf color exactly.
#  Does NOT change hue/saturation.
# ═══════════════════════════════════════════════════════════════════════════════

def remove_noise(img_array: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(img_array, d=5, sigmaColor=30, sigmaSpace=30)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _infer(image_bytes: bytes) -> dict:

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file. Please upload a JPG or PNG.")

    arr = np.array(img)

    # Stage 1 — validate
    quality = validate(arr)
    if not quality["valid"]:
        raise HTTPException(400, quality["reason"])

    # Stage 2 — background removal (OpenCV HSV mask + morphology)
    arr = remove_background(arr)

    # Stage 3 — noise removal (bilateral filter)
    arr = remove_noise(arr)

    # Stage 4 — resize + normalize (matches training)
    img = Image.fromarray(arr).resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)

    # Stage 5 — inference
    out  = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    cls  = int(np.argmax(out))
    conf = float(np.max(out))

    return {
        "disease":    CLASSES[cls],
        "confidence": round(conf, 4),
        "image_quality": {
            "blur_score":    quality["blur_score"],
            "brightness":    quality["brightness"],
            "leaf_coverage": quality["leaf_coverage"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Invalid file type. Please upload a JPG or PNG.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image too large. Max 10MB.")
    return await asyncio.get_event_loop().run_in_executor(_executor, _infer, data)

@app.get("/ping")
def ping():
    return {"message": "server is alive"}