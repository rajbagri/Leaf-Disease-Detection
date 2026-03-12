from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# ── Thread pool ───────────────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=6)

# ── Load ONNX model ───────────────────────────────────────────────────────────
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

print("Loading ONNX model...")

session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name

print(f"Model loaded. Input: '{INPUT_NAME}', Output: '{OUTPUT_NAME}'")

# ── Class labels ──────────────────────────────────────────────────────────────
CLASSES = [
    "Healthy Leaf",
    "Powdery Mildew",
    "Downy Mildew",
    "Rust",
    "Leaf Spot",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string for JSON response."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def arr_to_base64(arr: np.ndarray) -> str:
    """Convert numpy array (0-1 float) to base64 string."""
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return pil_to_base64(img)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Basic Image Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image(img_array: np.ndarray) -> dict:

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 5:
        return {"valid": False, "reason": "Image is too blurry. Please upload a clearer photo."}

    brightness = gray.mean()
    if brightness < 10:
        return {"valid": False, "reason": "Image is too dark. Please upload a well-lit photo."}
    if brightness > 250:
        return {"valid": False, "reason": "Image is completely overexposed. Please upload a better photo."}

    return {
        "valid":      True,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Leaf Detection
# ═══════════════════════════════════════════════════════════════════════════════

def validate_is_leaf(img_array: np.ndarray) -> dict:

    r, g, b      = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio  = np.sum((g > r) & (g > b)) / total_pixels

    hsv        = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230]))
    red_mask1  = cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230]))
    red_mask2  = cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230]))

    combined      = cv2.bitwise_or(cv2.bitwise_or(green_mask, brown_mask),
                                   cv2.bitwise_or(red_mask1,  red_mask2))
    leaf_coverage = np.sum(combined > 0) / total_pixels

    if green_ratio < 0.03 and leaf_coverage < 0.03:
        return {
            "valid":  False,
            "reason": "No leaf detected. Please upload a clear photo of a plant leaf.",
        }

    return {
        "valid":        True,
        "leaf_coverage": round(float(leaf_coverage), 2),
        "green_ratio":   round(float(green_ratio), 2),
        "mask":          combined,   # returned for debug visualization
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Preprocess (resize + /255.0 — matches training)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # shape: (256, 256, 3) — without batch dim for debug visibility


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE (used by /predict)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_inference(image_bytes: bytes) -> dict:

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    img_array = np.array(img)

    quality = validate_image(img_array)
    if not quality["valid"]:
        raise HTTPException(status_code=400, detail=quality["reason"])

    leaf_check = validate_is_leaf(img_array)
    if not leaf_check["valid"]:
        raise HTTPException(status_code=400, detail=leaf_check["reason"])

    arr         = preprocess(img)
    model_input = np.expand_dims(arr, axis=0)

    outputs         = session.run([OUTPUT_NAME], {INPUT_NAME: model_input})
    prediction      = outputs[0]
    predicted_class = int(np.argmax(prediction))
    confidence      = float(np.max(prediction))

    return {
        "disease":    CLASSES[predicted_class],
        "confidence": round(confidence, 4),
        "image_quality": {
            "blur_score":    quality.get("blur_score"),
            "brightness":    quality.get("brightness"),
            "leaf_coverage": leaf_check.get("leaf_coverage"),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DEBUG INFERENCE (used by /debug/preprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_debug(image_bytes: bytes) -> dict:

    stages = {}

    # ── Stage 1: Original image ──
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    original_arr = np.array(img)
    stages["1_original"] = {
        "description": "Original image as received",
        "size":        f"{img.width}x{img.height}",
        "image":       pil_to_base64(img),
    }

    # ── Stage 2: Quality check results ──
    quality = validate_image(original_arr)
    stages["2_quality_check"] = {
        "description": "Image quality metrics",
        "passed":      quality["valid"],
        "blur_score":  quality.get("blur_score"),
        "brightness":  quality.get("brightness"),
        "reason":      quality.get("reason", "passed"),
    }
    if not quality["valid"]:
        return {"stages": stages, "stopped_at": "quality_check", "reason": quality["reason"]}

    # ── Stage 3: Leaf detection mask ──
    leaf_check = validate_is_leaf(original_arr)

    # Build colored mask image: green = detected leaf area, black = non-leaf
    mask        = leaf_check.get("mask", np.zeros(original_arr.shape[:2], np.uint8))
    mask_rgb    = np.zeros_like(original_arr)
    mask_rgb[mask > 0] = [0, 200, 0]  # green overlay where leaf detected
    mask_img    = Image.fromarray(mask_rgb.astype(np.uint8))

    # Overlay mask on original (50% transparency)
    overlay     = Image.blend(img.resize(mask_img.size), mask_img, alpha=0.4)

    stages["3_leaf_detection"] = {
        "description":  "Green = detected leaf area. Black = detected as background/non-leaf.",
        "passed":       leaf_check["valid"],
        "leaf_coverage": leaf_check.get("leaf_coverage"),
        "green_ratio":  leaf_check.get("green_ratio"),
        "mask_image":   pil_to_base64(mask_img),
        "overlay_image": pil_to_base64(overlay),
        "reason":       leaf_check.get("reason", "passed"),
    }
    if not leaf_check["valid"]:
        return {"stages": stages, "stopped_at": "leaf_detection", "reason": leaf_check["reason"]}

    # ── Stage 4: Resized to 256x256 ──
    resized = img.resize((256, 256), Image.LANCZOS)
    stages["4_resized"] = {
        "description": "Image resized to 256x256 (model input size)",
        "size":        "256x256",
        "image":       pil_to_base64(resized),
    }

    # ── Stage 5: Normalized (what model actually sees) ──
    arr         = np.array(resized, dtype=np.float32) / 255.0
    model_input = np.expand_dims(arr, axis=0)

    # Visualize normalized image (multiply back by 255 for display)
    normalized_display = Image.fromarray((arr * 255).astype(np.uint8))
    stages["5_normalized"] = {
        "description": "After /255.0 normalization — this is exactly what the model sees",
        "pixel_min":   round(float(arr.min()), 4),
        "pixel_max":   round(float(arr.max()), 4),
        "pixel_mean":  round(float(arr.mean()), 4),
        "image":       pil_to_base64(normalized_display),
    }

    # ── Stage 6: Model prediction ──
    outputs         = session.run([OUTPUT_NAME], {INPUT_NAME: model_input})
    prediction      = outputs[0][0]
    predicted_class = int(np.argmax(prediction))
    confidence      = float(np.max(prediction))

    # All class scores
    class_scores = {
        CLASSES[i]: round(float(prediction[i]), 4)
        for i in range(len(CLASSES))
    }

    stages["6_prediction"] = {
        "description":  "Raw model output scores for all classes",
        "class_scores": class_scores,
        "predicted":    CLASSES[predicted_class],
        "confidence":   round(confidence, 4),
    }

    return {
        "stages":     stages,
        "final_result": {
            "disease":    CLASSES[predicted_class],
            "confidence": round(confidence, 4),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG image.")

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_inference, image_bytes)
    return result


@app.post("/debug/preprocess")
async def debug_preprocess(file: UploadFile = File(...)):
    """
    Returns base64 images of every preprocessing stage so you can
    visually inspect what the model actually receives.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large.")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_debug, image_bytes)
    return JSONResponse(content=result)


@app.get("/ping")
def ping():
    return {"message": "server is alive"}