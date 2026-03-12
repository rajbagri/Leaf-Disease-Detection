from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io
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
#  STEP 1 — Basic Image Validation (non-destructive checks only)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image(img_array: np.ndarray) -> dict:

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Blur — only reject extremely blurry images
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 5:
        return {"valid": False, "reason": "Image is too blurry. Please upload a clearer photo."}

    # Brightness — only reject completely dark or completely white images
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
#  STEP 2 — Leaf Detection (very relaxed — just reject obvious non-leaves)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_is_leaf(img_array: np.ndarray) -> dict:

    r, g, b      = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio  = np.sum((g > r) & (g > b)) / total_pixels

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Very wide range — covers green, yellow, brown, red (all leaf states)
    green_mask = cv2.inRange(hsv, np.array([10, 10, 10]), np.array([110, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([3,  10, 10]), np.array([30,  255, 230]))
    red_mask1  = cv2.inRange(hsv, np.array([0,  10, 10]), np.array([3,   255, 230]))
    red_mask2  = cv2.inRange(hsv, np.array([170,10, 10]), np.array([180, 255, 230]))

    combined      = cv2.bitwise_or(cv2.bitwise_or(green_mask, brown_mask),
                                   cv2.bitwise_or(red_mask1,  red_mask2))
    leaf_coverage = np.sum(combined > 0) / total_pixels

    # Very relaxed — only reject if both checks clearly fail
    if green_ratio < 0.03 and leaf_coverage < 0.03:
        return {
            "valid":  False,
            "reason": "No leaf detected. Please upload a clear photo of a plant leaf.",
        }

    return {"valid": True, "leaf_coverage": round(float(leaf_coverage), 2)}


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Preprocess exactly as model was trained (resize + /255.0 only)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, 256, 256, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def full_preprocess(image_bytes: bytes) -> tuple:

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a JPG or PNG.")

    img_array = np.array(img)

    # 1. Basic quality validation
    quality = validate_image(img_array)
    if not quality["valid"]:
        raise HTTPException(status_code=400, detail=quality["reason"])

    # 2. Leaf check
    leaf_check = validate_is_leaf(img_array)
    if not leaf_check["valid"]:
        raise HTTPException(status_code=400, detail=leaf_check["reason"])

    # 3. Preprocess — NO segmentation, NO enhancement, just resize + normalize
    model_input = preprocess(img)

    metadata = {
        "blur_score":    quality.get("blur_score"),
        "brightness":    quality.get("brightness"),
        "leaf_coverage": leaf_check.get("leaf_coverage"),
    }

    return model_input, metadata


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_inference(image_bytes: bytes) -> dict:

    model_input, metadata = full_preprocess(image_bytes)

    outputs         = session.run([OUTPUT_NAME], {INPUT_NAME: model_input})
    prediction      = outputs[0]
    predicted_class = int(np.argmax(prediction))
    confidence      = float(np.max(prediction))

    return {
        "disease":       CLASSES[predicted_class],
        "confidence":    round(confidence, 4),
        "image_quality": metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG or PNG image."
        )

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_inference, image_bytes)
    return result


@app.get("/ping")
def ping():
    return {"message": "server is alive"}