from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
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
#  STEP 1 — Image Quality Validation
# ═══════════════════════════════════════════════════════════════════════════════

def validate_image_quality(img_array: np.ndarray) -> dict:

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Blur detection via Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50:
        return {"valid": False, "reason": f"Image is too blurry (score: {blur_score:.1f}). Please upload a clearer photo."}

    # Brightness check
    brightness = gray.mean()
    if brightness < 30:
        return {"valid": False, "reason": "Image is too dark. Please upload a well-lit photo."}
    if brightness > 230:
        return {"valid": False, "reason": "Image is too bright / overexposed. Please upload a better photo."}

    # Contrast check
    contrast = gray.std()
    if contrast < 15:
        return {"valid": False, "reason": "Image has very low contrast. Please upload a clearer photo."}

    return {
        "valid":      True,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Leaf Validation (reject non-leaf images)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_is_leaf(img_array: np.ndarray) -> dict:

    r, g, b      = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    total_pixels = img_array.shape[0] * img_array.shape[1]
    green_ratio  = np.sum((g > r) & (g > b)) / total_pixels

    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Green range (healthy leaves)
    green_mask = cv2.inRange(hsv, np.array([25, 20, 20]),  np.array([100, 255, 255]))
    # Brown/yellow range (diseased leaves)
    brown_mask = cv2.inRange(hsv, np.array([5,  20, 20]),  np.array([25,  255, 200]))

    leaf_coverage = np.sum(cv2.bitwise_or(green_mask, brown_mask) > 0) / total_pixels

    if green_ratio < 0.10 and leaf_coverage < 0.10:
        return {
            "valid":  False,
            "reason": "No leaf detected in the image. Please upload a clear photo of a plant leaf.",
        }

    return {"valid": True, "leaf_coverage": round(float(leaf_coverage), 2)}


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Leaf Segmentation (GrabCut — isolate leaf, remove background)
# ═══════════════════════════════════════════════════════════════════════════════

def segment_leaf(img_array: np.ndarray) -> np.ndarray:

    try:
        img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        h, w      = img_bgr.shape[:2]
        mask      = np.zeros((h, w), np.uint8)
        rect      = (w // 8, h // 8, w * 6 // 8, h * 6 // 8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)

        # If mask too empty, skip segmentation
        if final_mask.sum() < (h * w * 0.05):
            return img_array

        segmented = img_bgr.copy()
        segmented[final_mask == 0] = [255, 255, 255]  # background → white

        return cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)

    except Exception:
        return img_array  # never crash — fallback to original


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Auto Enhancement (CLAHE + denoising + sharpening + saturation)
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_image(img: Image.Image) -> Image.Image:

    img_array = np.array(img)

    # CLAHE on L channel in LAB color space
    lab           = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b       = cv2.split(lab)
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_enhanced  = cv2.merge([clahe.apply(l), a, b])
    img_array     = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Fast denoising
    img_array = cv2.fastNlMeansDenoisingColored(
        img_array, None,
        h=5, hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    img = Image.fromarray(img_array)

    # Unsharp mask sharpening
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    # Saturation boost — makes disease spots more distinguishable
    img = ImageEnhance.Color(img).enhance(1.2)

    return img


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Normalize (match training preprocessing — simple /255.0 only)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(img: Image.Image) -> np.ndarray:

    img = img.resize((256, 256), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0   # matches original training
    return np.expand_dims(arr, axis=0)               # shape: (1, 256, 256, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def full_preprocess(image_bytes: bytes) -> tuple:

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a JPG or PNG.")

    img_array = np.array(img)

    # 1. Quality validation
    quality = validate_image_quality(img_array)
    if not quality["valid"]:
        raise HTTPException(status_code=400, detail=quality["reason"])

    # 2. Leaf validation
    leaf_check = validate_is_leaf(img_array)
    if not leaf_check["valid"]:
        raise HTTPException(status_code=400, detail=leaf_check["reason"])

    # 3. Segmentation
    img_array = segment_leaf(img_array)
    img       = Image.fromarray(img_array)

    # 4. Enhancement
    img = enhance_image(img)

    # 5. Normalize
    model_input = normalize(img)

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
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG image.")

    image_bytes = await file.read()

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10MB.")

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _run_inference, image_bytes)
    return result


@app.get("/ping")
def ping():
    return {"message": "server is alive"}