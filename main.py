import os
os.environ["U2NET_HOME"] = "/app/u2net_models"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
from rembg import remove, new_session
import cv2, io, asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
_executor = ThreadPoolExecutor(max_workers=3)

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading rembg u2netp...")
rembg_session = new_session("u2netp")

print("Loading ONNX model...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session     = ort.InferenceSession("model.onnx", sess_options=opts, providers=["CPUExecutionProvider"])
INPUT_NAME  = session.get_inputs()[0].name
OUTPUT_NAME = session.get_outputs()[0].name
print("Ready.")

CLASSES = ["Healthy Leaf", "Powdery Mildew", "Downy Mildew", "Rust", "Leaf Spot"]

# ── Validation ────────────────────────────────────────────────────────────────
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

    # Leaf check — wide HSV range covers healthy + all disease states
    hsv      = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    r, g, b  = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    total    = img_array.shape[0] * img_array.shape[1]
    g_ratio  = float(np.sum((g > r) & (g > b)) / total)

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
        "valid":      True,
        "blur_score": round(blur, 2),
        "brightness": round(brightness, 2),
        "leaf_coverage": round(coverage, 2),
    }

# ── Preprocessing pipeline ────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> tuple:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file. Please upload a JPG or PNG.")

    quality = validate(np.array(img))
    if not quality["valid"]:
        raise HTTPException(400, quality["reason"])

    # Background removal
    try:
        rgba = remove(img, session=rembg_session)
        bg   = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        img  = bg
    except Exception:
        pass  # fallback to original on failure

    # Noise removal (bilateral — color preserving, lightweight)
    img = Image.fromarray(
        cv2.bilateralFilter(np.array(img), d=5, sigmaColor=30, sigmaSpace=30)
    )

    # Resize + normalize
    img   = img.resize((256, 256), Image.LANCZOS)
    arr   = np.array(img, dtype=np.float32) / 255.0
    inp   = np.expand_dims(arr, axis=0)

    meta = {
        "blur_score":    quality["blur_score"],
        "brightness":    quality["brightness"],
        "leaf_coverage": quality["leaf_coverage"],
    }
    return inp, meta

# ── Inference ─────────────────────────────────────────────────────────────────
def _infer(image_bytes: bytes) -> dict:
    inp, meta   = preprocess(image_bytes)
    out         = session.run([OUTPUT_NAME], {INPUT_NAME: inp})[0]
    cls         = int(np.argmax(out))
    conf        = float(np.max(out))
    return {"disease": CLASSES[cls], "confidence": round(conf, 4), "image_quality": meta}

# ── Routes ────────────────────────────────────────────────────────────────────
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