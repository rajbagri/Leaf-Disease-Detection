from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

_executor = ThreadPoolExecutor(max_workers=3)


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

CLASSES = [
    "Healthy Leaf",
    "Powdery Mildew",
    "Downy Mildew",
    "Rust",
    "Leaf Spot",
]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    arr   = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # shape: (1, 256, 256, 3)

# ── Blocking inference (runs in thread pool) ──────────────────────────────────
def _run_inference(image_bytes: bytes) -> dict:
    processed = preprocess_image(image_bytes)
    outputs   = session.run([OUTPUT_NAME], {INPUT_NAME: processed})
    prediction      = outputs[0]                        # shape: (1, num_classes)
    predicted_class = int(np.argmax(prediction))
    confidence      = float(np.max(prediction))
    return {
        "disease":    CLASSES[predicted_class],
        "confidence": round(confidence, 4),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _run_inference, image_bytes)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/ping")
def ping():
    return {"message": "server is alive"}