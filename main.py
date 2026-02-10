from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import importlib.util
import sys
import kagglehub
import os

app = FastAPI()

# ===== DOWNLOAD MODEL ON STARTUP =====
print("Downloading model from KaggleHub...")

MODEL_BASE_PATH = kagglehub.model_download(
    "khanaamer/leaf-disease-detection-using-cnn-and-vit/tensorFlow2/default"
)

print("Model downloaded to:", MODEL_BASE_PATH)

# ===== FIND FILES AUTOMATICALLY (IMPORTANT FIX) =====
def find_file(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

CUSTOM_LAYER_FILE = find_file(MODEL_BASE_PATH, "cnn_vit_model.py")
MODEL_FILE = find_file(MODEL_BASE_PATH, "vit_dataset-1.h5")

print("Custom layer file:", CUSTOM_LAYER_FILE)
print("Model file:", MODEL_FILE)

if CUSTOM_LAYER_FILE is None or MODEL_FILE is None:
    raise FileNotFoundError("Could not locate model files in downloaded archive")

# ===== LOAD CUSTOM LAYER DYNAMICALLY =====
spec = importlib.util.spec_from_file_location("cnn_vit_model", CUSTOM_LAYER_FILE)
cnn_vit = importlib.util.module_from_spec(spec)
sys.modules["cnn_vit_model"] = cnn_vit
spec.loader.exec_module(cnn_vit)

TransformerBlock = cnn_vit.TransformerBlock

# ===== LOAD MODEL (INFERENCE ONLY) =====
with tf.keras.utils.custom_object_scope({
    "TransformerBlock": TransformerBlock
}):
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)

print("Model Loaded Successfully!")

# ===== PREPROCESS IMAGE =====
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))   # Model expects 256x256
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        prediction = model.predict(processed_img)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        CLASSES = [
            "Healthy Leaf",
            "Powdery Mildew",
            "Downy Mildew",
            "Rust",
            "Leaf Spot"
        ]

        return {
            "disease": CLASSES[predicted_class],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/ping")
def ping():
    return {"message": "server is alive"}
