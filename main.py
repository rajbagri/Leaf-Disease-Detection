from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import importlib.util
import sys

app = FastAPI()

# ===== PATHS =====
MODEL_FILE = r"C:\Users\hp\.cache\kagglehub\models\khanaamer\leaf-disease-detection-using-cnn-and-vit\tensorFlow2\default\1\saved_models\vit_dataset-1.h5"

CUSTOM_LAYER_FILE = r"C:\Users\hp\.cache\kagglehub\models\khanaamer\leaf-disease-detection-using-cnn-and-vit\tensorFlow2\default\1\Models\cnn_vit_model.py"

# ===== DYNAMIC LOAD OF CUSTOM LAYERS =====
spec = importlib.util.spec_from_file_location("cnn_vit_model", CUSTOM_LAYER_FILE)
cnn_vit = importlib.util.module_from_spec(spec)
sys.modules["cnn_vit_model"] = cnn_vit
spec.loader.exec_module(cnn_vit)

TransformerBlock = cnn_vit.TransformerBlock

# ===== LOAD MODEL (<<< IMPORTANT CHANGE HERE) =====
with tf.keras.utils.custom_object_scope({
    "TransformerBlock": TransformerBlock
}):
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)  # <-- FIX

print("Model Loaded Successfully!")

# ===== PREPROCESS IMAGE =====
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))   # <-- FIXED (was 224)
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
        return {
            "error": str(e)
        }


@app.get("/ping")
def ping():
    return {"message": "server is alive"}

