
import os
import sys
import importlib.util
import kagglehub
import tf2onnx
import tensorflow as tf

print("Downloading model from KaggleHub...")
MODEL_BASE_PATH = kagglehub.model_download(
    "khanaamer/leaf-disease-detection-using-cnn-and-vit/tensorFlow2/default"
)
print("Downloaded to:", MODEL_BASE_PATH)

def find_file(base_path, filename):
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

CUSTOM_LAYER_FILE = find_file(MODEL_BASE_PATH, "cnn_vit_model.py")
MODEL_FILE        = find_file(MODEL_BASE_PATH, "vit_dataset-1.h5")

print("Custom layer file:", CUSTOM_LAYER_FILE)
print("Model file       :", MODEL_FILE)

if not CUSTOM_LAYER_FILE or not MODEL_FILE:
    raise FileNotFoundError("Could not locate model files in downloaded archive")

spec    = importlib.util.spec_from_file_location("cnn_vit_model", CUSTOM_LAYER_FILE)
cnn_vit = importlib.util.module_from_spec(spec)
sys.modules["cnn_vit_model"] = cnn_vit
spec.loader.exec_module(cnn_vit)

TransformerBlock = cnn_vit.TransformerBlock

print("Loading Keras model...")
with tf.keras.utils.custom_object_scope({"TransformerBlock": TransformerBlock}):
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)

print("Model loaded. Input shape:", model.input_shape)

OUTPUT_PATH = "model.onnx"
print(f"Converting to ONNX → {OUTPUT_PATH} ...")

input_signature = [
    tf.TensorSpec(shape=[None, 256, 256, 3], dtype=tf.float32, name="input")
]

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=OUTPUT_PATH,
)

print(f"Done! ONNX model saved to: {OUTPUT_PATH}")
print("Now copy model.onnx into your project root and rebuild Docker.")