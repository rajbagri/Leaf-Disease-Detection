"""
Keras to ONNX Model Conversion Script
--------------------------------------
Downloads a pre-trained CNN + Vision Transformer (ViT) leaf disease detection
model from KaggleHub, loads it with its custom Keras layer definition, and
converts it to the ONNX format for cross-platform inference (e.g. via ONNX
Runtime in the FastAPI serving pipeline).

Output: model.onnx — a portable ONNX model ready for deployment.
"""

import os
import sys
import importlib.util
import kagglehub
import tensorflow as tf
import tf2onnx


# ---------------------------------------------------------------------------
# Stage 1: Download the model from KaggleHub
# ---------------------------------------------------------------------------

# Downloads the specified model version to a local cache directory and returns
# the path to the root of the downloaded files. KaggleHub handles authentication
# via the KAGGLE_USERNAME and KAGGLE_KEY environment variables (or ~/.kaggle/kaggle.json).
print("Downloading model from KaggleHub...")

MODEL_BASE_PATH = kagglehub.model_download(
    "khanaamer/leaf-disease-detection-using-cnn-and-vit/tensorFlow2/default"
)

print("Downloaded to:", MODEL_BASE_PATH)


# ---------------------------------------------------------------------------
# Stage 2: Locate required files within the downloaded directory tree
# ---------------------------------------------------------------------------

def find_file(base_path, filename):
    """
    Recursively searches a directory tree for a file with the given name.

    KaggleHub may nest downloaded files inside subdirectories whose names
    are not known ahead of time, so a recursive search is used rather than
    constructing a hard-coded path.

    Parameters:
        base_path (str): Root directory to start the search from.
        filename  (str): Exact filename to search for (case-sensitive).

    Returns:
        str | None: Full path to the file if found, or None if not found.
    """
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


# Path to the Python file that defines the custom TransformerBlock Keras layer.
# This file must be loaded before the .h5 model, because Keras needs the class
# definition registered in order to deserialize the model correctly.
CUSTOM_LAYER_FILE = find_file(MODEL_BASE_PATH, "cnn_vit_model.py")

# Path to the saved Keras model weights and architecture in HDF5 format.
MODEL_FILE = find_file(MODEL_BASE_PATH, "vit_dataset-1.h5")

print("Custom layer file:", CUSTOM_LAYER_FILE)
print("Model file:", MODEL_FILE)


# ---------------------------------------------------------------------------
# Stage 3: Dynamically load the custom Keras layer definition
# ---------------------------------------------------------------------------

# The model uses a custom layer class (TransformerBlock) that is not part of
# the standard Keras library. To load the .h5 model, Keras must be able to
# resolve this class by name. We load the source file at runtime using
# importlib so we do not need it as a installed package or local module.

# Build a module spec from the file path, then create and execute the module.
# Registering it in sys.modules under its logical name ('cnn_vit_model') ensures
# any internal imports within the file can also resolve correctly.
spec   = importlib.util.spec_from_file_location("cnn_vit_model", CUSTOM_LAYER_FILE)
cnn_vit = importlib.util.module_from_spec(spec)
sys.modules["cnn_vit_model"] = cnn_vit
spec.loader.exec_module(cnn_vit)

# Extract the custom layer class from the dynamically loaded module
TransformerBlock = cnn_vit.TransformerBlock


# ---------------------------------------------------------------------------
# Stage 4: Load the Keras model
# ---------------------------------------------------------------------------

print("Loading Keras model...")

# custom_object_scope makes TransformerBlock available to Keras by name during
# deserialization. Without this, Keras would raise an 'Unknown layer' error
# when it encounters 'TransformerBlock' in the .h5 config.
#
# compile=False skips recompiling the model with its original optimizer and
# loss function — not needed for inference or export, and avoids potential
# errors if the optimizer config is incompatible with the current TF version.
with tf.keras.utils.custom_object_scope({"TransformerBlock": TransformerBlock}):
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)

print("Model loaded")


# ---------------------------------------------------------------------------
# Stage 5: Convert and export to ONNX
# ---------------------------------------------------------------------------

# Destination path for the exported ONNX model file
OUTPUT_PATH = "model.onnx"

# Define the expected input tensor shape and type for the ONNX graph.
# Shape [None, 256, 256, 3] means:
#   None -- dynamic batch size (accepts any number of images per call)
#   256  -- image height in pixels
#   256  -- image width in pixels
#   3    -- RGB color channels
# dtype=tf.float32 matches the normalization applied in the preprocessing pipeline
# (pixel values scaled to [0.0, 1.0]).
input_signature = [
    tf.TensorSpec(shape=[None, 256, 256, 3], dtype=tf.float32, name="input")
]

# Convert the Keras model to ONNX format.
# opset=13 refers to the ONNX operator set version. Version 13 offers broad
# compatibility with modern ONNX runtimes while supporting all ops used by
# this CNN + ViT architecture (including attention and layer normalization).
tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=OUTPUT_PATH,
)

print("ONNX model saved as:", OUTPUT_PATH)