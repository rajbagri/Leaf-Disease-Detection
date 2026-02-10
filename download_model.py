import kagglehub

# Download latest version
path = kagglehub.model_download(
    "khanaamer/leaf-disease-detection-using-cnn-and-vit/tensorFlow2/default"
)

print("Path to model files:", path)
