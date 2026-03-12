import onnxruntime as ort
import psutil
import os

process = psutil.Process(os.getpid())

# RAM before loading
before = process.memory_info().rss / 1024 / 1024
print(f"RAM before loading: {before:.1f} MB")

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

# RAM after loading
after = process.memory_info().rss / 1024 / 1024
print(f"RAM after loading:  {after:.1f} MB")
print(f"Model uses RAM:     {after - before:.1f} MB")