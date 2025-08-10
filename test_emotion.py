# test_emotion.py
from inference_runner import run_inference_on_file
import os

# Choose one image from the uploads folder
image_name = "surprised1.jpg"  # change this to the file you want
image_path = os.path.join("uploads", image_name)

try:
    result = run_inference_on_file(image_path)
    print(f"Image: {image_name}")
    print(f"  Predicted: {result['label']}")
    print(f"  Scores: {result['scores']}")
    print(f"  Top-k: {result['topk']}")
except Exception as e:
    print(f"Error processing {image_name}: {e}")
