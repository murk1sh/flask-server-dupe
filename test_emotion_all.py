# test_emotion_all.py
from inference_runner import run_inference_on_file
import os
for image_name in os.listdir("uploads"):
    if image_name.endswith("-processed.png"): 
        continue
    image_path = os.path.join("uploads", image_name)
    try:
        result = run_inference_on_file(image_path)
        print(f"Image: {image_name}")
        print(f"  Predicted: {result['label']}")
        print(f"  Scores: {result['scores']}")
        print(f"  Top-k: {result['topk']}")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
