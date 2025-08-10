# inference_runner.py
import base64
import face_preprocess2
from emotion_inference import load_emotion_model, infer_emotions
from face_preprocess2 import process_face_image_from_base64
import cv2

# Load the model ONCE at import-time
model, processor, class_names, device = load_emotion_model("murk1sh/best_model_vit")
print(f"[INIT] Model loaded on {device}; classes: {class_names}")


"""
    Reads the saved image file, isolates the face, and runs emotion inference.
    Returns a dict: { 'label': str, 'scores': {label: prob}, 'topk': [(label, prob), ...] }
    Raises a ValueError if no face is detected/invalid image.
    """
def run_inference_on_file(image_path: str):
    
    # Read the file and convert to base64 string (no data URL header needed)
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    #import pdb; pdb.set_trace() for debug
    # Crop face (returns grayscale ndarray 224x224 by default if you set it so)
    face_nd = process_face_image_from_base64(b64, output_size=(224, 224))
    if face_nd is None:
        raise ValueError("No face detected or invalid image")

    # Run inference on the cropped face
    result = infer_emotions(face_nd, model, processor, class_names, device, top_k=1)
    if image_path is not None:
        output_filename = image_path + '-processed.png'
        org = (5, 249)  # Bottom-left corner of the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)  # Green color in BGR
        thickness = 2

        enlarged_image = cv2.copyMakeBorder(face_nd, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value = [255,255,255])

        cv2.putText(enlarged_image, result['label'], org, font, font_scale, color, thickness, cv2.LINE_AA)
        
        cv2.imwrite(output_filename, enlarged_image)
        
        print(f"Image successfully saved as {output_filename}")
    return result
