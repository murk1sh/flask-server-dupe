# face_preprocess2.py
import cv2
import numpy as np
import base64

# Load OpenCV's built-in Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def process_face_image_from_base64(base64_image_string: str, output_size: tuple = (224, 224),
                                   pad_ratio: float = 0.10, use_color: bool = False):
    try:
        # Support both data URLs and raw base64 strings
        if "," in base64_image_string:
            _, encoded = base64_image_string.split(",", 1)
        else:
            encoded = base64_image_string

        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            print("Could not decode image.")
            return None
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        print("No face detected.")
        return cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA) # if cant find a face, it just returns the greyscale image without crop

    # Choose largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    # Apply padding
    pad = int(pad_ratio * max(w, h))
    x0, y0 = x - pad, y - pad
    x1, y1 = x + w + pad, y + h + pad

    # Clamp to image bounds
    H, W = gray.shape
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)

    # Crop and resize
    crop_src = img if use_color else gray
    face_roi = crop_src[y0:y1, x0:x1]
    if face_roi.size == 0:
        print("Cropped face region is empty.")
        return None
    content = cv2.resize(face_roi, output_size, interpolation=cv2.INTER_AREA)
    

    return content
