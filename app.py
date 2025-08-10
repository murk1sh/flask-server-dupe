# app.py
# This version is updated to accept a base64 image string from the frontend.

import os
import random
import time
import base64 # Import the base64 library
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inference_runner import run_inference_on_file 

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS to allow requests from your frontend
CORS(app)

def predict_expression(image_path):
    result = run_inference_on_file(image_path)   # {'label': ..., 'scores': {...}, 'topk': [...]} # Need 2 add inference_runner in this folder
    return result['label']  # keep original behavior (a single string) to return to user

@app.route('/predict', methods=['POST'])
def upload_image():
    """
    Receives a base64 image string in a JSON payload, decodes it,
    saves it as a file, and returns a prediction.
    """
    print("predict endpoint was hit!")

    json_data = request.get_json()
    if not json_data or 'image' not in json_data:
        print("ERROR: No 'image' key in the JSON payload.")
        return jsonify({'error': 'Missing image data in request'}), 400

    base64_string = json_data['image']

    try:
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)

        filename = f"{int(time.time())}.png"
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, "wb") as f:
            f.write(image_data)

        print(f"File saved successfully from base64 to: {filepath}")

        # Run the prediction on the saved file
        prediction = predict_expression(filepath)
        print(f"Model returned prediction: '{prediction}'")

        # Return the prediction (unchanged response shape)
        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"ERROR: Failed to process base64 image. {e}")
        return jsonify({'error': 'Invalid image data or server error'}), 500


if __name__ == "__main__":
    app.run(debug=True, port=4000)
