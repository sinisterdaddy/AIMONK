from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Path to YOLOv5 model
MODEL_PATH = 'yolov5s.pt'  # Ensure this is your correct model path (e.g., yolov5s, yolov5m)

# Load YOLOv5 model using torch.hub
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

# Define helper function to process image from URL
def process_image_from_url(image_url):
    try:
        # Fetch image from URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Convert to RGB (if not already)
        image = image.convert("RGB")

        # Resize image to 640x640 for YOLOv5
        image_resized = image.resize((640, 640))

        return image_resized, None
    except Exception as e:
        return None, str(e)

# Helper function to extract predictions from YOLOv5 output
def run_inference(image):
    # Run inference using YOLOv5 (directly on PIL image)
    results = model(image)  # YOLOv5 model supports PIL images natively

    # Convert predictions to a pandas DataFrame
    predictions = results.pandas().xywh[0]  # Extracting the DataFrame for the first image

    return predictions

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image_url is part of the request JSON
        data = request.get_json()

        # Get the image_url from the JSON payload
        image_url = data.get('image_url')

        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        # Process the image from the URL
        image, error = process_image_from_url(image_url)
        if error:
            return jsonify({'error': error}), 500

        # Run inference
        predictions = run_inference(image)

        # Convert predictions into a list of dictionaries for response
        predictions_list = predictions.to_dict(orient="records")

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
