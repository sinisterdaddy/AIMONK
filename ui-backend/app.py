from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np

app = Flask(__name__)

AI_BACKEND_URL = 'http://127.0.0.1:5000/predict'  # AI backend service

# Set the upload folder for image files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Route to serve images in the uploads folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if the image is part of the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Get the image from the request
        image = request.files['image']

        # Ensure the filename is safe and get the file extension
        image_filename = image.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

        # Save the image to the specified path
        image.save(image_path)

        # Construct the image URL (this URL will be accessible from the browser)
        image_url = f'http://127.0.0.1:8000/uploads/{image_filename}'

        # Call AI backend for predictions
        response = requests.post(AI_BACKEND_URL, json={'image_url': image_url})

        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            
            # Open the uploaded image for annotation
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Set font for annotations (you may need to adjust or use a different font)
            font = ImageFont.load_default()

            # Draw the bounding boxes and class names
            for pred in predictions:
                # Coordinates of the bounding box
                x_center = pred['xcenter']
                y_center = pred['ycenter']
                width = pred['width']
                height = pred['height']
                
                # Convert to bounding box corner coordinates
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(y_center + height / 2)

                # Draw the bounding box
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                
                # Draw the class name
                draw.text((x_min, y_min - 10), pred['name'], fill="red", font=font)

            # Save the annotated image in memory
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='PNG')
            img_byte_array.seek(0)
            
            # Convert the image to base64 for sending it over HTTP
            img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

            # Return predictions along with the annotated image in base64 format
            return render_template('result.html', predictions=predictions, img_base64=img_base64)
        
        else:
            return jsonify({'error': 'Error in AI backend response'}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
