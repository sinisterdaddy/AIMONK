from flask import Flask, request, render_template, jsonify, send_from_directory
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OUTPUT_FOLDER = 'outputs'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)



AI_BACKEND_URL = 'http://13.233.159.36:8080/predict'  # AI backend service

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
        image_filename = image.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        image.save(image_path)

        # Construct the image URL
        image_url = f"http://{request.host}/uploads/{image_filename}"


        # Call AI backend for predictions
        response = requests.post(AI_BACKEND_URL, json={'image_url': image_url})

        if response.status_code == 200:
            predictions = response.json().get('predictions', [])

            # Open the uploaded image
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Set font for text
            font = ImageFont.load_default()

            # Get image dimensions
            img_width, img_height = img.size

            # Draw bounding boxes on the image
            for pred in predictions:
                # Scale coordinates to match original image dimensions
                x_center = pred['xcenter'] * img_width / 640
                y_center = pred['ycenter'] * img_height / 640
                width = pred['width'] * img_width / 640
                height = pred['height'] * img_height / 640

                # Convert center coordinates to corner coordinates
                x_min = int(x_center - width / 2)
                y_min = int(y_center - height / 2)
                x_max = int(x_center + width / 2)
                y_max = int(x_center + width / 2)

                # Draw the bounding box and class label
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min - 10), f"{pred['name']} ({pred['confidence']:.2f})", fill="red", font=font)

            # Save the annotated image
            annotated_image_filename = f"annotated_{image_filename}"
            annotated_image_path = os.path.join(OUTPUT_FOLDER, annotated_image_filename)
            img.save(annotated_image_path)

            # Save predictions as a JSON file
            json_filename = f"predictions_{os.path.splitext(image_filename)[0]}.json"
            json_path = os.path.join(OUTPUT_FOLDER, json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(predictions, json_file, indent=4)

            # Convert the image to base64 for displaying in the HTML
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='PNG')
            img_byte_array.seek(0)
            img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')

            # Render result page with predictions and download links
            return render_template(
                'result.html',
                predictions=predictions,
                json_data=json.dumps(predictions, indent=4),  # Pass JSON for display
                img_base64=img_base64,
                annotated_image_url=f"/outputs/{annotated_image_filename}",
                json_url=f"/outputs/{json_filename}"
            )
        else:
            return jsonify({'error': 'Error in AI backend response'}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outputs/<filename>')
def serve_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
