# utils.py

import torch
from PIL import Image
import numpy as np
from io import BytesIO

def process_image(image_bytes):
    """
    Convert the input image bytes to a tensor ready for YOLOv3.
    """
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image_resized = image.resize((640, 640))
    img_np = np.array(image_resized)
    img_np = img_np / 255.0  # Normalize to [0, 1]
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)  # Add batch dimension
    return img_tensor

def run_inference(model, image_tensor):
    """
    Run inference on the processed image tensor using the YOLOv3 model.
    """
    with torch.no_grad():
        outputs = model(image_tensor)
    return outputs
