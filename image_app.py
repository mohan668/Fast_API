import cv2
import torch
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Load the YOLOv8 model
model = YOLO('best.pt')  # Directly load the model using YOLO class


# Function to decode the base64 image and return it as a PIL Image
def decode_base64_image(image_base64):
    img_data = BytesIO(image_base64)
    img = Image.open(img_data)
    return img

@app.post("/detect-trash/")
async def detect_trash(file: UploadFile = File(...)):
    # Read the image
    img_bytes = await file.read()
    img = decode_base64_image(img_bytes)
    
    # Convert the PIL image to a NumPy array
    img_np = np.array(img)
    
    # Perform inference using YOLOv8 model
    results = model(img_np)  # Inference
    
    # Parse results
    predictions = results[0].boxes  # Get bounding boxes
    
    # Extract bounding box details (xyxy format, confidence, and class)
    boxes = predictions.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
    confidences = predictions.conf.cpu().numpy()  # Confidence scores
    classes = predictions.cls.cpu().numpy()  # Class indices
    
    # Draw bounding boxes on the image
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box, conf, cls in zip(boxes, confidences, classes):
        if conf > 0.5:  # You can adjust the confidence threshold here
            x1, y1, x2, y2 = map(int, box)  # Get the bounding box coordinates
            # Draw rectangle and label on image
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Trash {conf:.2f}"  # Change this if you want to display class labels
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert the image back to PIL for sending in response
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Convert the PIL image to bytes and send as response
    img_byte_array = BytesIO()
    img_pil.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)

    return StreamingResponse(img_byte_array, media_type="image/png")
