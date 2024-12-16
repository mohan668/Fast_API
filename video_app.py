import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Serve the static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the YOLOv8 model with your custom weights
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your trained weights

@app.get("/")
async def read_index():
    return {"message": "Welcome to the real-time detection app"}

@app.get("/real-time-detection/")
async def real_time_detection():
    # Open the default webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    # Function to generate video frames
    def generate_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference with YOLO
            results = model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
            classes = results[0].boxes.cls.cpu().numpy()  # Class indices

            # Draw bounding boxes on the frame
            for box, conf, cls in zip(boxes, confidences, classes):
                if conf > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Trash {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode the frame as a JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame in HTTP multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
