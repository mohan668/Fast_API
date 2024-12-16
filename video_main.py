import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace with the actual path to your best.pt model

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera index

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame from BGR to RGB (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference using YOLOv8 model
    results = model(frame_rgb)

    # Get the bounding boxes, confidences, and classes from the predictions
    predictions = results[0].boxes  # YOLOv8's box predictions
    boxes = predictions.xyxy.cpu().numpy()  # Get bounding boxes in xyxy format
    confidences = predictions.conf.cpu().numpy()  # Confidence scores
    classes = predictions.cls.cpu().numpy()  # Class indices

    # Draw bounding boxes on the frame
    for box, conf, cls in zip(boxes, confidences, classes):
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box)  # Get the bounding box coordinates
            # Draw a rectangle around the detected object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label the bounding box
            label = f"Trash {conf:.2f}"  # Add class label if necessary
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-Time Trash Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
