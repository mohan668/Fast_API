# Real-Time Object Detection with YOLOv8 and FastAPI

## Project Overview
This project implements a real-time object detection application using YOLOv8 (Ultralytics) and FastAPI. The application captures video from your webcam, performs object detection on each frame, and streams the result with bounding boxes overlaid on the detected objects. The processed video stream is served via a FastAPI endpoint and can be accessed through a web browser.

### Key Features:
- **Real-time Object Detection**: Utilizes YOLOv8 to detect objects in video frames.
- **Streaming**: Streams the processed video with detected objects back to the browser in real-time.
- **FastAPI Backend**: Handles requests and serves static files (HTML, images, etc.) through FastAPI.
- **Easy Setup**: Simple installation process with dependencies listed in a `requirements.txt` file.

## Project Structure
```
your_project/
├── static/
│   └── index.html          # Static HTML page to view the real-time stream
├── video_app.py            # FastAPI backend script
├── best.pt                 # YOLOv8 custom-trained model weights (replace with your model)
└── requirements.txt        # List of required Python packages
```

### Files:
1. **`video_app.py`**: This is the FastAPI application script that contains the backend logic for handling real-time video streaming and object detection.
2. **`index.html`**: A basic HTML file placed inside the `static/` folder to view the real-time video stream.
3. **`best.pt`**: This file contains the trained weights for the YOLOv8 model. Replace it with your custom-trained model file.

## Requirements

Before running the project, you need to install the required packages. You can install them using the following command:

### 1. Install the Dependencies:
Create a virtual environment (optional but recommended), then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. `requirements.txt` contents:
```
fastapi
opencv-python
uvicorn
ultralytics
```

If you don't want to use a `requirements.txt` file, you can manually install the dependencies using `pip`:

```bash
pip install fastapi opencv-python uvicorn ultralytics
```

## Setup and Run

### 1. Clone the Repository (or copy the files):
Ensure that the `index.html`, `video_app.py`, and `best.pt` files are in the correct locations in your project directory.

### 2. Running the Application:

To start the FastAPI server, run the following command from your project directory:

```bash
uvicorn video_app:app --reload
```

This will start the FastAPI server at `http://127.0.0.1:8000`.

### 3. Access the Application:

- **Real-Time Detection**: Open your browser and go to `http://127.0.0.1:8000/real-time-detection/` to view the webcam video stream with YOLOv8 object detection.
- **Static HTML Page**: Open the `index.html` file by visiting `http://127.0.0.1:8000/static/index.html` to view the live stream.

## YOLOv8 Model

In this project, we use YOLOv8 for object detection. Replace the `best.pt` file with your custom-trained YOLOv8 model weights. If you haven't trained your model yet, you can use the [Ultralytics YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics) for training your custom object detection model.

### Example of Model Loading:

In `video_app.py`:
```python
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your trained weights
```

## Troubleshooting

1. **404 Not Found for `index.html`**: Ensure that the `index.html` file is placed inside the `static/` folder and that the folder is correctly mounted in FastAPI.
2. **Error: 'StreamingResponse' is not defined**: Ensure that `StreamingResponse` is imported from `fastapi.responses` in the `video_app.py` file.
3. **Camera Not Found**: If you're using an external webcam or the default webcam (index 0) doesn't work, try changing the index in the `cv2.VideoCapture(0)` line of `video_app.py` to another value (e.g., `cv2.VideoCapture(1)`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

### Key Sections of the README:

1. Project Overview: Describes the purpose of the project and the core technologies used (YOLOv8 and FastAPI).
2. Project Structure: Lists the directory structure to explain where the various files should be placed.
3. Installation: Instructions for installing required Python packages either through `requirements.txt` or manually.
4. Setup and Run: Step-by-step guide for setting up and running the application.
5. YOLOv8 Model: Explanation of how to use a custom YOLOv8 model with the project.
6. Troubleshooting**: Common errors users might encounter and how to solve them.
7. License: Project license information.
