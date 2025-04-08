from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import torch  # Import torch to check GPU availability

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())

# Initialize Flask app
app = Flask(__name__)

# Load the trained YOLOv9 segmentation model
MODEL_PATH = 'yoloseg9.pt'  # Update with the correct path to your model
model = YOLO(MODEL_PATH)

# OpenCV VideoCapture (webcam)
cap = cv2.VideoCapture(0)  # Change to correct device if needed (0 is default webcam)

# Resize factor for the webcam feed to improve performance
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# Class names corresponding to indices 0, 1, 2
CLASS_NAMES = ['Fire', 'Light Fire', 'Smoke']  # Adjust this to your model's class names

def generate_frames():
    prev_time = 0
    last_result = None
    frame_count = 0
    INFERENCE_EVERY_N_FRAMES = 2  # Try every 2 frames for better FPS

    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        original_h, original_w = frame.shape[:2]

        # Resize for inference (smaller = faster)
        inference_frame = cv2.resize(frame, (320, 320))

        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            results = model.predict(inference_frame, imgsz=320, conf=0.3, stream=False)
            last_result = results[0]
        frame_count += 1

        # Draw scaled mask on original frame
        if last_result and last_result.masks is not None:
            for i, seg in enumerate(last_result.masks.xy):
                # Scale the segmentation points back to original frame size
                scale_x = original_w / 320
                scale_y = original_h / 320
                scaled_points = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in seg], dtype=np.int32)

                cv2.polylines(frame, [scaled_points], isClosed=True, color=(0, 255, 0), thickness=2)

                class_id = int(last_result.boxes.cls[i].item())
                if class_id in [0, 1, 2]:
                    class_name = CLASS_NAMES[class_id]
                    cv2.putText(frame, class_name, tuple(scaled_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # --- FPS Calculation ---
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        # Convert to JPEG for browser streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
