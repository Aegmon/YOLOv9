from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained YOLOv9 segmentation model
MODEL_PATH = 'best.pt'  # Update with the correct path to your model
model = YOLO(MODEL_PATH)

# OpenCV VideoCapture (webcam)
cap = cv2.VideoCapture(0)  # Change to correct device if needed (0 is default webcam)
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Print the current FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {fps}")
# Resize factor for the webcam feed to improve performance
RESIZE_WIDTH = 120
RESIZE_HEIGHT = 120

# Class names corresponding to indices 0, 1, 2
CLASS_NAMES = ['Fire', 'Light Fire', 'Smoke']  # Adjust this to your model's class names

# Minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # Only show detections with >= 70% confidence

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame to improve performance
            frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # Convert resized frame to PIL image for YOLO inference
            pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

            # Perform inference using YOLOv9 model
            results = model(pil_img)

            # Draw segmentation masks and class labels on the frame
            result = results[0]
            if result.masks is not None:
                for i, seg in enumerate(result.masks.xy):
                    points = np.array(seg, dtype=np.int32)
                    class_id = int(result.boxes.cls[i].item())  # Cast to int to avoid float index error
                    confidence = result.boxes.conf[i].item()  # Get the confidence score for this detection

                    # Only display class labels for classes with indices 0, 1, or 2 and confidence >= 70%
                    if confidence >= CONFIDENCE_THRESHOLD and class_id in [0, 1, 2]:  # Check confidence
                        class_name = CLASS_NAMES[class_id]
                        # Draw the segmentation mask only if confidence is above the threshold
                        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                        # Display class name and confidence on the frame
                        cv2.putText(frame, f'{class_name}: {confidence*100:.2f}%', tuple(points[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame back to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to be displayed in the web page
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
