from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import torch 
import sys
import bcrypt
import hashlib
import serial  
import serial.tools.list_ports
import logging
from pathlib import Path
from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify
from flaskwebgui import FlaskUI
from waitress import serve 

print("CUDA available:", torch.cuda.is_available())
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)  # PyInstaller temp folder
else:
    BASE_DIR = Path(__file__).parent.resolve()
USER_DB = BASE_DIR / "users.txt"

app = Flask(__name__)

fire_alert_triggered = False

light_fire_alert_triggered = False

app.secret_key = "supersecretkey"
MODEL_PATH = 'yoloseg9.pt'
model = YOLO(MODEL_PATH)

def find_arduino():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "Arduino" in port.description:
            return port.device
    return None

arduino_port = find_arduino()
if arduino_port:
    try:
        arduino = serial.Serial(port=arduino_port, baudrate=9600, timeout=1)
        print(f"✅ Arduino connected on {arduino_port}!")
    except serial.SerialException as e:
        print(f"❌ Error opening {arduino_port}: {e}")
        arduino = None
else:
    print("⚠️ No Arduino detected. Check connections.")
    arduino = None

# Ensure users.txt exists
if not USER_DB.exists():
    with open(USER_DB, "w") as f:
        f.write("admin,email=admin@email.com,password=" + hashlib.sha256("Admin@1234".encode()).hexdigest() + "\n")

def get_users():
    users = {}
    with open(USER_DB, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                name, email, password = parts
                users[email] = password
    return users

def register_user(email, password):
    users = get_users()
    if email in users:
        return False
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with open(USER_DB, "a") as f:
        f.write(f"Admin User,{email},{hashed_password}\n")
    return True

def authenticate_user(email, password):
    users = get_users()
    return email in users and bcrypt.checkpw(password.encode(), users[email].encode())

cap = cv2.VideoCapture(0)  


RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480


CLASS_NAMES = ['Fire', 'Light Fire', 'Smoke']  

def generate_frames():
    global fire_alert_triggered, light_fire_alert_triggered

    prev_time = 0
    last_result = None
    frame_count = 0
    INFERENCE_EVERY_N_FRAMES = 2

    # Alert confirmation mechanism
    fire_detected_frames = 0
    light_fire_detected_frames = 0
    CONFIRMATION_THRESHOLD = 3  # Confirm alert after 3 consecutive detections

    while True:
        success, frame = cap.read()
        if not success:
            break

        curr_time = time.time()
        original_h, original_w = frame.shape[:2]
        inference_frame = cv2.resize(frame, (320, 320))

        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            results = model.predict(inference_frame, imgsz=320, conf=0.8, stream=False)
            last_result = results[0]
        frame_count += 1

        detected_fire = False
        detected_light_fire = False

        if last_result and last_result.boxes is not None:
            boxes = last_result.boxes.xyxy.cpu().numpy()
            classes = last_result.boxes.cls.cpu().numpy()

            scale_x = original_w / 320
            scale_y = original_h / 320

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                class_id = int(classes[i])
                if class_id in [0, 1, 2]:
                    class_name = CLASS_NAMES[class_id]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    # Detection logic
                    if class_name == "Fire":
                        detected_fire = True
                    elif class_name == "Light Fire":
                        detected_light_fire = True

        # Alert validation logic
        if detected_fire:
            fire_detected_frames += 1
            light_fire_detected_frames = 0
        elif detected_light_fire:
            light_fire_detected_frames += 1
            fire_detected_frames = 0
        else:
            fire_detected_frames = 0
            light_fire_detected_frames = 0

        # Alert trigger
        if fire_detected_frames >= CONFIRMATION_THRESHOLD:
            fire_alert_triggered = True
            if arduino and arduino.is_open:
                    arduino.write(b'ALARM\n')
            print("🔥 Fire Detected! Alarm Triggered.")
            light_fire_alert_triggered = False
        elif light_fire_detected_frames >= CONFIRMATION_THRESHOLD:
            light_fire_alert_triggered = True
            fire_alert_triggered = False
            if arduino and arduino.is_open:
                arduino.write(b'STOP\n')
            print("✅ No Fire Detected. Resetting Alerts.")
        else:
            fire_alert_triggered = False
            light_fire_alert_triggered = False
            if arduino and arduino.is_open:
                 arduino.write(b'STOP\n')
            print("✅ No Fire Detected. Resetting Alerts.")
        # --- FPS Display ---
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

     
@app.route('/video_feed') 
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if authenticate_user(email, password):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        return render_template("dashboard.html", error="Invalid credentials!")
    return render_template("dashboard.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if register_user(email, password):
            return redirect(url_for("login"))
        return render_template("register.html", error="User already exists!")
    return render_template("register.html")
@app.route("/dashboard")
def dashboard():
    if "logged_in" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/check_alert")
def check_alert():
    return jsonify({
        "fire_alert": fire_alert_triggered,
        "light_fire_alert": light_fire_alert_triggered
    })

def get_device_location():
    return {"lat": 37.7749, "lng": -122.4194}

def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    FlaskUI(app=app, server="flask").run()
    # run_flask()

