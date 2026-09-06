from flask import Flask, render_template, request, Response, jsonify
import cv2
import numpy as np
import face_recognition
import os
import datetime
import csv

app = Flask(__name__)

# Initialize Camera
camera = cv2.VideoCapture(0)
known_faces = []
known_names = []
faces_folder = "faces"
csv_file = "attendance.csv"

# Ensure necessary folders exist
if not os.path.exists(faces_folder):
    os.makedirs(faces_folder)

# Ensure CSV file exists
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time", "Day"])

# Load known faces
def load_known_faces():
    global known_faces, known_names
    known_faces = []
    known_names = []
    for filename in os.listdir(faces_folder):
        img_path = os.path.join(faces_folder, filename)
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(filename.split(".")[0])

load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_user', methods=['POST'])
def add_user():
    name = request.form['name'].strip()

    if not name:
        return jsonify({"message": "Name cannot be empty!"})

    # Check if user already exists
    if f"{name}.jpg" in os.listdir(faces_folder):
        return jsonify({"message": f"User {name} already exists!"})

    success, frame = camera.read()
    if success:
        img_path = os.path.join(faces_folder, f"{name}.jpg")
        cv2.imwrite(img_path, frame)

        # Reload face data
        load_known_faces()
        return jsonify({"message": f"User {name} added successfully!"})
    else:
        return jsonify({"message": "Failed to capture image. Try again!"})

@app.route('/recognize_face')
def recognize_face():
    success, frame = camera.read()
    
    if not success:
        return jsonify({"message": "Error accessing camera!"})

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    name = "Unknown"
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, encoding)
        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Save attendance
            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([name, str(datetime.date.today()), datetime.datetime.now().strftime("%H:%M:%S"), datetime.datetime.now().strftime("%A")])

            return jsonify({
                "name": name,
                "date": str(datetime.date.today()),
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "day": datetime.datetime.now().strftime("%A"),
                "message": f"Welcome, {name}!"
            })

    return jsonify({"name": "Unknown", "message": "Face not recognized! Please add user."})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == "__main__":
    app.run(debug=True)
