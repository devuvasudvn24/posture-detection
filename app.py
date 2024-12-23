import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from flask import Flask, render_template, Response

app = Flask(__name__)

MODEL_PATH = "cnn_lstm_posture_model.h5"
model = load_model(MODEL_PATH)

class_mapping = {0: 'sitting', 1: 'standing', 2: 'walking'}

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Preprocessing function
def preprocess_landmarks(landmarks_sequence):
    return np.array(landmarks_sequence)[np.newaxis, ..., np.newaxis]

# Video capture
def generate_frames():
    cap = cv2.VideoCapture(0)
    sequence = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            sequence.append(landmarks)
            if len(sequence) > 30:
                sequence.pop(0)
            if len(sequence) == 30:
                preprocessed_sequence = preprocess_landmarks(sequence)
                predictions = model.predict(preprocessed_sequence)
                predicted_class = np.argmax(predictions)
                posture = class_mapping[predicted_class]
                cv2.putText(frame, f"Posture: {posture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No pose detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame as JPEG and return
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to display video
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML page with video feed

if __name__ == '__main__':
    app.run(debug=True)
