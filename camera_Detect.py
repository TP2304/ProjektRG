from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detection_result = {
    "detected": False,
    "x": 0,
    "y": 0,
    "z": 0
}

cap = cv2.VideoCapture(0)

def detect_objects(frame):
    global detection_result
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        detection_result["detected"] = True
        x, y, w, h = faces[0]
        detection_result["x"] = x + w // 2
        detection_result["y"] = y + h // 2
        detection_result["z"] = w
    else:
        detection_result["detected"] = False
        detection_result["x"] = detection_result["y"] = detection_result["z"] = 0

@app.route('/detection')
def detection():
    return jsonify(detection_result)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Perform detection
            detect_objects(frame)

            if detection_result["detected"]:
                x = detection_result["x"] - detection_result["z"] // 2
                y = detection_result["y"] - detection_result["z"] // 2
                w = detection_result["z"]
                h = detection_result["z"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
