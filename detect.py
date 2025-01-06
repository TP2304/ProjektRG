from flask import Flask, jsonify, Response
import cv2

app = Flask(__name__)

# Simulated detection data
detection_result = {
    "detected": False,
    "x": 0,
    "y": 0,
    "z": 0
}

# Video capture
cap = cv2.VideoCapture(0)

@app.route('/detection')
def detection():
    # Return detection results as JSON
    return jsonify(detection_result)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
