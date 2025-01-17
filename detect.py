from flask import Flask, jsonify, Response
import cv2

app = Flask(__name__)

# Detection data with movement flags
detection_result = {
    "detected": True,
    "x": 1,
    "y": 2,
    "z": 3,
    "forward": False,  # Simulate forward movement
    "backward": True # Simulate backward movement
}

# Video capture
cap = cv2.VideoCapture(0)

@app.route('/detection') # API endpoint za detection data
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
