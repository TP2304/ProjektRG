from flask import Flask, jsonify
from flask_cors import CORS
import face_recognition
import atexit

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})  # Allow requests from your live server

# Load known face encodings and names
known_face_encodings = []
known_face_names = ["Ana", "Tatjana", "Stefanija"]

# Path to the static image
STATIC_IMAGE_PATH = r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\ana.jpg"

# Load images of known faces
known_faces = {
    "Ana": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\ana.jpg",
    "Tatjana": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\tatjana.jpg",
    "Stefanija": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\stefanija.jpg"
}

# Load and encode known faces
for name, path in known_faces.items():
    try:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        print(f"Loaded face for {name}")
    except Exception as e:
        print(f"Error loading face for {name}: {e}")

# Static image face detection result
detection_result = {"detected": False, "name": "None"}


@app.route('/detection', methods=['GET'])
def detection():
    """Perform face detection and recognition on the static image."""
    global detection_result
    try:
        # Load the static image
        image = face_recognition.load_image_file(STATIC_IMAGE_PATH)
        rgb_frame = image[:, :, ::-1]  # Convert to RGB for face recognition

        # Detect faces and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("No faces detected.")
            detection_result = {"detected": False, "name": "No Face Detected"}
            return jsonify(detection_result)

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if not face_encodings:
            print("No face encodings found.")
            detection_result = {"detected": False, "name": "No Face Detected"}
            return jsonify(detection_result)

        detection_result = {"detected": False, "name": "None"}  # Reset detection result

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                detection_result = {"detected": True, "name": known_face_names[match_index]}
                print(f"Match found: {known_face_names[match_index]}")
                break

    except Exception as e:
        print(f"Error during detection: {e}")
        detection_result = {"detected": False, "name": "Error"}

    return jsonify(detection_result)


# Gracefully handle shutdown
@atexit.register
def cleanup():
    print("Server shutting down. Resources released.")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
