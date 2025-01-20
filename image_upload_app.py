# app.py (Flask Backend)
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import face_recognition
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# Paths to known faces (replace with your actual image paths)
known_faces = {
    "Tatjana": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_t.jpg",
            "Ana": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_an.jpg",
            "Stefanija": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_s.jpg"
}

# Load known face encodings
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for name, path in known_faces.items():
        image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

load_known_faces()

@app.route('/detect_face', methods=['POST'])
def detect_face():
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"})

    # Save the uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    file.save(filepath)

    # Load and process the image
    uploaded_image = face_recognition.load_image_file(filepath)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)

    if len(face_encodings) == 0:
        return jsonify({"message": "No face detected"})

    # Match faces with known faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            return jsonify({"name": name})

    return jsonify({"message": "Unknown person detected"})

if __name__ == '__main__':
    app.run(debug=True)
