from flask import Flask, jsonify, request
import face_recognition
import numpy as np
import os

app = Flask(__name__)

known_face_encodings = []
known_face_names = ["Ana", "Tatjana", "Stefanija"]

known_faces = {
    "Ana": "C:\\Users\\stefi\\Desktop\\projektSRDS\\Projekt_z_sliki\\slika_an.jpg",
    "Tatjana": "C:\\Users\\stefi\\Desktop\\projektSRDS\\Projekt_z_sliki\\slika_t.jpg",
    "Stefanija": "C:\\Users\\stefi\\Desktop\\projektSRDS\\Projekt_z_sliki\\slika_s.jpg"
}

for name, path in known_faces.items():
    try:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            continue

        image = face_recognition.load_image_file(path)
        face_encoding = face_recognition.face_encodings(image)

        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            print(f"[INFO] Loaded encoding for {name}.")
        else:
            print(f"[WARNING] No faces detected in {name}'s image.")

    except Exception as e:
        print(f"[ERROR] Could not process {name}: {e}")

if not known_face_encodings:
    print("[CRITICAL ERROR] No valid face encodings loaded! Check your images.")

detection_result = {"detected": False, "name": "None"}


@app.route('/detection', methods=['GET'])
def detection():
    """Perform face detection and recognition."""
    global detection_result

    test_image_path = "C:\\Users\\stefi\\Desktop\\projektSRDS\\Projekt_z_sliki\\slika_an.jpg"

    if not os.path.exists(test_image_path):
        return jsonify({"error": f"Test image not found: {test_image_path}"})

    try:
        image = face_recognition.load_image_file(test_image_path)
        rgb_image = image[:, :, ::-1] 
    except Exception as e:
        return jsonify({"error": f"Failed to load test image: {str(e)}"})

    face_locations = face_recognition.face_locations(rgb_image)
    print(f"[INFO] Detected {len(face_locations)} faces in test image.")

    if not face_locations:
        return jsonify({"detected": False, "name": "None"})

    try:
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    except Exception as e:
        return jsonify({"error": f"Face encoding failed: {str(e)}"})

    if not face_encodings:
        print("[WARNING] No face encodings found in the test image.")
        return jsonify({"detected": False, "name": "None"})

    if not known_face_encodings:
        print("[CRITICAL ERROR] No known face encodings available for comparison.")
        return jsonify({"error": "No known face encodings available."})

    detection_result = {"detected": False, "name": "None"}

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances) 
            detection_result = {"detected": True, "name": known_face_names[best_match_index]}
            break

    return jsonify(detection_result)


if __name__ == '__main__':
    app.run(debug=False)  
