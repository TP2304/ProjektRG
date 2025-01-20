from flask import Flask, request, jsonify
import os
import face_recognition
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Known faces and their encodings
known_faces = {
"Ana": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\ana.jpg",
    "Tatjana": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\tatjana.jpg",
    "Stefanija": r"C:\Users\PC\Desktop\prva_razlicica_RG\ProjektRG\stefanija.jpg"
}
known_face_encodings = []
known_face_names = []

for name, path in known_faces.items():
    try:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)
    except Exception as e:
        print(f"Error encoding {name}: {e}")

@app.route('/detection', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if not file:
        return jsonify({"error": "File upload failed"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]

            results.append({
                "name": name,
                "location": face_location  # [top, right, bottom, left]
            })

        os.remove(file_path)
        return jsonify({"detected": True, "results": results})
    except Exception as e:
        os.remove(file_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
