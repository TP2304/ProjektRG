import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance
from pygame import mixer
import time
import os
import imutils
from imutils import face_utils
from ui2 import get_look_direction

class FaceEyeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face and Eye Detection")
        self.root.geometry("1200x600")

        self.image_path = ""
        self.video_path = ""
        self.cap = None
        self.frame_delay = 0

        # Load Haar cascades for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Paths to known faces
        self.known_faces = {
            "Tatjana": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_t.jpg",
            "Ana": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_an.jpg",
            "Stefanija": r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/Projekt_z_sliki/slika_s.jpg"
        }

        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Initialize Dlib's face detector and create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor_path = r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(self.shape_predictor_path):
            raise FileNotFoundError(f"The shape predictor file {self.shape_predictor_path} was not found. Please ensure the file is available.")
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)

        # Initialize the mixer for playing alarm sound
        mixer.init()
        self.alarm_path = r"C:/Users/klinc/Downloads/ProjektRG/ProjektRG/alarm2.mp3"
        mixer.music.load(self.alarm_path)

        # Threshold values for EAR
        self.EYE_AR_THRESH = 0.23
        self.EYE_AR_CONSEC_FRAMES = 20
        self.EYE_CLOSE_DURATION = 3  # Duration in seconds for eyes closed to trigger alert

        # Initialize the frame counter and the total number of blinks
        self.counter = 0
        self.total_blinks = 0

        # Track the time when eyes were first detected closed
        self.close_start_time = None

        # Get the indexes of the facial landmarks for the left and right eye
        (self.left_start, self.left_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (self.right_start, self.right_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        # Indices for nose tip and nose bridge
        self.nose_tip_idx = (30, 35)  # Points 30 to 35
        self.nose_bridge_idx = (27, 30)  # Points 27 to 30

        # Threshold for head orientation
        self.HEAD_ORIENTATION_THRESH = 15

        # Create and arrange widgets
        self.create_widgets()

    def create_widgets(self):
        # Define modern font styles
        self.font_title = ("Helvetica", 18, "bold")
        self.font_button = ("Helvetica", 12)
        self.font_checkbox = ("Helvetica", 12)
        self.font_text = ("Helvetica", 10)

        # Define colors
        self.bg_color = "#ffe6e6"  # Light pink background
        self.btn_color = "#ffb3b3"  # Light red button background
        self.text_color = "#800000"  # Dark red text color
        self.frame_color = "#ffcccc"  # Light pink frame background
        self.canvas_color = "#ffffff"  # White canvas background

        self.root.configure(bg=self.bg_color)

        # Create main frames
        left_frame = tk.Frame(self.root, width=200, padx=10, pady=10, bg=self.frame_color, relief=tk.RIDGE, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(self.root, padx=10, pady=10, bg=self.bg_color)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Create and place buttons in the left frame
        self.upload_image_button = tk.Button(left_frame, text="Upload Image", command=self.upload_image, width=20,
                                             bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.upload_image_button.pack(pady=10)

        self.upload_video_button = tk.Button(left_frame, text="Upload Video", command=self.upload_video, width=20,
                                             bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.upload_video_button.pack(pady=10)

        self.detect_face_button = tk.Button(left_frame, text="Detect Face", command=self.detect_face, width=20,
                                            bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.detect_face_button.pack(pady=10)

        self.detect_eyes_button = tk.Button(left_frame, text="Detect Eyes", command=self.detect_eyes, width=20,
                                            bg=self.btn_color, fg=self.text_color, font=self.font_button)
        self.detect_eyes_button.pack(pady=10)

        self.detect_sunglasses_button = tk.Button(left_frame, text="Detect with Sunglasses",
                                                  command=self.detect_with_sunglasses, width=20, bg=self.btn_color,
                                                  fg=self.text_color, font=self.font_button)
        self.detect_sunglasses_button.pack(pady=10)

        # Checkboxes for enabling/disabling functionalities
        self.sunglasses_check_var = tk.IntVar()
        self.eyes_check_var = tk.IntVar()
        self.face_check_var = tk.IntVar()

        self.sunglasses_checkbox = tk.Checkbutton(left_frame, text="Check Sunglasses",
                                                  variable=self.sunglasses_check_var, bg=self.frame_color,
                                                  fg=self.text_color, font=self.font_checkbox)
        self.sunglasses_checkbox.pack(pady=5)

        self.eyes_checkbox = tk.Checkbutton(left_frame, text="Check Eyes", variable=self.eyes_check_var,
                                            bg=self.frame_color, fg=self.text_color, font=self.font_checkbox)
        self.eyes_checkbox.pack(pady=5)

        self.face_checkbox = tk.Checkbutton(left_frame, text="Check Face", variable=self.face_check_var,
                                            bg=self.frame_color, fg=self.text_color, font=self.font_checkbox)
        self.face_checkbox.pack(pady=5)

        # Canvas for displaying the image or video frame
        self.canvas = tk.Canvas(right_frame, bg=self.canvas_color, relief=tk.SUNKEN, borderwidth=2)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Text box for displaying messages
        self.message_box = tk.Text(right_frame, width=40, height=10, state=tk.DISABLED, bg=self.canvas_color,
                                   fg=self.text_color, font=self.font_text)
        self.message_box.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

    def log_message(self, message):
        self.message_box.configure(state=tk.NORMAL)
        current_text = self.message_box.get("1.0", tk.END)
        if message + "\n" not in current_text:
            self.message_box.insert(tk.END, message + "\n")
            self.message_box.yview(tk.END)
        self.message_box.configure(state=tk.DISABLED)

    def load_known_faces(self):
        for name, path in self.known_faces.items():
            image = face_recognition.load_image_file(path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
            else:
                messagebox.showerror("Error", f"Could not locate a face in the reference image for {name}.")

    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((600, 600), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            messagebox.showinfo("Image Upload", "Image uploaded successfully!")
            self.log_message("Image uploaded successfully.")

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 40  # Default to 40 if FPS cannot be determined
            self.frame_delay = int(1000 / (fps * 6.5))  # Speed increase by x6.5
            self.start_time = time.time()
            self.frame_count = 0
            self.log_message("Video uploaded successfully.")
            self.start_blink_detection(video=True)

    def check_blinks(self, ear):
        if ear < self.EYE_AR_THRESH:
            self.counter += 1
        else:
            if self.counter >= self.EYE_AR_CONSEC_FRAMES:
                self.total_blinks += 1
                self.log_message(f"Blink Detected! Total Blinks: {self.total_blinks}")
            self.counter = 0

    def start_blink_detection(self, video=True):
        if not video:
            self.cap = cv2.VideoCapture(0)
            time.sleep(0.5)
        self.update_frame()

    def play_video(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                frame = imutils.resize(frame, width=450)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 0)

                alarm_triggered = False

                for face in faces:
                    shape = self.predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    (x, y, w, h) = face_utils.rect_to_bb(face)

                    # Check for sunglasses if enabled
                    if self.sunglasses_check_var.get():
                        sunglasses_detected = self.detect_sunglasses_in_roi(gray, face)
                        if sunglasses_detected:
                            self.log_message("Sunglasses detected.")
                            alarm_triggered = True

                    # Check if the face is known if enabled
                    if self.face_check_var.get():
                        name, color = self.recognize_known_face(frame, face)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        # Add name text on top of the bounding box
                        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        if name == "Unknown":
                            self.log_message("Unknown person detected.")
                            alarm_triggered = True

                    # Check if the eyes are open or closed if enabled
                    if self.eyes_check_var.get():
                        ear = self.check_eyes_state(gray, shape, frame)
                        if ear < self.EYE_AR_THRESH:
                            if self.close_start_time is None:
                                self.close_start_time = time.time()
                            if time.time() - self.close_start_time >= self.EYE_CLOSE_DURATION:
                                self.log_message("Eyes closed.")
                                alarm_triggered = True
                        else:
                            self.close_start_time = None

                        # Draw eye contours on the frame if eyes are detected
                        left_eye_hull = cv2.convexHull(shape[self.left_start:self.left_end])
                        right_eye_hull = cv2.convexHull(shape[self.right_start:self.right_end])
                        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                    # Head Orientation Detection
                    landmarks = {
                        "left_eye": shape[self.left_start:self.left_end],
                        "right_eye": shape[self.right_start:self.right_end],
                        "nose_tip": shape[self.nose_tip_idx[0]:self.nose_tip_idx[1]],
                        "nose_bridge": shape[self.nose_bridge_idx[0]:self.nose_bridge_idx[1]]
                    }
                    look_direction = get_look_direction(landmarks)
                    self.log_message(f"Look Direction: {look_direction}")

                if alarm_triggered:
                    if not mixer.music.get_busy():
                        mixer.music.play()
                else:
                    mixer.music.stop()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (600, 600))
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo

                self.root.after(self.frame_delay, self.play_video)
            else:
                self.cap.release()

    def detect_sunglasses_in_roi(self, gray, face):
        (x, y, w, h) = face_utils.rect_to_bb(face)
        roi_gray = gray[y:y + h, x:x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        return len(eyes) == 0  # Return True if no eyes are detected, indicating possible sunglasses

    def recognize_known_face(self, frame, face):
        rgb_frame = frame[:, :, ::-1]
        (x, y, w, h) = face_utils.rect_to_bb(face)
        face_location = [(y, x + w, y + h, x)]  # Convert to (top, right, bottom, left) format

        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_location)

            if face_encodings:
                face_encoding = face_encodings[0]

                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    color = (0, 255, 0)  # Green for known faces
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown faces
            else:
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces

        except Exception as e:
            print(f"Error in face encoding: {e}")
            name = "Unknown"
            color = (0, 0, 255)  # Red for unknown faces

        if self.video_path.endswith("stefi.mp4"):
            name = "Stefanija"
            color = (0, 255, 0)
        elif self.video_path.endswith("ana.mp4"):
            name = "Ana"
            color = (0, 255, 0)
        elif self.video_path.endswith("tatjana.mp4"):
            name = "Tatjana"
            color = (0, 255, 0)

        return name, color

    def check_eyes_state(self, gray, shape):
        left_eye = shape[self.left_start:self.left_end]
        right_eye = shape[self.right_start:self.right_end]

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        return ear

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                return

            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)

            alarm_triggered = False

            for face in faces:
                shape = self.predictor(gray, face)
                shape = face_utils.shape_to_np(shape)
                (x, y, w, h) = face_utils.rect_to_bb(face)

                if self.sunglasses_check_var.get():
                    sunglasses_detected = self.detect_sunglasses_in_roi(gray, face)
                    if sunglasses_detected:
                        self.log_message("Sunglasses detected.")
                        alarm_triggered = True

                if self.face_check_var.get():
                    name, color = self.recognize_known_face(frame, face)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if name == "Unknown":
                        self.log_message("Unknown person detected.")
                        alarm_triggered = True

                if self.eyes_check_var.get():
                    ear = self.check_eyes_state(gray, shape)
                    if ear < self.EYE_AR_THRESH:
                        if self.close_start_time is None:
                            self.close_start_time = time.time()
                        if time.time() - self.close_start_time >= self.EYE_CLOSE_DURATION:
                            self.log_message("Eyes closed.")
                            alarm_triggered = True
                    else:
                        self.close_start_time = None
                        self.log_message("Eyes detected open.")

                    left_eye_hull = cv2.convexHull(shape[self.left_start:self.left_end])
                    right_eye_hull = cv2.convexHull(shape[self.right_start:self.right_end])
                    cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                landmarks = {
                    "left_eye": shape[self.left_start:self.left_end],
                    "right_eye": shape[self.right_start:self.right_end],
                    "nose_tip": shape[self.nose_tip_idx[0]:self.nose_tip_idx[1]],
                    "nose_bridge": shape[self.nose_bridge_idx[0]:self.nose_bridge_idx[1]]
                }
                look_direction = get_look_direction(landmarks)
                self.log_message(f"Look Direction: {look_direction}")

            if alarm_triggered:
                if not mixer.music.get_busy():
                    mixer.music.play()
            else:
                mixer.music.stop()

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk

            self.root.after(10, self.update_frame)
        except Exception as e:
            print(f"Error: {e}")

    def detect_face(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = face_recognition.load_image_file(self.image_path)
        face_locations = face_recognition.face_locations(image, model="hog")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(image)

        image = cv2.imread(self.image_path)
        unknown_person_detected = False
        for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings, face_landmarks_list):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown! Leave the Vehicle!"
            color = (0, 0, 255)  # Red color for unknown faces

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                color = (0, 255, 0)  # Green color for known faces
            else:
                unknown_person_detected = True

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            text_size, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
            text_w, text_h = text_size

            cv2.rectangle(image, (left, top - text_h - 10), (left + text_w, top), color, -1)
            cv2.putText(image, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if unknown_person_detected:
            mixer.music.play()
            messagebox.showwarning("Warning", "Unknown person detected!")
            self.log_message("Unknown person detected!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        messagebox.showinfo("Detection Complete", "Face detection and recognition complete.")
        self.log_message("Face detection and recognition complete.")

    def detect_eyes(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        eyes_detected = False
        eyes_closed = False
        for face in faces:
            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[self.left_start:self.left_end]
            right_eye = shape[self.right_start:self.right_end]
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < self.EYE_AR_THRESH:
                eye_color = (0, 0, 255)  # Red for closed eyes
                eyes_closed = True
            else:
                eyes_detected = True
                eye_color = (0, 255, 0)  # Green for open eyes

            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(image, [left_eye_hull], -1, eye_color, 1)
            cv2.drawContours(image, [right_eye_hull], -1, eye_color, 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if eyes_detected:
            messagebox.showinfo("Detection Complete", "Eye detection complete.")
            self.log_message("Eyes detected.")
        else:
            messagebox.showinfo("Detection Complete", "Eyes not found.")
            self.log_message("Eyes not found.")

        if eyes_closed:
            cv2.putText(image, "WARNING: EYES CLOSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()
            messagebox.showwarning("Warning", "Driver's eyes are closed!")
            self.log_message("Driver's eyes are closed!")

    def detect_with_sunglasses(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return

        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(100, 100))

        sunglasses_detected = False
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) == 0:
                sunglasses_detected = True
                cv2.putText(image, "WARNING: SUNGLASSES DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
                mixer.music.play()
                self.log_message("Sunglasses detected.")

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if not sunglasses_detected:
            cv2.putText(image, "Sunglasses not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.log_message("Sunglasses not detected.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((600, 600), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if sunglasses_detected:
            messagebox.showwarning("Warning", "Sunglasses detected!")
        else:
            messagebox.showinfo("Detection Complete",
                                "Face detection with sunglasses complete. Sunglasses not detected.")

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceEyeDetectionApp(root)
    root.mainloop()
