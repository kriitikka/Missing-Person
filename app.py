import cv2
import face_recognition
import pickle
import os
from datetime import datetime
import numpy as np
import serial
import time

# === CONFIGURATION ===
ENCODINGS_PATH = "encodings.pickle"
SNAPSHOT_DIR = "detections"
LABEL_SMOOTHING_FRAMES = 5
MIN_FACE_SIZE = 80
FRAME_SCALE = 0.5
FACE_RESIZE_DIM = (150, 150)
SERIAL_PORT = "COM8"
BAUD_RATE = 9600

# === Initialize Serial Communication ===
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE)
    time.sleep(2)
    print(f"[INFO] Arduino connected on {SERIAL_PORT}")
except:
    arduino = None
    print("[WARNING] Arduino not connected. Proceeding without it.")

last_color_sent = None

# === Load Face Encodings ===
def load_encodings():
    with open(ENCODINGS_PATH, "rb") as file:
        data = pickle.load(file)
    return data["encodings"], data["names"]

# === Save Detected Snapshot ===
def save_snapshot(frame, name):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"{name}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    print(f"[INFO] Snapshot saved: {path}")

# === Crop & Resize Detected Face ===
def crop_and_resize_face(frame, location):
    top, right, bottom, left = location
    face = frame[max(0, top):min(frame.shape[0], bottom),
                 max(0, left):min(frame.shape[1], right)]
    if face.size == 0:
        return None
    return cv2.resize(face, FACE_RESIZE_DIM)

# === MAIN FUNCTION ===
def main():
    known_encodings, known_names = load_encodings()
    print("[INFO] Face encodings loaded.")

    video = cv2.VideoCapture(0)
    label_buffer = {}

    global last_color_sent

    while True:
        ret, frame = video.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        found_known_face = False

        for face_location in face_locations:
            top_s, right_s, bottom_s, left_s = face_location
            top, right, bottom, left = [int(v / FRAME_SCALE) for v in [top_s, right_s, bottom_s, left_s]]

            if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
                continue

            face_crop = crop_and_resize_face(frame, (top, right, bottom, left))
            if face_crop is None:
                continue

            rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_face_crop)
            if not face_encoding:
                continue
            face_encoding = face_encoding[0]

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances) if face_distances.size else None

            name = "Unknown"

            if best_match_index is not None and matches[best_match_index]:
                name = known_names[best_match_index]
                found_known_face = True
                save_snapshot(frame, name)

            box_key = (top, right, bottom, left)
            label_buffer.setdefault(box_key, []).append(name)
            if len(label_buffer[box_key]) > LABEL_SMOOTHING_FRAMES:
                label_buffer[box_key].pop(0)
            stable_name = max(set(label_buffer[box_key]), key=label_buffer[box_key].count)

            color = (0, 255, 0) if stable_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, stable_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # === Arduino Control ===
        if arduino:
            if found_known_face:
                if last_color_sent != 'A':
                    arduino.write(b'A\n')
                    last_color_sent = 'A'
                    print("[DEBUG] Known face — Sent 'A'")
            else:
                if last_color_sent != 'O':
                    arduino.write(b'O\n')
                    last_color_sent = 'O'
                    print("[DEBUG] No or Unknown face — Sent 'O'")

        cv2.imshow("Real-time Missing Person Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
