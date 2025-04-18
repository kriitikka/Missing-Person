import face_recognition
import cv2
import os
import pickle

dataset_path = "dataset"  # or the actual folder where your data folders (arkdata, etc.) are
known_encodings = []
known_names = []

print("🔍 Encoding images from:", dataset_path)

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"🧠 Processing {person_name}...")

    for img_name in os.listdir(person_folder):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(person_folder, img_name)
            image = cv2.imread(path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(person_name.upper())

# Save as encodings.pickle
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("✅ Done! Encodings saved to encodings.pickle")
