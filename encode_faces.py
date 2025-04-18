import face_recognition
import os
import cv2
import pickle

# Paths
BASE_DIR = 'dataset'
ENCODING_FILE = 'face_encodings.pkl'

known_encodings = []
known_names = []

print("🔍 Encoding faces...")

# Go through each augmented folder
for person_folder in os.listdir(BASE_DIR):
    if not person_folder.endswith('_augmented'):
        continue
    
    person_name = person_folder.replace('_augmented', '').upper()
    folder_path = os.path.join(BASE_DIR, person_folder)
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            continue

        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

# Save to pickle
with open(ENCODING_FILE, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

print(f"✅ Encoded {len(known_encodings)} faces and saved to '{ENCODING_FILE}'")
