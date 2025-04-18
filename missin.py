import face_recognition
import cv2
import os
from datetime import datetime
import yagmail
import geocoder

# === CONFIGURATION ===
EMAIL_SENDER = "arkjakki@gmail.com"        # Replace with your email
EMAIL_PASSWORD = "kraq mohd igao kiyd"    # Replace with app password
EMAIL_RECEIVER = "arkjakki@gmail.com"      # Receiver email

# === STEP 1: Load known faces ===
print("[*] Loading known faces from dataset...")
known_encodings = []
known_names = []
dataset_path = 'dataset'

for filename in os.listdir(dataset_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[!] Could not read: {filename}")
            continue

        encodings = face_recognition.face_encodings(img)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].split('(')[0].strip().upper()
            known_names.append(name)
        else:
            print(f"[!] No face found in: {filename}")

print(f"[+] Loaded {len(known_encodings)} known face(s).")

# === STEP 2: Setup webcam ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("[*] Webcam started. Press 'q' to quit.")

# === To track who has already been notified ===
notified_names = set()

# === STEP 3: Real-time loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("[!] Failed to grab frame.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match = face_distances.argmin() if len(face_distances) > 0 else None

        if best_match is not None and matches[best_match]:
            name = known_names[best_match]

            if name not in notified_names:
                # Get location and time
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                g = geocoder.ip('me')
                location = g.city + ", " + g.country if g.ok else "Unknown"

                print(f"\n[✔️] Match Found: {name}")
                print(f"Time: {now}")
                print(f"Location: {location}")

                try:
                    yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
                    yag.send(
                        to=EMAIL_RECEIVER,
                        subject=f"Missing Person Found: {name}",
                        contents=[
                            f"Name: {name}",
                            f"Time: {now}",
                            f"Location: {location}",
                            "This person was detected via the live webcam feed."
                        ]
                    )
                    print("[📧] Email sent.")
                    notified_names.add(name)  # Avoid multiple notifications
                except Exception as e:
                    print(f"[!] Failed to send email: {e}")

        # Draw box on face
        top, right, bottom, left = [v * 4 for v in face_location]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Missing Person Detector", frame)

    # Quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()