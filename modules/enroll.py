import cv2
import os
import json

import os
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH         = os.path.join(BASE_DIR, "config", "haarcascade_frontalface_default.xml")
PROFILE_CASCADE_PATH = os.path.join(BASE_DIR, "config", "haarcascade_profileface.xml")
FACES_DIR    = os.path.join(BASE_DIR, "data", "faces")
LABEL_MAP    = os.path.join(BASE_DIR, "data", "models", "label_map.json")
ENROLL_SECS   = 10
FPS           = 30
FACE_RESIZE   = (200, 200)

os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "models"), exist_ok=True)

def preprocess(face_crop):
    gray  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_RESIZE)

    # I've applied CLAHE to normalize light
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)

def enroll_person(person_id, name, role="student"):
   
    detector_frontal = cv2.CascadeClassifier(CASCADE_PATH)
    detector_profile = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    save_dir = os.path.join(FACES_DIR, str(person_id))
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP) as f:
            existing = json.load(f)
        
        if str(person_id) in existing:
            existing_name = existing[str(person_id)]["name"]
            print(f"\n[WARNING] ID {person_id} is already enrolled as '{existing_name}'")
            choice = input("Do you want to re-enroll and overwrite their images? (yes / no): ").strip().lower()
            
            if choice != "yes":
                print("[ENROLL] Cancelled. Existing record kept.")
                cap.release()
                return

            # Clear old frames before re-enrolling
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
            print(f"[ENROLL] Old frames cleared. Starting fresh for ID {person_id}.")

    total_frames = ENROLL_SECS * FPS
    saved = 0
    captured = 0

    print(f"\n[ENROLL] Starting in 3 seconds — hold each angle for 2 seconds:")
    print(f"         FRONT (2s) → LEFT (2s) → RIGHT (2s) → UP (2s) → DOWN (2s)")
    print(f"         Exaggerate your side turns so the profile cascade can capture them.")
    cv2.waitKey(3000)

    while captured < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        captured += 1

        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try frontal cascade first, fall back to profile cascade
        faces = detector_frontal.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            faces = detector_profile.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

        if len(faces) == 0:
            cv2.putText(frame, "No face detected — keep moving slowly",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            x, y, w, h = faces[0]

            
            pad = 40
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])
            

            crop     = preprocess(frame[y1:y2, x1:x2])
            filename = os.path.join(save_dir, f"{person_id}_{saved:04d}.jpg")
            cv2.imwrite(filename, crop)
            saved += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved: {saved}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        progress = int((captured / total_frames) * 640)
        cv2.rectangle(frame, (0, 460), (progress, 480), (255, 165, 0), -1)
        cv2.putText(frame, f"Recording... {captured}/{total_frames}",
                    (20, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[ENROLL] Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[ENROLL] Done. {saved} frames saved for {name} (ID: {person_id})")

    
    if os.path.exists(LABEL_MAP):
        with open(LABEL_MAP) as f:
            label_data = json.load(f)
    else:
        label_data = {}

    label_data[str(person_id)] = {"name": name, "role": role}

    with open(LABEL_MAP, "w") as f:
        json.dump(label_data, f, indent=2)

    print(f"[ENROLL] Label map updated → {LABEL_MAP}")


if __name__ == "__main__":
    print("=== Face Enrollment ===")
    role      = input("Role (student / teacher): ").strip().lower()
    person_id = int(input("Enter registration number: "))
    name      = input("Enter full name: ").strip()

    enroll_person(person_id, name, role)