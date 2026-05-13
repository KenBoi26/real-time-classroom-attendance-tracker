import cv2
import os
import json
import numpy as np
from modules.camera import get_camera

# ── Paths ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "config", "haarcascade_frontalface_default.xml")
MODEL_PATH   = os.path.join(BASE_DIR, "data", "models", "lbph_model.yml")
LABEL_MAP    = os.path.join(BASE_DIR, "data", "models", "label_map.json")

# ── Settings ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 70   # lower = stricter; tune this in Step 14
FACE_RESIZE          = (200, 200)
PAD                  = 40

def load_recognizer():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("[DETECT] Model not found. Run train.py first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABEL_MAP) as f:
        label_data = json.load(f)

    # flip to {int_id: name} for quick lookup during recognition
    id_to_name = {int(k): v["name"] for k, v in label_data.items()}

    print("[DETECT] Model and label map loaded.")
    return recognizer, id_to_name

def preprocess(face_crop):
    gray    = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FACE_RESIZE)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(resized)

def run_detection():
    recognizer, id_to_name = load_recognizer()
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cap      = get_camera(0)

    print("[DETECT] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # ── Padded crop ──────────────────────────────
            x1 = max(x - PAD, 0)
            y1 = max(y - PAD, 0)
            x2 = min(x + w + PAD, frame.shape[1])
            y2 = min(y + h + PAD, frame.shape[0])

            crop       = preprocess(frame[y1:y2, x1:x2])
            label, confidence = recognizer.predict(crop)

            # ── Decide known vs unknown ──────────────────
            if confidence < CONFIDENCE_THRESHOLD:
                name       = id_to_name.get(label, "Unknown")
                box_color  = (0, 255, 0)   # green — recognised
                label_text = f"{name}  ({confidence:.1f})"
            else:
                name       = "Unknown"
                box_color  = (0, 0, 255)   # red — not recognised
                label_text = f"Unknown  ({confidence:.1f})"

            # ── Draw box and label ───────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()