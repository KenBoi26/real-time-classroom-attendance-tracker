import cv2
import os
import json
import numpy as np

# ── Paths ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR   = os.path.join(BASE_DIR, "data", "faces")
MODELS_DIR  = os.path.join(BASE_DIR, "data", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_model.yml")
LABEL_MAP   = os.path.join(MODELS_DIR, "label_map.json")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_training_data():
    faces  = []
    labels = []

    if not os.path.exists(LABEL_MAP):
        print("[TRAIN] No label map found. Enroll people first.")
        return [], []

    with open(LABEL_MAP) as f:
        label_data = json.load(f)

    for person_id, info in label_data.items():
        person_dir = os.path.join(FACES_DIR, person_id)

        if not os.path.exists(person_dir):
            print(f"[TRAIN] Warning: no frames folder found for ID {person_id}, skipping.")
            continue

        frame_files = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]

        if len(frame_files) == 0:
            print(f"[TRAIN] Warning: 0 frames for {info['name']}, skipping.")
            continue

        for filename in frame_files:
            img_path = os.path.join(person_dir, filename)
            img      = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            faces.append(img)
            labels.append(int(person_id))

        print(f"[TRAIN] Loaded {len(frame_files)} frames for {info['name']} (ID: {person_id})")

    return faces, labels

def train_model():
    print("\n[TRAIN] Loading training data...")
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("[TRAIN] No training data found. Exiting.")
        return

    print(f"\n[TRAIN] Training on {len(faces)} total frames across {len(set(labels))} people...")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    print(f"[TRAIN] Model saved → {MODEL_PATH}")
    print(f"[TRAIN] Training complete.")

if __name__ == "__main__":
    train_model()

