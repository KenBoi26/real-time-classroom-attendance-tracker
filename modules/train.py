import cv2
import os
import json
import random
import shutil
import numpy as np
from collections import defaultdict

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES_DIR   = os.path.join(BASE_DIR, "data", "faces")
MODELS_DIR  = os.path.join(BASE_DIR, "data", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "lbph_model.yml")
LABEL_MAP   = os.path.join(MODELS_DIR, "label_map.json")

os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42


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


def stratified_split(faces, labels, train_ratio, val_ratio, test_ratio, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    rng = random.Random(seed)

    label_to_indices = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_indices[lbl].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for lbl, indices in label_to_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        n_test  = max(1, n - n_train - n_val)

        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val
            if n_test < 0:
                n_val  = n - n_train
                n_test = 0

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:n_train + n_val + n_test])

    def gather(idxs):
        return ([faces[i] for i in idxs], [labels[i] for i in idxs])

    return gather(train_idx), gather(val_idx), gather(test_idx)


def evaluate(recognizer, faces, labels, id_to_name, set_name="Validation"):
    if len(faces) == 0:
        print(f"[{set_name.upper()}] No samples to evaluate.")
        return 0.0

    correct = 0
    total   = len(faces)

    person_correct = defaultdict(int)
    person_total   = defaultdict(int)

    for img, true_label in zip(faces, labels):
        pred_label, confidence = recognizer.predict(img)
        person_total[true_label] += 1
        if pred_label == true_label:
            correct += 1
            person_correct[true_label] += 1

    overall_acc = (correct / total) * 100

    print(f"\n{'=' * 60}")
    print(f"   {set_name.upper()} SET EVALUATION")
    print(f"{'=' * 60}")
    print(f"  {'Name':<25} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print(f"{'-' * 60}")

    for lbl in sorted(person_total.keys()):
        name = id_to_name.get(lbl, f"ID {lbl}")
        c    = person_correct[lbl]
        t    = person_total[lbl]
        acc  = (c / t) * 100 if t > 0 else 0.0
        print(f"  {name:<25} {c:<10} {t:<10} {acc:.1f}%")

    print(f"{'-' * 60}")
    print(f"  {'OVERALL':<25} {correct:<10} {total:<10} {overall_acc:.1f}%")
    print(f"{'=' * 60}")

    return overall_acc


def train_model():
    print("\n[TRAIN] Loading training data...")
    faces, labels = load_training_data()

    if len(faces) == 0:
        print("[TRAIN] No training data found. Exiting.")
        return

    with open(LABEL_MAP) as f:
        label_data = json.load(f)
    id_to_name = {int(k): v["name"] for k, v in label_data.items()}

    print(f"\n[TRAIN] Splitting data → Train {TRAIN_RATIO*100:.0f}% | "
          f"Val {VAL_RATIO*100:.0f}% | Test {TEST_RATIO*100:.0f}%  "
          f"(seed={RANDOM_SEED})")

    (train_faces, train_labels), \
    (val_faces,   val_labels),   \
    (test_faces,  test_labels) = stratified_split(
        faces, labels,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        seed=RANDOM_SEED
    )

    n_people = len(set(labels))
    print(f"[TRAIN]   Train : {len(train_faces)} samples")
    print(f"[TRAIN]   Val   : {len(val_faces)} samples")
    print(f"[TRAIN]   Test  : {len(test_faces)} samples")
    print(f"[TRAIN]   People: {n_people}")

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    if os.path.exists(MODEL_PATH):
        print(f"\n[TRAIN] Updating existing model with {len(train_faces)} new samples "
              f"across {len(set(train_labels))} people...")
        recognizer.read(MODEL_PATH)
        recognizer.update(train_faces, np.array(train_labels))
    else:
        print(f"\n[TRAIN] Training NEW model on {len(train_faces)} samples "
              f"across {len(set(train_labels))} people...")
        recognizer.train(train_faces, np.array(train_labels))

    val_acc = evaluate(recognizer, val_faces, val_labels, id_to_name,
                       set_name="Validation")

    test_acc = evaluate(recognizer, test_faces, test_labels, id_to_name,
                        set_name="Test")

    recognizer.save(MODEL_PATH)
    print(f"\n[TRAIN] Model saved → {MODEL_PATH}")

    print("\n[TRAIN] Cleaning up raw face images to save space...")
    for person_id in set(labels):
        person_dir = os.path.join(FACES_DIR, str(person_id))
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            print(f"  Deleted photos for ID {person_id}")

    print(f"\n{'=' * 60}")
    print(f"   TRAINING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Train samples  : {len(train_faces)}")
    print(f"  Val samples    : {len(val_faces)}")
    print(f"  Test samples   : {len(test_faces)}")
    print(f"  Val accuracy   : {val_acc:.1f}%")
    print(f"  Test accuracy  : {test_acc:.1f}%")
    print(f"{'=' * 60}")
    print(f"[TRAIN] Training complete.")


if __name__ == "__main__":
    train_model()
