import cv2
import os
import json

# ── Paths ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXIT_ZONE_PATH  = os.path.join(BASE_DIR, "config", "exit_zone.json")


def setup_exit_zone():
    """Open camera, let admin draw a rectangle over the door area, save to JSON."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[EXIT ZONE] Could not read from camera.")
        return None

    print("[EXIT ZONE] Draw a rectangle over the exit/door area, then press ENTER or SPACE.")
    roi = cv2.selectROI("Draw Exit Zone", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Draw Exit Zone")

    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    if w == 0 or h == 0:
        print("[EXIT ZONE] No region selected. Cancelled.")
        return None

    zone = {"x": x, "y": y, "w": w, "h": h}

    os.makedirs(os.path.dirname(EXIT_ZONE_PATH), exist_ok=True)
    with open(EXIT_ZONE_PATH, "w") as f:
        json.dump(zone, f, indent=2)

    print(f"[EXIT ZONE] Saved → {EXIT_ZONE_PATH}")
    print(f"[EXIT ZONE] Coordinates: x={x}, y={y}, w={w}, h={h}")
    return zone


def load_exit_zone():
    """Load exit zone rectangle from JSON. Raises FileNotFoundError if missing."""
    if not os.path.exists(EXIT_ZONE_PATH):
        raise FileNotFoundError(
            f"[EXIT ZONE] No exit zone config found at {EXIT_ZONE_PATH}. "
            "Run exit zone setup first."
        )

    with open(EXIT_ZONE_PATH) as f:
        zone = json.load(f)

    print(f"[EXIT ZONE] Loaded zone: x={zone['x']}, y={zone['y']}, "
          f"w={zone['w']}, h={zone['h']}")
    return zone


def is_inside_exit_zone(cx, cy, zone):
    """Return True if point (cx, cy) falls inside the exit zone rectangle."""
    return (zone["x"] <= cx <= zone["x"] + zone["w"] and
            zone["y"] <= cy <= zone["y"] + zone["h"])


if __name__ == "__main__":
    setup_exit_zone()
