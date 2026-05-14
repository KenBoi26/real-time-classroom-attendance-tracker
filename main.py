import cv2
import os
import sys
import json
import time
from datetime import datetime

# ── Paths ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH  = os.path.join(BASE_DIR, "config", "haarcascade_frontalface_default.xml")
LABEL_MAP     = os.path.join(BASE_DIR, "data", "models", "label_map.json")

# ── Settings ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 70
FACE_RESIZE          = (200, 200)
PAD                  = 40
MIN_FACE_SIZE        = (60, 60)  # increased to prevent detecting background noise
DASHBOARD_INTERVAL   = 1      # seconds between dashboard refreshes
EXIT_ZONE_COLOR      = (0, 255, 255)   # yellow rectangle on frame

# ── Imports from project modules ─────────────────────────
from modules.detector   import load_recognizer, preprocess
from modules.exit_zone  import setup_exit_zone, load_exit_zone, is_inside_exit_zone
from modules.tracker    import init_tracker, update_tracker, sweep_grace_periods
from modules.presence   import calculate_presence
from modules.dashboard  import print_dashboard
from modules.reporter   import save_attendance_report, save_teacher_report
from modules.camera     import get_camera


def load_label_roles():
    """Load label_map.json to get {int_id: role} mapping."""
    if not os.path.exists(LABEL_MAP):
        return {}
    with open(LABEL_MAP) as f:
        label_data = json.load(f)
    return {int(k): v.get("role", "student") for k, v in label_data.items()}


def start_session():
    """Step 12: Start a tracking session — the main loop."""
    # ── Load model, cascade, exit zone ───────────────────
    try:
        recognizer, id_to_name = load_recognizer()
    except FileNotFoundError as e:
        print(e)
        return

    try:
        zone = load_exit_zone()
    except FileNotFoundError as e:
        print(e)
        return

    # ── Load label map for role lookup ───────────────────
    id_to_role = load_label_roles()

    # ── Load full label map for teacher validation ───────
    if not os.path.exists(LABEL_MAP):
        print("[SESSION] No label map found. Enroll people first.")
        return
    with open(LABEL_MAP) as f:
        label_data = json.load(f)

    # ── Show enrolled teachers and let user pick by serial number ──
    teachers_list = [(tid, info) for tid, info in label_data.items()
                     if info.get("role", "student") == "teacher"]

    if not teachers_list:
        print("[SESSION] No teachers enrolled. Enroll a teacher first.")
        return

    print("\n  Enrolled teachers:")
    for i, (tid, info) in enumerate(teachers_list, start=1):
        print(f"    {i}. {info['name']}  (ID: {tid})")

    while True:
        try:
            choice = int(input(f"  Select teacher (1-{len(teachers_list)}): ").strip())
        except ValueError:
            print("  Invalid input. Please enter a number.")
            continue
        if choice < 1 or choice > len(teachers_list):
            print(f"  Please enter a number between 1 and {len(teachers_list)}.")
            continue
        break

    teacher_id, teacher_info = teachers_list[choice - 1]
    teacher_name = teacher_info["name"]
    print(f"  Welcome, {teacher_name}!")

    # ── Prompt for section name ──────────────────────────
    section_name = input("  Enter section name (e.g. CSE-6A): ").strip()

    # ── Prompt for class duration ────────────────────────
    while True:
        try:
            class_duration = int(input("  Enter class duration in seconds: ").strip())
            if class_duration <= 0:
                print("  Duration must be positive.")
                continue
            break
        except ValueError:
            print("  Invalid input. Please enter an integer.")

    detector   = cv2.CascadeClassifier(CASCADE_PATH)
    tracker    = init_tracker()

    cap = get_camera(0)

    session_start   = time.time()
    last_dashboard  = 0.0

    print(f"[SESSION] Session started — Section: {section_name}, Duration: {class_duration}s")
    print("[SESSION] Press Q on the video window to stop.")

    # ── Main loop ────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now  = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=MIN_FACE_SIZE
        )

        # ── Process each detected face ──────────────────
        for (x, y, w, h) in faces:
            # Padded crop (same as detector.py)
            x1 = max(x - PAD, 0)
            y1 = max(y - PAD, 0)
            x2 = min(x + w + PAD, frame.shape[1])
            y2 = min(y + h + PAD, frame.shape[0])

            crop             = preprocess(frame[y1:y2, x1:x2])
            label, confidence = recognizer.predict(crop)

            if confidence < CONFIDENCE_THRESHOLD:
                name      = id_to_name.get(label, "Unknown")
                role      = id_to_role.get(label, "student")
                box_color = (0, 255, 0)
                label_text = f"{name} ({confidence:.1f})"

                # Face center for position tracking
                cx = x + w // 2
                cy = y + h // 2

                update_tracker(tracker, label, name, role, cx, cy, zone, now)
            else:
                box_color  = (0, 0, 255)
                label_text = f"Unknown ({confidence:.1f})"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # ── Sweep grace periods every frame ─────────────
        sweep_grace_periods(tracker, now)

        # ── Draw exit zone rectangle ────────────────────
        zx, zy = zone["x"], zone["y"]
        zw, zh = zone["w"], zone["h"]
        cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), EXIT_ZONE_COLOR, 2)
        cv2.putText(frame, "EXIT ZONE", (zx, zy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, EXIT_ZONE_COLOR, 2)

        # ── Dashboard refresh ───────────────────────────
        if now - last_dashboard >= DASHBOARD_INTERVAL:
            print_dashboard(tracker, session_start, now,
                            section_name=section_name,
                            teacher_name=teacher_name)
            last_dashboard = now

        # ── Show frame ──────────────────────────────────
        cv2.imshow("Attendance Session", frame)

        # Auto-stop when class duration is reached
        if (now - session_start) >= class_duration:
            print("\n[SESSION] Class duration reached. Stopping automatically.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Session end ──────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    session_end  = time.time()
    session_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("\n[SESSION] Session ended.")
    print(f"[SESSION] Duration: {session_end - session_start:.1f} seconds")

    # ── Step 9: final presence calculation ───────────────
    results = calculate_presence(tracker, session_start, session_end,
                                 class_duration=class_duration)

    # ── Print final summary ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"   FINAL ATTENDANCE REPORT — {section_name}")
    print("=" * 60)
    print(f"  {'Name':<20} {'IN Time (s)':<14} {'Presence %':<14} {'Verdict'}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:<20} {r['total_in']:<14} {r['presence_pct']:<14} {r['verdict']}")
    print("=" * 60)

    # ── Step 13: save reports ────────────────────────────
    save_attendance_report(results, session_date, section_name=section_name)
    save_teacher_report(results, session_date, section_name=section_name)

    print("\n[SESSION] All done.")


def main_menu():
    """Simple terminal menu — entry point."""
    while True:
        print("\n" + "=" * 40)
        print("  Classroom Attendance System")
        print("=" * 40)
        print("  1. Enroll a person")
        print("  2. Train model")
        print("  3. Setup exit zone")
        print("  4. Start session")
        print("  Q. Quit")
        print("=" * 40)

        choice = input("  Select option: ").strip().lower()

        if choice == "1":
            from modules.enroll import enroll_person
            
            src_choice = input("  Enroll from (1) Webcam or (2) Video file in data/videos: ").strip()
            video_path = None
            if src_choice == "2":
                videos_dir = os.path.join(BASE_DIR, "data", "videos")
                if not os.path.exists(videos_dir):
                    os.makedirs(videos_dir)
                videos = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                if not videos:
                    print("  No videos found in data/videos/")
                    continue
                
                print("  Available videos:")
                for i, v in enumerate(videos, start=1):
                    print(f"    {i}. {v}")
                
                try:
                    v_choice = int(input(f"  Select video (1-{len(videos)}): ").strip())
                    video_filename = videos[v_choice - 1]
                    video_path = os.path.join(videos_dir, video_filename)
                except (ValueError, IndexError):
                    print("  Invalid selection.")
                    continue
                
                base_name = os.path.splitext(video_filename)[0]
                if " - " in base_name:
                    try:
                        person_id = int(base_name.split(" - ")[0].strip())
                        name = base_name.split(" - ")[1].strip()
                        print(f"  Auto-detected -> ID: {person_id}, Name: {name}")
                    except ValueError:
                        person_id = int(input("  Enter registration number: "))
                        name      = input("  Enter full name: ").strip()
                else:
                    person_id = int(input("  Enter registration number: "))
                    name      = input("  Enter full name: ").strip()
            else:
                person_id = int(input("  Enter registration number: "))
                name      = input("  Enter full name: ").strip()
                
            role = input("  Role (student / teacher): ").strip().lower()
                
            enroll_person(person_id, name, role, video_path)

        elif choice == "2":
            from modules.train import train_model
            train_model()

        elif choice == "3":
            setup_exit_zone()

        elif choice == "4":
            start_session()

        elif choice == "q":
            print("  Goodbye!")
            sys.exit(0)

        else:
            print("  Invalid option. Try again.")


if __name__ == "__main__":
    main_menu()
