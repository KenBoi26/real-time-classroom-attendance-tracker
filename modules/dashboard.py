import os
import time

DASHBOARD_INTERVAL = 1


def _compute_running_presence(entry, session_start, now):
    elapsed = now - session_start
    if elapsed <= 0:
        return 0.0

    logs     = entry["logs"]
    total_in = 0.0
    in_time  = None

    for timestamp, event in logs:
        if event == "IN":
            in_time = timestamp
        elif event == "OUT" and in_time is not None:
            total_in += timestamp - in_time
            in_time = None

    if in_time is not None:
        total_in += now - in_time

    return (total_in / elapsed) * 100


def print_dashboard(tracker, session_start, now, section_name="", teacher_name=""):
    os.system("cls" if os.name == "nt" else "clear")

    elapsed   = now - session_start
    hours     = int(elapsed // 3600)
    minutes   = int((elapsed % 3600) // 60)
    seconds   = int(elapsed % 60)

    print("=" * 60)
    print("       ATTENDANCE TRACKER — LIVE DASHBOARD")
    if section_name or teacher_name:
        print(f"       Section: {section_name}    Teacher: {teacher_name}")
    print("=" * 60)
    print(f"  Session elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("-" * 60)
    print(f"  {'Name':<20} {'Status':<10} {'Presence %':<12}")
    print("-" * 60)

    if not tracker:
        print("  (no one recognised yet)")
    else:
        for person_id in sorted(tracker, key=lambda k: tracker[k]["name"]):
            entry   = tracker[person_id]
            pct     = _compute_running_presence(entry, session_start, now)
            status  = entry["status"]
            print(f"  {entry['name']:<20} {status:<10} {pct:>6.1f}%")

    print("-" * 60)
    print("  Press Q on the video window to stop the session.")
    print("=" * 60)
