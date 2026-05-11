import csv
import os
from datetime import datetime

# ── Paths ────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORTS_DIR, exist_ok=True)


def save_attendance_report(results, session_date, section_name=""):
    """
    Step 13: Write attendance_<section>_<date>.csv with all students.
    Columns: Name, Total IN Duration (s), Presence %, Verdict, Event Log
    """
    prefix = f"attendance_{section_name}_" if section_name else "attendance_"
    filename = os.path.join(REPORTS_DIR, f"{prefix}{session_date}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Total IN Duration (s)", "Presence %", "Verdict", "Event Log"])

        for r in results:
            # Format each (timestamp, event) as "HH:MM:SS IN" / "HH:MM:SS OUT"
            log_entries = []
            for ts, event in r.get("logs", []):
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                log_entries.append(f"{time_str} {event}")
            event_log = " | ".join(log_entries)

            writer.writerow([r["name"], r["total_in"], r["presence_pct"], r["verdict"], event_log])

    print(f"[REPORT] Attendance report saved → {filename}")
    return filename


def save_teacher_report(results, session_date, section_name=""):
    """
    Step 13: Write teacher_<section>_<date>.csv with teacher records only.
    Columns: Name, Time Arrived, Total Duration Present (s)
    """
    teachers = [r for r in results if r["role"] == "teacher"]

    if not teachers:
        print("[REPORT] No teacher records found — skipping teacher report.")
        return None

    prefix = f"teacher_{section_name}_" if section_name else "teacher_"
    filename = os.path.join(REPORTS_DIR, f"{prefix}{session_date}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time Arrived", "Total Duration Present (s)"])

        for r in teachers:
            if r["first_in"] is not None:
                arrived = datetime.fromtimestamp(r["first_in"]).strftime("%H:%M:%S")
            else:
                arrived = "N/A"
            writer.writerow([r["name"], arrived, r["total_in"]])

    print(f"[REPORT] Teacher report saved → {filename}")
    return filename
