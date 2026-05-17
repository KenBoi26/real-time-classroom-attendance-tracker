PRESENCE_THRESHOLD = 90


def calculate_presence(tracker, session_start, session_end, class_duration=None):
    session_duration = class_duration if class_duration else (session_end - session_start)
    if session_duration <= 0:
        return []

    results = []

    for person_id, entry in tracker.items():
        logs = entry["logs"]
        total_in = 0.0
        in_time  = None

        for timestamp, event in logs:
            if event == "IN":
                in_time = timestamp
            elif event == "OUT" and in_time is not None:
                total_in += timestamp - in_time
                in_time = None

        if in_time is not None:
            total_in += session_end - in_time

        presence_pct = (total_in / session_duration) * 100
        verdict      = "Present" if presence_pct >= PRESENCE_THRESHOLD else "Absent"

        first_in = None
        for timestamp, event in logs:
            if event == "IN":
                first_in = timestamp
                break

        results.append({
            "person_id":    person_id,
            "name":         entry["name"],
            "role":         entry["role"],
            "total_in":     round(total_in, 1),
            "presence_pct": round(presence_pct, 1),
            "verdict":      verdict,
            "first_in":     first_in,
            "logs":         logs,
        })

    results.sort(key=lambda r: r["name"])
    return results
