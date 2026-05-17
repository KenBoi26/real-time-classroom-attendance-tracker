import time

GRACE_PERIOD = 5
PENDING_IN_DURATION = 5.0
GAP_TOLERANCE = 2.0


def init_tracker():
    return {}


def update_tracker(tracker, person_id, name, role, cx, cy, zone, now):
    if person_id not in tracker:
        tracker[person_id] = {
            "name":                 name,
            "role":                 role,
            "status":               "PENDING",
            "logs":                 [],
            "exit_zone_last_seen":  None,
            "last_seen_time":       now,
            "continuous_since":     now,
        }
        return

    entry = tracker[person_id]
    
    if now - entry["last_seen_time"] > GAP_TOLERANCE:
        entry["continuous_since"] = now
        
    entry["last_seen_time"] = now

    if entry["status"] in ["OUT", "PENDING"]:
        if now - entry["continuous_since"] >= PENDING_IN_DURATION:
            entry["status"] = "IN"
            entry["logs"].append((entry["continuous_since"], "IN"))
            entry["exit_zone_last_seen"] = None
        return

    from modules.exit_zone import is_inside_exit_zone

    if is_inside_exit_zone(cx, cy, zone):
        entry["exit_zone_last_seen"] = now
    else:
        entry["exit_zone_last_seen"] = None


def sweep_grace_periods(tracker, now):
    for person_id, entry in tracker.items():
        if entry["status"] != "IN":
            continue

        if entry["exit_zone_last_seen"] is None:
            continue

        time_since_exit_seen = now - entry["exit_zone_last_seen"]

        if time_since_exit_seen >= GRACE_PERIOD:
            entry["status"] = "OUT"
            entry["logs"].append((now, "OUT"))
            entry["exit_zone_last_seen"] = None
