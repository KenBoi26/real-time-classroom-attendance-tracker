import time

GRACE_PERIOD = 5    # seconds after exit zone sighting before marking OUT
PENDING_IN_DURATION = 5.0  # seconds of continuous detection to mark as IN
GAP_TOLERANCE = 2.0        # seconds of allowed missing frames without resetting continuous timer


def init_tracker():
    """Return an empty tracker dict. People are added when first recognised."""
    return {}


def update_tracker(tracker, person_id, name, role, cx, cy, zone, now):
    """
    Update tracker for a recognised person detected at (cx, cy) at time `now`.

    Handles:
      - Step 8: late arrivals (first-time add with IN log)
      - Step 6: position check against exit zone
      - Step 7: grace period entry / cancellation
    """
    # ── Step 8: first time seeing this person — late arrival ─────────
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
    
    # ── Reset continuous timer if gap in detection is too large ─────
    if now - entry["last_seen_time"] > GAP_TOLERANCE:
        entry["continuous_since"] = now
        
    entry["last_seen_time"] = now

    # ── If previously OUT or PENDING and now reappearing ────────────
    if entry["status"] in ["OUT", "PENDING"]:
        # Only mark IN if they've been continuously seen for PENDING_IN_DURATION
        if now - entry["continuous_since"] >= PENDING_IN_DURATION:
            entry["status"] = "IN"
            entry["logs"].append((entry["continuous_since"], "IN"))
            entry["exit_zone_last_seen"] = None
        return

    # ── Step 6: check position against exit zone ────────────────────
    from modules.exit_zone import is_inside_exit_zone

    if is_inside_exit_zone(cx, cy, zone):
        # Inside exit zone — record this timestamp
        entry["exit_zone_last_seen"] = now
    else:
        # Outside exit zone — cancel any pending exit timer
        entry["exit_zone_last_seen"] = None


def sweep_grace_periods(tracker, now):
    """
    Step 7: mark someone OUT only if they were last seen in the exit zone
    AND have not been detected anywhere in the frame for GRACE_PERIOD seconds
    since that exit zone sighting.
    If they were never in the exit zone, do nothing regardless of disappearance.
    """
    for person_id, entry in tracker.items():
        if entry["status"] != "IN":
            continue

        # Only consider people who have a pending exit zone sighting
        if entry["exit_zone_last_seen"] is None:
            continue

        # Time since they were last seen in the exit zone
        time_since_exit_seen = now - entry["exit_zone_last_seen"]

        # They were in the exit zone and haven't reappeared for GRACE_PERIOD
        if time_since_exit_seen >= GRACE_PERIOD:
            entry["status"] = "OUT"
            entry["logs"].append((now, "OUT"))
            entry["exit_zone_last_seen"] = None
