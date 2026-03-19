import re
from pathlib import Path
from typing import Dict, List


WARNING_BUCKETS = (
    "illegal_lane_index",
    "no_connection_next_edge",
    "emergency_stop_no_connection",
    "junction_collision",
    "lanechange_collision",
    "emergency_braking_high",
    "other",
)
EMERGENCY_BRAKE_THRESHOLD = 20.0
_DECEL_PATTERN = re.compile(r"decel=([-+]?\d+(?:\.\d+)?)")


def empty_warning_counts() -> Dict[str, int]:
    return {bucket: 0 for bucket in WARNING_BUCKETS}


def classify_warning_line(line: str) -> str:
    text = (line or "").strip()
    lower = text.lower()
    if not lower.startswith(("warning:", "error:")):
        return ""
    if "no lane with index" in lower:
        return "illegal_lane_index"
    if "there is no connection to the next edge" in lower and "performs emergency stop" in lower:
        return "emergency_stop_no_connection"
    if "there is no connection to the next edge" in lower:
        return "no_connection_next_edge"
    if "junction collision" in lower:
        return "junction_collision"
    if "stage=lanechange" in lower:
        return "lanechange_collision"
    if "performs emergency braking" in lower:
        match = _DECEL_PATTERN.search(lower)
        decel = float(match.group(1)) if match else 0.0
        if abs(decel) >= EMERGENCY_BRAKE_THRESHOLD:
            return "emergency_braking_high"
    return "other"


def parse_warning_counts(log_path: str) -> Dict[str, int]:
    counts = empty_warning_counts()
    path = Path(log_path)
    if not log_path or not path.is_file():
        return counts

    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        bucket = classify_warning_line(line)
        if bucket:
            counts[bucket] += 1
    return counts


def summarize_episode_warnings(episode_id: str, risky_mode: bool, log_path: str) -> Dict:
    return {
        "episode_id": str(episode_id),
        "risky_mode": bool(risky_mode),
        "sumo_log_path": str(log_path or ""),
        "buckets": parse_warning_counts(log_path),
    }


def aggregate_warning_records(records: List[Dict]) -> Dict:
    def _build_scope(scope_name: str, subset: List[Dict]) -> Dict:
        episode_count = len(subset)
        payload = {"episode_count": episode_count}
        for bucket in WARNING_BUCKETS:
            count = sum(int(record.get("buckets", {}).get(bucket, 0)) for record in subset)
            episodes_with_warning = sum(1 for record in subset if int(record.get("buckets", {}).get(bucket, 0)) > 0)
            payload[bucket] = {
                "count": count,
                "episodes_with_warning": episodes_with_warning,
                "avg_per_episode": float(count / episode_count) if episode_count else 0.0,
            }
        return payload

    normal_records = [record for record in records if not bool(record.get("risky_mode", False))]
    risky_records = [record for record in records if bool(record.get("risky_mode", False))]
    return {
        "overall": _build_scope("overall", records),
        "normal": _build_scope("normal", normal_records),
        "risky": _build_scope("risky", risky_records),
        "episodes": list(records),
    }
