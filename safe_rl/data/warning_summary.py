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
TOTAL_DIMENSIONS = (
    "traci_command_errors",
    "sumo_runtime_warnings",
    "collisions",
    "route_lane_structural_warnings",
)
STRUCTURAL_BUCKETS = (
    "illegal_lane_index",
    "no_connection_next_edge",
    "emergency_stop_no_connection",
)
EMERGENCY_BRAKE_THRESHOLD = 20.0
RISKY_STRUCTURAL_DOMINANCE_THRESHOLD = 0.5
_DECEL_PATTERN = re.compile(r"decel=([-+]?\d+(?:\.\d+)?)")


def empty_warning_counts() -> Dict[str, int]:
    return {bucket: 0 for bucket in WARNING_BUCKETS}


def empty_totals() -> Dict[str, int]:
    return {dimension: 0 for dimension in TOTAL_DIMENSIONS}


def _line_dimensions(lower: str, bucket: str) -> Dict[str, int]:
    dimensions = empty_totals()
    if lower.startswith("error:"):
        dimensions["traci_command_errors"] += 1
    if lower.startswith("warning:"):
        dimensions["sumo_runtime_warnings"] += 1
    if bucket in STRUCTURAL_BUCKETS:
        dimensions["route_lane_structural_warnings"] += 1
    if bucket in ("junction_collision", "lanechange_collision") or " collision " in lower or "collision with" in lower:
        dimensions["collisions"] += 1
    return dimensions


def classify_log_line(line: str) -> Dict[str, object]:
    text = (line or "").strip()
    lower = text.lower()
    if not lower.startswith(("warning:", "error:")):
        return {"bucket": "", "dimensions": empty_totals()}

    bucket = "other"
    if "no lane with index" in lower:
        bucket = "illegal_lane_index"
    elif "there is no connection to the next edge" in lower and "performs emergency stop" in lower:
        bucket = "emergency_stop_no_connection"
    elif "there is no connection to the next edge" in lower:
        bucket = "no_connection_next_edge"
    elif "junction collision" in lower:
        bucket = "junction_collision"
    elif "stage=lanechange" in lower:
        bucket = "lanechange_collision"
    elif "performs emergency braking" in lower:
        match = _DECEL_PATTERN.search(lower)
        decel = float(match.group(1)) if match else 0.0
        if abs(decel) >= EMERGENCY_BRAKE_THRESHOLD:
            bucket = "emergency_braking_high"

    return {
        "bucket": bucket,
        "dimensions": _line_dimensions(lower, bucket),
    }


def parse_warning_counts(log_path: str) -> Dict[str, Dict[str, int]]:
    counts = empty_warning_counts()
    totals = empty_totals()
    path = Path(log_path)
    if not log_path or not path.is_file():
        return {"buckets": counts, "totals": totals}

    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        classified = classify_log_line(line)
        bucket = str(classified["bucket"] or "")
        if bucket:
            counts[bucket] += 1
        for key, value in classified["dimensions"].items():
            totals[key] += int(value)
    return {"buckets": counts, "totals": totals}


def summarize_episode_warnings(episode_id: str, risky_mode: bool, log_path: str) -> Dict:
    parsed = parse_warning_counts(log_path)
    return {
        "episode_id": str(episode_id),
        "risky_mode": bool(risky_mode),
        "sumo_log_path": str(log_path or ""),
        "buckets": parsed["buckets"],
        "totals": parsed["totals"],
    }


def _aggregate_metric(subset: List[Dict], key: str, from_field: str) -> Dict[str, float]:
    episode_count = len(subset)
    count = sum(int(record.get(from_field, {}).get(key, 0)) for record in subset)
    episodes_with_value = sum(1 for record in subset if int(record.get(from_field, {}).get(key, 0)) > 0)
    return {
        "count": count,
        "episodes_with_warning": episodes_with_value,
        "avg_per_episode": float(count / episode_count) if episode_count else 0.0,
    }


def _build_scope(subset: List[Dict]) -> Dict:
    payload = {"episode_count": len(subset)}
    payload["totals"] = {
        dimension: _aggregate_metric(subset, dimension, "totals")
        for dimension in TOTAL_DIMENSIONS
    }
    for bucket in WARNING_BUCKETS:
        payload[bucket] = _aggregate_metric(subset, bucket, "buckets")
    return payload


def _build_acceptance(summary: Dict) -> Dict:
    normal = summary["normal"]
    risky = summary["risky"]
    normal_structural = sum(int(normal[bucket]["count"]) for bucket in STRUCTURAL_BUCKETS)
    risky_structural = sum(int(risky[bucket]["count"]) for bucket in STRUCTURAL_BUCKETS)
    risky_warning_total = int(risky["totals"]["sumo_runtime_warnings"]["count"])
    risky_structural_share = float(risky_structural / risky_warning_total) if risky_warning_total > 0 else 0.0

    checks = {
        "illegal_lane_index_normal_zero": int(normal["illegal_lane_index"]["count"]) == 0,
        "illegal_lane_index_risky_zero": int(risky["illegal_lane_index"]["count"]) == 0,
        "route_lane_structural_normal_zero": normal_structural == 0,
        "route_lane_structural_risky_non_dominant": risky_structural_share <= RISKY_STRUCTURAL_DOMINANCE_THRESHOLD,
    }
    return {
        "checks": checks,
        "metrics": {
            "normal_route_lane_structural_total": normal_structural,
            "risky_route_lane_structural_total": risky_structural,
            "risky_route_lane_structural_share": risky_structural_share,
            "risky_structural_dominance_threshold": RISKY_STRUCTURAL_DOMINANCE_THRESHOLD,
        },
        "passed": all(checks.values()),
    }


def aggregate_warning_records(records: List[Dict]) -> Dict:
    normal_records = [record for record in records if not bool(record.get("risky_mode", False))]
    risky_records = [record for record in records if bool(record.get("risky_mode", False))]
    summary = {
        "overall": _build_scope(records),
        "normal": _build_scope(normal_records),
        "risky": _build_scope(risky_records),
        "episodes": list(records),
    }
    summary["acceptance"] = _build_acceptance(summary)
    return summary
