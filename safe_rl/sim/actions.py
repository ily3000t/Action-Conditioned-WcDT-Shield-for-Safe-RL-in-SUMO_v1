from dataclasses import dataclass
from typing import Dict, List


LONGITUDINAL_VALUES = (-1, 0, 1)  # DECEL, KEEP, ACCEL
LATERAL_VALUES = (-1, 0, 1)  # LEFT, KEEP, RIGHT


@dataclass(frozen=True)
class DiscreteAction:
    action_id: int
    longitudinal: int
    lateral: int


def _build_actions() -> List[DiscreteAction]:
    actions: List[DiscreteAction] = []
    action_id = 0
    for lon in LONGITUDINAL_VALUES:
        for lat in LATERAL_VALUES:
            actions.append(DiscreteAction(action_id=action_id, longitudinal=lon, lateral=lat))
            action_id += 1
    return actions


_ACTIONS = _build_actions()
_ACTION_BY_ID: Dict[int, DiscreteAction] = {a.action_id: a for a in _ACTIONS}
_ACTION_TO_ID: Dict[tuple, int] = {(a.longitudinal, a.lateral): a.action_id for a in _ACTIONS}


def all_action_ids() -> List[int]:
    return list(_ACTION_BY_ID.keys())


def decode_action(action_id: int) -> DiscreteAction:
    if action_id not in _ACTION_BY_ID:
        raise ValueError(f"Unknown action id: {action_id}")
    return _ACTION_BY_ID[action_id]


def encode_action(longitudinal: int, lateral: int) -> int:
    key = (longitudinal, lateral)
    if key not in _ACTION_TO_ID:
        raise ValueError(f"Unknown action tuple: {key}")
    return _ACTION_TO_ID[key]


def action_name(action_id: int) -> str:
    action = decode_action(action_id)
    lon_name = {-1: "DECEL", 0: "KEEP", 1: "ACCEL"}[action.longitudinal]
    lat_name = {-1: "LEFT", 0: "KEEP", 1: "RIGHT"}[action.lateral]
    return f"{lon_name}_{lat_name}"


def action_distance(a: int, b: int) -> int:
    aa = decode_action(a)
    bb = decode_action(b)
    return abs(aa.longitudinal - bb.longitudinal) + abs(aa.lateral - bb.lateral)


def fallback_action_id() -> int:
    # DECEL + KEEP as required by phase-1 fallback strategy.
    return encode_action(-1, 0)


def neighboring_actions(action_id: int) -> List[int]:
    center = decode_action(action_id)
    candidates: List[int] = [action_id]
    for d_lon, d_lat in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 0), (1, -1), (1, 1)):
        lon = max(-1, min(1, center.longitudinal + d_lon))
        lat = max(-1, min(1, center.lateral + d_lat))
        candidate = encode_action(lon, lat)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates
