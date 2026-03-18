import math
from typing import List

import numpy as np

from safe_rl.data.risk import compute_min_distance, compute_min_ttc, get_ego_vehicle
from safe_rl.data.types import SceneState


BASE_FEATURE_DIM = 24
ACTION_DIM = 9


def _safe_last(history_scene: List[SceneState]) -> SceneState:
    if not history_scene:
        raise ValueError("history_scene must not be empty")
    return history_scene[-1]


def scene_statistics(scene: SceneState) -> np.ndarray:
    ego = get_ego_vehicle(scene)
    min_dist = compute_min_distance(scene)
    min_ttc = compute_min_ttc(scene)
    num_vehicles = len(scene.vehicles)
    lane_ids = [v.lane_id for v in scene.vehicles]
    mean_lane = float(sum(lane_ids) / max(1, len(lane_ids)))
    mean_speed = float(sum(math.sqrt(v.vx * v.vx + v.vy * v.vy) for v in scene.vehicles) / max(1, len(scene.vehicles)))
    return np.array(
        [
            ego.x,
            ego.y,
            ego.vx,
            ego.vy,
            ego.heading,
            float(ego.lane_id),
            min_dist,
            min_ttc,
            float(num_vehicles),
            mean_lane,
            mean_speed,
            float(len(scene.traffic_lights)),
        ],
        dtype=np.float32,
    )


def encode_history(history_scene: List[SceneState]) -> np.ndarray:
    scene = _safe_last(history_scene)
    current = scene_statistics(scene)

    if len(history_scene) >= 2:
        previous = scene_statistics(history_scene[-2])
        delta = current - previous
    else:
        delta = np.zeros_like(current)

    feature = np.concatenate([current, delta], axis=0)
    if feature.shape[0] != BASE_FEATURE_DIM:
        raise RuntimeError(f"Unexpected feature dimension: {feature.shape[0]}")
    return feature


def action_one_hot(action_id: int) -> np.ndarray:
    one_hot = np.zeros((ACTION_DIM,), dtype=np.float32)
    if 0 <= action_id < ACTION_DIM:
        one_hot[action_id] = 1.0
    return one_hot


def history_action_feature(history_scene: List[SceneState], action_id: int) -> np.ndarray:
    return np.concatenate([encode_history(history_scene), action_one_hot(action_id)], axis=0)
