import math
from typing import Iterable, Tuple

from safe_rl.data.types import RiskLabels, SceneState, VehicleState


def _placeholder_ego(scene: SceneState) -> VehicleState:
    ego_id = scene.ego_id if scene.ego_id else "ego"
    return VehicleState(
        vehicle_id=ego_id,
        x=0.0,
        y=0.0,
        vx=0.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        heading=0.0,
        lane_id=0,
    )


def get_ego_vehicle(scene: SceneState) -> VehicleState:
    for vehicle in scene.vehicles:
        if vehicle.vehicle_id == scene.ego_id:
            return vehicle
    if scene.vehicles:
        return scene.vehicles[0]
    return _placeholder_ego(scene)


def compute_min_distance(scene: SceneState) -> float:
    ego = get_ego_vehicle(scene)
    min_distance = math.inf
    for vehicle in scene.vehicles:
        if vehicle.vehicle_id == ego.vehicle_id:
            continue
        dx = vehicle.x - ego.x
        dy = vehicle.y - ego.y
        distance = math.sqrt(dx * dx + dy * dy)
        min_distance = min(min_distance, distance)
    return float(min_distance if min_distance != math.inf else 1e6)


def compute_min_ttc(scene: SceneState) -> float:
    ego = get_ego_vehicle(scene)
    min_ttc = math.inf
    for vehicle in scene.vehicles:
        if vehicle.vehicle_id == ego.vehicle_id:
            continue
        dx = vehicle.x - ego.x
        if dx <= 0:
            continue
        rel_speed = ego.vx - vehicle.vx
        if rel_speed <= 1e-3:
            continue
        ttc = dx / rel_speed
        min_ttc = min(min_ttc, ttc)
    return float(min_ttc if min_ttc != math.inf else 1e6)


def detect_collision(scene: SceneState, collision_distance: float = 2.5) -> bool:
    return compute_min_distance(scene) <= collision_distance


def aggregate_future_risk(
    future_scenes: Iterable[SceneState],
    ttc_threshold: float,
    lane_violation: bool = False,
) -> RiskLabels:
    collision = False
    min_ttc = math.inf
    min_distance = math.inf

    for scene in future_scenes:
        scene_min_ttc = compute_min_ttc(scene)
        scene_min_distance = compute_min_distance(scene)
        min_ttc = min(min_ttc, scene_min_ttc)
        min_distance = min(min_distance, scene_min_distance)
        if detect_collision(scene):
            collision = True

    if min_ttc == math.inf:
        min_ttc = 1e6
    if min_distance == math.inf:
        min_distance = 1e6

    ttc_risk = min_ttc < ttc_threshold
    overall = max(
        1.0 if collision else 0.0,
        0.7 if ttc_risk else 0.0,
        0.5 if lane_violation else 0.0,
    )
    return RiskLabels(
        collision=collision,
        ttc_risk=ttc_risk,
        lane_violation=lane_violation,
        overall_risk=overall,
        min_ttc=float(min_ttc),
        min_distance=float(min_distance),
    )


def risk_targets(labels: RiskLabels) -> Tuple[float, float, float, float]:
    return (
        float(labels.collision),
        float(labels.ttc_risk),
        float(labels.lane_violation),
        float(labels.overall_risk),
    )
