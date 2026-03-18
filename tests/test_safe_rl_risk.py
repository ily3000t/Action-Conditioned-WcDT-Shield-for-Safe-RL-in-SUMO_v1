from safe_rl.data.risk import compute_min_distance, compute_min_ttc, get_ego_vehicle
from safe_rl.data.types import SceneState


def test_empty_scene_uses_placeholder_ego():
    scene = SceneState(timestamp=0.0, ego_id="ego", vehicles=[])
    ego = get_ego_vehicle(scene)
    assert ego.vehicle_id == "ego"
    assert compute_min_distance(scene) == 1e6
    assert compute_min_ttc(scene) == 1e6
