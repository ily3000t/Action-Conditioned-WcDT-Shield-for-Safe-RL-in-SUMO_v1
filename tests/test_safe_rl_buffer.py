from pathlib import Path

from safe_rl.buffer.intervention_buffer import InterventionBuffer
from safe_rl.data.types import InterventionRecord, SceneState, VehicleState


def _dummy_scene(step: int) -> SceneState:
    ego = VehicleState(
        vehicle_id="ego",
        x=float(step),
        y=4.0,
        vx=20.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        heading=0.0,
        lane_id=1,
    )
    other = VehicleState(
        vehicle_id="lead",
        x=float(step) + 10.0,
        y=4.0,
        vx=18.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        heading=0.0,
        lane_id=1,
    )
    return SceneState(timestamp=float(step), ego_id="ego", vehicles=[ego, other], traffic_lights=[])


def test_intervention_buffer_push_sample_save_load():
    buffer = InterventionBuffer(capacity=10)
    record = InterventionRecord(
        history_scene=[_dummy_scene(0), _dummy_scene(1)],
        raw_action=4,
        final_action=1,
        raw_risk=0.9,
        final_risk=0.2,
        reason="risk_threshold_exceeded",
    )
    buffer.push(record)

    sampled = buffer.sample(1)
    assert len(sampled) == 1
    assert sampled[0].raw_action == 4

    save_dir = Path("safe_rl_output") / "test_tmp"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "buffer.pkl"
    buffer.save(str(save_path))

    loaded = InterventionBuffer(capacity=10)
    loaded.load(str(save_path))
    assert len(loaded) == 1
    assert loaded.all_records()[0].final_action == 1



def test_intervention_buffer_stats_include_replacement_diagnostics():
    buffer = InterventionBuffer(capacity=10)
    buffer.push(
        InterventionRecord(
            history_scene=[_dummy_scene(0)],
            raw_action=4,
            final_action=1,
            raw_risk=0.8,
            final_risk=0.3,
            reason="risk_threshold_exceeded",
        )
    )
    buffer.push(
        InterventionRecord(
            history_scene=[_dummy_scene(1)],
            raw_action=4,
            final_action=4,
            raw_risk=0.7,
            final_risk=0.4,
            reason="all_candidates_high_risk_or_uncertain",
        )
    )

    stats = buffer.stats()
    assert stats["size"] == 2.0
    assert stats["replacement_count"] == 1.0
    assert stats["replacement_same_as_raw_count"] == 1.0
    assert stats["fallback_action_count"] == 1.0
