from pathlib import Path

from safe_rl.config.config import SimConfig
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.sim.libsumo_backend import LibsumoBackend


class _FatalTraCIError(Exception):
    pass


class _BrokenLibsumo:
    FatalTraCIError = _FatalTraCIError

    class simulation:
        @staticmethod
        def getMinExpectedNumber():
            return 0

    def simulationStep(self):
        raise _FatalTraCIError("Connection closed by SUMO.")


class _DummyController:
    def apply_action(self, _action_id):
        return {"lane_violation": False}

    def build_scene(self):
        return SceneState(timestamp=0.0, ego_id="ego", vehicles=[])


def test_libsumo_backend_handles_fatal_simulation_step():
    config = SimConfig(force_mock=False, runtime_log_dir="safe_rl_output/test_artifacts")
    backend = LibsumoBackend(config)
    backend._use_mock = False
    backend._started = True
    backend._session_active = True
    backend._connection_healthy = True
    backend._libsumo = _BrokenLibsumo()
    backend._controller = _DummyController()
    backend._runtime_log_path = Path("safe_rl_output/test_artifacts/libsumo_runtime.log")
    backend._last_risk_meta = {
        "actual_event": "unsafe_merge",
        "target_vehicle_id": "merge_seed",
        "requested_event": "unsafe_merge",
    }
    backend._last_scene = SceneState(
        timestamp=1.2,
        ego_id="ego",
        vehicles=[VehicleState("ego", 10.0, 4.0, 21.0, 0.0, 0.0, 0.0, 0.0, 1)],
    )

    result = backend.step(4)

    assert result.done is True
    assert result.task_reward == -10.0
    assert result.scene == backend._last_scene
    assert result.info["collision"] is True
    assert result.info["terminated_by_sumo"] is True
    assert result.info["termination_reason"] == "sumo_connection_closed"
    assert result.info["risk_event"] == "unsafe_merge"
    assert result.info["sumo_log_path"].endswith("libsumo_runtime.log")
    assert backend._connection_healthy is False
    assert backend._session_active is False
