import json
from pathlib import Path

from safe_rl.config.config import SafeRLConfig
from safe_rl.data.collector import SumoDataCollector
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend


def _scene() -> SceneState:
    return SceneState(
        timestamp=0.0,
        ego_id="ego",
        vehicles=[
            VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1),
            VehicleState("lead", 25.0, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1),
        ],
    )


class _FlakyBackend(ISumoBackend):
    def __init__(self):
        self.reset_calls = 0
        self.close_calls = 0

    @property
    def runtime_log_path(self) -> str:
        return "safe_rl_output/test_artifacts/traci_runtime.log"

    def start(self):
        return None

    def reset(self, seed=None):
        self.reset_calls += 1
        if self.reset_calls == 2:
            raise RuntimeError("synthetic backend crash")
        return _scene()

    def step(self, action_id: int) -> BackendStepResult:
        return BackendStepResult(scene=_scene(), task_reward=0.5, done=True, info={"collision": False})

    def inject_risk_event(self, event_type=None):
        return None

    def get_state(self) -> SceneState:
        return _scene()

    def close(self):
        self.close_calls += 1


def test_collector_skips_failed_episode_and_saves_report():
    config = SafeRLConfig()
    config.sim.normal_episodes = 2
    config.sim.risky_episodes = 0
    config.sim.episode_steps = 1
    config.dataset.raw_log_dir = "safe_rl_output/test_artifacts/raw"
    config.dataset.dataset_dir = "safe_rl_output/test_artifacts/datasets"

    collector = SumoDataCollector(backend=_FlakyBackend(), config=config)
    episodes = collector.collect()

    report_path = Path("safe_rl_output/test_artifacts/collector_failures_test.json")
    collector.save_failure_report(str(report_path))
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(episodes) == 1
    assert payload["successful_episodes"] == 1
    assert payload["failed_episodes"] == 1
    assert payload["failures"][0]["episode_id"] == "ep_00001"
    assert payload["failures"][0]["reason"] == "episode_exception"
