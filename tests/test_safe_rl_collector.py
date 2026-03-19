import json
from pathlib import Path

from safe_rl.config.config import SafeRLConfig
from safe_rl.data.collector import SumoDataCollector
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend


ARTIFACT_ROOT = Path("safe_rl_output/test_artifacts")


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
        self._runtime_log_path = ARTIFACT_ROOT / "collector_default.log"
        self._episode_id = ""
        self._risky_mode = False

    @property
    def runtime_log_path(self) -> str:
        return str(self._runtime_log_path)

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        self._episode_id = episode_id
        self._risky_mode = risky_mode
        self._runtime_log_path = ARTIFACT_ROOT / f"{episode_id}.log"
        self._runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        if risky_mode:
            self._runtime_log_path.write_text(
                "Warning: Vehicle 'ego' performs emergency stop at the end of lane ':merge_0_0' because there is no connection to the next edge.\n",
                encoding="utf-8",
            )
        else:
            self._runtime_log_path.write_text(
                "Error: Answered with error to command 0xc4: No lane with index '1' on road 'ramp_in'.\n",
                encoding="utf-8",
            )

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


def test_collector_skips_failed_episode_and_saves_reports():
    config = SafeRLConfig()
    config.sim.normal_episodes = 1
    config.sim.risky_episodes = 1
    config.sim.episode_steps = 1
    config.dataset.raw_log_dir = "safe_rl_output/test_artifacts/raw"
    config.dataset.dataset_dir = "safe_rl_output/test_artifacts/datasets"

    collector = SumoDataCollector(backend=_FlakyBackend(), config=config)
    episodes = collector.collect()

    failure_report_path = ARTIFACT_ROOT / "collector_failures_test.json"
    warning_report_path = ARTIFACT_ROOT / "collector_warning_summary_test.json"
    collector.save_failure_report(str(failure_report_path))
    collector.save_warning_summary(str(warning_report_path))

    failure_payload = json.loads(failure_report_path.read_text(encoding="utf-8"))
    warning_payload = json.loads(warning_report_path.read_text(encoding="utf-8"))

    assert len(episodes) == 1
    assert failure_payload["successful_episodes"] == 1
    assert failure_payload["failed_episodes"] == 1
    assert failure_payload["failures"][0]["episode_id"] == "ep_00001"
    assert failure_payload["failures"][0]["reason"] == "episode_exception"

    assert warning_payload["normal"]["illegal_lane_index"]["count"] == 1
    assert warning_payload["risky"]["emergency_stop_no_connection"]["count"] == 1
    assert warning_payload["overall"]["illegal_lane_index"]["episodes_with_warning"] == 1
    assert warning_payload["overall"]["emergency_stop_no_connection"]["episodes_with_warning"] == 1
