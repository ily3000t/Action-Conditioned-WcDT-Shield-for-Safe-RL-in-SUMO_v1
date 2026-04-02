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
                "\n".join(
                    [
                        "Warning: Vehicle 'ego' performs emergency stop at the end of lane ':merge_0_0' because there is no connection to the next edge.",
                        "Warning: Vehicle 'ego'; junction collision with vehicle 'merge_seed', lane=':merge_0_0', gap=-1.00, latGap=0.00, time=4.20, stage=move.",
                    ]
                ) + "\n",
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
    assert warning_payload["risky"]["junction_collision"]["count"] == 1
    assert warning_payload["overall"]["totals"]["traci_command_errors"]["count"] == 1
    assert warning_payload["overall"]["totals"]["sumo_runtime_warnings"]["count"] == 2
    assert warning_payload["overall"]["totals"]["collisions"]["count"] == 1
    assert warning_payload["overall"]["totals"]["route_lane_structural_warnings"]["count"] == 2
    assert warning_payload["acceptance"]["checks"]["illegal_lane_index_normal_zero"] is False
    assert warning_payload["acceptance"]["passed"] is False


class _ProbeBackend(ISumoBackend):
    def __init__(self, name: str):
        self._name = name
        self._runtime_log_path = ARTIFACT_ROOT / f"{name}.log"
        self._episode_id = ""
        self._risky_mode = False
        self._step_index = 0
        self._last_action = 4

    @property
    def runtime_log_path(self) -> str:
        return str(self._runtime_log_path)

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        self._episode_id = episode_id
        self._risky_mode = risky_mode
        self._runtime_log_path = ARTIFACT_ROOT / f"{self._name}_{episode_id}.log"
        self._runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_log_path.write_text("", encoding="utf-8")

    def start(self):
        return None

    def reset(self, seed=None):
        self._step_index = 0
        self._last_action = 4
        return self.get_state()

    def _scene(self, action_id: int) -> SceneState:
        ego_x = float(self._step_index * 2)
        if action_id == 8:
            gap = 0.5
        elif action_id == 7:
            gap = 2.0
        else:
            gap = 6.0 + float(action_id)
        lead_x = ego_x + gap
        return SceneState(
            timestamp=float(self._step_index) * 0.1,
            ego_id="ego",
            vehicles=[
                VehicleState("ego", ego_x, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1),
                VehicleState("lead", lead_x, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1),
            ],
        )

    def step(self, action_id: int) -> BackendStepResult:
        self._last_action = int(action_id)
        self._step_index += 1
        collision = int(action_id) == 8
        lane_violation = int(action_id) in (6, 7)
        return BackendStepResult(
            scene=self._scene(int(action_id)),
            task_reward=1.0 - 0.1 * float(abs(int(action_id) - 4)),
            done=self._step_index >= 5,
            info={"collision": collision, "lane_violation": lane_violation, "teleport": False},
        )

    def inject_risk_event(self, event_type=None):
        return None

    def get_state(self) -> SceneState:
        return self._scene(self._last_action)

    def close(self):
        return None


def test_collector_stage1_probe_generates_same_state_pairs():
    config = SafeRLConfig()
    config.sim.normal_episodes = 0
    config.sim.risky_episodes = 1
    config.sim.episode_steps = 6
    config.sim.history_steps = 2
    config.sim.risk_event_prob = 0.0
    config.stage1_collection.probe_warmup_steps = 2
    config.stage1_collection.initial_risk_event_step = 2
    config.stage1_collection.min_gap_between_risk_events = 2
    config.dataset.raw_log_dir = "safe_rl_output/test_artifacts/raw_probe"
    config.dataset.dataset_dir = "safe_rl_output/test_artifacts/datasets_probe"

    collector = SumoDataCollector(
        backend=_ProbeBackend("main"),
        config=config,
        probe_backend=_ProbeBackend("probe"),
    )
    episodes = collector.collect()

    assert len(episodes) == 1
    assert collector.probe_summary["episodes_probed"] == 1
    assert collector.probe_summary["selected_by_event_window"] > 0
    assert collector.probe_summary["pairs_created"] > 0
    assert collector.probe_pairs
    assert all(sample.source == "stage1_probe_same_state" for sample in collector.probe_pairs)
    assert any(bool(sample.meta.get("trusted_for_spread", False)) for sample in collector.probe_pairs)
    assert collector.probe_events
    assert any(event.get("status") == "ok" for event in collector.probe_events)
    assert collector.bucket_summary()["episodes_by_bucket"]["clean_risky"] == 1
    assert collector.bucket_summary()["episodes_too_short_for_probe"] == 0
    assert episodes[0].meta["collection_bucket"] == "clean_risky"
