from safe_rl.config.config import PPOConfig, SimConfig
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.rl.env import SafeDrivingEnv
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend


class _RecordingBackend(ISumoBackend):
    def __init__(self):
        self.contexts = []
        self.closed = False
        self.current_log_path = ""
        self.scene = SceneState(
            timestamp=0.0,
            ego_id="ego",
            vehicles=[VehicleState("ego", 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1)],
        )

    @property
    def runtime_log_path(self) -> str:
        return self.current_log_path

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        self.contexts.append((episode_id, risky_mode))
        self.current_log_path = f"safe_rl_output/test_artifacts/{episode_id}.log"

    def get_runtime_diagnostics(self):
        return {
            "runtime_log_path": self.current_log_path,
            "last_reset_status": {"restart_count": 0, "restarted": False, "load_attempted": False, "load_failed": False},
        }

    def start(self):
        return None

    def reset(self, seed=None):
        return self.scene

    def step(self, action_id: int) -> BackendStepResult:
        _ = action_id
        return BackendStepResult(
            scene=self.scene,
            task_reward=1.0,
            done=True,
            info={"ego_speed": 5.0, "sumo_log_path": self.current_log_path},
        )

    def inject_risk_event(self, event_type=None):
        _ = event_type

    def get_state(self):
        return self.scene

    def close(self):
        self.closed = True



def test_env_reset_sets_episode_context_and_records_episode_prefix():
    backend = _RecordingBackend()
    env = SafeDrivingEnv(
        backend=backend,
        sim_config=SimConfig(history_steps=2, episode_steps=5),
        ppo_config=PPOConfig(total_timesteps=10),
        shield=None,
        episode_prefix="stage3_train",
    )

    _, reset_info = env.reset(options={"risky_mode": True})
    _, _, terminated, truncated, step_info = env.step(4)

    assert backend.contexts == [("stage3_train_ep_000001", True)]
    assert reset_info["episode_id"] == "stage3_train_ep_000001"
    assert step_info["episode_id"] == "stage3_train_ep_000001"
    assert terminated is True
    assert truncated is False

    records = env.get_session_records()
    assert len(records) == 1
    assert records[0]["episode_id"] == "stage3_train_ep_000001"
    assert records[0]["sumo_log_path"].endswith("stage3_train_ep_000001.log")

    env.close()
    assert backend.closed is True
