from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from safe_rl.config.config import PPOConfig, SimConfig
from safe_rl.data.risk import compute_min_distance, compute_min_ttc, detect_collision
from safe_rl.data.types import SceneState, ShieldDecision
from safe_rl.models.features import BASE_FEATURE_DIM, encode_history
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim import ISumoBackend

try:
    import gymnasium as gym
    from gymnasium import spaces

    BaseEnv = gym.Env
except Exception:
    gym = None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    spaces = _Spaces()
    BaseEnv = object


class SafeDrivingEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: ISumoBackend,
        sim_config: SimConfig,
        ppo_config: PPOConfig,
        shield: Optional[SafetyShield] = None,
    ):
        self.backend = backend
        self.sim_config = sim_config
        self.ppo_config = ppo_config
        self.shield = shield
        self.history: deque = deque(maxlen=sim_config.history_steps)
        self.step_count = 0
        self.episode_interventions = 0
        self.episode_collisions = 0
        self.last_transition: Optional[Dict[str, Any]] = None

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(BASE_FEATURE_DIM,),
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        risky_mode = bool(options.get("risky_mode", False))
        scene = self.backend.reset(seed=seed)
        self.history.clear()
        for _ in range(self.sim_config.history_steps):
            self.history.append(scene)
        self.step_count = 0
        self.episode_interventions = 0
        self.episode_collisions = 0
        self.last_transition = None

        if risky_mode:
            self.backend.inject_risk_event()

        observation = self._current_observation()
        info = {"risky_mode": risky_mode}
        return observation, info

    def step(self, action: int):
        raw_action = int(action)
        history_before = list(self.history)

        if self.shield is not None:
            decision = self.shield.select_action(history_before, raw_action)
        else:
            decision = ShieldDecision(
                raw_action=raw_action,
                final_action=raw_action,
                intervened=False,
                reason="shield_disabled",
                risk_raw=0.0,
                risk_final=0.0,
                candidate_risks={raw_action: 0.0},
            )

        result = self.backend.step(decision.final_action)
        self.history.append(result.scene)
        self.step_count += 1

        collision = bool(result.info.get("collision", detect_collision(result.scene)))
        if decision.intervened:
            self.episode_interventions += 1
        if collision:
            self.episode_collisions += 1

        reward = float(result.task_reward - self.ppo_config.intervene_penalty * float(decision.intervened))

        info = {
            "risk_raw": decision.risk_raw,
            "risk_final": decision.risk_final,
            "intervened": decision.intervened,
            "intervention_reason": decision.reason,
            "candidate_risks": decision.candidate_risks,
            "task_reward": float(result.task_reward),
            "reward": reward,
            "collision": collision,
            "ttc": compute_min_ttc(result.scene),
            "min_distance": compute_min_distance(result.scene),
            "lane_violation": bool(result.info.get("lane_violation", False)),
            "shield_meta": decision.meta,
            "ego_speed": float(result.info.get("ego_speed", 0.0)),
        }

        self.last_transition = {
            "history_scene": history_before,
            "raw_action": raw_action,
            "final_action": decision.final_action,
            "decision": decision,
            "info": info,
        }

        observation = self._current_observation()
        terminated = bool(result.done)
        truncated = bool(self.step_count >= self.sim_config.episode_steps)

        return observation, reward, terminated, truncated, info

    def _current_observation(self) -> np.ndarray:
        return encode_history(list(self.history)).astype(np.float32)

    def get_history(self) -> List[SceneState]:
        return list(self.history)

    def close(self):
        self.backend.close()


def create_env(backend: ISumoBackend, sim_config: SimConfig, ppo_config: PPOConfig, shield: Optional[SafetyShield]):
    return SafeDrivingEnv(backend=backend, sim_config=sim_config, ppo_config=ppo_config, shield=shield)
