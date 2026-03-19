from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from safe_rl.data.types import SceneState


@dataclass
class BackendStepResult:
    scene: SceneState
    task_reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ISumoBackend(ABC):
    @property
    def runtime_log_path(self) -> str:
        return ""

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        _ = (episode_id, risky_mode)

    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> SceneState:
        raise NotImplementedError

    @abstractmethod
    def step(self, action_id: int) -> BackendStepResult:
        raise NotImplementedError

    @abstractmethod
    def inject_risk_event(self, event_type: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> SceneState:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
