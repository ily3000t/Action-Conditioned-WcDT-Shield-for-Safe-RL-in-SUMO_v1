from typing import Any, Dict, List, Optional


class BackendSessionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        backend_type: str,
        episode_id: str = "",
        seed: Optional[int] = None,
        sumo_log_path: str = "",
        runtime_args: Optional[List[str]] = None,
        exception_type: str = "",
        exception_text: str = "",
    ):
        self.backend_type = str(backend_type)
        self.episode_id = str(episode_id)
        self.seed = seed
        self.sumo_log_path = str(sumo_log_path)
        self.runtime_args = list(runtime_args or [])
        self.exception_type = str(exception_type)
        self.exception_text = str(exception_text)
        self.message = str(message)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return (
            f"{self.message} | backend={self.backend_type} | episode_id={self.episode_id} | seed={self.seed} | "
            f"sumo_log_path={self.sumo_log_path} | runtime_args={self.runtime_args} | "
            f"exception_type={self.exception_type} | exception_text={self.exception_text}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "backend_type": self.backend_type,
            "episode_id": self.episode_id,
            "seed": self.seed,
            "sumo_log_path": self.sumo_log_path,
            "runtime_args": list(self.runtime_args),
            "exception_type": self.exception_type,
            "exception_text": self.exception_text,
        }


class BackendStartError(BackendSessionError):
    pass


class BackendResetError(BackendSessionError):
    pass
