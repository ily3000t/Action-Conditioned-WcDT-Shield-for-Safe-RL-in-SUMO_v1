import logging
from pathlib import Path
from typing import Optional

from safe_rl.config.config import SimConfig
from safe_rl.data.types import SceneState
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend
from safe_rl.sim.mock_core import MockTrafficCore
from safe_rl.sim.real_control import RealSumoController
from safe_rl.sim.sumo_utils import (
    maybe_build_network_from_plain,
    prepare_sumo_python_path,
    resolve_sumo_binary,
)


_LOGGER = logging.getLogger(__name__)


class TraciBackend(ISumoBackend):
    def __init__(self, config: SimConfig):
        self.config = config
        self._traci = None
        self._controller: Optional[RealSumoController] = None
        self._sumo_binary = None
        self._started = False
        self._session_active = False
        self._connection_healthy = False
        self._cfg_path: Optional[Path] = None
        self._runtime_log_path: Optional[Path] = None
        self._current_episode_id: Optional[str] = None
        self._current_risky_mode: bool = False
        self._last_risk_meta = None
        self._last_scene: Optional[SceneState] = None
        self._mock = MockTrafficCore(
            episode_steps=config.episode_steps,
            step_length=config.step_length,
            seed=config.random_seed,
        )
        self._use_mock = True

    @property
    def runtime_log_path(self) -> str:
        if self._runtime_log_path is None:
            return ""
        return str(self._runtime_log_path)

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        self._current_episode_id = str(episode_id)
        self._current_risky_mode = bool(risky_mode)
        if not self._use_mock:
            self._runtime_log_path = self._resolve_runtime_log_path()

    def start(self):
        if self.config.force_mock:
            _LOGGER.warning("force_mock=true, using mock backend.")
            self._started = True
            self._use_mock = True
            return

        cfg_path = Path(self.config.sumo_cfg)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        self._cfg_path = cfg_path
        self._runtime_log_path = self._resolve_runtime_log_path()

        prepare_sumo_python_path(self.config)

        try:
            import traci  # type: ignore

            self._traci = traci
            self._controller = RealSumoController(self._traci, self.config, _LOGGER)
            self._sumo_binary = resolve_sumo_binary(self.config, use_gui=self.config.use_gui)

            if cfg_path.is_file():
                ok, message = maybe_build_network_from_plain(cfg_path, self.config)
                if not ok:
                    _LOGGER.warning("SUMO cfg check failed (%s), fallback to mock backend.", message)
                else:
                    _LOGGER.info(message)
                    self._start_real_session(seed=self.config.random_seed)
                    self._use_mock = False
                    _LOGGER.info(
                        "TraCI backend started in real SUMO mode with cfg=%s, log=%s",
                        cfg_path,
                        self.runtime_log_path,
                    )
            else:
                _LOGGER.warning("SUMO cfg not found (%s), fallback to mock backend.", cfg_path)
        except Exception as exc:
            _LOGGER.warning("TraCI unavailable (%s), fallback to mock backend.", exc)
        self._started = True

    def _resolve_runtime_log_path(self) -> Path:
        log_dir = Path(self.config.runtime_log_dir)
        if not log_dir.is_absolute():
            log_dir = (Path.cwd() / log_dir).resolve()
        if self._current_episode_id:
            log_dir = log_dir / "episodes"
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir / f"{self._current_episode_id}.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "traci_runtime.log"

    def _runtime_args(self, seed: Optional[int]):
        cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
        args = [
            "-c",
            str(cfg_path),
            "--seed",
            str(int(seed if seed is not None else self.config.random_seed)),
            "--log",
            str(self._runtime_log_path),
            "--collision.action",
            str(self.config.collision_action),
            "--collision.stoptime",
            str(self.config.collision_stoptime),
            "--collision.check-junctions",
            "true" if self.config.collision_check_junctions else "false",
        ]
        return args

    def _start_real_session(self, seed: Optional[int]):
        if self._runtime_log_path is None:
            self._runtime_log_path = self._resolve_runtime_log_path()
        self._traci.start([self._sumo_binary] + self._runtime_args(seed))
        self._session_active = True
        self._connection_healthy = True

    def _restart_real_session(self, seed: Optional[int]):
        try:
            self._traci.close()
        except Exception:
            pass
        self._start_real_session(seed)

    def _load_with_seed(self, seed: Optional[int]):
        self._traci.load(self._runtime_args(seed))

    def _warmup_after_reset(self):
        max_steps = max(10, int(5.0 / max(self.config.step_length, 1e-3)))
        try:
            if self._controller is not None and self._controller.warmup_until_ego(max_steps=max_steps):
                return
        except Exception as exc:
            if self._is_fatal_traci_error(exc):
                self._mark_connection_closed()
                _LOGGER.warning(
                    "TraCI reset warmup lost SUMO connection (%s). See log: %s",
                    exc,
                    self.runtime_log_path,
                )
                return
            raise
        _LOGGER.warning(
            "TraCI reset warmup finished without ego vehicle '%s' (steps=%d). Continue with placeholder scene.",
            self.config.ego_vehicle_id,
            max_steps,
        )

    def reset(self, seed: Optional[int] = None):
        if not self._started:
            self.start()
        if self._use_mock:
            scene = self._mock.reset(seed=seed)
            self._last_scene = scene
            return scene

        self._last_risk_meta = None
        self._runtime_log_path = self._resolve_runtime_log_path()
        seed_value = int(seed if seed is not None else self.config.random_seed)
        try:
            if not self._session_active or not self._connection_healthy:
                self._restart_real_session(seed_value)
            else:
                self._load_with_seed(seed_value)
        except Exception as exc:
            _LOGGER.warning(
                "TraCI load/restart failed (%s), retrying restart with log=%s.",
                exc,
                self.runtime_log_path,
            )
            self._restart_real_session(seed_value)

        self._warmup_after_reset()
        scene = self.get_state()
        self._last_scene = scene
        return scene

    def step(self, action_id: int) -> BackendStepResult:
        if self._use_mock:
            scene, task_reward, done, info = self._mock.step(action_id)
            self._last_scene = scene
            return BackendStepResult(scene=scene, task_reward=task_reward, done=done, info=info)

        action_meta = self._controller.apply_action(action_id)
        try:
            self._traci.simulationStep()
        except Exception as exc:
            if self._is_fatal_traci_error(exc):
                return self._handle_fatal_step(exc, action_meta)
            raise

        scene = self.get_state()
        self._last_scene = scene
        done = self._traci.simulation.getMinExpectedNumber() <= 0
        info = self._controller.summarize_step(scene, action_meta, self._last_risk_meta)
        info["sumo_log_path"] = self.runtime_log_path
        task_reward = float(info.get("ego_speed", 0.0) * self.config.step_length * 0.1)
        self._last_risk_meta = None
        return BackendStepResult(scene=scene, task_reward=task_reward, done=done, info=info)

    def inject_risk_event(self, event_type: Optional[str] = None):
        if self._use_mock:
            self._mock.inject_risk_event(event_type)
            return
        self._last_risk_meta = self._controller.inject_risk_event(event_type)

    def get_state(self):
        if self._use_mock:
            scene = self._mock.get_scene(timestamp=self._mock.step_index * self.config.step_length)
            self._last_scene = scene
            return scene
        if not self._connection_healthy or not self._session_active or self._controller is None:
            return self._fallback_scene()
        scene = self._controller.build_scene()
        self._last_scene = scene
        return scene

    def close(self):
        if not self._started:
            return
        if not self._use_mock and self._traci is not None and self._session_active:
            try:
                self._traci.close()
            except Exception:
                pass
        self._session_active = False
        self._connection_healthy = False
        self._started = False

    def _handle_fatal_step(self, exc: Exception, action_meta: dict) -> BackendStepResult:
        self._mark_connection_closed()
        scene = self._fallback_scene()
        info = {
            "collision": True,
            "ego_speed": self._fallback_ego_speed(scene),
            "lane_violation": bool(action_meta.get("lane_violation", False)),
            "risk_event": self._last_risk_meta.get("actual_event", "") if self._last_risk_meta else "",
            "risk_target_vehicle": self._last_risk_meta.get("target_vehicle_id", "") if self._last_risk_meta else "",
            "risk_requested_event": self._last_risk_meta.get("requested_event", "") if self._last_risk_meta else "",
            "risk_skipped_reason": self._last_risk_meta.get("skipped_reason", "") if self._last_risk_meta else "",
            "terminated_by_sumo": True,
            "termination_reason": "sumo_connection_closed",
            "sumo_exception": str(exc),
            "sumo_log_path": self.runtime_log_path,
        }
        skipped = str(action_meta.get("lane_change_skipped_reason", "") or "")
        if skipped:
            info["lane_change_skipped_reason"] = skipped
        self._last_risk_meta = None
        _LOGGER.warning(
            "SUMO closed TraCI connection during simulationStep (%s). Episode terminated early. Log: %s",
            exc,
            self.runtime_log_path,
        )
        return BackendStepResult(scene=scene, task_reward=-10.0, done=True, info=info)

    def _fallback_scene(self) -> SceneState:
        if self._last_scene is not None:
            return self._last_scene
        return SceneState(timestamp=0.0, ego_id=self.config.ego_vehicle_id, vehicles=[])

    def _fallback_ego_speed(self, scene: SceneState) -> float:
        for vehicle in scene.vehicles:
            if vehicle.vehicle_id == scene.ego_id:
                return float(vehicle.vx)
        return 0.0

    def _mark_connection_closed(self):
        self._session_active = False
        self._connection_healthy = False

    def _is_fatal_traci_error(self, exc: Exception) -> bool:
        fatal_cls = getattr(getattr(self._traci, "exceptions", None), "FatalTraCIError", None)
        return fatal_cls is not None and isinstance(exc, fatal_cls)


