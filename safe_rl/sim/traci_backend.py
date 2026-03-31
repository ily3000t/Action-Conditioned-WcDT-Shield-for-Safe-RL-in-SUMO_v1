import datetime as dt
import itertools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from safe_rl.config.config import SimConfig
from safe_rl.data.types import SceneState
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend
from safe_rl.sim.exceptions import BackendResetError, BackendStartError
from safe_rl.sim.mock_core import MockTrafficCore
from safe_rl.sim.real_control import RealSumoController
from safe_rl.sim.sumo_utils import (
    maybe_build_network_from_plain,
    prepare_sumo_python_path,
    resolve_sumo_binary,
)


_LOGGER = logging.getLogger(__name__)
_CONNECTION_LABEL_COUNTER = itertools.count(1)


class TraciBackend(ISumoBackend):
    def __init__(self, config: SimConfig):
        self.config = config
        self._traci = None
        self._traci_conn = None
        self._connection_label = f"safe_rl_traci_{next(_CONNECTION_LABEL_COUNTER)}"
        self._controller: Optional[RealSumoController] = None
        self._sumo_binary: Optional[str] = None
        self._started = False
        self._session_active = False
        self._connection_healthy = False
        self._cfg_path: Optional[Path] = None
        self._runtime_log_path: Optional[Path] = None
        self._current_episode_id: Optional[str] = None
        self._current_risky_mode: bool = False
        self._episode_session_index: int = 0
        self._last_risk_meta = None
        self._last_scene: Optional[SceneState] = None
        self._last_seed: Optional[int] = None
        self._last_runtime_args: List[str] = []
        self._last_reset_status: Dict[str, Any] = {}
        self._session_events: List[Dict[str, Any]] = []
        self._session_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None
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
        self._episode_session_index = 1 if self._current_episode_id else 0
        if not self._use_mock:
            self._runtime_log_path = self._resolve_runtime_log_path()

    def set_session_event_sink(self, sink: Optional[Callable[[Dict[str, Any]], None]]):
        self._session_event_sink = sink

    def get_runtime_diagnostics(self) -> Dict[str, Any]:
        cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
        return {
            "backend": "traci",
            "use_mock": bool(self._use_mock),
            "sumo_binary": str(self._sumo_binary or ""),
            "sumo_cfg": str(cfg_path),
            "collision_action": str(self.config.collision_action),
            "collision_stoptime": float(self.config.collision_stoptime),
            "collision_check_junctions": bool(self.config.collision_check_junctions),
            "runtime_log_path": self.runtime_log_path,
            "episode_id": str(self._current_episode_id or ""),
            "risky_mode": bool(self._current_risky_mode),
            "session_active": bool(self._session_active),
            "connection_healthy": bool(self._connection_healthy),
            "session_index": int(self._episode_session_index),
            "last_seed": self._last_seed,
            "last_runtime_args": list(self._last_runtime_args),
            "last_reset_status": dict(self._last_reset_status),
            "sim_time": self._safe_sim_time(),
        }

    def get_session_events(self, clear: bool = False) -> List[Dict[str, Any]]:
        events = [dict(event) for event in self._session_events]
        if clear:
            self._session_events = []
        return events

    def start(self):
        if self.config.force_mock:
            _LOGGER.warning("force_mock=true, using mock backend.")
            self._started = True
            self._use_mock = True
            self._record_session_event("start_mock_backend")
            return

        cfg_path = Path(self.config.sumo_cfg)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        self._cfg_path = cfg_path
        self._runtime_log_path = self._resolve_runtime_log_path()

        prepare_sumo_python_path(self.config)

        try:
            import traci  # type: ignore
        except Exception as exc:
            _LOGGER.warning("TraCI unavailable (%s), fallback to mock backend.", exc)
            self._use_mock = True
            self._started = True
            self._record_session_event("start_mock_backend", exc=exc, reason="traci_import_failed")
            return

        self._traci = traci
        self._traci_conn = None
        self._controller = None
        self._sumo_binary = resolve_sumo_binary(self.config, use_gui=self.config.use_gui)

        if not cfg_path.is_file():
            _LOGGER.warning("SUMO cfg not found (%s), fallback to mock backend.", cfg_path)
            self._use_mock = True
            self._started = True
            self._record_session_event("start_mock_backend", reason="missing_cfg", cfg_path=str(cfg_path))
            return

        ok, message = maybe_build_network_from_plain(cfg_path, self.config)
        if not ok:
            _LOGGER.warning("SUMO cfg check failed (%s), fallback to mock backend.", message)
            self._use_mock = True
            self._started = True
            self._record_session_event("start_mock_backend", reason="cfg_check_failed", exception_text=str(message))
            return

        _LOGGER.info(message)
        try:
            self._start_real_session(seed=self.config.random_seed, reason="initial_start", bump_session_log=False)
        except Exception as exc:
            self._record_session_event("initial_start_failed", exc=exc)
            raise self._make_start_error("TraCI initial start failed", exc, self.config.random_seed) from exc

        self._use_mock = False
        self._started = True
        _LOGGER.info(
            "TraCI backend started in real SUMO mode with cfg=%s, log=%s",
            cfg_path,
            self.runtime_log_path,
        )

    def reset(self, seed: Optional[int] = None) -> SceneState:
        if not self._started:
            self.start()
        if self._use_mock:
            scene = self._mock.reset(seed=seed)
            self._last_scene = scene
            self._last_reset_status = {
                "seed": None if seed is None else int(seed),
                "load_attempted": False,
                "load_failed": False,
                "restarted": False,
                "restart_count": 0,
                "reset_error": "",
                "ego_found": True,
            }
            return scene

        self._last_risk_meta = None
        self._runtime_log_path = self._resolve_runtime_log_path()
        seed_value = int(seed if seed is not None else self.config.random_seed)
        self._last_reset_status = {
            "seed": seed_value,
            "episode_id": str(self._current_episode_id or ""),
            "risky_mode": bool(self._current_risky_mode),
            "load_attempted": False,
            "load_failed": False,
            "restarted": False,
            "restart_count": 0,
            "warmup_fatal": False,
            "reset_error": "",
            "ego_found": False,
        }
        restart_attempted = False

        try:
            if not self._session_active or not self._connection_healthy:
                restart_attempted = True
                self._last_reset_status["restarted"] = True
                self._last_reset_status["restart_count"] = 1
                self._restart_real_session(seed_value, cause="pre_reset_unhealthy", bump_session_log=False)
            else:
                self._last_reset_status["load_attempted"] = True
                try:
                    self._load_with_seed(seed_value)
                except Exception as exc:
                    self._last_reset_status["load_failed"] = True
                    self._mark_connection_closed()
                    self._record_session_event("reset_load_failed", exc=exc)
                    restart_attempted = True
                    self._last_reset_status["restarted"] = True
                    self._last_reset_status["restart_count"] = 1
                    self._restart_real_session(seed_value, cause="load_failure", bump_session_log=True)
        except Exception as exc:
            self._last_reset_status["reset_error"] = str(exc)
            self._record_session_event("reset_failed", exc=exc, stage="load_or_restart")
            raise self._make_reset_error("TraCI reset failed", exc, seed_value) from exc

        warmup_status = self._warmup_after_reset()
        if warmup_status["fatal"]:
            self._last_reset_status["warmup_fatal"] = True
            if restart_attempted:
                exc = RuntimeError("warmup_connection_lost")
                self._last_reset_status["reset_error"] = str(exc)
                self._record_session_event("reset_failed", exc=exc, stage="warmup")
                raise self._make_reset_error("TraCI reset failed after warmup connection loss", exc, seed_value)

            try:
                restart_attempted = True
                self._last_reset_status["restarted"] = True
                self._last_reset_status["restart_count"] = 1
                self._restart_real_session(seed_value, cause="warmup_connection_lost", bump_session_log=True)
            except Exception as exc:
                self._last_reset_status["reset_error"] = str(exc)
                self._record_session_event("reset_failed", exc=exc, stage="warmup_restart")
                raise self._make_reset_error("TraCI reset warmup restart failed", exc, seed_value) from exc

            warmup_status = self._warmup_after_reset()
            if warmup_status["fatal"]:
                exc = RuntimeError("warmup_connection_lost")
                self._last_reset_status["reset_error"] = str(exc)
                self._record_session_event("reset_failed", exc=exc, stage="warmup_retry")
                raise self._make_reset_error("TraCI reset failed after warmup retry", exc, seed_value)

        self._last_reset_status["ego_found"] = bool(warmup_status["ego_found"])
        scene = self.get_state()
        self._last_scene = scene
        self._record_session_event("reset_completed", seed=seed_value)
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

    def get_state(self) -> SceneState:
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
        if not self._use_mock and self._session_active:
            try:
                self._close_traci_connection()
            except Exception:
                pass
        self._record_session_event("close_backend")
        self._session_active = False
        self._connection_healthy = False
        self._started = False

    def _resolve_runtime_log_path(self) -> Path:
        log_dir = Path(self.config.runtime_log_dir)
        if not log_dir.is_absolute():
            log_dir = (Path.cwd() / log_dir).resolve()
        if self._current_episode_id:
            log_dir = log_dir / "episodes"
            log_dir.mkdir(parents=True, exist_ok=True)
            session_index = max(1, int(self._episode_session_index))
            return log_dir / f"{self._current_episode_id}_sess_{session_index:02d}.log"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "traci_runtime.log"

    def _advance_episode_session_log(self):
        if not self._current_episode_id:
            return
        self._episode_session_index = max(1, int(self._episode_session_index)) + 1
        self._runtime_log_path = self._resolve_runtime_log_path()
        self._record_session_event("session_log_rotated", session_index=self._episode_session_index)

    def _runtime_args(self, seed: Optional[int]) -> List[str]:
        cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
        return [
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

    def _open_traci_connection(self, args: List[str]):
        try:
            self._traci.start(args, label=self._connection_label)
        except TypeError:
            self._traci.start(args)
            return self._traci
        return self._resolve_traci_connection()

    def _resolve_traci_connection(self):
        get_connection = getattr(self._traci, "getConnection", None)
        if callable(get_connection):
            return get_connection(self._connection_label)
        return self._traci

    def _ensure_traci_connection(self):
        if self._traci_conn is None:
            self._traci_conn = self._resolve_traci_connection()
        return self._traci_conn

    def _close_traci_connection(self):
        conn = self._traci_conn
        self._traci_conn = None
        self._controller = None
        if conn is not None and hasattr(conn, "close"):
            try:
                conn.close(False)
                return
            except TypeError:
                conn.close()
                return
        if self._traci is not None and hasattr(self._traci, "close"):
            try:
                self._traci.close(False)
            except TypeError:
                self._traci.close()

    def _start_real_session(self, seed: Optional[int], reason: str, bump_session_log: bool):
        if bump_session_log:
            self._advance_episode_session_log()
        elif self._runtime_log_path is None:
            self._runtime_log_path = self._resolve_runtime_log_path()
        args = [self._sumo_binary] + self._runtime_args(seed)
        self._last_seed = int(seed if seed is not None else self.config.random_seed)
        self._last_runtime_args = list(args)
        self._traci_conn = self._open_traci_connection(args)
        if hasattr(self._traci_conn, "vehicle") and hasattr(self._traci_conn, "simulation"):
            self._controller = RealSumoController(self._traci_conn, self.config, _LOGGER)
        self._session_active = True
        self._connection_healthy = True
        self._record_session_event("start_real_session", reason=reason, seed=self._last_seed, connection_label=self._connection_label)

    def _restart_real_session(self, seed: Optional[int], cause: str, bump_session_log: bool):
        self._record_session_event("restart_real_session", cause=cause, seed=int(seed if seed is not None else self.config.random_seed))
        try:
            self._close_traci_connection()
        except Exception:
            pass
        self._start_real_session(seed, reason=f"restart:{cause}", bump_session_log=bump_session_log)

    def _load_with_seed(self, seed: Optional[int]):
        args = self._runtime_args(seed)
        self._last_seed = int(seed if seed is not None else self.config.random_seed)
        self._last_runtime_args = list(args)
        self._ensure_traci_connection().load(args)
        self._record_session_event("load_session", seed=self._last_seed, connection_label=self._connection_label)

    def _warmup_after_reset(self) -> Dict[str, bool]:
        max_steps = max(10, int(5.0 / max(self.config.step_length, 1e-3)))
        try:
            if self._controller is not None and self._controller.warmup_until_ego(max_steps=max_steps):
                return {"fatal": False, "ego_found": True}
        except Exception as exc:
            if self._is_fatal_traci_error(exc):
                self._mark_connection_closed()
                self._record_session_event("warmup_connection_lost", exc=exc)
                _LOGGER.warning(
                    "TraCI reset warmup lost SUMO connection (%s). See log: %s",
                    exc,
                    self.runtime_log_path,
                )
                return {"fatal": True, "ego_found": False}
            raise
        self._record_session_event("warmup_finished_without_ego", max_steps=max_steps)
        _LOGGER.warning(
            "TraCI reset warmup finished without ego vehicle '%s' (steps=%d). Continue with placeholder scene.",
            self.config.ego_vehicle_id,
            max_steps,
        )
        return {"fatal": False, "ego_found": False}

    def _handle_fatal_step(self, exc: Exception, action_meta: dict) -> BackendStepResult:
        self._mark_connection_closed()
        self._record_session_event("fatal_step", exc=exc)
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
            "sumo_exception_type": exc.__class__.__name__,
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

    def _safe_sim_time(self) -> float:
        if self._use_mock:
            return float(self._mock.step_index * self.config.step_length)
        try:
            if self._traci is not None and self._session_active:
                return float(self._traci.simulation.getTime())
        except Exception:
            pass
        if self._last_scene is not None:
            return float(self._last_scene.timestamp)
        return 0.0

    def _record_session_event(self, event_type: str, exc: Optional[Exception] = None, **payload: Any):
        cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
        restart_count = int(self._last_reset_status.get("restart_count", 0))
        event: Dict[str, Any] = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "source": "backend",
            "backend_type": "traci",
            "event": str(event_type),
            "episode_id": str(self._current_episode_id or ""),
            "risky_mode": bool(self._current_risky_mode),
            "collision_action": str(self.config.collision_action),
            "runtime_log_path": self.runtime_log_path,
            "restart_count": restart_count,
            "sumo_binary": str(self._sumo_binary or ""),
            "sumo_cfg": str(cfg_path),
            "session_index": int(self._episode_session_index),
            "sim_time_at_event": self._safe_sim_time(),
        }
        if exc is not None:
            event["exception_type"] = exc.__class__.__name__
            event["exception_text"] = str(exc)
            event["sim_time_at_failure"] = event["sim_time_at_event"]
        event.update(payload)
        self._session_events.append(event)
        if self._session_event_sink is not None:
            try:
                self._session_event_sink(dict(event))
            except Exception:
                _LOGGER.warning("Stage3 session event sink failed for event=%s", event_type)

    def _make_start_error(self, message: str, exc: Exception, seed: int) -> BackendStartError:
        return BackendStartError(
            message,
            backend_type="traci",
            episode_id=str(self._current_episode_id or ""),
            seed=seed,
            sumo_log_path=self.runtime_log_path,
            runtime_args=self._last_runtime_args,
            exception_type=exc.__class__.__name__,
            exception_text=str(exc),
        )

    def _make_reset_error(self, message: str, exc: Exception, seed: int) -> BackendResetError:
        return BackendResetError(
            message,
            backend_type="traci",
            episode_id=str(self._current_episode_id or ""),
            seed=seed,
            sumo_log_path=self.runtime_log_path,
            runtime_args=self._last_runtime_args,
            exception_type=exc.__class__.__name__,
            exception_text=str(exc),
        )

    def _is_fatal_traci_error(self, exc: Exception) -> bool:
        fatal_cls = getattr(getattr(self._traci, "exceptions", None), "FatalTraCIError", None)
        return fatal_cls is not None and isinstance(exc, fatal_cls)



