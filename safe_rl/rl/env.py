from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from safe_rl.config.config import PPOConfig, SimConfig
from safe_rl.data.risk import compute_min_distance, compute_min_ttc, detect_collision, get_ego_vehicle
from safe_rl.data.types import SceneState, ShieldDecision
from safe_rl.models.features import BASE_FEATURE_DIM, encode_history
from safe_rl.shield.safety_shield import SafetyShield
from safe_rl.sim.actions import action_name, decode_action
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
        episode_prefix: str = "env",
        session_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.backend = backend
        self.sim_config = sim_config
        self.ppo_config = ppo_config
        self.shield = shield
        self.episode_prefix = self._sanitize_episode_prefix(episode_prefix)
        self.session_event_sink = session_event_sink
        self.history: deque = deque(maxlen=sim_config.history_steps)
        self.step_count = 0
        self.episode_interventions = 0
        self.episode_collisions = 0
        self.last_transition: Optional[Dict[str, Any]] = None
        self.episode_index = 0
        self.session_records: List[Dict[str, Any]] = []
        self._current_episode_record: Optional[Dict[str, Any]] = None

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

        if self._current_episode_record is not None:
            self._finalize_episode(closed_early=True, termination_reason="reset_before_terminal")

        self.episode_index += 1
        episode_id = f"{self.episode_prefix}_ep_{self.episode_index:06d}"
        self.backend.set_episode_context(episode_id, risky_mode)

        self._current_episode_record = {
            "episode_id": episode_id,
            "reset_seed": None if seed is None else int(seed),
            "risky_mode": risky_mode,
            "sumo_log_path": self.backend.runtime_log_path,
            "restarted": False,
            "restart_count": 0,
            "reset_failed": False,
            "fatal_step_terminated": False,
        }
        self._emit_session_event(
            event="episode_reset_started",
            episode_id=episode_id,
            reset_seed=None if seed is None else int(seed),
            risky_mode=risky_mode,
        )

        try:
            scene = self.backend.reset(seed=seed)
        except Exception as exc:
            diagnostics = self.backend.get_runtime_diagnostics()
            reset_status = diagnostics.get("last_reset_status", {})
            self._current_episode_record.update(
                {
                    "reset_failed": True,
                    "reset_error_type": exc.__class__.__name__,
                    "reset_error_message": str(exc),
                    "sumo_log_path": str(diagnostics.get("runtime_log_path") or self.backend.runtime_log_path),
                    "backend_diagnostics": diagnostics,
                    "restart_count": int(reset_status.get("restart_count", 0)),
                    "restarted": bool(reset_status.get("restarted", False)),
                    "termination_reason": "reset_error",
                }
            )
            self._emit_session_event(
                event="episode_reset_failed",
                episode_id=episode_id,
                reset_seed=None if seed is None else int(seed),
                risky_mode=risky_mode,
                exception_type=exc.__class__.__name__,
                exception_text=str(exc),
            )
            self.session_records.append(dict(self._current_episode_record))
            self._current_episode_record = None
            raise

        diagnostics = self.backend.get_runtime_diagnostics()
        reset_status = diagnostics.get("last_reset_status", {})
        self._current_episode_record.update(
            {
                "sumo_log_path": str(diagnostics.get("runtime_log_path") or self.backend.runtime_log_path),
                "restart_count": int(reset_status.get("restart_count", 0)),
                "restarted": bool(reset_status.get("restarted", False)),
                "load_attempted": bool(reset_status.get("load_attempted", False)),
                "load_failed": bool(reset_status.get("load_failed", False)),
            }
        )
        self._emit_session_event(
            event="episode_reset_completed",
            episode_id=episode_id,
            reset_seed=None if seed is None else int(seed),
            risky_mode=risky_mode,
            restart_count=int(reset_status.get("restart_count", 0)),
            restarted=bool(reset_status.get("restarted", False)),
        )

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
        info = {
            "risky_mode": risky_mode,
            "episode_id": episode_id,
            "sumo_log_path": str(diagnostics.get("runtime_log_path") or self.backend.runtime_log_path),
            "scenario_source": str(self.sim_config.sumo_cfg),
            "reset_seed": None if seed is None else int(seed),
        }
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

        shield_called = self.shield is not None
        replacement_happened = bool(decision.final_action != raw_action)
        fallback_used = bool(decision.meta.get("fallback_used", False))
        evaluated_candidate_count = int(decision.meta.get("evaluated_candidate_count", len(decision.candidate_risks)))
        shield_blocked = bool(decision.meta.get("shield_blocked", shield_called and decision.reason != "raw_action_safe"))
        replacement_same_as_raw = int(bool(decision.intervened and decision.final_action == raw_action))
        ego_vehicle = get_ego_vehicle(result.scene)
        ego_lane_index = int(getattr(ego_vehicle, "lane_id", 0))
        raw_action_type = str(decision.meta.get("raw_action_type", action_name(raw_action)))
        final_action_type = str(decision.meta.get("final_action_type", action_name(decision.final_action)))
        lane_change_involved = bool(decision.meta.get("lane_change_involved", decode_action(raw_action).lateral != 0 or decode_action(decision.final_action).lateral != 0))

        info = {
            "raw_action": raw_action,
            "final_action": decision.final_action,
            "executed_action": decision.final_action,
            "risk_raw": decision.risk_raw,
            "risk_final": decision.risk_final,
            "risk_reduction": float(decision.risk_raw - decision.risk_final),
            "intervened": decision.intervened,
            "intervention_reason": decision.reason,
            "candidate_risks": decision.candidate_risks,
            "candidate_evaluations": decision.meta.get("candidate_evaluations", []),
            "task_reward": float(result.task_reward),
            "reward": reward,
            "collision": collision,
            "ttc": compute_min_ttc(result.scene),
            "min_distance": compute_min_distance(result.scene),
            "lane_violation": bool(result.info.get("lane_violation", False)),
            "shield_meta": decision.meta,
            "ego_speed": float(result.info.get("ego_speed", 0.0)),
            "ego_lane_id": str(getattr(ego_vehicle, "lane_id", 0)),
            "ego_lane_index": ego_lane_index,
            "episode_id": self._current_episode_record.get("episode_id", "") if self._current_episode_record else "",
            "risky_mode": bool(self._current_episode_record.get("risky_mode", False)) if self._current_episode_record else False,
            "scenario_source": str(self.sim_config.sumo_cfg),
            "replacement_happened": replacement_happened,
            "fallback_used": fallback_used,
            "evaluated_candidate_count": evaluated_candidate_count,
            "chosen_candidate_index": int(decision.meta.get("chosen_candidate_index", -1)),
            "chosen_candidate_rank_by_risk": int(decision.meta.get("chosen_candidate_rank_by_risk", -1)),
            "raw_action_type": raw_action_type,
            "final_action_type": final_action_type,
            "lane_change_involved": lane_change_involved,
            "constraint_reason": str(decision.meta.get("constraint_reason", "")),
            "replacement_margin": float(decision.meta.get("replacement_margin", 0.0)),
            "shield_called_steps": int(shield_called),
            "shield_candidate_evaluated_steps": int(shield_called and evaluated_candidate_count > 0),
            "shield_blocked_steps": int(shield_blocked),
            "shield_replaced_steps": int(replacement_happened),
            "replacement_count": int(replacement_happened),
            "replacement_same_as_raw_count": replacement_same_as_raw,
            "fallback_action_count": int(fallback_used),
            "terminated_by_sumo": bool(result.info.get("terminated_by_sumo", False)),
            "termination_reason": str(result.info.get("termination_reason", "")),
            "sumo_exception_type": str(result.info.get("sumo_exception_type", "")),
            "sumo_exception": str(result.info.get("sumo_exception", "")),
            "sumo_log_path": str(result.info.get("sumo_log_path", self.backend.runtime_log_path)),
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

        if terminated or truncated:
            self._finalize_episode(terminated=terminated, truncated=truncated, info=info)

        return observation, reward, terminated, truncated, info

    def _current_observation(self) -> np.ndarray:
        return encode_history(list(self.history)).astype(np.float32)

    def get_history(self) -> List[SceneState]:
        return list(self.history)

    def get_session_records(self) -> List[Dict[str, Any]]:
        return [dict(record) for record in self.session_records]

    def close(self):
        if self._current_episode_record is not None:
            self._finalize_episode(closed_early=True, termination_reason="env_closed")
        self.backend.close()

    def _sanitize_episode_prefix(self, value: str) -> str:
        text = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in str(value or "env"))
        return text.strip("_") or "env"

    def _emit_session_event(self, event: str, **payload: Any):
        if self.session_event_sink is None:
            return
        diagnostics = self.backend.get_runtime_diagnostics()
        reset_status = diagnostics.get("last_reset_status", {})
        sim_time = self._current_sim_time(diagnostics)
        event_payload = {
            "source": "env",
            "event": str(event),
            "episode_id": str(payload.get("episode_id") or (self._current_episode_record or {}).get("episode_id", "")),
            "risky_mode": bool(payload.get("risky_mode") if "risky_mode" in payload else (self._current_episode_record or {}).get("risky_mode", False)),
            "backend_type": str(diagnostics.get("backend") or ""),
            "collision_action": str(diagnostics.get("collision_action") or ""),
            "runtime_log_path": str(diagnostics.get("runtime_log_path") or self.backend.runtime_log_path),
            "restart_count": int(payload.get("restart_count", reset_status.get("restart_count", 0))),
            "sumo_binary": str(diagnostics.get("sumo_binary") or ""),
            "sumo_cfg": str(diagnostics.get("sumo_cfg") or ""),
            "sim_time_at_event": sim_time,
        }
        exception_type = str(payload.get("exception_type") or "")
        exception_text = str(payload.get("exception_text") or "")
        if exception_type or exception_text:
            event_payload["exception_type"] = exception_type
            event_payload["exception_text"] = exception_text
            event_payload["sim_time_at_failure"] = sim_time
        event_payload.update(payload)
        self.session_event_sink(event_payload)

    def _current_sim_time(self, diagnostics: Dict[str, Any]) -> float:
        try:
            if self.history:
                return float(self.history[-1].timestamp)
        except Exception:
            pass
        sim_time = diagnostics.get("sim_time")
        if sim_time is None:
            return 0.0
        return float(sim_time)

    def _finalize_episode(
        self,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
        closed_early: bool = False,
        termination_reason: str = "",
    ):
        if self._current_episode_record is None:
            return

        info = info or {}
        diagnostics = self.backend.get_runtime_diagnostics()
        reset_status = diagnostics.get("last_reset_status", {})
        reason = str(
            termination_reason
            or info.get("termination_reason")
            or ("episode_limit" if truncated else "backend_done" if terminated else "")
        )

        record = dict(self._current_episode_record)
        record.update(
            {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "closed_early": bool(closed_early),
                "episode_steps": int(self.step_count),
                "episode_interventions": int(self.episode_interventions),
                "episode_collisions": int(self.episode_collisions),
                "fatal_step_terminated": bool(info.get("terminated_by_sumo", False)),
                "termination_reason": reason,
                "sumo_exception_type": str(info.get("sumo_exception_type", "")),
                "sumo_exception": str(info.get("sumo_exception", "")),
                "sumo_log_path": str(info.get("sumo_log_path") or diagnostics.get("runtime_log_path") or self.backend.runtime_log_path),
                "restart_count": int(reset_status.get("restart_count", record.get("restart_count", 0))),
                "restarted": bool(reset_status.get("restarted", record.get("restarted", False))),
                "sim_time_at_end": self._current_sim_time(diagnostics),
            }
        )
        self._emit_session_event(
            event="episode_completed",
            episode_id=record.get("episode_id", ""),
            risky_mode=bool(record.get("risky_mode", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            closed_early=bool(closed_early),
            episode_steps=int(self.step_count),
            episode_interventions=int(self.episode_interventions),
            episode_collisions=int(self.episode_collisions),
            termination_reason=reason,
            exception_type=str(info.get("sumo_exception_type", "")),
            exception_text=str(info.get("sumo_exception", "")),
        )
        self.session_records.append(record)
        self._current_episode_record = None



def create_env(
    backend: ISumoBackend,
    sim_config: SimConfig,
    ppo_config: PPOConfig,
    shield: Optional[SafetyShield],
    episode_prefix: str = "env",
    session_event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    return SafeDrivingEnv(
        backend=backend,
        sim_config=sim_config,
        ppo_config=ppo_config,
        shield=shield,
        episode_prefix=episode_prefix,
        session_event_sink=session_event_sink,
    )
