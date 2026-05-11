import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from safe_rl.config import load_safe_rl_config
from safe_rl.data.risk import compute_min_distance, compute_min_ttc, detect_collision
from safe_rl.data.stage1_probe import KEEP_KEEP_ACTION, STRUCTURAL_SKIP_REASONS
from safe_rl.data.types import SceneState
from safe_rl.sim.actions import action_name
from safe_rl.sim.factory import create_backend


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_episode_payload(run_dir: Path, episode_id: str) -> Dict[str, Any]:
    raw_dir = Path(run_dir) / "raw"
    candidate = raw_dir / str(episode_id)
    if candidate.suffix.lower() != ".json":
        candidate = raw_dir / f"{episode_id}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Episode payload not found: {candidate}")
    with candidate.open("r", encoding="utf-8") as f:
        return dict(json.load(f) or {})


def _build_event_map(schedule: Sequence[Dict[str, Any]]) -> Dict[int, List[str]]:
    event_map: Dict[int, List[str]] = {}
    for item in list(schedule or []):
        step_index = int(item.get("before_step", -1))
        if step_index < 0:
            continue
        event_type = str(item.get("event_type", "") or "")
        event_map.setdefault(step_index, []).append(event_type)
    return event_map


def _inject_events_before_step(backend: Any, event_map: Dict[int, List[str]], step_index: int):
    for event_type in list(event_map.get(int(step_index), []) or []):
        backend.inject_risk_event(event_type or None)


def _step_proxy_metrics(scenes: Sequence[SceneState], infos: Sequence[Dict[str, Any]], ttc_threshold: float) -> Dict[str, Any]:
    collision = any(bool(dict(info or {}).get("collision", False)) for info in infos) or any(detect_collision(scene) for scene in scenes)
    min_ttc = min((compute_min_ttc(scene) for scene in scenes), default=1e6)
    min_distance = min((compute_min_distance(scene) for scene in scenes), default=1e6)
    lane_conflict = any(
        bool(dict(info or {}).get("lane_violation", False))
        or str(dict(info or {}).get("lane_change_skipped_reason", "")).strip()
        for info in infos
    )
    route_structural_flag = any(
        str(dict(info or {}).get("lane_change_skipped_reason", "") or dict(info or {}).get("risk_skipped_reason", "")).strip().lower()
        in STRUCTURAL_SKIP_REASONS
        for info in infos
    )
    teleport_flag = any(bool(dict(info or {}).get("teleport", False) or dict(info or {}).get("sim_teleport", False)) for info in infos)

    distance_term = 1.0 if float(min_distance) < 3.0 else max(0.0, 1.0 - float(min_distance) / 30.0)
    ttc_term = 1.0 if float(min_ttc) < 1.5 else max(0.0, 1.0 - float(min_ttc) / 8.0)
    overall_proxy_risk = max(
        1.0 if collision else 0.0,
        0.7 if float(min_ttc) < float(ttc_threshold) else 0.0,
        0.5 if lane_conflict else 0.0,
        1.0 if route_structural_flag or teleport_flag else 0.0,
        float(distance_term),
        float(ttc_term),
    )
    return {
        "collision": bool(collision),
        "min_ttc": float(min_ttc),
        "min_distance": float(min_distance),
        "lane_conflict": bool(lane_conflict),
        "route_structural_flag": bool(route_structural_flag),
        "teleport_flag": bool(teleport_flag),
        "overall_proxy_risk": float(overall_proxy_risk),
    }


def _saturation_reasons(metrics: Dict[str, Any], saturation_threshold: float = 0.999) -> List[str]:
    reasons: List[str] = []
    risk = _safe_float(metrics.get("overall_proxy_risk", 0.0), 0.0)
    if risk < float(saturation_threshold):
        return reasons
    if bool(metrics.get("teleport_flag", False)):
        reasons.append("teleport")
    if bool(metrics.get("route_structural_flag", False)):
        reasons.append("route_structural")
    if bool(metrics.get("collision", False)):
        reasons.append("collision")
    if bool(metrics.get("lane_conflict", False)):
        reasons.append("lane_conflict")
    if _safe_float(metrics.get("min_ttc", 1e6), 1e6) < 1.5:
        reasons.append("min_ttc_low")
    if _safe_float(metrics.get("min_distance", 1e6), 1e6) < 3.0:
        reasons.append("min_distance_low")
    if not reasons:
        reasons.append("high_risk_saturated")
    return reasons


def _replay_prefix(
    backend: Any,
    raw_action_prefix: Sequence[int],
    event_map: Dict[int, List[str]],
    target_step_index: int,
):
    for prefix_step in range(int(target_step_index)):
        _inject_events_before_step(backend, event_map=event_map, step_index=prefix_step)
        action = int(raw_action_prefix[prefix_step]) if prefix_step < len(raw_action_prefix) else int(KEEP_KEEP_ACTION)
        backend.step(action)


def _rollout_branch(
    backend: Any,
    raw_action_prefix: Sequence[int],
    event_map: Dict[int, List[str]],
    step_index: int,
    action_id: int,
    horizon: int,
    episode_steps: int,
    ttc_threshold: float,
    sleep_ms: int = 0,
) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    scenes: List[SceneState] = []
    infos: List[Dict[str, Any]] = []
    reward_sum = 0.0
    done = False

    max_horizon = max(1, int(horizon))
    for offset in range(max_horizon):
        current_step = int(step_index) + int(offset)
        if current_step >= int(episode_steps):
            break
        _inject_events_before_step(backend, event_map=event_map, step_index=current_step)
        if offset == 0:
            chosen_action = int(action_id)
        else:
            chosen_action = int(raw_action_prefix[current_step]) if current_step < len(raw_action_prefix) else int(KEEP_KEEP_ACTION)

        result = backend.step(chosen_action)
        scene = result.scene
        info = dict(result.info or {})
        scenes.append(scene)
        infos.append(info)
        reward_sum += float(result.task_reward)
        metrics = _step_proxy_metrics([scene], [info], ttc_threshold=ttc_threshold)
        traces.append(
            {
                "step_index": int(current_step),
                "action_id": int(chosen_action),
                "action_name": str(action_name(int(chosen_action))),
                "task_reward": float(result.task_reward),
                **metrics,
            }
        )
        if int(sleep_ms) > 0:
            time.sleep(float(max(0, int(sleep_ms))) / 1000.0)
        done = bool(result.done)
        if done:
            break

    branch_metrics = _step_proxy_metrics(scenes=scenes, infos=infos, ttc_threshold=ttc_threshold)
    return {
        "action_id": int(action_id),
        "action_name": str(action_name(int(action_id))),
        "horizon_requested": int(horizon),
        "steps_executed": int(len(traces)),
        "reward_sum": float(reward_sum),
        "done": bool(done),
        "proxy_metrics": branch_metrics,
        "trace": traces,
    }


def _run_compare_ab(
    run_id: str,
    run_dir: Path,
    episode_id: str,
    config_path: str,
    use_gui: bool,
    step_index: int,
    action_a: int,
    action_b: int,
    horizon: int,
    output_root: Path,
    sleep_ms: int = 0,
) -> Dict[str, Any]:
    if int(horizon) <= 0:
        raise ValueError("horizon must be > 0 for compare_ab mode.")

    episode_payload = _load_episode_payload(run_dir=run_dir, episode_id=episode_id)
    steps = list(episode_payload.get("steps", []) or [])
    meta = dict(episode_payload.get("meta", {}) or {})
    raw_action_prefix = [int(v) for v in list(meta.get("raw_action_prefix", []) or [])]
    schedule = list(meta.get("risk_event_schedule", []) or [])
    event_map = _build_event_map(schedule)
    episode_steps = max(int(len(steps)), int(len(raw_action_prefix)))
    if episode_steps <= 0:
        raise ValueError("Episode has no steps to replay.")
    if int(step_index) < 0 or int(step_index) >= int(episode_steps):
        raise ValueError(f"step-index out of range: {step_index}, episode_steps={episode_steps}")

    stage_config = load_safe_rl_config(config_path)
    stage_config.sim.use_gui = bool(use_gui)

    branch_outputs: Dict[str, Any] = {}
    for key, action_id in (("a", int(action_a)), ("b", int(action_b))):
        backend = create_backend(stage_config.sim)
        try:
            backend.start()
            backend.set_episode_context(f"{episode_id}_compare_{key}", bool(episode_payload.get("risky_mode", False)))
            backend.reset(seed=int(meta.get("episode_seed", stage_config.sim.random_seed)))
            _replay_prefix(
                backend=backend,
                raw_action_prefix=raw_action_prefix,
                event_map=event_map,
                target_step_index=int(step_index),
            )
            branch_outputs[key] = _rollout_branch(
                backend=backend,
                raw_action_prefix=raw_action_prefix,
                event_map=event_map,
                step_index=int(step_index),
                action_id=int(action_id),
                horizon=int(horizon),
                episode_steps=int(episode_steps),
                ttc_threshold=float(stage_config.dataset.ttc_threshold),
                sleep_ms=int(sleep_ms),
            )
        finally:
            backend.close()

    a_metrics = dict(branch_outputs["a"]["proxy_metrics"] or {})
    b_metrics = dict(branch_outputs["b"]["proxy_metrics"] or {})
    preferred_by_risk = "tie"
    if float(a_metrics.get("overall_proxy_risk", 0.0)) < float(b_metrics.get("overall_proxy_risk", 0.0)):
        preferred_by_risk = "a"
    elif float(a_metrics.get("overall_proxy_risk", 0.0)) > float(b_metrics.get("overall_proxy_risk", 0.0)):
        preferred_by_risk = "b"

    comparison = {
        "delta_overall_proxy_risk_b_minus_a": float(b_metrics.get("overall_proxy_risk", 0.0))
        - float(a_metrics.get("overall_proxy_risk", 0.0)),
        "delta_min_ttc_b_minus_a": float(b_metrics.get("min_ttc", 0.0)) - float(a_metrics.get("min_ttc", 0.0)),
        "delta_min_distance_b_minus_a": float(b_metrics.get("min_distance", 0.0))
        - float(a_metrics.get("min_distance", 0.0)),
        "delta_reward_sum_b_minus_a": float(branch_outputs["b"].get("reward_sum", 0.0))
        - float(branch_outputs["a"].get("reward_sum", 0.0)),
        "collision_a": bool(a_metrics.get("collision", False)),
        "collision_b": bool(b_metrics.get("collision", False)),
        "preferred_by_proxy_risk": str(preferred_by_risk),
    }
    saturation_threshold = 0.999
    both_saturated = (
        float(a_metrics.get("overall_proxy_risk", 0.0)) >= saturation_threshold
        and float(b_metrics.get("overall_proxy_risk", 0.0)) >= saturation_threshold
    )
    saturation_reason_a = _saturation_reasons(a_metrics, saturation_threshold=saturation_threshold)
    saturation_reason_b = _saturation_reasons(b_metrics, saturation_threshold=saturation_threshold)
    action_sensitive = not (
        bool(both_saturated)
        and abs(float(comparison["delta_overall_proxy_risk_b_minus_a"])) <= 1e-6
    )
    diagnosis = "action_sensitive_proxy_risk_divergence"
    if bool(both_saturated):
        structural_or_teleport_a = any(item in ("teleport", "route_structural") for item in saturation_reason_a)
        structural_or_teleport_b = any(item in ("teleport", "route_structural") for item in saturation_reason_b)
        if structural_or_teleport_a and structural_or_teleport_b:
            diagnosis = "both_branches_saturated_by_structural_or_teleport"
        else:
            diagnosis = "both_branches_saturated"
    elif str(preferred_by_risk) == "tie":
        diagnosis = "action_insensitive_tie"
    comparison.update(
        {
            "both_saturated": bool(both_saturated),
            "saturation_reason_a": list(saturation_reason_a),
            "saturation_reason_b": list(saturation_reason_b),
            "action_sensitive": bool(action_sensitive),
            "diagnosis": str(diagnosis),
        }
    )

    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"compare_ab_{episode_id}_step_{int(step_index):04d}_a{int(action_a)}_b{int(action_b)}.json"
    )
    payload = {
        "mode": "compare_ab",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "config_path": str(config_path),
        "use_gui": bool(use_gui),
        "step_index": int(step_index),
        "horizon": int(horizon),
        "action_a": int(action_a),
        "action_b": int(action_b),
        "branch_a": branch_outputs["a"],
        "branch_b": branch_outputs["b"],
        "a": branch_outputs["a"].get("proxy_metrics", {}),
        "b": branch_outputs["b"].get("proxy_metrics", {}),
        "comparison": comparison,
        "diagnosis": str(diagnosis),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


def _run_raw_replay(
    run_id: str,
    run_dir: Path,
    episode_id: str,
    config_path: str,
    use_gui: bool,
    until_step: int,
    output_root: Path,
    sleep_ms: int = 0,
) -> Dict[str, Any]:
    episode_payload = _load_episode_payload(run_dir=run_dir, episode_id=episode_id)
    steps = list(episode_payload.get("steps", []) or [])
    meta = dict(episode_payload.get("meta", {}) or {})
    raw_action_prefix = [int(v) for v in list(meta.get("raw_action_prefix", []) or [])]
    schedule = list(meta.get("risk_event_schedule", []) or [])
    event_map = _build_event_map(schedule)
    episode_steps = max(int(len(steps)), int(len(raw_action_prefix)))
    if episode_steps <= 0:
        raise ValueError("Episode has no steps to replay.")

    last_step = int(until_step)
    if last_step < 0:
        last_step = int(episode_steps - 1)
    last_step = min(int(last_step), int(episode_steps - 1))

    stage_config = load_safe_rl_config(config_path)
    stage_config.sim.use_gui = bool(use_gui)

    backend = create_backend(stage_config.sim)
    traces: List[Dict[str, Any]] = []
    try:
        backend.start()
        backend.set_episode_context(f"{episode_id}_raw_replay", bool(episode_payload.get("risky_mode", False)))
        backend.reset(seed=int(meta.get("episode_seed", stage_config.sim.random_seed)))

        for step_index in range(last_step + 1):
            _inject_events_before_step(backend, event_map=event_map, step_index=step_index)
            action = int(raw_action_prefix[step_index]) if step_index < len(raw_action_prefix) else int(KEEP_KEEP_ACTION)
            result = backend.step(action)
            info = dict(result.info or {})
            metrics = _step_proxy_metrics([result.scene], [info], ttc_threshold=float(stage_config.dataset.ttc_threshold))
            traces.append(
                {
                    "step_index": int(step_index),
                    "action_id": int(action),
                    "action_name": str(action_name(int(action))),
                    "task_reward": float(result.task_reward),
                    "risk_events_applied": list(event_map.get(step_index, []) or []),
                    **metrics,
                }
            )
            if int(sleep_ms) > 0:
                time.sleep(float(max(0, int(sleep_ms))) / 1000.0)
            if bool(result.done):
                break
    finally:
        backend.close()

    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"raw_replay_{episode_id}.json"
    payload = {
        "mode": "raw_replay",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "config_path": str(config_path),
        "use_gui": bool(use_gui),
        "requested_until_step": int(until_step),
        "executed_steps": int(len(traces)),
        "trace": traces,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


def run_stage1_sumo_replay(
    run_id: str,
    mode: str,
    episode_id: str,
    config_path: str = "safe_rl/config/default_safe_rl.yaml",
    run_root: Path = Path("safe_rl_output/runs"),
    output_root: Path = Path("qualitative_results/stage1_sumo_replay"),
    use_gui: bool = True,
    until_step: int = -1,
    step_index: int = -1,
    action_a: int = 0,
    action_b: int = 8,
    horizon: int = 20,
    sleep_ms: int = 0,
) -> Dict[str, Any]:
    run_dir = Path(run_root) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "raw_replay":
        return _run_raw_replay(
            run_id=run_id,
            run_dir=run_dir,
            episode_id=episode_id,
            config_path=config_path,
            use_gui=use_gui,
            until_step=int(until_step),
            output_root=Path(output_root),
            sleep_ms=int(sleep_ms),
        )

    if normalized_mode == "compare_ab":
        return _run_compare_ab(
            run_id=run_id,
            run_dir=run_dir,
            episode_id=episode_id,
            config_path=config_path,
            use_gui=use_gui,
            step_index=int(step_index),
            action_a=int(action_a),
            action_b=int(action_b),
            horizon=int(horizon),
            output_root=Path(output_root),
            sleep_ms=int(sleep_ms),
        )

    raise ValueError(f"Unsupported mode: {mode}. expected raw_replay or compare_ab.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Stage1 episodes in SUMO for manual data inspection.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--mode", choices=("raw_replay", "compare_ab"), required=True, help="Replay mode")
    parser.add_argument("--episode-id", required=True, help="Episode id, e.g. ep_00037")
    parser.add_argument("--config", default="safe_rl/config/default_safe_rl.yaml", help="Config path")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--output-root", default="qualitative_results/stage1_sumo_replay", help="Output root directory")
    parser.add_argument("--until-step", type=int, default=-1, help="raw_replay only. replay up to this step (inclusive)")
    parser.add_argument("--step-index", type=int, default=-1, help="compare_ab only. apply A/B at this step index")
    parser.add_argument("--action-a", type=int, default=0, help="compare_ab only. action id for branch A")
    parser.add_argument("--action-b", type=int, default=8, help="compare_ab only. action id for branch B")
    parser.add_argument("--horizon", type=int, default=20, help="compare_ab only. rollout horizon after A/B action")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Optional sleep between steps for GUI observation")
    parser.add_argument("--use-gui", dest="use_gui", action="store_true", help="Use SUMO GUI for replay")
    parser.add_argument("--no-gui", dest="use_gui", action="store_false", help="Disable SUMO GUI for replay")
    parser.set_defaults(use_gui=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = run_stage1_sumo_replay(
        run_id=args.run_id,
        mode=args.mode,
        episode_id=args.episode_id,
        config_path=str(args.config),
        run_root=Path(args.run_root),
        output_root=Path(args.output_root),
        use_gui=bool(args.use_gui),
        until_step=int(args.until_step),
        step_index=int(args.step_index),
        action_a=int(args.action_a),
        action_b=int(args.action_b),
        horizon=int(args.horizon),
        sleep_ms=int(args.sleep_ms),
    )
    print(f"[stage1_sumo_replay] mode={args.mode} episode={args.episode_id}")
    print(f"[stage1_sumo_replay] output={payload.get('output_path', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
