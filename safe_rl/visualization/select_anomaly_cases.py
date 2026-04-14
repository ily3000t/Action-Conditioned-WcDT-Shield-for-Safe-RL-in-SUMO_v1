import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

try:
    from safe_rl.visualization.replay_episode import load_pair_payload, normalize_heading_to_degrees
except Exception:  # pragma: no cover
    from replay_episode import load_pair_payload, normalize_heading_to_degrees  # type: ignore


DEFAULT_RULES_PATH = Path("safe_rl/visualization/anomaly_rules.default.yaml")


def load_anomaly_rules(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def select_anomaly_cases(
    run_id: str,
    trace_dir: Optional[str] = None,
    run_root: Path = Path("safe_rl_output/runs"),
    rules_path: Path = DEFAULT_RULES_PATH,
    top_k: int = 20,
    output_root: Path = Path("qualitative_results/anomaly_cases"),
) -> Dict[str, Any]:
    run_dir = Path(run_root) / str(run_id)
    reports_dir = run_dir / "reports"
    resolved_trace_dir = _resolve_trace_dir(reports_dir=reports_dir, trace_dir=trace_dir)
    pair_files = _resolve_pair_files(resolved_trace_dir)
    rules = load_anomaly_rules(rules_path)

    scored_cases: List[Dict[str, Any]] = []
    for pair_path in pair_files:
        payload = load_pair_payload(pair_path)
        metrics = _compute_case_metrics(payload)
        matched_rules = _match_rules(metrics=metrics, rules=rules)
        if not matched_rules:
            continue
        score = _score_case(metrics=metrics, matched_rules=matched_rules)
        scored_cases.append(
            {
                "pair_file": str(pair_path),
                "pair_index": int(payload.get("pair_index", -1)),
                "seed": int(payload.get("seed", -1)),
                "distilled_unavailable": bool(payload.get("distilled_unavailable", True)),
                "matched_rules": matched_rules,
                "metrics": metrics,
                "score": float(score),
            }
        )

    scored_cases.sort(
        key=lambda item: (
            -int(len(item.get("matched_rules", []))),
            -float(item.get("score", 0.0)),
            int(item.get("pair_index", 1_000_000)),
        )
    )
    selected = scored_cases[: max(0, int(top_k))]
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank

    payload = {
        "run_id": str(run_id),
        "trace_dir": str(resolved_trace_dir),
        "rules_path": str(rules_path),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total_pairs_seen": int(len(pair_files)),
        "selected_count": int(len(selected)),
        "cases": selected,
    }
    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "anomaly_cases.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


def _resolve_trace_dir(reports_dir: Path, trace_dir: Optional[str]) -> Path:
    if trace_dir:
        explicit = Path(trace_dir)
        if explicit.exists():
            return explicit
        candidate = reports_dir / str(trace_dir)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Trace dir not found: {trace_dir}")

    preferred = ["stage5_trace_capture_default", "stage5_trace_capture_cost", "shield_trace"]
    for name in preferred:
        candidate = reports_dir / name
        if candidate.exists() and (candidate / "trace_summary.json").exists():
            return candidate

    discovered = sorted(
        [
            item
            for item in reports_dir.iterdir()
            if item.is_dir() and (item / "trace_summary.json").exists()
        ],
        key=lambda path: path.name,
    )
    if not discovered:
        raise FileNotFoundError(f"No trace directory with trace_summary.json found under {reports_dir}")
    return discovered[0]


def _resolve_pair_files(trace_dir: Path) -> List[Path]:
    trace_summary_path = trace_dir / "trace_summary.json"
    pair_files: List[Path] = []
    if trace_summary_path.exists():
        with trace_summary_path.open("r", encoding="utf-8") as f:
            trace_summary = dict(json.load(f) or {})
        for raw_path in list(trace_summary.get("pair_files", []) or []):
            path = Path(raw_path)
            if not path.is_absolute():
                path = trace_dir / path
            if path.exists():
                pair_files.append(path)
    if pair_files:
        return pair_files
    return sorted(trace_dir.glob("pair_*_seed_*.json"))


def _compute_case_metrics(pair_payload: Dict[str, Any]) -> Dict[str, Any]:
    baseline_steps = [dict(item) for item in list(pair_payload.get("baseline_steps", []) or [])]
    shielded_steps = [dict(item) for item in list(pair_payload.get("shielded_steps", []) or [])]
    steps = max(1, len(shielded_steps))

    replacement_count = sum(1 for item in shielded_steps if bool(item.get("replacement_happened", False)))
    shield_blocked_steps = 0
    near_risk_steps = 0
    min_ttc = math.inf
    min_distance = math.inf
    for item in shielded_steps:
        constraint_reason = str(item.get("constraint_reason", "") or "")
        block_trigger = str(item.get("block_trigger", "") or "")
        blocked_by_candidate = any(
            str(candidate.get("constraint_reason", "") or "") == "blocked_by_margin"
            for candidate in list(item.get("candidate_evaluations", []) or [])
        )
        if "blocked" in constraint_reason or block_trigger not in ("", "none") or blocked_by_candidate:
            shield_blocked_steps += 1
        ttc = _safe_float(item.get("min_ttc", item.get("ttc", math.inf)), math.inf)
        distance = _safe_float(item.get("min_distance", math.inf), math.inf)
        min_ttc = min(min_ttc, ttc)
        min_distance = min(min_distance, distance)
        if ttc < 1.5 or distance < 3.0:
            near_risk_steps += 1

    baseline_task_reward = _mean_task_reward(baseline_steps, default=_safe_float(pair_payload.get("baseline_reward", 0.0), 0.0))
    shielded_task_reward = _mean_task_reward(shielded_steps, default=_safe_float(pair_payload.get("shielded_reward", 0.0), 0.0))
    heading_stats = _heading_stats(shielded_steps)
    vx_stats = _negative_vx_stats(shielded_steps)

    return {
        "step_count": int(steps),
        "shield_blocked_steps": int(shield_blocked_steps),
        "replacement_count": int(replacement_count),
        "replacement_rate": float(replacement_count / max(1, steps)),
        "near_risk_step_rate": float(near_risk_steps / max(1, steps)),
        "min_ttc": float(min_ttc if math.isfinite(min_ttc) else 0.0),
        "min_distance": float(min_distance if math.isfinite(min_distance) else 0.0),
        "mean_task_reward_baseline": float(baseline_task_reward),
        "mean_task_reward_shielded": float(shielded_task_reward),
        "mean_task_reward_delta": float(shielded_task_reward - baseline_task_reward),
        "heading_abs_deg_max": float(heading_stats["abs_max_deg"]),
        "heading_abnormal_consecutive_steps": int(heading_stats["abnormal_consecutive_max"]),
        "heading_jump_count": int(heading_stats["jump_count"]),
        "negative_vx_consecutive_steps": int(vx_stats["consecutive_max"]),
    }


def _mean_task_reward(steps: Sequence[Dict[str, Any]], default: float = 0.0) -> float:
    if not steps:
        return float(default)
    values = [_safe_float(item.get("task_reward", item.get("reward", default)), default) for item in steps]
    return float(sum(values) / max(1, len(values)))


def _heading_stats(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    headings_deg: List[float] = []
    for step in steps:
        heading = _ego_state_value(step, "heading")
        headings_deg.append(normalize_heading_to_degrees(heading))

    abs_max = max((abs(value) for value in headings_deg), default=0.0)
    abnormal_consecutive = 0
    abnormal_consecutive_max = 0
    jump_count = 0
    for idx, heading in enumerate(headings_deg):
        if abs(heading) > 45.0:
            abnormal_consecutive += 1
            abnormal_consecutive_max = max(abnormal_consecutive_max, abnormal_consecutive)
        else:
            abnormal_consecutive = 0
        if idx > 0 and abs(heading - headings_deg[idx - 1]) > 60.0:
            jump_count += 1

    return {
        "abs_max_deg": float(abs_max),
        "abnormal_consecutive_max": int(abnormal_consecutive_max),
        "jump_count": int(jump_count),
    }


def _negative_vx_stats(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    consecutive = 0
    consecutive_max = 0
    for step in steps:
        vx = _safe_float(_ego_state_value(step, "vx"), 0.0)
        if vx < -0.5:
            consecutive += 1
            consecutive_max = max(consecutive_max, consecutive)
        else:
            consecutive = 0
    return {"consecutive_max": int(consecutive_max)}


def _ego_state_value(step: Dict[str, Any], key: str) -> float:
    history_scene = list(step.get("history_scene", []) or [])
    if not history_scene:
        return 0.0
    scene = dict(history_scene[-1] or {})
    ego_id = str(scene.get("ego_id", "ego"))
    for vehicle in list(scene.get("vehicles", []) or []):
        if str(vehicle.get("vehicle_id", "")) == ego_id:
            return _safe_float(vehicle.get(key, 0.0), 0.0)
    return 0.0


def _match_rules(metrics: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
    matched: List[str] = []

    blocked_cfg = dict(rules.get("high_blocked_low_replacement", {}) or {})
    if bool(blocked_cfg.get("enabled", True)):
        if (
            int(metrics.get("shield_blocked_steps", 0)) >= int(blocked_cfg.get("shield_blocked_steps_min", 80))
            and float(metrics.get("replacement_rate", 0.0)) <= float(blocked_cfg.get("replacement_rate_max", 0.15))
        ):
            matched.append("high_blocked_low_replacement")

    near_risk_cfg = dict(rules.get("high_near_risk", {}) or {})
    if bool(near_risk_cfg.get("enabled", True)):
        if (
            float(metrics.get("near_risk_step_rate", 0.0)) >= float(near_risk_cfg.get("near_risk_step_rate_min", 0.05))
            or float(metrics.get("min_ttc", 0.0)) < float(near_risk_cfg.get("min_ttc_max", 1.5))
            or float(metrics.get("min_distance", 0.0)) < float(near_risk_cfg.get("min_distance_max", 3.0))
        ):
            matched.append("high_near_risk")

    reward_cfg = dict(rules.get("low_task_reward", {}) or {})
    if bool(reward_cfg.get("enabled", True)):
        if float(metrics.get("mean_task_reward_delta", 0.0)) <= float(reward_cfg.get("mean_task_reward_delta_max", -0.02)):
            matched.append("low_task_reward")

    heading_cfg = dict(rules.get("heading_anomaly", {}) or {})
    if bool(heading_cfg.get("enabled", True)):
        abs_threshold = float(heading_cfg.get("heading_abs_deg_threshold", 45.0))
        jump_threshold = float(heading_cfg.get("heading_jump_deg_threshold", 60.0))
        min_consecutive = int(heading_cfg.get("min_consecutive_steps", 5))
        min_jump_count = int(heading_cfg.get("min_jump_count", 3))
        if (
            float(metrics.get("heading_abs_deg_max", 0.0)) >= abs_threshold
            and int(metrics.get("heading_abnormal_consecutive_steps", 0)) >= min_consecutive
        ) or (
            float(metrics.get("heading_abs_deg_max", 0.0)) >= abs_threshold
            and int(metrics.get("heading_jump_count", 0)) >= min_jump_count
            and jump_threshold >= 0.0
        ):
            matched.append("heading_anomaly")

    vx_cfg = dict(rules.get("negative_vx_anomaly", {}) or {})
    if bool(vx_cfg.get("enabled", True)):
        min_consecutive = int(vx_cfg.get("min_consecutive_steps", 5))
        if int(metrics.get("negative_vx_consecutive_steps", 0)) >= min_consecutive:
            matched.append("negative_vx_anomaly")

    return matched


def _score_case(metrics: Dict[str, Any], matched_rules: Sequence[str]) -> float:
    reward_penalty = abs(min(0.0, float(metrics.get("mean_task_reward_delta", 0.0))))
    score = (
        len(list(matched_rules)) * 100.0
        + float(metrics.get("shield_blocked_steps", 0)) * 0.2
        + float(metrics.get("near_risk_step_rate", 0.0)) * 50.0
        + reward_penalty * 100.0
    )
    return float(score)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select anomalous stage5 trace cases for replay visualization.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--trace-dir", default="", help="Trace directory name under run reports, or an absolute path")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--rules", default=str(DEFAULT_RULES_PATH), help="Anomaly rules YAML path")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K anomaly cases to keep")
    parser.add_argument(
        "--output-root",
        default="qualitative_results/anomaly_cases",
        help="Output root for anomaly_cases.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = select_anomaly_cases(
        run_id=args.run_id,
        trace_dir=args.trace_dir or None,
        run_root=Path(args.run_root),
        rules_path=Path(args.rules),
        top_k=int(args.top_k),
        output_root=Path(args.output_root),
    )
    print(f"[select_anomaly_cases] total_pairs={result['total_pairs_seen']} selected={result['selected_count']}")
    print(f"[select_anomaly_cases] output={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
