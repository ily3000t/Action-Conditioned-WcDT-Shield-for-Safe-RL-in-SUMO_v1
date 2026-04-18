from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

BASE_STEPS: tuple[tuple[str, str, str], ...] = (
    (
        "Stage 4 - collect intervention buffer (self-recovery candidates)",
        "safe_rl/config/default_safe_rl.yaml",
        "stage4",
    ),
    (
        "Stage 2 - retrain world/light models with latest pair sources",
        "safe_rl/config/default_safe_rl.yaml",
        "stage2",
    ),
)
STAGE5_STEP: tuple[str, str, str] = (
    "Stage 5 - distill and evaluate after Stage2 recovery",
    "safe_rl/config/default_safe_rl.yaml",
    "stage5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAFE_RL closed-loop revalidation sequence: stage4 -> stage2 -> stage5.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to reuse across all steps. If omitted, auto-pick the latest run under safe_rl_output/runs.",
    )
    parser.add_argument(
        "--python",
        dest="python_cmd",
        default=sys.executable,
        help="Python executable to use. Default: current interpreter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()



def build_command(python_cmd: str, config_path: str, stage: str, run_id: str) -> list[str]:
    return [
        python_cmd,
        "safe_rl_main.py",
        "--config",
        config_path,
        "--stage",
        stage,
        "--run-id",
        run_id,
    ]



def format_command(parts: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(parts))



def run_step(repo_root: Path, step_name: str, command: Sequence[str], dry_run: bool) -> int:
    print("=" * 60)
    print(f"[SAFE_RL] {step_name}")
    print("=" * 60)
    print(format_command(command))
    print()

    if dry_run:
        return 0

    completed = subprocess.run(list(command), cwd=repo_root)
    if completed.returncode != 0:
        print()
        print(f"[SAFE_RL] Step failed: {step_name}")
        return completed.returncode
    return 0


def _stage2_report_path(repo_root: Path, run_id: str) -> Path:
    return repo_root / "safe_rl_output" / "runs" / run_id / "reports" / "stage2_training_report.json"


def should_run_stage5_from_stage2_report_payload(payload: Dict[str, Any]) -> Tuple[bool, str, str]:
    stage2_health = dict(payload.get("stage2_pair_source_health", {}) or {})
    model_quality = dict(stage2_health.get("model_quality", {}) or {})
    status = str(model_quality.get("status", "")).strip().lower()
    if status in {"healthy", "degraded"}:
        return True, status, ""
    if not status:
        return False, "unknown", "missing_model_quality_status"
    message = str(model_quality.get("message", "") or "")
    if message:
        return False, status, message
    return False, status, f"stage2_model_quality_status={status}"


def should_run_stage5_from_stage2_report_path(report_path: Path) -> Tuple[bool, str, str]:
    if not report_path.exists():
        return False, "missing", f"missing_stage2_training_report: {report_path}"
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "invalid", f"failed_to_parse_stage2_training_report: {exc}"
    return should_run_stage5_from_stage2_report_payload(payload)


def summarize_stage2_resolution_progress(payload: Dict[str, Any]) -> Dict[str, Any]:
    world_pair_ft = dict(dict(payload.get("pair_finetune_metrics", {}) or {}).get("world", {}) or {})
    epoch_metrics = list(world_pair_ft.get("epoch_metrics", []) or [])
    active_counts = [float(dict(epoch or {}).get("stage4_aux_active_pair_count", 0.0) or 0.0) for epoch in epoch_metrics]
    resolution_losses = [float(dict(epoch or {}).get("stage4_aux_resolution_loss", 0.0) or 0.0) for epoch in epoch_metrics]
    below_score_margin_fractions = [
        float(dict(epoch or {}).get("stage4_aux_below_score_margin_fraction", 0.0) or 0.0)
        for epoch in epoch_metrics
    ]
    active_below_score_margin_fractions = [
        fraction for fraction, active in zip(below_score_margin_fractions, active_counts) if float(active) > 0.0
    ]
    has_active_pairs = any(float(item) > 0.0 for item in active_counts)
    has_resolution_loss = any(float(item) > 0.0 for item in resolution_losses)
    has_below_score_margin = any(float(item) > 0.0 for item in active_below_score_margin_fractions)
    below_score_margin_trend_down = False
    if len(active_below_score_margin_fractions) >= 2:
        below_score_margin_trend_down = (
            float(active_below_score_margin_fractions[-1]) < float(active_below_score_margin_fractions[0]) - 1e-9
        )

    stage4_aux_unique_after = float(
        dict(payload.get("stage4_aux_unique_score_count_before_after", {}) or {}).get("after", 0.0) or 0.0
    )
    world_pair_ft_best_epoch = int(payload.get("world_pair_ft_best_epoch", -1) or -1)
    return {
        "stage4_aux_active_pair_count_max": float(max(active_counts) if active_counts else 0.0),
        "stage4_aux_resolution_loss_max": float(max(resolution_losses) if resolution_losses else 0.0),
        "stage4_aux_below_score_margin_fraction_first": (
            float(active_below_score_margin_fractions[0]) if active_below_score_margin_fractions else 0.0
        ),
        "stage4_aux_below_score_margin_fraction_last": (
            float(active_below_score_margin_fractions[-1]) if active_below_score_margin_fractions else 0.0
        ),
        "stage4_aux_unique_after": float(stage4_aux_unique_after),
        "world_pair_ft_best_epoch": int(world_pair_ft_best_epoch),
        "has_active_pairs": bool(has_active_pairs),
        "has_resolution_loss": bool(has_resolution_loss),
        "has_below_score_margin": bool(has_below_score_margin),
        "below_score_margin_trend_down": bool(below_score_margin_trend_down),
    }


def print_stage2_resolution_progress(summary: Dict[str, Any]) -> None:
    print("[SAFE_RL] Stage2 score-space resolution checks:")
    print(
        "[SAFE_RL] 1) active_pairs>0: "
        f"{summary.get('has_active_pairs')} (max_active={summary.get('stage4_aux_active_pair_count_max')})"
    )
    print(
        "[SAFE_RL] 2) resolution_loss>0: "
        f"{summary.get('has_resolution_loss')} (max_resolution={summary.get('stage4_aux_resolution_loss_max')})"
    )
    print(
        "[SAFE_RL] 3) below_score_margin_fraction>0: "
        f"{summary.get('has_below_score_margin')} "
        f"(first={summary.get('stage4_aux_below_score_margin_fraction_first')}, "
        f"last={summary.get('stage4_aux_below_score_margin_fraction_last')}, "
        f"trend_down={summary.get('below_score_margin_trend_down')})"
    )
    print(
        "[SAFE_RL] 4) stage4_aux_unique_after>8: "
        f"{float(summary.get('stage4_aux_unique_after', 0.0)) > 8.0} "
        f"(value={summary.get('stage4_aux_unique_after')})"
    )
    print(
        "[SAFE_RL] 5) world_pair_ft_best_epoch!=-1: "
        f"{int(summary.get('world_pair_ft_best_epoch', -1)) != -1} "
        f"(value={summary.get('world_pair_ft_best_epoch')})"
    )
    print()



def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_id = str(args.run_id or "").strip()
    if not run_id:
        runs_root = repo_root / "safe_rl_output" / "runs"
        if not runs_root.exists():
            print(f"[SAFE_RL] No runs directory found: {runs_root}")
            return 2
        all_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
        timestamp_like = [path for path in all_dirs if re.match(r"^\d{8}_\d{6}$", path.name)]
        candidates = sorted(timestamp_like, key=lambda item: item.name)
        if not candidates:
            candidates = sorted(all_dirs, key=lambda item: item.stat().st_mtime)
        if not candidates:
            print(f"[SAFE_RL] No run directories found under: {runs_root}")
            return 2
        run_id = candidates[-1].name

    print(f"[SAFE_RL] Repo root: {repo_root}")
    print(f"[SAFE_RL] Run ID: {run_id}")
    print(f"[SAFE_RL] Python: {args.python_cmd}")
    print()

    for step_name, config_path, stage in BASE_STEPS:
        command = build_command(args.python_cmd, config_path, stage, run_id)
        exit_code = run_step(repo_root, step_name, command, args.dry_run)
        if exit_code != 0:
            return exit_code

    stage5_name, stage5_config, stage5_stage = STAGE5_STEP
    stage5_command = build_command(args.python_cmd, stage5_config, stage5_stage, run_id)
    if args.dry_run:
        print("[SAFE_RL] Dry-run mode: Stage5 is conditionally executed when Stage2 model quality is healthy/degraded.")
        print(format_command(stage5_command))
        print("[SAFE_RL] Closed-loop dry-run completed successfully.")
        return 0

    stage2_report_path = _stage2_report_path(repo_root, run_id)
    stage2_report_payload: Dict[str, Any] = {}
    if stage2_report_path.exists():
        try:
            stage2_report_payload = json.loads(stage2_report_path.read_text(encoding="utf-8"))
        except Exception:
            stage2_report_payload = {}
    if stage2_report_payload:
        print_stage2_resolution_progress(summarize_stage2_resolution_progress(stage2_report_payload))
        run_stage5, stage2_status, skip_reason = should_run_stage5_from_stage2_report_payload(stage2_report_payload)
    else:
        run_stage5, stage2_status, skip_reason = should_run_stage5_from_stage2_report_path(stage2_report_path)
    if not run_stage5:
        print(f"[SAFE_RL] Stage5 skipped: stage2_status={stage2_status}, reason={skip_reason}")
        print("[SAFE_RL] Closed-loop revalidation completed (conditional Stage5 skipped).")
        return 0

    exit_code = run_step(repo_root, stage5_name, stage5_command, dry_run=False)
    if exit_code != 0:
        return exit_code

    print("[SAFE_RL] Closed-loop revalidation steps completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SAFE_RL] Interrupted by user.")
        raise SystemExit(130)
