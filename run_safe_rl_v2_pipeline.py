from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

STEPS: tuple[tuple[str, str, str], ...] = (
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
    (
        "Stage 5 - distill and evaluate after Stage2 recovery",
        "safe_rl/config/default_safe_rl.yaml",
        "stage5",
    ),
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

    for step_name, config_path, stage in STEPS:
        command = build_command(args.python_cmd, config_path, stage, run_id)
        exit_code = run_step(repo_root, step_name, command, args.dry_run)
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
