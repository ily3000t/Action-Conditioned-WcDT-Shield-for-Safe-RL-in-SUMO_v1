from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_RUN_ID = "20260320_210439"

STEPS: tuple[tuple[str, str, str], ...] = (
    (
        "Stage 1 - build pointwise and stage1 probe data",
        "safe_rl/config/default_safe_rl.yaml",
        "stage1",
    ),
    (
        "Stage 5 - bootstrap strong pairs",
        "safe_rl/config/stage5_pair_bootstrap.yaml",
        "stage5",
    ),
    (
        "Stage 2 - world-focused v2 training",
        "safe_rl/config/stage2_v2_world_pair_focus.yaml",
        "stage2",
    ),
    (
        "Stage 5 - held-out after-trace validation",
        "safe_rl/config/shield_trace_holdout_c1.yaml",
        "stage5",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the recommended 4-step SAFE_RL v2 pipeline in sequence.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=f"Run ID to reuse across all 4 steps. Default: {DEFAULT_RUN_ID}",
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

    print(f"[SAFE_RL] Repo root: {repo_root}")
    print(f"[SAFE_RL] Run ID: {args.run_id}")
    print(f"[SAFE_RL] Python: {args.python_cmd}")
    print()

    for step_name, config_path, stage in STEPS:
        command = build_command(args.python_cmd, config_path, stage, args.run_id)
        exit_code = run_step(repo_root, step_name, command, args.dry_run)
        if exit_code != 0:
            return exit_code

    print("[SAFE_RL] All 4 steps completed successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SAFE_RL] Interrupted by user.")
        raise SystemExit(130)
