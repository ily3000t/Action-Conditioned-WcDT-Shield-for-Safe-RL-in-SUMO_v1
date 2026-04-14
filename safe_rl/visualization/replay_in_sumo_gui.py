import argparse
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import yaml


def build_gui_replay_override_payload(
    seed: int,
    mode: str,
    trace_dir_name: str,
) -> Dict:
    return {
        "sim": {"use_gui": True},
        "shield_trace": {
            "enabled": True,
            "save_pair_traces": True,
            "trace_dir_name": trace_dir_name,
            "seed_list": [int(seed)],
        },
        "eval": {
            "eval_episodes": 1,
            "seed_list": [int(seed)],
        },
        "tensorboard": {
            "run_name": f"sumo_gui_replay_{mode}_{int(seed)}",
        },
    }


def write_gui_replay_override_file(
    run_id: str,
    seed: int,
    mode: str,
    output_dir: Path = Path("safe_rl_output/gui_replay_overrides"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir_name = f"sumo_gui_replay_{mode}_{int(seed)}"
    payload = build_gui_replay_override_payload(seed=seed, mode=mode, trace_dir_name=trace_dir_name)
    override_path = output_dir / f"{run_id}_{mode}_{int(seed)}.yaml"
    with override_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
    return override_path


def build_gui_replay_command(config_path: str, run_id: str) -> Sequence[str]:
    return [
        "python",
        "safe_rl_main.py",
        "--config",
        str(config_path),
        "--stage",
        "stage5",
        "--run-id",
        str(run_id),
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one selected stage5 case in SUMO GUI.")
    parser.add_argument("--run-id", required=True, help="Existing SAFE_RL run id")
    parser.add_argument("--seed", type=int, required=True, help="Seed for the selected case")
    parser.add_argument(
        "--mode",
        choices=("baseline", "shielded", "distilled"),
        required=True,
        help="Selected replay mode for case-focused debugging",
    )
    parser.add_argument(
        "--base-config",
        default="safe_rl/config/default_safe_rl.yaml",
        help="Base config. The script writes a temporary GUI override YAML.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute stage5 immediately. Without this flag, only print the command.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    override_path = write_gui_replay_override_file(
        run_id=args.run_id,
        seed=int(args.seed),
        mode=str(args.mode),
    )

    # base-config may be replaced by override for complete isolation; keep command explicit for reproducibility.
    command = build_gui_replay_command(config_path=str(override_path), run_id=str(args.run_id))
    print("[replay_in_sumo_gui] mode controls case interpretation; stage5 still evaluates baseline/shielded (+distilled when available).")
    print(f"[replay_in_sumo_gui] override_config={override_path}")
    print("[replay_in_sumo_gui] command:")
    print(" ".join(command))

    if args.execute:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
