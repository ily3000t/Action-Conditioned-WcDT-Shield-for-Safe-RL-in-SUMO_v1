import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


def build_gui_replay_override_payload(
    seed: int,
    mode: str,
    trace_dir_name: str,
    scenario_source: str = "",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
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
    if str(scenario_source or "").strip():
        payload["sim"]["sumo_cfg"] = str(scenario_source)
    return payload


def write_gui_replay_override_file(
    run_id: str,
    seed: int,
    mode: str,
    base_config_path: Path,
    trace_dir: str = "stage5_trace_capture_default",
    output_dir: Path = Path("safe_rl_output/gui_replay_overrides"),
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir_name = f"sumo_gui_replay_{mode}_{int(seed)}"
    scenario_source = _resolve_scenario_source(run_id=run_id, seed=seed, trace_dir=trace_dir)
    override = build_gui_replay_override_payload(
        seed=seed,
        mode=mode,
        trace_dir_name=trace_dir_name,
        scenario_source=scenario_source,
    )
    base_payload = _read_yaml(base_config_path)
    merged_payload = _deep_merge(base_payload, override)
    merged_payload.setdefault("tensorboard", {})
    merged_payload["tensorboard"]["run_name"] = f"sumo_gui_replay_{mode}_{int(seed)}"

    override_path = output_dir / f"{run_id}_{mode}_{int(seed)}.yaml"
    with override_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged_payload, f, sort_keys=False, allow_unicode=False)
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


def _resolve_scenario_source(run_id: str, seed: int, trace_dir: str) -> str:
    reports_root = Path("safe_rl_output/runs") / str(run_id) / "reports"
    trace_path = Path(trace_dir)
    if not trace_path.is_absolute():
        trace_path = reports_root / trace_dir
    if not trace_path.exists():
        return ""

    pair_path = trace_path / f"pair_00_seed_{int(seed)}.json"
    if not pair_path.exists():
        seed_matches = sorted(trace_path.glob(f"pair_*_seed_{int(seed)}.json"))
        if seed_matches:
            pair_path = seed_matches[0]
    if not pair_path.exists():
        return ""

    try:
        with pair_path.open("r", encoding="utf-8") as f:
            payload = dict(json.load(f) or {})
        return str(payload.get("scenario_source", "") or "")
    except Exception:
        return ""


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    for key, value in dict(update or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged.get(key, {})), value)
        else:
            merged[key] = value
    return merged


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
        help="Base config used for merged GUI replay override.",
    )
    parser.add_argument(
        "--trace-dir",
        default="stage5_trace_capture_default",
        help="Trace dir name under reports for scenario lookup.",
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
        base_config_path=Path(args.base_config),
        trace_dir=str(args.trace_dir),
    )

    command = build_gui_replay_command(config_path=str(override_path), run_id=str(args.run_id))
    print("[replay_in_sumo_gui] mode controls case interpretation; stage5 still evaluates baseline/shielded (+distilled when available).")
    print(f"[replay_in_sumo_gui] base_config={args.base_config}")
    print(f"[replay_in_sumo_gui] trace_dir={args.trace_dir}")
    print(f"[replay_in_sumo_gui] override_config={override_path}")
    print("[replay_in_sumo_gui] command:")
    print(" ".join(command))

    if args.execute:
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
