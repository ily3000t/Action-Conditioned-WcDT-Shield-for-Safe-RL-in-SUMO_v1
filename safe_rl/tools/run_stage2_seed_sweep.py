from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml


class Stage2SeedSweepError(RuntimeError):
    pass


def _parse_seeds(raw: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        raise Stage2SeedSweepError("seeds is required")
    values: List[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise Stage2SeedSweepError(f"invalid seed value: {item}") from exc
    if not values:
        raise Stage2SeedSweepError("no valid seeds parsed")
    return values


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f) or {})


def _write_yaml(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def run_stage2_seed_sweep(
    run_id: str,
    base_config: str,
    seeds: List[int],
    max_runs: int = 5,
    python_cmd: str = sys.executable,
    repo_root: Path = Path("."),
    output_root: Path = Path("safe_rl_output"),
) -> Dict[str, Any]:
    run_id_text = str(run_id or "").strip()
    if not run_id_text:
        raise Stage2SeedSweepError("run_id is required")

    base_config_path = Path(str(base_config or "").strip())
    if not base_config_path.exists():
        raise Stage2SeedSweepError(f"base config not found: {base_config_path}")
    base_config_path = base_config_path.resolve()

    if max_runs <= 0:
        raise Stage2SeedSweepError(f"max-runs must be positive, got={max_runs}")
    if not seeds:
        raise Stage2SeedSweepError("seed list is empty")

    run_root = output_root / "runs" / run_id_text
    reports_dir = run_root / "reports"
    temp_config_dir = run_root / "tmp" / "stage2_seed_sweep_configs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    temp_config_dir.mkdir(parents=True, exist_ok=True)
    latest_snapshot_path = run_root / "snapshots" / "stage2_healthy" / "latest_snapshot.json"
    stage2_report_path = reports_dir / "stage2_training_report.json"

    base_payload = _read_yaml(base_config_path)
    if not isinstance(base_payload, dict):
        raise Stage2SeedSweepError(f"base config payload is not a dict: {base_config_path}")

    attempts: List[Dict[str, Any]] = []
    successful_seed: int | None = None

    for index, seed in enumerate(seeds):
        if index >= int(max_runs):
            break

        payload = deepcopy(base_payload)
        world_model = dict(payload.get("world_model", {}) or {})
        world_model["pair_ft_random_seed"] = int(seed)
        payload["world_model"] = world_model

        tensorboard = dict(payload.get("tensorboard", {}) or {})
        base_run_name = str(tensorboard.get("run_name", "") or "").strip()
        if base_run_name:
            tensorboard["run_name"] = f"{base_run_name}_seed{seed}"
        else:
            tensorboard["run_name"] = f"stage2_seed_sweep_seed{seed}"
        payload["tensorboard"] = tensorboard

        config_path = temp_config_dir / f"{base_config_path.stem}_seed{seed}.yaml"
        _write_yaml(config_path, payload)

        command = [
            str(python_cmd),
            "safe_rl_main.py",
            "--config",
            str(config_path).replace("\\", "/"),
            "--stage",
            "stage2",
            "--run-id",
            run_id_text,
        ]
        result = subprocess.run(command, cwd=str(repo_root), check=False)

        report_snapshot_path = reports_dir / f"stage2_training_report_seed{seed}.json"
        if stage2_report_path.exists():
            shutil.copy2(stage2_report_path, report_snapshot_path)
        else:
            report_snapshot_path = Path("")

        snapshot_exists = latest_snapshot_path.exists()
        attempt = {
            "seed": int(seed),
            "config_path": str(config_path),
            "command": command,
            "return_code": int(result.returncode),
            "stage2_report_snapshot_path": str(report_snapshot_path) if str(report_snapshot_path) else "",
            "latest_snapshot_exists": bool(snapshot_exists),
        }
        attempts.append(attempt)

        if snapshot_exists:
            successful_seed = int(seed)
            break

    summary: Dict[str, Any] = {
        "run_id": run_id_text,
        "base_config_path": str(base_config_path),
        "seeds_requested": [int(item) for item in seeds],
        "max_runs": int(max_runs),
        "attempt_count": int(len(attempts)),
        "attempts": attempts,
        "results": attempts,
        "latest_snapshot_path": str(latest_snapshot_path),
        "successful_seed": successful_seed,
        "status": "success" if successful_seed is not None else "failed",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = reports_dir / "stage2_seed_sweep_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic Stage2 seed sweep and stop when latest healthy snapshot appears.",
    )
    parser.add_argument("--run-id", required=True, help="Run id under safe_rl_output/runs/<run_id>.")
    parser.add_argument("--base-config", required=True, help="Base config path used as template.")
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated seed list (example: 7,13,21,33,57).",
    )
    parser.add_argument("--max-runs", type=int, default=5, help="Maximum number of seeds to run.")
    parser.add_argument(
        "--python-cmd",
        default=sys.executable,
        help="Python executable used to invoke safe_rl_main.py.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    try:
        seeds = _parse_seeds(str(args.seeds))
        summary = run_stage2_seed_sweep(
            run_id=str(args.run_id),
            base_config=str(args.base_config),
            seeds=seeds,
            max_runs=int(args.max_runs),
            python_cmd=str(args.python_cmd),
            repo_root=Path(".").resolve(),
            output_root=Path("safe_rl_output"),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary.get("status") == "success" else 1
    except Exception as exc:
        print(f"[run_stage2_seed_sweep] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
