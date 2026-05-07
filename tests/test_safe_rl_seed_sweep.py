import json
from pathlib import Path
import uuid

import yaml

from safe_rl.tools.run_stage2_seed_sweep import run_stage2_seed_sweep


def _write_base_config(path: Path):
    payload = {
        "world_model": {
            "pair_ft_random_seed": 42,
        },
        "tensorboard": {
            "run_name": "stage2_w006_det",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _local_tmp_dir(tag: str) -> Path:
    path = Path("safe_rl_output/test_artifacts") / f"{tag}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_seed_sweep_stops_when_latest_snapshot_found(monkeypatch):
    repo_root = _local_tmp_dir("seed_sweep_hit") / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    output_root = repo_root / "safe_rl_output"
    run_id = "ut_seed_sweep_hit"
    base_config = repo_root / "base.yaml"
    _write_base_config(base_config)

    def _fake_run(command, cwd=None, check=False):  # noqa: ANN001
        _ = (cwd, check)
        config_path = Path(command[command.index("--config") + 1])
        run_id_text = str(command[command.index("--run-id") + 1])
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        seed = int(((payload.get("world_model") or {}).get("pair_ft_random_seed", 0)))

        reports_dir = output_root / "runs" / run_id_text / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "stage2_training_report.json").write_text(
            json.dumps({"seed": seed}),
            encoding="utf-8",
        )
        if seed == 13:
            latest_path = output_root / "runs" / run_id_text / "snapshots" / "stage2_healthy" / "latest_snapshot.json"
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(
                json.dumps({"run_id": run_id_text, "snapshot_dir": "dummy"}),
                encoding="utf-8",
            )

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr("safe_rl.tools.run_stage2_seed_sweep.subprocess.run", _fake_run)

    summary = run_stage2_seed_sweep(
        run_id=run_id,
        base_config=str(base_config),
        seeds=[7, 13, 21],
        max_runs=5,
        python_cmd="python",
        repo_root=repo_root,
        output_root=output_root,
    )
    assert summary["status"] == "success"
    assert summary["successful_seed"] == 13
    assert summary["attempt_count"] == 2
    assert (output_root / "runs" / run_id / "reports" / "stage2_training_report_seed7.json").exists()
    assert (output_root / "runs" / run_id / "reports" / "stage2_training_report_seed13.json").exists()
    assert Path(summary["summary_path"]).exists()


def test_seed_sweep_respects_max_runs_when_no_snapshot(monkeypatch):
    repo_root = _local_tmp_dir("seed_sweep_miss") / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    output_root = repo_root / "safe_rl_output"
    run_id = "ut_seed_sweep_miss"
    base_config = repo_root / "base.yaml"
    _write_base_config(base_config)

    def _fake_run(command, cwd=None, check=False):  # noqa: ANN001
        _ = (cwd, check)
        run_id_text = str(command[command.index("--run-id") + 1])
        reports_dir = output_root / "runs" / run_id_text / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "stage2_training_report.json").write_text(
            json.dumps({"ok": True}),
            encoding="utf-8",
        )

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr("safe_rl.tools.run_stage2_seed_sweep.subprocess.run", _fake_run)

    summary = run_stage2_seed_sweep(
        run_id=run_id,
        base_config=str(base_config),
        seeds=[7, 13, 21],
        max_runs=2,
        python_cmd="python",
        repo_root=repo_root,
        output_root=output_root,
    )
    assert summary["status"] == "failed"
    assert summary["successful_seed"] is None
    assert summary["attempt_count"] == 2
    assert Path(summary["summary_path"]).exists()
