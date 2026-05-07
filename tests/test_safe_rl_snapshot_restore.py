import json
import uuid
from pathlib import Path

import pytest

from safe_rl.tools.restore_stage2_snapshot import SnapshotRestoreError, restore_stage2_snapshot


def _build_snapshot_fixture(run_id: str) -> dict:
    run_root = Path("safe_rl_output") / "runs" / run_id
    reports_dir = run_root / "reports"
    models_dir = run_root / "models"
    snapshot_dir = run_root / "snapshots" / "stage2_healthy" / "20260506_010101"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    (models_dir / "world_model.pt").write_text("active_world_old", encoding="utf-8")
    (models_dir / "light_risk.pt").write_text("active_light_old", encoding="utf-8")
    (reports_dir / "stage2_training_report.json").write_text(
        json.dumps({"stage2_pair_source_health": {"model_quality": {"status": "critical"}}}),
        encoding="utf-8",
    )
    (reports_dir / "risk_v2_eval_summary.json").write_text(json.dumps({"active": "old"}), encoding="utf-8")

    snapshot_report_path = snapshot_dir / "stage2_training_report.json"
    snapshot_world_path = snapshot_dir / "world_model.pt"
    snapshot_light_path = snapshot_dir / "light_risk.pt"
    snapshot_risk_summary_path = snapshot_dir / "risk_v2_eval_summary.json"
    snapshot_report_path.write_text(
        json.dumps({"stage2_pair_source_health": {"model_quality": {"status": "healthy"}}}),
        encoding="utf-8",
    )
    snapshot_world_path.write_text("snapshot_world", encoding="utf-8")
    snapshot_light_path.write_text("snapshot_light", encoding="utf-8")
    snapshot_risk_summary_path.write_text(json.dumps({"snapshot": "risk"}), encoding="utf-8")

    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "snapshot_type": "stage2_healthy",
                "snapshot_source": "promoted_candidate",
                "run_id": run_id,
                "stage2_training_report_path": str(snapshot_report_path),
                "world_model_path": str(snapshot_world_path),
                "light_model_path": str(snapshot_light_path),
                "risk_v2_eval_summary_path": str(snapshot_risk_summary_path),
            }
        ),
        encoding="utf-8",
    )
    latest_path = run_root / "snapshots" / "stage2_healthy" / "latest_snapshot.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "snapshot_source": "promoted_candidate",
                "snapshot_dir": str(snapshot_dir),
                "snapshot_manifest_path": str(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    return {
        "run_root": run_root,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "snapshot_dir": snapshot_dir,
        "manifest_path": manifest_path,
        "latest_path": latest_path,
    }


def test_restore_stage2_snapshot_latest_success():
    run_id = f"ut_restore_ok_{uuid.uuid4().hex[:8]}"
    fixture = _build_snapshot_fixture(run_id)

    audit = restore_stage2_snapshot(run_id=run_id, snapshot="latest", strict=True)
    assert audit["status"] == "restored"
    assert audit["resolved_snapshot_dir"] == str(fixture["snapshot_dir"])
    assert audit["snapshot_source"] == "promoted_candidate"
    assert audit["source_snapshot_dir"] == str(fixture["snapshot_dir"])
    assert (fixture["models_dir"] / "world_model.pt").read_text(encoding="utf-8") == "snapshot_world"
    assert (fixture["models_dir"] / "light_risk.pt").read_text(encoding="utf-8") == "snapshot_light"
    restored_report = json.loads((fixture["reports_dir"] / "stage2_training_report.json").read_text(encoding="utf-8"))
    assert restored_report["stage2_pair_source_health"]["model_quality"]["status"] == "healthy"
    restored_risk = json.loads((fixture["reports_dir"] / "risk_v2_eval_summary.json").read_text(encoding="utf-8"))
    assert restored_risk["snapshot"] == "risk"
    assert (fixture["reports_dir"] / "stage2_active_snapshot_restore.json").exists()


def test_restore_stage2_snapshot_fails_when_required_file_missing():
    run_id = f"ut_restore_missing_{uuid.uuid4().hex[:8]}"
    fixture = _build_snapshot_fixture(run_id)
    (fixture["snapshot_dir"] / "world_model.pt").unlink()

    with pytest.raises(SnapshotRestoreError, match="world_model"):
        restore_stage2_snapshot(run_id=run_id, snapshot="latest", strict=True)

    audit = json.loads((fixture["reports_dir"] / "stage2_active_snapshot_restore.json").read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert any("world_model" in item for item in audit["errors"])


def test_restore_stage2_snapshot_strict_run_id_mismatch_fails():
    run_id = f"ut_restore_mismatch_{uuid.uuid4().hex[:8]}"
    fixture = _build_snapshot_fixture(run_id)
    bad_manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    bad_manifest["run_id"] = "another_run_id"
    fixture["manifest_path"].write_text(json.dumps(bad_manifest), encoding="utf-8")

    with pytest.raises(SnapshotRestoreError, match="run_id mismatch"):
        restore_stage2_snapshot(run_id=run_id, snapshot="latest", strict=True)

    audit = json.loads((fixture["reports_dir"] / "stage2_active_snapshot_restore.json").read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert any("mismatch" in item for item in audit["errors"])
