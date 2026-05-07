from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


class SnapshotRestoreError(RuntimeError):
    pass


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _resolve_snapshot_dir(run_root: Path, snapshot: str) -> Tuple[Path, Path]:
    snapshot_root = run_root / "snapshots" / "stage2_healthy"
    if str(snapshot).strip().lower() == "latest":
        latest_snapshot_path = snapshot_root / "latest_snapshot.json"
        if not latest_snapshot_path.exists():
            raise SnapshotRestoreError(f"latest snapshot pointer not found: {latest_snapshot_path}")
        latest_payload = _read_json(latest_snapshot_path)
        snapshot_dir_text = str(latest_payload.get("snapshot_dir", "") or "").strip()
        if not snapshot_dir_text:
            raise SnapshotRestoreError(f"latest snapshot pointer missing snapshot_dir: {latest_snapshot_path}")
        snapshot_dir = Path(snapshot_dir_text)
        return snapshot_dir, latest_snapshot_path

    raw = str(snapshot or "").strip()
    if not raw:
        raise SnapshotRestoreError("snapshot argument is empty")
    candidate = Path(raw)
    if not candidate.is_absolute():
        if candidate.exists():
            resolved = candidate.resolve()
        else:
            resolved = (snapshot_root / raw).resolve()
    else:
        resolved = candidate
    if resolved.is_file():
        if resolved.name.lower() == "snapshot_manifest.json":
            resolved = resolved.parent
        else:
            raise SnapshotRestoreError(f"snapshot path points to a file, expected snapshot dir: {resolved}")
    return resolved, Path("")


def _resolve_snapshot_manifest_path(snapshot_dir: Path) -> Path:
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not manifest_path.exists():
        raise SnapshotRestoreError(f"snapshot manifest not found: {manifest_path}")
    return manifest_path


def restore_stage2_snapshot(
    run_id: str,
    snapshot: str = "latest",
    strict: bool = False,
    output_root: Path = Path("safe_rl_output"),
) -> Dict[str, Any]:
    run_id_text = str(run_id or "").strip()
    if not run_id_text:
        raise SnapshotRestoreError("run_id is required")

    run_root = output_root / "runs" / run_id_text
    reports_dir = run_root / "reports"
    models_dir = run_root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    audit_path = reports_dir / "stage2_active_snapshot_restore.json"

    audit: Dict[str, Any] = {
        "run_id": run_id_text,
        "requested_snapshot": str(snapshot),
        "strict": bool(strict),
        "restored_at": dt.datetime.now().isoformat(timespec="seconds"),
        "status": "failed",
        "snapshot_source": "",
        "source_snapshot_dir": "",
        "resolved_snapshot_dir": "",
        "resolved_snapshot_manifest_path": "",
        "resolved_latest_pointer_path": "",
        "restored_files": {},
        "errors": [],
    }

    try:
        if not run_root.exists():
            raise SnapshotRestoreError(f"run root not found: {run_root}")

        snapshot_dir, latest_pointer_path = _resolve_snapshot_dir(run_root=run_root, snapshot=snapshot)
        audit["resolved_snapshot_dir"] = str(snapshot_dir)
        audit["source_snapshot_dir"] = str(snapshot_dir)
        if str(latest_pointer_path):
            audit["resolved_latest_pointer_path"] = str(latest_pointer_path)
        if not snapshot_dir.exists():
            raise SnapshotRestoreError(f"snapshot dir not found: {snapshot_dir}")

        manifest_path = _resolve_snapshot_manifest_path(snapshot_dir)
        manifest = _read_json(manifest_path)
        audit["resolved_snapshot_manifest_path"] = str(manifest_path)
        if str(latest_pointer_path):
            try:
                latest_payload = _read_json(latest_pointer_path)
                audit["snapshot_source"] = str(latest_payload.get("snapshot_source", "") or "")
            except Exception:
                pass
        if not audit["snapshot_source"]:
            audit["snapshot_source"] = str(manifest.get("snapshot_source", "") or "")

        manifest_run_id = str(manifest.get("run_id", "") or "").strip()
        if manifest_run_id and manifest_run_id != run_id_text:
            raise SnapshotRestoreError(
                f"snapshot run_id mismatch: snapshot={manifest_run_id}, requested={run_id_text}"
            )
        if strict and str(manifest.get("snapshot_type", "") or "").strip().lower() != "stage2_healthy":
            raise SnapshotRestoreError(
                f"strict restore requires snapshot_type=stage2_healthy, got={manifest.get('snapshot_type')}"
            )

        def _path_from_manifest(key: str, fallback_name: str) -> Path:
            path_text = str(manifest.get(key, "") or "").strip()
            if path_text:
                return Path(path_text)
            return snapshot_dir / fallback_name

        source_stage2_report = _path_from_manifest("stage2_training_report_path", "stage2_training_report.json")
        source_world_model = _path_from_manifest("world_model_path", "world_model.pt")
        source_light_model = _path_from_manifest("light_model_path", "light_risk.pt")
        source_risk_summary = _path_from_manifest("risk_v2_eval_summary_path", "risk_v2_eval_summary.json")

        required_sources = {
            "stage2_training_report": source_stage2_report,
            "world_model": source_world_model,
            "light_model": source_light_model,
        }
        for label, path in required_sources.items():
            if not path.exists():
                raise SnapshotRestoreError(f"snapshot required file missing ({label}): {path}")
            if strict and not str(path.resolve()).startswith(str(snapshot_dir.resolve())):
                raise SnapshotRestoreError(f"strict restore rejects file outside snapshot dir ({label}): {path}")

        targets = {
            "stage2_training_report": reports_dir / "stage2_training_report.json",
            "world_model": models_dir / "world_model.pt",
            "light_model": models_dir / "light_risk.pt",
        }
        optional_targets = {
            "risk_v2_eval_summary": reports_dir / "risk_v2_eval_summary.json",
        }

        restored_files: Dict[str, Any] = {}
        for label, source_path in required_sources.items():
            target_path = targets[label]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            restored_files[label] = {
                "source": str(source_path),
                "target": str(target_path),
                "restored": True,
            }

        if source_risk_summary.exists():
            target_path = optional_targets["risk_v2_eval_summary"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_risk_summary, target_path)
            restored_files["risk_v2_eval_summary"] = {
                "source": str(source_risk_summary),
                "target": str(target_path),
                "restored": True,
            }
        else:
            restored_files["risk_v2_eval_summary"] = {
                "source": str(source_risk_summary),
                "target": str(optional_targets["risk_v2_eval_summary"]),
                "restored": False,
            }

        audit["status"] = "restored"
        audit["restored_files"] = restored_files
        _write_json(audit_path, audit)
        return audit
    except Exception as exc:
        audit["status"] = "failed"
        audit["errors"] = [str(exc)]
        _write_json(audit_path, audit)
        if isinstance(exc, SnapshotRestoreError):
            raise
        raise SnapshotRestoreError(str(exc))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore Stage2 healthy snapshot artifacts into active run paths.")
    parser.add_argument("--run-id", required=True, help="Run id under safe_rl_output/runs/<run_id>")
    parser.add_argument(
        "--snapshot",
        default="latest",
        help="Snapshot selector: latest or explicit snapshot dir path (or snapshot_manifest.json path).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation (snapshot_type, in-snapshot file path checks).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    try:
        result = restore_stage2_snapshot(
            run_id=str(args.run_id),
            snapshot=str(args.snapshot),
            strict=bool(args.strict),
        )
        print(
            json.dumps(
                {
                    "status": result.get("status", "unknown"),
                    "run_id": result.get("run_id", ""),
                    "resolved_snapshot_dir": result.get("resolved_snapshot_dir", ""),
                    "audit_path": str(
                        Path("safe_rl_output")
                        / "runs"
                        / str(args.run_id)
                        / "reports"
                        / "stage2_active_snapshot_restore.json"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"[restore_stage2_snapshot] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
