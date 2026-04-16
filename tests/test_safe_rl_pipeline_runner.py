import json
import uuid
from pathlib import Path

from run_safe_rl_v2_pipeline import (
    should_run_stage5_from_stage2_report_path,
    should_run_stage5_from_stage2_report_payload,
)


def test_should_run_stage5_from_payload_accepts_degraded_and_healthy():
    degraded_payload = {
        "stage2_pair_source_health": {
            "model_quality": {
                "status": "degraded",
                "message": "ok to continue",
            }
        }
    }
    healthy_payload = {
        "stage2_pair_source_health": {
            "model_quality": {
                "status": "healthy",
            }
        }
    }
    assert should_run_stage5_from_stage2_report_payload(degraded_payload)[0] is True
    assert should_run_stage5_from_stage2_report_payload(healthy_payload)[0] is True


def test_should_run_stage5_from_payload_blocks_critical():
    payload = {
        "stage2_pair_source_health": {
            "model_quality": {
                "status": "critical",
                "message": "blocked by quality gate",
            }
        }
    }
    allowed, status, reason = should_run_stage5_from_stage2_report_payload(payload)
    assert allowed is False
    assert status == "critical"
    assert "blocked by quality gate" in reason


def _local_tmp_dir(tag: str) -> Path:
    path = Path("safe_rl_output/test_artifacts") / f"{tag}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_should_run_stage5_from_report_path_missing_or_invalid():
    tmp_dir = _local_tmp_dir("pipeline_runner")
    missing_path = tmp_dir / "missing_stage2_training_report.json"
    allowed_missing, status_missing, reason_missing = should_run_stage5_from_stage2_report_path(missing_path)
    assert allowed_missing is False
    assert status_missing == "missing"
    assert "missing_stage2_training_report" in reason_missing

    invalid_path = tmp_dir / "invalid_stage2_training_report.json"
    invalid_path.write_text("{bad json", encoding="utf-8")
    allowed_invalid, status_invalid, reason_invalid = should_run_stage5_from_stage2_report_path(invalid_path)
    assert allowed_invalid is False
    assert status_invalid == "invalid"
    assert "failed_to_parse_stage2_training_report" in reason_invalid


def test_should_run_stage5_from_report_path_reads_json():
    tmp_dir = _local_tmp_dir("pipeline_runner_report")
    report_path = tmp_dir / "stage2_training_report.json"
    report_path.write_text(
        json.dumps(
            {
                "stage2_pair_source_health": {
                    "model_quality": {
                        "status": "degraded",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    allowed, status, reason = should_run_stage5_from_stage2_report_path(report_path)
    assert allowed is True
    assert status == "degraded"
    assert reason == ""
