import json
import uuid
from pathlib import Path

import pytest

from run_safe_rl_v2_pipeline import main as runner_main
from run_safe_rl_v2_pipeline import (
    print_stage2_resolution_progress,
    print_stage2_probe_progress,
    summarize_stage2_probe_progress,
    summarize_stage2_resolution_progress,
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


def test_summarize_stage2_resolution_progress_reports_ordered_checks(capsys):
    payload = {
        "pair_finetune_metrics": {
            "world": {
                "epoch_metrics": [
                    {
                        "stage4_aux_active_pair_count": 0.0,
                        "stage4_aux_resolution_loss": 0.0,
                        "stage4_aux_below_score_margin_fraction": 0.0,
                    },
                    {
                        "stage4_aux_active_pair_count": 64.0,
                        "stage4_aux_resolution_loss": 0.12,
                        "stage4_aux_below_score_margin_fraction": 0.75,
                    },
                    {
                        "stage4_aux_active_pair_count": 64.0,
                        "stage4_aux_resolution_loss": 0.08,
                        "stage4_aux_below_score_margin_fraction": 0.40,
                    },
                ]
            }
        },
        "stage4_aux_unique_score_count_before_after": {"after": 9.0},
        "world_pair_ft_best_epoch": 1,
    }
    summary = summarize_stage2_resolution_progress(payload)
    assert summary["has_active_pairs"] is True
    assert summary["has_resolution_loss"] is True
    assert summary["has_below_score_margin"] is True
    assert summary["below_score_margin_trend_down"] is True
    assert summary["stage4_aux_unique_after"] == 9.0
    assert summary["world_pair_ft_best_epoch"] == 1

    print_stage2_resolution_progress(summary)
    captured = capsys.readouterr()
    assert "Stage2 score-space resolution checks" in captured.out
    assert "active_pairs>0: True" in captured.out
    assert "resolution_loss>0: True" in captured.out


def test_summarize_stage2_probe_progress_reports_main_metrics(capsys):
    payload = {
        "stage1_probe_pairs_created": 312,
        "stage1_probe_unique_score_count_before_after": {"before": 9.0, "after": 11.0},
        "ranking_metrics": {"world": {"unique_score_count": 11.0}},
        "model_quality_metric_source": "stage1_probe",
        "stage2_pair_source_health": {"model_quality": {"status": "degraded"}},
    }
    summary = summarize_stage2_probe_progress(payload)
    assert summary["stage1_probe_pairs_created"] == 312
    assert summary["stage1_probe_unique_after"] == 11.0
    assert summary["world_unique_score_count"] == 11.0
    assert summary["model_quality_status"] == "degraded"
    assert summary["model_quality_metric_source"] == "stage1_probe"

    print_stage2_probe_progress(summary)
    captured = capsys.readouterr()
    assert "Stage2 Stage1-probe recovery checks" in captured.out
    assert "stage1_probe_pairs_created: 312" in captured.out
    assert "stage1_probe_unique_after: 11.0" in captured.out
    assert "world_unique_score_count: 11.0" in captured.out


def test_runner_stage1_probe_recovery_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_safe_rl_v2_pipeline.py",
            "--mode",
            "stage1_probe_recovery",
            "--run-id",
            "20260414_200057",
            "--dry-run",
        ],
    )
    code = runner_main()
    captured = capsys.readouterr()
    assert code == 0
    assert "Mode: stage1_probe_recovery" in captured.out
    assert "--stage stage1" in captured.out
    assert "--stage stage2" in captured.out
    assert "Stage5 is intentionally not attempted in this mode by default." in captured.out


def test_runner_stage1_probe_recovery_executes_stage1_stage2_only(monkeypatch, capsys):
    calls = []

    def _fake_run_step(repo_root, step_name, command, dry_run):
        _ = repo_root
        calls.append((step_name, list(command), dry_run))
        return 0

    def _fake_stage2_report_path(repo_root, run_id):
        _ = repo_root
        return _local_tmp_dir(f"stage2_report_{run_id}") / "stage2_training_report.json"

    report_path = _fake_stage2_report_path(Path("."), "20260414_200057")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "stage1_probe_pairs_created": 300,
                "stage1_probe_unique_score_count_before_after": {"after": 10.0},
                "ranking_metrics": {"world": {"unique_score_count": 10.0}},
                "model_quality_metric_source": "stage1_probe",
                "stage2_pair_source_health": {"model_quality": {"status": "critical"}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("run_safe_rl_v2_pipeline.run_step", _fake_run_step)
    monkeypatch.setattr("run_safe_rl_v2_pipeline._stage2_report_path", _fake_stage2_report_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_safe_rl_v2_pipeline.py",
            "--mode",
            "stage1_probe_recovery",
            "--run-id",
            "20260414_200057",
        ],
    )
    code = runner_main()
    captured = capsys.readouterr()
    assert code == 0
    assert len(calls) == 2
    assert "--stage" in calls[0][1]
    assert calls[0][1][calls[0][1].index("--stage") + 1] == "stage1"
    assert calls[1][1][calls[1][1].index("--stage") + 1] == "stage2"
    assert all("--stage stage5" not in " ".join(item[1]) for item in calls)
    assert "Stage1-probe recovery sequence completed" in captured.out
