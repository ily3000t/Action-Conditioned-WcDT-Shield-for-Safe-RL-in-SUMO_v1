import json
import math
import pickle
import uuid
from pathlib import Path

import pytest

from safe_rl.visualization.export_paired_gif import export_paired_gifs
from safe_rl.visualization.replay_in_sumo_gui import write_gui_replay_override_file
from safe_rl.visualization.replay_episode import load_pair_payload, normalize_heading_to_degrees, render_pair_gif
from safe_rl.visualization.select_anomaly_cases import select_anomaly_cases


def _tmp_dir(tag: str) -> Path:
    path = Path("safe_rl_output/test_artifacts") / f"{tag}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pair_payload(seed: int = 42, include_distilled: bool = False):
    payload = {
        "pair_index": 0,
        "seed": seed,
        "baseline_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 4,
                "raw_risk": 0.2,
                "final_risk": 0.2,
                "min_ttc": 4.0,
                "min_distance": 12.0,
                "task_reward": 0.20,
                "history_scene": [
                    {
                        "timestamp": 0.0,
                        "ego_id": "ego",
                        "vehicles": [{"vehicle_id": "ego", "x": 0.0, "y": 0.0, "vx": 10.0, "heading": 0.0}],
                        "lane_polylines": [[[0.0, 0.0], [100.0, 0.0]]],
                    }
                ],
            }
        ],
        "shielded_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 4,
                "raw_risk": 0.25,
                "final_risk": 0.24,
                "min_ttc": 1.2,
                "min_distance": 2.6,
                "replacement_happened": False,
                "constraint_reason": "blocked_by_margin",
                "task_reward": 0.12,
                "history_scene": [
                    {
                        "timestamp": 0.0,
                        "ego_id": "ego",
                        "vehicles": [{"vehicle_id": "ego", "x": 2.0, "y": 0.0, "vx": -0.8, "heading": math.pi / 2}],
                        "lane_polylines": [[[0.0, 0.0], [100.0, 0.0]]],
                    }
                ],
            }
        ],
    }
    if include_distilled:
        payload["distilled_episode_id"] = "distilled_ep"
        payload["distilled_steps"] = [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 4,
                "raw_risk": 0.22,
                "final_risk": 0.20,
                "min_ttc": 2.0,
                "min_distance": 5.0,
                "task_reward": 0.18,
                "history_scene": [
                    {
                        "timestamp": 0.0,
                        "ego_id": "ego",
                        "vehicles": [{"vehicle_id": "ego", "x": 1.0, "y": 0.0, "vx": 8.0, "heading": 0.0}],
                        "lane_polylines": [[[0.0, 0.0], [100.0, 0.0]]],
                    }
                ],
            }
        ]
    return payload


def test_heading_unit_normalization_supports_deg_and_rad():
    assert normalize_heading_to_degrees(90.0) == 90.0
    assert normalize_heading_to_degrees(math.pi / 2.0) == 90.0


def test_render_pair_gif_smoke_and_distilled_fallback():
    tmp_dir = _tmp_dir("viz_render")
    pair_path = tmp_dir / "pair_00_seed_42.json"
    pair_path.write_text(json.dumps(_pair_payload(include_distilled=False), ensure_ascii=False), encoding="utf-8")

    payload = load_pair_payload(pair_path)
    assert payload["distilled_unavailable"] is True
    assert payload["distilled_steps"] == []
    assert set(payload["aligned_steps"][0].keys()) == {"step_index", "baseline", "shielded", "distilled"}

    output_path = tmp_dir / "pair_00_seed_42.gif"
    render_pair_gif(payload, output_path=output_path, mode="auto", fps=4)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_select_anomaly_cases_rules_and_sorting():
    run_id = f"ut_viz_anomaly_{uuid.uuid4().hex[:8]}"
    run_dir = _tmp_dir("viz_anomaly_run") / run_id
    trace_dir = run_dir / "reports" / "stage5_trace_capture_default"
    trace_dir.mkdir(parents=True, exist_ok=True)

    pair_bad = _pair_payload(seed=42, include_distilled=False)
    pair_bad["shielded_steps"] = pair_bad["shielded_steps"] * 90
    pair_bad["baseline_steps"] = pair_bad["baseline_steps"] * 90
    pair_bad_path = trace_dir / "pair_00_seed_42.json"
    pair_bad_path.write_text(json.dumps(pair_bad, ensure_ascii=False), encoding="utf-8")

    pair_ok = _pair_payload(seed=123, include_distilled=True)
    pair_ok["shielded_steps"][0]["constraint_reason"] = ""
    pair_ok["shielded_steps"][0]["replacement_happened"] = True
    pair_ok["shielded_steps"][0]["min_ttc"] = 4.0
    pair_ok["shielded_steps"][0]["min_distance"] = 10.0
    pair_ok["shielded_steps"][0]["task_reward"] = 0.19
    pair_ok_path = trace_dir / "pair_01_seed_123.json"
    pair_ok_path.write_text(json.dumps(pair_ok, ensure_ascii=False), encoding="utf-8")

    (trace_dir / "trace_summary.json").write_text(
        json.dumps({"pair_files": [str(pair_bad_path), str(pair_ok_path)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = select_anomaly_cases(
        run_id=run_id,
        trace_dir=str(trace_dir),
        run_root=run_dir.parent,
        top_k=5,
        output_root=Path("qualitative_results/anomaly_cases"),
    )
    assert result["selected_count"] >= 1
    assert Path(result["output_path"]).exists()
    first = result["cases"][0]
    assert Path(first["pair_file"]).name == "pair_00_seed_42.json"
    assert "high_blocked_low_replacement" in first["matched_rules"]
    assert "high_near_risk" in first["matched_rules"]


def test_export_paired_gifs_tolerates_old_payload_without_distilled_steps():
    run_id = f"ut_viz_export_{uuid.uuid4().hex[:8]}"
    run_root = _tmp_dir("viz_export_run")
    trace_dir = run_root / run_id / "reports" / "stage5_trace_capture_default"
    trace_dir.mkdir(parents=True, exist_ok=True)

    legacy_payload = _pair_payload(seed=42, include_distilled=False)
    pair_path = trace_dir / "pair_00_seed_42.json"
    pair_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")
    (trace_dir / "trace_summary.json").write_text(
        json.dumps({"pair_files": [str(pair_path)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = export_paired_gifs(
        run_id=run_id,
        trace_dir=str(trace_dir),
        run_root=run_root,
        top_k=1,
        output_root=Path("qualitative_results/stage5_replays"),
        fps=4,
        mode="auto",
    )
    assert summary["exported_count"] == 1
    assert summary["failed_count"] == 0
    assert Path(summary["index_path"]).exists()
    exported = summary["exported"][0]
    assert Path(exported["gif_path"]).exists()
    assert exported["distilled_unavailable"] is True


def test_select_anomaly_cases_fallback_mode_when_no_rule_match():
    run_id = f"ut_viz_fallback_{uuid.uuid4().hex[:8]}"
    run_root = _tmp_dir("viz_fallback_run")
    trace_dir = run_root / run_id / "reports" / "stage5_trace_capture_default"
    trace_dir.mkdir(parents=True, exist_ok=True)

    payload = _pair_payload(seed=42, include_distilled=True)
    payload["shielded_steps"][0]["constraint_reason"] = ""
    payload["shielded_steps"][0]["block_trigger"] = "none"
    payload["shielded_steps"][0]["replacement_happened"] = False
    payload["shielded_steps"][0]["min_ttc"] = 8.0
    payload["shielded_steps"][0]["min_distance"] = 15.0
    payload["shielded_steps"][0]["task_reward"] = 0.20
    payload["baseline_steps"][0]["task_reward"] = 0.20
    payload["baseline_reward"] = 0.20
    payload["shielded_reward"] = 0.19  # only mild penalized drop

    pair_path = trace_dir / "pair_00_seed_42.json"
    pair_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (trace_dir / "trace_summary.json").write_text(
        json.dumps({"pair_files": [str(pair_path)]}, ensure_ascii=False),
        encoding="utf-8",
    )

    result = select_anomaly_cases(
        run_id=run_id,
        trace_dir=str(trace_dir),
        run_root=run_root,
        top_k=5,
        output_root=Path("qualitative_results/anomaly_cases"),
    )
    assert result["selected_count"] == 1
    assert result["selection_mode"] == "fallback_top_signal"
    assert "fallback_top_signal" in result["cases"][0]["matched_rules"]


def test_replay_in_sumo_gui_writes_merged_override_with_trace_scenario():
    run_id = f"ut_viz_gui_{uuid.uuid4().hex[:8]}"
    run_root = _tmp_dir("viz_gui_run")
    trace_dir = run_root / "safe_rl_output" / "runs" / run_id / "reports" / "stage5_trace_capture_default"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_path = trace_dir / "pair_00_seed_123.json"
    payload = _pair_payload(seed=123, include_distilled=True)
    payload["scenario_source"] = "scenarios/highway_merge/highway_merge.sumocfg"
    pair_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # create minimal base config under current workspace
    base_config = run_root / "base_gui.yaml"
    base_config.write_text("sim:\n  use_gui: false\n  sumo_cfg: scenarios/highway_merge/highway_merge.sumocfg\n", encoding="utf-8")

    cwd = Path.cwd()
    try:
        # script resolves trace under safe_rl_output/runs/<run_id>/reports
        # symlink-like structure by switching cwd to temp run root folder
        # keep test portable by creating expected relative tree and changing cwd.
        (run_root / "safe_rl_output").mkdir(exist_ok=True)
        # Already created full path above, now execute from run_root.
        import os

        os.chdir(run_root)
        override = write_gui_replay_override_file(
            run_id=run_id,
            seed=123,
            mode="shielded",
            base_config_path=Path("base_gui.yaml"),
            trace_dir="stage5_trace_capture_default",
            output_dir=Path("gui_overrides"),
        )
        data = json.loads(json.dumps(__import__("yaml").safe_load(override.read_text(encoding="utf-8"))))
        assert data["sim"]["use_gui"] is True
        assert data["sim"]["sumo_cfg"] == "scenarios/highway_merge/highway_merge.sumocfg"
        assert data["eval"]["eval_episodes"] == 1
        assert data["eval"]["seed_list"] == [123]
    finally:
        import os

        os.chdir(cwd)


def _build_stage1_history(history_steps: int, base_x: float):
    from safe_rl.data.types import SceneState, VehicleState

    history = []
    for step in range(history_steps):
        ego = VehicleState(
            vehicle_id="ego",
            x=base_x + 1.5 * float(step),
            y=0.0,
            vx=10.0,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            heading=0.0,
            lane_id=1,
        )
        history.append(
            SceneState(
                timestamp=0.1 * float(step),
                ego_id="ego",
                vehicles=[ego],
                lane_polylines=[[[0.0, 0.0], [120.0, 0.0]]],
            )
        )
    return history


def _make_stage1_probe_pair(history_steps: int, idx: int, target_gap: float):
    from safe_rl.data.types import RiskPairSample

    target_risk_a = 0.50 + float(target_gap) / 2.0
    target_risk_b = 0.50 - float(target_gap) / 2.0
    return RiskPairSample(
        history_scene=_build_stage1_history(history_steps=history_steps, base_x=float(idx)),
        action_a=1,
        action_b=2,
        preferred_action=2,
        source="stage1_probe_same_state",
        meta={
            "episode_id": f"ep_{idx:05d}",
            "step_index": idx,
            "history_hash": f"h{idx}",
            "target_risk_a": float(target_risk_a),
            "target_risk_b": float(target_risk_b),
            "target_gap": float(target_gap),
            "trusted_for_spread": bool(idx % 2 == 0),
            "boundary_pair": bool(idx % 3 == 0),
        },
    )


def _create_stage1_probe_run_fixture(run_root: Path, run_id: str, target_gaps: list[float]) -> Path:
    pytest.importorskip("torch")
    from safe_rl.config import load_safe_rl_config
    from safe_rl.models.world_model import WorldModelTrainer

    run_dir = Path(run_root) / run_id
    (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    pairs = [
        _make_stage1_probe_pair(history_steps=int(config.sim.history_steps), idx=idx, target_gap=gap)
        for idx, gap in enumerate(target_gaps)
    ]
    with (run_dir / "datasets" / "pairs_stage1_probe.pkl").open("wb") as f:
        pickle.dump(pairs, f)

    trainer = WorldModelTrainer(config=config.world_model, history_steps=int(config.sim.history_steps), device="cpu")
    trainer.save(str(run_dir / "models" / "world_model.pt"))

    stage2_report = {
        "pair_finetune_metrics": {
            "world": {
                "selection_path": "legacy_tieaware",
                "selection_reason": "legacy_stage1_probe_unique_higher",
                "best_epoch_stage1_unique": 10.0,
                "best_epoch_eval_unique": 10.0,
                "stage1_probe_unique_score_count_before_after": {"before": 9.0, "after": 10.0},
                "epoch_metrics": [
                    {"stage1_probe_below_score_margin_fraction": 0.55},
                    {"stage1_probe_below_score_margin_fraction": 0.40},
                ],
            }
        },
        "stage2_pair_source_health": {"model_quality": {"status": "critical", "metric_source": "stage1_probe"}},
        "model_quality_gate_metrics": {"world_unique_score_count": 10.0},
    }
    (run_dir / "reports" / "stage2_training_report.json").write_text(
        json.dumps(stage2_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_stage1_probe_diagnostics_smoke_with_optional_baseline():
    pytest.importorskip("torch")
    from safe_rl.visualization.stage1_probe_diagnostics import generate_stage1_probe_diagnostics

    run_root = _tmp_dir("viz_stage1_diag_run_root")
    current_run_id = f"ut_viz_stage1_diag_cur_{uuid.uuid4().hex[:8]}"
    baseline_run_id = f"ut_viz_stage1_diag_base_{uuid.uuid4().hex[:8]}"

    _create_stage1_probe_run_fixture(
        run_root=run_root,
        run_id=current_run_id,
        target_gaps=[0.009, 0.011, 0.013, 0.017, 0.024, 0.031],
    )
    _create_stage1_probe_run_fixture(
        run_root=run_root,
        run_id=baseline_run_id,
        target_gaps=[0.009, 0.012, 0.014, 0.018, 0.023, 0.028],
    )

    result = generate_stage1_probe_diagnostics(
        run_id=current_run_id,
        baseline_run_id=baseline_run_id,
        run_root=run_root,
        output_root=Path("qualitative_results/stage1_probe_diagnostics"),
        config_path="safe_rl/config/default_safe_rl.yaml",
        device="cpu",
        batch_size=4,
    )
    assert Path(result["output_path"]).exists()
    assert result["current"]["record_count"] > 0
    assert "by_gap_bin" in result["current"]["summary"]
    assert "gap_0.008_0.012" in result["current"]["summary"]["by_gap_bin"]
    assert Path(result["current"]["plots"]["target_gap_hist_overall"]).exists()
    assert "baseline" in result and result["baseline"].get("run_id") == baseline_run_id
    assert Path(result["comparison_plots"]["current_vs_baseline_predicted_gap_hist"]).exists()


def test_select_stage1_probe_samples_gap_bucket_export():
    pytest.importorskip("torch")
    from safe_rl.visualization.select_stage1_probe_samples import select_stage1_probe_samples

    run_root = _tmp_dir("viz_stage1_samples_run_root")
    run_id = f"ut_viz_stage1_samples_{uuid.uuid4().hex[:8]}"
    _create_stage1_probe_run_fixture(
        run_root=run_root,
        run_id=run_id,
        target_gaps=[0.009, 0.010, 0.0125, 0.016, 0.021, 0.029, 0.036],
    )

    payload = select_stage1_probe_samples(
        run_id=run_id,
        run_root=run_root,
        output_root=Path("qualitative_results/stage1_probe_diagnostics"),
        config_path="safe_rl/config/default_safe_rl.yaml",
        device="cpu",
        batch_size=4,
        top_k=5,
    )
    assert payload["selected_count"] > 0
    assert Path(payload["output_path"]).exists()
    assert sum(payload["selected_count_by_gap_bin"].values()) == payload["selected_count"]
    assert all("gap_bin" in item for item in payload["cases"])
