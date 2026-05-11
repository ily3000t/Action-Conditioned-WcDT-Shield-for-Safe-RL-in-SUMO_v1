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


def _create_stage1_data_audit_fixture(run_root: Path, run_id: str) -> Path:
    from safe_rl.data.types import RiskPairSample

    run_dir = Path(run_root) / run_id
    (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    pairs = []
    for idx in range(6):
        risk_a = 0.65 + 0.05 * float(idx)
        risk_b = 0.62 + 0.03 * float(idx)
        pairs.append(
            RiskPairSample(
                history_scene=_build_stage1_history(history_steps=3, base_x=float(idx)),
                action_a=0,
                action_b=8,
                preferred_action=(0 if idx % 2 == 0 else 8),
                source="stage1_probe_same_state",
                meta={
                    "episode_id": f"ep_{idx:05d}",
                    "step_index": int(10 + idx),
                    "target_risk_a": float(risk_a),
                    "target_risk_b": float(risk_b),
                    "target_gap": float(abs(risk_a - risk_b)),
                    "trusted_for_spread": bool(idx % 2 == 0),
                    "boundary_pair": bool(idx % 3 == 0),
                },
            )
        )

    with (run_dir / "datasets" / "pairs_stage1_probe.pkl").open("wb") as f:
        pickle.dump(pairs, f)

    events = {
        "events": [
            {
                "episode_id": "ep_00000",
                "step_index": 10,
                "status": "ok",
                "pairs_created": 2,
                "candidate_count": 3,
                "candidates": [
                    {"candidate_action": 0, "overall_proxy_risk": 0.72, "min_ttc": 3.1, "min_distance": 9.0},
                    {"candidate_action": 4, "overall_proxy_risk": 0.88, "min_ttc": 2.1, "min_distance": 5.0},
                    {"candidate_action": 8, "overall_proxy_risk": 1.00, "min_ttc": 1.0, "min_distance": 1.0},
                ],
            }
        ]
    }
    (run_dir / "reports" / "stage1_probe_events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "pairs_created": 6,
        "pairs_dropped_small_gap": 30,
        "pairs_kept_strong_signal": 4,
        "pairs_capped_by_budget": 2,
        "pairs_boundary_appended": 1,
    }
    (run_dir / "reports" / "stage1_probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def test_stage1_data_audit_generates_required_artifacts_and_summary_fields():
    from safe_rl.visualization.stage1_data_audit import generate_stage1_data_audit

    run_root = _tmp_dir("viz_stage1_audit")
    run_id = f"ut_viz_stage1_audit_{uuid.uuid4().hex[:8]}"
    _create_stage1_data_audit_fixture(run_root=run_root, run_id=run_id)

    result = generate_stage1_data_audit(
        run_id=run_id,
        run_root=run_root,
        output_root=Path("qualitative_results/stage1_data_audit"),
        bins=16,
    )

    assert Path(result["output_path"]).exists()
    for key in (
        "target_risk_hist_ecdf",
        "target_risk_bin_occupancy",
        "target_risk_trusted_vs_all_bin_occupancy",
        "candidate_risk_hist",
        "candidate_bin_occupancy",
        "pair_target_gap_hist",
        "pair_bin_heatmap",
    ):
        assert Path(result["files"][key]).exists()

    summary = dict(result["summary"] or {})
    required = {
        "pair_count",
        "candidate_count",
        "pairs_dropped_small_gap",
        "pairs_kept_strong_signal",
        "pairs_capped_by_budget",
        "pairs_boundary_appended",
        "target_risk_q01",
        "target_risk_q10",
        "target_risk_q50",
        "target_risk_q90",
        "target_risk_q99",
        "pair_all_bin_nonempty",
        "pair_all_bin_effective",
        "pair_trusted_bin_nonempty",
        "pair_trusted_bin_effective",
        "candidate_bin_nonempty",
        "candidate_bin_effective",
        "preferred_a",
        "preferred_b",
        "tie_like",
    }
    assert required.issubset(set(summary.keys()))
    assert int(summary["pair_all_bin_nonempty"]) <= 16
    assert float(summary["pair_all_bin_effective"]) >= 1.0
    assert int(summary["preferred_a"]) + int(summary["preferred_b"]) + int(summary["tie_like"]) == int(summary["pair_count"])


def test_stage1_data_audit_missing_files_fail_fast():
    from safe_rl.visualization.stage1_data_audit import generate_stage1_data_audit

    run_root = _tmp_dir("viz_stage1_audit_missing")
    run_id = f"ut_viz_stage1_missing_{uuid.uuid4().hex[:8]}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        generate_stage1_data_audit(run_id=run_id, run_root=run_root)


class _Stage1ReplayBackend:
    def __init__(self):
        from safe_rl.data.types import SceneState, VehicleState
        from safe_rl.sim.backend_interface import BackendStepResult

        self._SceneState = SceneState
        self._VehicleState = VehicleState
        self._BackendStepResult = BackendStepResult
        self._step_index = 0
        self._last_action = 4
        self.injected_events = []

    def start(self):
        return None

    def close(self):
        return None

    def set_episode_context(self, episode_id: str, risky_mode: bool):
        _ = (episode_id, risky_mode)

    def reset(self, seed=None):
        _ = seed
        self._step_index = 0
        self._last_action = 4
        return self.get_state()

    def inject_risk_event(self, event_type=None):
        self.injected_events.append(event_type)
        return None

    def _scene(self, action_id: int):
        gap = 1.0 if int(action_id) == 8 else 10.0
        ego_x = 2.0 * float(self._step_index)
        return self._SceneState(
            timestamp=0.1 * float(self._step_index),
            ego_id="ego",
            vehicles=[
                self._VehicleState("ego", ego_x, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1),
                self._VehicleState("lead", ego_x + gap, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1),
            ],
        )

    def get_state(self):
        return self._scene(self._last_action)

    def step(self, action_id: int):
        self._last_action = int(action_id)
        self._step_index += 1
        collision = bool(int(action_id) == 8)
        return self._BackendStepResult(
            scene=self._scene(int(action_id)),
            task_reward=1.0 - 0.1 * float(abs(int(action_id) - 4)),
            done=self._step_index >= 8,
            info={"collision": collision, "lane_violation": bool(int(action_id) in (7, 8)), "teleport": collision},
        )


def _create_stage1_raw_episode_fixture(run_root: Path, run_id: str, episode_id: str = "ep_00037") -> Path:
    run_dir = Path(run_root) / run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_id": episode_id,
        "risky_mode": True,
        "meta": {
            "episode_seed": 7,
            "raw_action_prefix": [4, 4, 4, 4, 4, 4, 4, 4],
            "risk_event_schedule": [{"before_step": 2, "event_type": "hard_brake"}],
        },
        "steps": [{"step_index": idx} for idx in range(8)],
    }
    (raw_dir / f"{episode_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def test_stage1_sumo_replay_raw_and_compare_ab_outputs(monkeypatch):
    from safe_rl.visualization.stage1_sumo_replay import run_stage1_sumo_replay

    run_root = _tmp_dir("viz_stage1_replay")
    run_id = f"ut_viz_stage1_replay_{uuid.uuid4().hex[:8]}"
    _create_stage1_raw_episode_fixture(run_root=run_root, run_id=run_id)

    monkeypatch.setattr("safe_rl.visualization.stage1_sumo_replay.create_backend", lambda _sim_cfg: _Stage1ReplayBackend())

    raw_payload = run_stage1_sumo_replay(
        run_id=run_id,
        mode="raw_replay",
        episode_id="ep_00037",
        run_root=run_root,
        output_root=Path("qualitative_results/stage1_sumo_replay"),
        use_gui=False,
        until_step=5,
    )
    assert Path(raw_payload["output_path"]).exists()
    assert int(raw_payload["executed_steps"]) > 0
    assert raw_payload["trace"][0]["action_id"] == 4

    cmp_payload = run_stage1_sumo_replay(
        run_id=run_id,
        mode="compare_ab",
        episode_id="ep_00037",
        run_root=run_root,
        output_root=Path("qualitative_results/stage1_sumo_replay"),
        use_gui=False,
        step_index=2,
        action_a=0,
        action_b=8,
        horizon=4,
    )
    assert Path(cmp_payload["output_path"]).exists()
    assert "branch_a" in cmp_payload and "branch_b" in cmp_payload
    assert "comparison" in cmp_payload
    assert "overall_proxy_risk" in cmp_payload["branch_a"]["proxy_metrics"]


def test_stage1_sumo_replay_validate_step_and_horizon(monkeypatch):
    from safe_rl.visualization.stage1_sumo_replay import run_stage1_sumo_replay

    run_root = _tmp_dir("viz_stage1_replay_validate")
    run_id = f"ut_viz_stage1_replay_val_{uuid.uuid4().hex[:8]}"
    _create_stage1_raw_episode_fixture(run_root=run_root, run_id=run_id)
    monkeypatch.setattr("safe_rl.visualization.stage1_sumo_replay.create_backend", lambda _sim_cfg: _Stage1ReplayBackend())

    with pytest.raises(ValueError):
        run_stage1_sumo_replay(
            run_id=run_id,
            mode="compare_ab",
            episode_id="ep_00037",
            run_root=run_root,
            use_gui=False,
            step_index=2,
            action_a=0,
            action_b=8,
            horizon=0,
        )

    with pytest.raises(ValueError):
        run_stage1_sumo_replay(
            run_id=run_id,
            mode="compare_ab",
            episode_id="ep_00037",
            run_root=run_root,
            use_gui=False,
            step_index=99,
            action_a=0,
            action_b=8,
            horizon=3,
        )
