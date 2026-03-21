import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_rl.buffer import InterventionBuffer
from safe_rl.config.config import SafeRLConfig, ShieldSweepVariant
from safe_rl.pipeline.pipeline import SafeRLPipeline
from safe_rl.rl.ppo import HeuristicPolicy
from safe_rl_main import parse_args


def _tiny_config() -> SafeRLConfig:
    config = SafeRLConfig()
    config.sim.backend = "traci"
    config.sim.force_mock = True
    config.sim.episode_steps = 5
    config.sim.history_steps = 2
    config.sim.future_steps = 2
    config.sim.normal_episodes = 1
    config.sim.risky_episodes = 1
    config.sim.random_seed = 7

    config.dataset.train_ratio = 0.6
    config.dataset.val_ratio = 0.2

    config.light_risk.epochs = 1
    config.world_model.epochs = 1
    config.ppo.use_sb3 = False
    config.ppo.total_timesteps = 20
    config.eval.eval_episodes = 1

    config.tensorboard.enabled = False
    return config


def test_cli_stage_requires_run_id_for_single_stage():
    with pytest.raises(SystemExit):
        parse_args(["--stage", "stage2"])



def test_cli_stage_all_allows_missing_run_id():
    args = parse_args(["--stage", "all"])
    assert args.stage == "all"
    assert args.run_id is None



def test_stage4_missing_dependencies_fails_fast():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage4_{uuid.uuid4().hex[:8]}"

    with pytest.raises(FileNotFoundError):
        pipeline.run(stage="stage4", run_id=run_id)



def test_stage1_creates_manifest_and_datasets():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage1_{uuid.uuid4().hex[:8]}"

    result = pipeline.run(stage="stage1", run_id=run_id)
    run_root = result["run_root"]

    manifest_path = f"{run_root}/manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["stage_done"]["stage1"] is True
    assert manifest["stage_done"]["stage2"] is False

    assert pipeline.train_pkl.exists()
    assert pipeline.val_pkl.exists()
    assert pipeline.test_pkl.exists()
    assert Path(result["stage1"]["warning_summary_report"]).exists()



def test_policy_artifact_heuristic_roundtrip():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_policy_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="all", run_id=run_id)

    meta = pipeline._save_policy_artifact(HeuristicPolicy())
    assert meta["policy_type"] == "heuristic"

    loaded_policy = pipeline._load_policy_artifact()
    assert isinstance(loaded_policy, HeuristicPolicy)



def test_run_scoped_runtime_log_dir_and_report_paths():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_logs_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="all", run_id=run_id)

    scoped_config = pipeline._config_with_run_paths()

    assert Path(scoped_config.sim.runtime_log_dir) == pipeline.sumo_logs_dir
    assert pipeline.collector_failure_report_path.name == "collector_failures.json"
    assert pipeline.warning_summary_report_path.name == "warning_summary.json"
    assert pipeline.stage3_runtime_config_path.name == "stage3_runtime_config.json"
    assert pipeline.stage3_session_events_path.name == "stage3_session_events.json"
    assert pipeline.stage4_buffer_report_path.name == "stage4_buffer_report.json"
    assert pipeline.stage5_paired_episode_results_path.name == "stage5_paired_episode_results.json"



def test_stage3_writes_runtime_and_incremental_session_reports():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage3_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage3", run_id=run_id)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    pipeline.light_model_path.touch()
    pipeline.world_model_path.touch()
    pipeline._build_predictors_from_saved_models = lambda: (None, None)

    result = pipeline.run(stage="stage3", run_id=run_id)

    runtime_report_path = Path(result["stage3"]["runtime_report"])
    session_report_path = Path(result["stage3"]["session_events_report"])
    assert runtime_report_path.exists()
    assert session_report_path.exists()

    runtime_payload = json.loads(runtime_report_path.read_text(encoding="utf-8"))
    session_payload = json.loads(session_report_path.read_text(encoding="utf-8"))

    assert runtime_payload["sim_config"]["collision_action"] == "teleport"
    assert runtime_payload["backend"]["backend"] == "traci"
    assert session_payload["stage"] == "stage3"
    assert session_payload["metadata"]["runtime_report_path"].endswith("stage3_runtime_config.json")
    assert session_payload["metadata"]["runtime_report"]["sim_config"]["collision_action"] == "teleport"
    assert session_payload["event_count"] >= 3

    event_names = [event["event"] for event in session_payload["events"]]
    assert "episode_reset_started" in event_names
    assert "episode_reset_completed" in event_names
    assert "episode_completed" in event_names
    assert any(event.get("source") == "env" for event in session_payload["events"])
    assert any(event.get("episode_id", "").startswith("stage3_train_ep_") for event in session_payload["events"])



def test_stage4_result_includes_buffer_policy_metadata(monkeypatch):
    config = _tiny_config()
    config.eval.eval_episodes = 2
    config.eval.seed_list = [11, 22]
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage4_meta_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage4", run_id=run_id)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    pipeline.light_model_path.touch()
    pipeline.world_model_path.touch()
    pipeline._save_policy_artifact(HeuristicPolicy())
    pipeline._build_predictors_from_saved_models = lambda: (None, None)

    buffer = InterventionBuffer()
    pipeline.collect_interventions = lambda *args, **kwargs: buffer

    result = pipeline.run(stage="stage4", run_id=run_id)["stage4"]
    assert result["buffer_policy_source"].endswith("policy_meta.json")
    assert result["buffer_policy_type"] == "heuristic"
    assert result["buffer_eval_seeds"] == [11, 22]
    assert result["buffer_risky_mode"] is True
    assert result["buffer_scenario_source"] == config.sim.sumo_cfg
    assert Path(pipeline.stage4_buffer_report_path).exists()



def test_evaluate_uses_same_policy_for_shield_off_on_and_shared_seeds(monkeypatch):
    config = _tiny_config()
    config.eval.eval_episodes = 3
    config.eval.seed_list = [5, 9]
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_eval_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    class _DummyBackend:
        def start(self):
            return None

        def close(self):
            return None

    class _DummyEnv:
        def __init__(self, prefix, shield):
            self.prefix = prefix
            self.shield = shield

        def close(self):
            return None

    class _CaptureEvaluator:
        instance = None

        def __init__(self, cfg):
            _ = cfg
            self.calls = []
            _CaptureEvaluator.instance = self

        def evaluate_policy(self, env, policy, episodes, risky_mode=True, tb_writer=None, tb_prefix="", seeds=None):
            self.calls.append(
                {
                    "env": env,
                    "policy": policy,
                    "episodes": episodes,
                    "risky_mode": risky_mode,
                    "tb_prefix": tb_prefix,
                    "seeds": list(seeds or []),
                }
            )
            if tb_prefix == "baseline":
                return {
                    "collision_rate": 0.2,
                    "intervention_rate": 0.0,
                    "mean_risk_reduction": 0.0,
                    "mean_raw_risk": 0.6,
                    "mean_final_risk": 0.6,
                    "avg_speed": 10.0,
                    "mean_reward": 1.0,
                    "success_rate": 0.5,
                    "replacement_count": 0.0,
                    "episode_details": [
                        {"episode_id": "base_ep_0", "seed": 5, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 1, "mean_reward": 1.0, "mean_raw_risk": 0.6, "interventions": 0, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.0},
                        {"episode_id": "base_ep_1", "seed": 9, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.1, "mean_raw_risk": 0.5, "interventions": 0, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.0},
                        {"episode_id": "base_ep_2", "seed": 5, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 0.9, "mean_raw_risk": 0.4, "interventions": 0, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.0},
                    ],
                }
            return {
                "collision_rate": 0.1,
                "intervention_rate": 0.3,
                "mean_risk_reduction": 0.2,
                "mean_raw_risk": 0.7,
                "mean_final_risk": 0.5,
                "avg_speed": 11.0,
                "mean_reward": 1.5,
                "success_rate": 0.7,
                "replacement_count": 2.0,
                "episode_details": [
                    {"episode_id": "shield_ep_0", "seed": 5, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.5, "mean_raw_risk": 0.7, "mean_final_risk": 0.4, "interventions": 2, "replacement_count": 1, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.3},
                    {"episode_id": "shield_ep_1", "seed": 9, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.6, "mean_raw_risk": 0.6, "mean_final_risk": 0.5, "interventions": 1, "replacement_count": 1, "replacement_same_as_raw_count": 0, "fallback_action_count": 1, "mean_risk_reduction": 0.1},
                    {"episode_id": "shield_ep_2", "seed": 5, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.4, "mean_raw_risk": 0.5, "mean_final_risk": 0.4, "interventions": 0, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.1},
                ],
            }

        def compare_baseline_and_shielded(self, baseline, shielded):
            _ = (baseline, shielded)
            return {"collision_reduction": 0.5, "efficiency_drop": -0.1}

        def evaluate_acceptance(self, delta_metrics):
            _ = delta_metrics
            return True

        def evaluate_world_model(self, world_predictor, samples):
            _ = (world_predictor, samples)
            return {"traj_ade": 0.0, "risk_acc": 1.0, "risk_mae": 0.0}

    def _fake_create_backend(_sim_config):
        return _DummyBackend()

    def _fake_create_env(_backend, _sim_config, _ppo_config, shield=None, episode_prefix=""):
        return _DummyEnv(episode_prefix, shield)

    monkeypatch.setattr("safe_rl.pipeline.pipeline.create_backend", _fake_create_backend)
    monkeypatch.setattr("safe_rl.pipeline.pipeline.SafeRLEvaluator", _CaptureEvaluator)
    monkeypatch.setattr("safe_rl.rl.env.create_env", _fake_create_env)

    policy = object()
    result = pipeline.evaluate(
        stage_config=config,
        shield=SimpleNamespace(name="shield"),
        shielded_policy=policy,
        world_predictor=None,
        test_samples=[],
        distilled_policy=None,
        tb_writer=None,
    )

    calls = _CaptureEvaluator.instance.calls
    assert len(calls) == 2
    assert calls[0]["tb_prefix"] == "baseline"
    assert calls[1]["tb_prefix"] == "shielded"
    assert calls[0]["policy"] is policy
    assert calls[1]["policy"] is policy
    assert calls[0]["seeds"] == [5, 9, 5]
    assert calls[1]["seeds"] == [5, 9, 5]
    assert result["comparison_mode"] == "same_policy_shield_off_vs_on"
    assert result["paired_eval"] is True
    assert result["paired_risky_mode"] is True
    assert result["paired_scenario_source"] == config.sim.sumo_cfg
    assert result["policy_source"].endswith("policy_meta.json")
    assert result["shield_contribution_validated"] is True
    assert result["attribution_passed"] is True
    assert Path(result["stage5_paired_episode_results_path"]).exists()
    paired_payload = json.loads(Path(result["stage5_paired_episode_results_path"]).read_text(encoding="utf-8"))
    assert len(paired_payload["pairs"]) == 3
    assert paired_payload["pairs"][0]["baseline_episode_id"] == "base_ep_0"
    assert paired_payload["pairs"][0]["shielded_episode_id"] == "shield_ep_0"
    assert paired_payload["pairs"][0]["replacement_count"] == 1



def test_attribution_passed_requires_real_action_replacement(monkeypatch):
    config = _tiny_config()
    config.eval.eval_episodes = 1
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_attr_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    class _DummyBackend:
        def start(self):
            return None

        def close(self):
            return None

    class _DummyEnv:
        def close(self):
            return None

    class _CaptureEvaluator:
        def __init__(self, cfg):
            _ = cfg

        def evaluate_policy(self, env, policy, episodes, risky_mode=True, tb_writer=None, tb_prefix="", seeds=None):
            _ = (env, policy, episodes, risky_mode, tb_writer, seeds)
            if tb_prefix == "baseline":
                return {
                    "collision_rate": 0.2,
                    "intervention_rate": 0.0,
                    "mean_risk_reduction": 0.0,
                    "mean_raw_risk": 0.6,
                    "mean_final_risk": 0.6,
                    "avg_speed": 10.0,
                    "mean_reward": 1.0,
                    "success_rate": 0.5,
                    "replacement_count": 0.0,
                    "episode_details": [{"episode_id": "base_ep", "seed": 7, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.0, "mean_raw_risk": 0.6, "interventions": 0, "replacement_count": 0, "replacement_same_as_raw_count": 0, "fallback_action_count": 0, "mean_risk_reduction": 0.0}],
                }
            return {
                "collision_rate": 0.1,
                "intervention_rate": 0.5,
                "mean_risk_reduction": 0.2,
                "mean_raw_risk": 0.7,
                "mean_final_risk": 0.5,
                "avg_speed": 11.0,
                "mean_reward": 1.5,
                "success_rate": 0.7,
                "replacement_count": 0.0,
                "replacement_same_as_raw_count": 2.0,
                "episode_details": [{"episode_id": "shield_ep", "seed": 7, "risky_mode": True, "scenario_source": config.sim.sumo_cfg, "collisions": 0, "mean_reward": 1.5, "mean_raw_risk": 0.7, "mean_final_risk": 0.5, "interventions": 2, "replacement_count": 0, "replacement_same_as_raw_count": 2, "fallback_action_count": 0, "mean_risk_reduction": 0.2}],
            }

        def compare_baseline_and_shielded(self, baseline, shielded):
            _ = (baseline, shielded)
            return {"collision_reduction": 0.5, "efficiency_drop": -0.1}

        def evaluate_acceptance(self, delta_metrics):
            _ = delta_metrics
            return True

        def evaluate_world_model(self, world_predictor, samples):
            _ = (world_predictor, samples)
            return {"traj_ade": 0.0, "risk_acc": 1.0, "risk_mae": 0.0}

    monkeypatch.setattr("safe_rl.pipeline.pipeline.create_backend", lambda _sim_config: _DummyBackend())
    monkeypatch.setattr("safe_rl.pipeline.pipeline.SafeRLEvaluator", _CaptureEvaluator)
    monkeypatch.setattr("safe_rl.rl.env.create_env", lambda *_args, **_kwargs: _DummyEnv())

    result = pipeline.evaluate(
        stage_config=config,
        shield=SimpleNamespace(name="shield"),
        shielded_policy=object(),
        world_predictor=None,
        test_samples=[],
        distilled_policy=None,
        tb_writer=None,
    )

    assert result["performance_passed"] is True
    assert result["attribution_passed"] is False
    assert result["shield_contribution_validated"] is False



def test_stage4_shield_sweep_writes_variant_buffers_and_reports():
    config = _tiny_config()
    config.eval.eval_episodes = 2
    config.eval.seed_list = [11, 22]
    config.shield_sweep.enabled = True
    config.shield_sweep.variants = [
        ShieldSweepVariant(name="A", risk_threshold=0.20, uncertainty_threshold=0.60, coarse_top_k=7),
        ShieldSweepVariant(name="B", risk_threshold=0.25, uncertainty_threshold=0.50, coarse_top_k=6),
        ShieldSweepVariant(name="C", risk_threshold=0.30, uncertainty_threshold=0.45, coarse_top_k=5),
    ]
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage4_sweep_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage4", run_id=run_id)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    pipeline.light_model_path.touch()
    pipeline.world_model_path.touch()
    pipeline._save_policy_artifact(HeuristicPolicy())
    pipeline._build_predictors_from_saved_models = lambda: (None, None)

    def _fake_collect(*args, save_path=None, **kwargs):
        _ = (args, kwargs)
        buffer = InterventionBuffer()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            buffer.save(str(save_path))
        return buffer

    pipeline.collect_interventions = _fake_collect

    result = pipeline.run(stage="stage4", run_id=run_id)["stage4"]

    assert result["mode"] == "shield_sweep"
    assert len(result["variants"]) == 3
    assert not pipeline.buffer_path.exists()
    assert Path(pipeline.stage4_buffer_report_path).exists()
    for variant in result["variants"]:
        assert Path(variant["buffer_path"]).exists()
        assert Path(variant["stage4_buffer_report_path"]).exists()


def test_stage5_shield_sweep_writes_summary_and_variant_reports(monkeypatch):
    config = _tiny_config()
    config.eval.eval_episodes = 2
    config.eval.seed_list = [5, 9]
    config.shield_sweep.enabled = True
    config.shield_sweep.target_intervention_min = 0.05
    config.shield_sweep.target_intervention_max = 0.30
    config.shield_sweep.min_avg_speed = 10.0
    config.shield_sweep.variants = [
        ShieldSweepVariant(name="A", risk_threshold=0.20, uncertainty_threshold=0.60, coarse_top_k=7),
        ShieldSweepVariant(name="B", risk_threshold=0.25, uncertainty_threshold=0.50, coarse_top_k=6),
        ShieldSweepVariant(name="C", risk_threshold=0.30, uncertainty_threshold=0.45, coarse_top_k=5),
    ]
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_sweep_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    pipeline.datasets_dir.mkdir(parents=True, exist_ok=True)
    pipeline.test_pkl.touch()
    pipeline.light_model_path.touch()
    pipeline.world_model_path.touch()
    pipeline._save_policy_artifact(HeuristicPolicy())
    policy_meta = pipeline._read_policy_artifact_meta()
    pipeline._load_dataset_splits = lambda: ([], [], [])
    pipeline._build_predictors_from_saved_models = lambda: (None, None)

    base_metadata = pipeline._build_stage4_buffer_metadata(config, policy_meta)
    for variant in config.shield_sweep.variants:
        variant_name = variant.name
        buffer_path = pipeline._shield_sweep_variant_buffer_path(variant_name)
        report_path = pipeline._shield_sweep_variant_stage4_report_path(variant_name)
        buffer = InterventionBuffer()
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        buffer.save(str(buffer_path))
        pipeline._write_json(
            report_path,
            {
                **base_metadata,
                "mode": "shield_sweep_variant",
                "variant_name": variant_name,
                "risk_threshold": variant.risk_threshold,
                "uncertainty_threshold": variant.uncertainty_threshold,
                "coarse_top_k": variant.coarse_top_k,
                "buffer_path": str(buffer_path),
                "buffer_stats": {"size": 0.0},
            },
        )

    metrics_by_threshold = {
        0.20: {
            "intervention_rate": 0.0,
            "replacement_count": 0.0,
            "replacement_same_as_raw_count": 0.0,
            "fallback_action_count": 0.0,
            "mean_raw_risk": 0.40,
            "mean_final_risk": 0.40,
            "mean_risk_reduction": 0.0,
            "avg_speed": 20.0,
            "mean_reward": 1.2,
            "collision_rate": 0.0,
            "success_rate": 1.0,
            "performance_passed": True,
        },
        0.25: {
            "intervention_rate": 0.20,
            "replacement_count": 6.0,
            "replacement_same_as_raw_count": 0.0,
            "fallback_action_count": 2.0,
            "mean_raw_risk": 0.60,
            "mean_final_risk": 0.40,
            "mean_risk_reduction": 0.20,
            "avg_speed": 12.0,
            "mean_reward": 1.5,
            "collision_rate": 0.0,
            "success_rate": 1.0,
            "performance_passed": True,
        },
        0.30: {
            "intervention_rate": 1.0,
            "replacement_count": 10.0,
            "replacement_same_as_raw_count": 0.0,
            "fallback_action_count": 10.0,
            "mean_raw_risk": 0.80,
            "mean_final_risk": 0.30,
            "mean_risk_reduction": 0.50,
            "avg_speed": 5.0,
            "mean_reward": -0.2,
            "collision_rate": 0.0,
            "success_rate": 1.0,
            "performance_passed": False,
        },
    }

    def _fake_evaluate(stage_config, shield, shielded_policy, world_predictor, test_samples, distilled_policy=None, tb_writer=None, paired_results_path=None):
        _ = (shield, shielded_policy, world_predictor, test_samples, distilled_policy, tb_writer)
        key = round(float(stage_config.shield.risk_threshold), 2)
        metrics = metrics_by_threshold[key]
        if paired_results_path is not None:
            paired_results_path.parent.mkdir(parents=True, exist_ok=True)
            paired_results_path.write_text(json.dumps({"pairs": [{"pair_index": 0, "replacement_count": int(metrics["replacement_count"])}]}, ensure_ascii=False), encoding="utf-8")
        shielded = {
            "intervention_rate": metrics["intervention_rate"],
            "replacement_count": metrics["replacement_count"],
            "replacement_same_as_raw_count": metrics["replacement_same_as_raw_count"],
            "fallback_action_count": metrics["fallback_action_count"],
            "mean_raw_risk": metrics["mean_raw_risk"],
            "mean_final_risk": metrics["mean_final_risk"],
            "mean_risk_reduction": metrics["mean_risk_reduction"],
            "avg_speed": metrics["avg_speed"],
            "mean_reward": metrics["mean_reward"],
            "collision_rate": metrics["collision_rate"],
            "success_rate": metrics["success_rate"],
        }
        performance_passed = metrics["performance_passed"]
        attribution_passed = metrics["intervention_rate"] > 0.0 and metrics["mean_risk_reduction"] > 0.0 and metrics["replacement_count"] > 0.0
        return {
            "comparison_mode": "same_policy_shield_off_vs_on",
            "policy_source": str(pipeline.policy_meta_path),
            "paired_eval": True,
            "paired_risky_mode": True,
            "paired_scenario_source": stage_config.sim.sumo_cfg,
            "evaluation_seeds": [5, 9],
            "system_baseline": {"collision_rate": 0.0, "avg_speed": 20.0},
            "system_shielded": shielded,
            "delta": {"collision_reduction": 0.0, "efficiency_drop": 0.0},
            "acceptance_passed": performance_passed,
            "performance_passed": performance_passed,
            "attribution_passed": attribution_passed,
            "shield_contribution_validated": attribution_passed,
            "stage5_paired_episode_results_path": str(paired_results_path),
        }

    pipeline.evaluate = _fake_evaluate

    result = pipeline.run(stage="stage5", run_id=run_id)["stage5"]

    assert result["mode"] == "shield_sweep"
    assert result["paired_eval"] is True
    assert Path(result["shield_sweep_summary_path"]).exists()
    assert Path(pipeline.report_path).exists()

    summary_payload = json.loads(Path(result["shield_sweep_summary_path"]).read_text(encoding="utf-8"))
    assert len(summary_payload["variants"]) == 3

    by_name = {entry["variant_name"]: entry for entry in summary_payload["variants"]}
    assert by_name["A"]["intervention_band_passed"] is False
    assert by_name["A"]["replacement_passed"] is False
    assert by_name["B"]["intervention_band_passed"] is True
    assert by_name["B"]["efficiency_guard_passed"] is True
    assert by_name["B"]["risk_reduction_passed"] is True
    assert by_name["C"]["fallback_dominant"] is True
    assert by_name["C"]["efficiency_guard_passed"] is False
    for entry in summary_payload["variants"]:
        assert Path(entry["stage4_buffer_report_path"]).exists()
        assert Path(entry["stage5_report_path"]).exists()
        assert Path(entry["stage5_paired_episode_results_path"]).exists()
