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


def _local_test_tmp_dir(tag: str) -> Path:
    path = Path("safe_rl_output/test_artifacts") / f"{tag}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    assert Path(result["stage1"]["stage1_probe_summary_report"]).exists()
    assert Path(result["stage1"]["stage1_bucket_summary_report"]).exists()
    assert Path(result["stage1"]["stage1_probe_events_report"]).exists()
    assert Path(result["stage1"]["pairs_stage1_probe"]).exists()



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
    assert result["distill_supervision_path"].endswith("distill_supervision.json")
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


def test_stage5_trace_writes_pair_files_and_summary(monkeypatch):
    config = _tiny_config()
    config.eval.eval_episodes = 3
    config.eval.seed_list = [42, 123, 2024]
    config.shield.risk_threshold = 0.30
    config.shield.uncertainty_threshold = 0.45
    config.shield.coarse_top_k = 5
    config.shield_trace.enabled = True
    config.shield_trace.seed_list = [42, 123, 2024]
    config.shield_trace.trace_dir_name = "shield_trace"

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_trace_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    class _DummyBackend:
        def start(self):
            return None

        def close(self):
            return None

    class _DummyEnv:
        def close(self):
            return None

    class _TraceEvaluator:
        def __init__(self, cfg):
            _ = cfg

        def evaluate_policy(self, env, policy, episodes, risky_mode=True, tb_writer=None, tb_prefix="", seeds=None, collect_step_traces=False):
            _ = (env, policy, episodes, risky_mode, tb_writer, collect_step_traces)
            if tb_prefix == "baseline":
                return {
                    "collision_rate": 0.0,
                    "intervention_rate": 0.0,
                    "mean_risk_reduction": 0.0,
                    "mean_raw_risk": 0.2,
                    "mean_final_risk": 0.2,
                    "avg_speed": 10.0,
                    "mean_reward": 1.0,
                    "success_rate": 1.0,
                    "replacement_count": 0.0,
                    "episode_details": [
                        {
                            "episode_id": "base_ep_0",
                            "seed": 42,
                            "risky_mode": True,
                            "scenario_source": config.sim.sumo_cfg,
                            "collisions": 0,
                            "mean_reward": 1.0,
                            "mean_raw_risk": 0.2,
                            "mean_final_risk": 0.2,
                            "interventions": 0,
                            "replacement_count": 0,
                            "replacement_same_as_raw_count": 0,
                            "fallback_action_count": 0,
                            "mean_risk_reduction": 0.0,
                            "step_trace": [
                                {"step_index": 0, "raw_action": 4, "final_action": 4, "executed_action": 4, "replacement_happened": False, "fallback_used": False, "chosen_candidate_index": -1, "chosen_candidate_rank_by_risk": -1, "raw_risk": 0.2, "final_risk": 0.2, "risk_reduction": 0.0, "candidate_evaluations": [], "raw_action_type": "KEEP_KEEP", "final_action_type": "KEEP_KEEP", "lane_change_involved": False, "ego_lane_id": "1", "ego_lane_index": 1, "ego_speed": 20.0, "ttc": 5.0, "min_distance": 12.0, "collision": False, "constraint_reason": "", "replacement_margin": 0.0},
                                {"step_index": 1, "raw_action": 4, "final_action": 4, "executed_action": 4, "replacement_happened": False, "fallback_used": False, "chosen_candidate_index": -1, "chosen_candidate_rank_by_risk": -1, "raw_risk": 0.2, "final_risk": 0.2, "risk_reduction": 0.0, "candidate_evaluations": [], "raw_action_type": "KEEP_KEEP", "final_action_type": "KEEP_KEEP", "lane_change_involved": False, "ego_lane_id": "1", "ego_lane_index": 1, "ego_speed": 19.0, "ttc": 4.0, "min_distance": 10.0, "collision": False, "constraint_reason": "", "replacement_margin": 0.0},
                            ],
                        }
                    ],
                }
            return {
                "collision_rate": 1.0,
                "intervention_rate": 1.0,
                "mean_risk_reduction": 0.1,
                "mean_raw_risk": 0.5,
                "mean_final_risk": 0.4,
                "avg_speed": 8.0,
                "mean_reward": 0.5,
                "success_rate": 0.0,
                "replacement_count": 1.0,
                "episode_details": [
                    {
                        "episode_id": "shield_ep_0",
                        "seed": 42,
                        "risky_mode": True,
                        "scenario_source": config.sim.sumo_cfg,
                        "collisions": 1,
                        "mean_reward": 0.5,
                        "mean_raw_risk": 0.5,
                        "mean_final_risk": 0.4,
                        "interventions": 1,
                        "replacement_count": 1,
                        "replacement_same_as_raw_count": 0,
                        "fallback_action_count": 0,
                        "mean_risk_reduction": 0.1,
                        "step_trace": [
                            {"step_index": 0, "raw_action": 3, "final_action": 4, "executed_action": 4, "replacement_happened": True, "fallback_used": False, "chosen_candidate_index": 1, "chosen_candidate_rank_by_risk": 0, "raw_risk": 0.5, "final_risk": 0.4, "risk_reduction": 0.1, "candidate_evaluations": [{"action_id": 3, "action_type": "KEEP_LEFT", "distance_to_raw": 0, "coarse_risk": 0.5, "fine_risk": 0.5, "uncertainty": 0.1, "selected": False, "safe_under_threshold": False, "evaluated": True, "constraint_reason": ""}, {"action_id": 4, "action_type": "KEEP_KEEP", "distance_to_raw": 1, "coarse_risk": 0.4, "fine_risk": 0.4, "uncertainty": 0.1, "selected": True, "safe_under_threshold": True, "evaluated": True, "constraint_reason": "merge_lateral_guard"}], "raw_action_type": "KEEP_LEFT", "final_action_type": "KEEP_KEEP", "lane_change_involved": True, "ego_lane_id": "1", "ego_lane_index": 1, "ego_speed": 18.0, "ttc": 2.0, "min_distance": 6.0, "collision": False, "constraint_reason": "merge_lateral_guard", "replacement_margin": 0.1},
                            {"step_index": 1, "raw_action": 4, "final_action": 4, "executed_action": 4, "replacement_happened": False, "fallback_used": False, "chosen_candidate_index": 0, "chosen_candidate_rank_by_risk": 0, "raw_risk": 0.4, "final_risk": 0.4, "risk_reduction": 0.0, "candidate_evaluations": [], "raw_action_type": "KEEP_KEEP", "final_action_type": "KEEP_KEEP", "lane_change_involved": False, "ego_lane_id": "1", "ego_lane_index": 1, "ego_speed": 0.0, "ttc": 0.5, "min_distance": 0.5, "collision": True, "constraint_reason": "", "replacement_margin": 0.0},
                        ],
                    }
                ],
            }

        def compare_baseline_and_shielded(self, baseline, shielded):
            _ = (baseline, shielded)
            return {"collision_reduction": -1.0, "efficiency_drop": 0.2}

        def evaluate_acceptance(self, delta_metrics):
            _ = delta_metrics
            return False

        def evaluate_world_model(self, world_predictor, samples):
            _ = (world_predictor, samples)
            return {"traj_ade": 0.0, "risk_acc": 1.0, "risk_mae": 0.0}

    monkeypatch.setattr("safe_rl.pipeline.pipeline.create_backend", lambda _sim_config: _DummyBackend())
    monkeypatch.setattr("safe_rl.pipeline.pipeline.SafeRLEvaluator", _TraceEvaluator)
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

    assert Path(result["shield_trace_summary_path"]).exists()
    trace_summary = json.loads(Path(result["shield_trace_summary_path"]).read_text(encoding="utf-8"))
    assert trace_summary["variant_name"] == "C_baseline"
    assert trace_summary["regression_pair_count"] == 1
    assert trace_summary["pairs_with_lane_change_replacement"] == 1
    assert trace_summary["pairs_with_merge_guard_triggered"] == 1
    assert trace_summary["effective_shield_config"]["effective_raw_passthrough_threshold"] == pytest.approx(0.20)
    assert trace_summary["blocked_by_margin_count"] == 0
    assert trace_summary["raw_passthrough_count"] == 0
    assert trace_summary["merge_lateral_guard_block_count"] == 1
    assert trace_summary["candidate_selected_count"] == 1

    pair_path = Path(trace_summary["pair_files"][0])
    assert pair_path.exists()
    pair_payload = json.loads(pair_path.read_text(encoding="utf-8"))
    assert pair_payload["regression_pair"] is True
    assert pair_payload["first_replacement_step"] == 0
    assert pair_payload["collision_step_shielded"] == 1
    assert pair_payload["replacement_count"] == 1
    assert pair_payload["blocked_by_margin_count"] == 0
    assert pair_payload["raw_passthrough_count"] == 0
    assert pair_payload["merge_lateral_guard_block_count"] == 1
    assert pair_payload["candidate_selected_count"] == 1
    assert "history_hash" in pair_payload
    assert "ego_lane_id" in pair_payload
    assert isinstance(pair_payload["neighbor_summary"], list)
    assert pair_payload["aligned_steps"][0]["shielded"]["chosen_candidate_index"] == 1
    assert Path(result["shield_trace_tuning_summary_path"]).exists()
    tuning_summary = json.loads(Path(result["shield_trace_tuning_summary_path"]).read_text(encoding="utf-8"))
    assert tuning_summary["baseline_available"] is True
    assert tuning_summary["variants"][0]["variant_name"] == "C_baseline"
    assert tuning_summary["variants"][0]["effective_shield_config"]["effective_raw_passthrough_threshold"] == pytest.approx(0.20)
    assert Path(result["shield_margin_analysis_summary_path"]).exists()
    assert Path(pipeline.risk_v2_eval_summary_path).exists()
    risk_v2_summary = json.loads(Path(pipeline.risk_v2_eval_summary_path).read_text(encoding="utf-8"))
    assert risk_v2_summary["after_trace_metrics"]["ANCHOR"]["variant_name"] == "C_baseline"
    assert risk_v2_summary["after_trace_metrics"]["BOUNDARY"]["variant_name"] == "C_baseline"
    assert risk_v2_summary["after_trace_metrics"]["CONSERVATIVE"]["variant_name"] == "C_baseline"
    assert risk_v2_summary["after_trace_metrics"]["HOLDOUT"] is None
    assert risk_v2_summary["after_trace_metrics_complete"] is False
    assert risk_v2_summary["margin_near_threshold_band_ratio_before_after"]["ANCHOR"]["after"] == pytest.approx(0.0)
    assert risk_v2_summary["margin_near_threshold_band_ratio_before_after"]["BOUNDARY"]["after"] == pytest.approx(0.0)
    assert risk_v2_summary["margin_near_threshold_band_ratio_before_after"]["CONSERVATIVE"]["after"] == pytest.approx(0.0)
    assert risk_v2_summary["margin_near_threshold_band_ratio_before_after"]["HOLDOUT"]["after"] is None
    margin_summary = json.loads(Path(result["shield_margin_analysis_summary_path"]).read_text(encoding="utf-8"))
    assert margin_summary["variants"][0]["variant_name"] == "C_baseline"
    assert margin_summary["variants"][0]["replacement_step_count"] == 1
    assert margin_summary["variants"][0]["replacement_margin_mean"] == pytest.approx(0.1)
    assert Path(trace_summary["margin_analysis_path"]).exists()
    assert trace_summary["unique_margin_count"] == 1
    assert trace_summary["replacement_margin_stats"]["mean"] == pytest.approx(0.1)
    assert trace_summary["margin_near_threshold_band_ratio"] == pytest.approx(0.0)
    assert "best_margin_stats" in trace_summary
    assert trace_summary["best_margin_stats"]["mean"] == pytest.approx(0.05)
    assert trace_summary["best_margin_unique_count"] == 2
    assert trace_summary["no_safe_candidate_count"] == 0
    assert trace_summary["raw_already_best_count"] == 0


def test_shield_trace_tuning_summary_aggregates_baseline_and_c1():
    config = _tiny_config()
    config.shield_trace.enabled = True
    config.shield_trace.trace_dir_name = "shield_trace_c1"

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_trace_tuning_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.reports_dir is not None
    baseline_dir = pipeline.reports_dir / "shield_trace"
    current_dir = pipeline.reports_dir / "shield_trace_c1"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)

    baseline_pair_path = baseline_dir / "pair_00_seed_42.json"
    current_pair_path = current_dir / "pair_00_seed_42.json"
    baseline_pair = {
        "seed": 42,
        "baseline_collision": False,
        "shielded_collision": False,
        "baseline_reward": 0.30,
        "shielded_reward": 0.24,
        "intervention_count": 150,
        "replacement_count": 150,
        "fallback_action_count": 0,
        "mean_risk_reduction": 0.05,
    }
    current_pair = {
        "seed": 42,
        "baseline_collision": False,
        "shielded_collision": False,
        "baseline_reward": 0.30,
        "shielded_reward": 0.27,
        "intervention_count": 90,
        "replacement_count": 90,
        "fallback_action_count": 0,
        "mean_risk_reduction": 0.03,
    }
    pipeline._write_json(baseline_pair_path, baseline_pair)
    pipeline._write_json(current_pair_path, current_pair)
    pipeline._write_json(
        baseline_dir / "trace_summary.json",
        {
            "variant_name": "C",
            "seeds": [42],
            "regression_pair_count": 0,
            "pair_files": [str(baseline_pair_path)],
        },
    )
    pipeline._write_json(
        current_dir / "trace_summary.json",
        {
            "variant_name": "C1",
            "seeds": [42],
            "regression_pair_count": 0,
            "pair_files": [str(current_pair_path)],
        },
    )

    payload = pipeline._write_shield_trace_tuning_summary()
    assert payload is not None
    assert Path(payload["summary_path"]).exists()

    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    assert summary["baseline_available"] is True
    by_name = {entry["variant_name"]: entry for entry in summary["variants"]}
    assert by_name["C_baseline"]["mean_intervention_count"] == 150.0
    assert by_name["C_baseline"]["mean_reward_gap_to_baseline_policy"] == pytest.approx(-0.06)
    assert by_name["C1"]["mean_intervention_count"] == 90.0
    assert by_name["C1"]["mean_reward_gap_to_baseline_policy"] == pytest.approx(-0.03)


def test_shield_trace_tuning_summary_supports_c_strong_variant():
    config = _tiny_config()
    config.shield_trace.enabled = True
    config.shield_trace.trace_dir_name = "shield_trace_c_strong"
    config.shield.risk_threshold = 0.30
    config.shield.raw_passthrough_risk_threshold = 0.30
    config.shield.replacement_min_risk_margin = 0.15
    config.shield.merge_override_margin = 0.12

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_trace_strong_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.reports_dir is not None
    strong_dir = pipeline.reports_dir / "shield_trace_c_strong"
    strong_dir.mkdir(parents=True, exist_ok=True)
    pair_path = strong_dir / "pair_00_seed_42.json"
    pipeline._write_json(
        pair_path,
        {
            "seed": 42,
            "baseline_collision": False,
            "shielded_collision": False,
            "baseline_reward": 0.30,
            "shielded_reward": 0.26,
            "intervention_count": 40,
            "replacement_count": 30,
            "fallback_action_count": 0,
            "mean_risk_reduction": 0.02,
            "blocked_by_margin_count": 12,
            "raw_passthrough_count": 8,
            "merge_lateral_guard_block_count": 4,
            "candidate_selected_count": 30,
        },
    )
    pipeline._write_json(
        strong_dir / "trace_summary.json",
        {
            "variant_name": "C_strong",
            "seeds": [42],
            "effective_shield_config": {
                "risk_threshold": 0.30,
                "uncertainty_threshold": 0.45,
                "replacement_min_risk_margin": 0.15,
                "raw_passthrough_risk_threshold": 0.30,
                "effective_raw_passthrough_threshold": 0.30,
                "merge_override_margin": 0.12,
            },
            "regression_pair_count": 0,
            "blocked_by_margin_count": 12,
            "raw_passthrough_count": 8,
            "merge_lateral_guard_block_count": 4,
            "candidate_selected_count": 30,
            "pair_files": [str(pair_path)],
        },
    )

    payload = pipeline._write_shield_trace_tuning_summary()
    assert payload is not None
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    by_name = {entry["variant_name"]: entry for entry in summary["variants"]}
    assert by_name["C_strong"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.15)
    assert by_name["C_strong"]["blocked_by_margin_count"] == 12
    assert by_name["C_strong"]["raw_passthrough_count"] == 8
    assert by_name["C_strong"]["merge_lateral_guard_block_count"] == 4
    assert by_name["C_strong"]["candidate_selected_count"] == 30



def test_shield_trace_tuning_summary_supports_legacy_trace_file_names():
    config = _tiny_config()
    config.shield_trace.enabled = True
    config.shield_trace.trace_dir_name = "shield_trace_d1"

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_trace_legacy_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.reports_dir is not None
    legacy_dir = pipeline.reports_dir / "shield_trace_d1"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    pair_path = legacy_dir / "d1pair_00_seed_42.json"
    pipeline._write_json(
        pair_path,
        {
            "seed": 42,
            "baseline_collision": False,
            "shielded_collision": False,
            "intervention_count": 12,
            "replacement_count": 12,
            "fallback_action_count": 0,
            "mean_risk_reduction": 0.02,
            "baseline_reward": 0.20,
            "shielded_reward": 0.18,
            "blocked_by_margin_count": 3,
            "raw_passthrough_count": 1,
            "merge_lateral_guard_block_count": 2,
            "candidate_selected_count": 12,
            "shielded_steps": [
                {
                    "step_index": 0,
                    "raw_action": 3,
                    "final_action": 4,
                    "executed_action": 4,
                    "replacement_happened": True,
                    "fallback_used": False,
                    "chosen_candidate_index": 1,
                    "chosen_candidate_rank_by_risk": 0,
                    "raw_risk": 0.302,
                    "final_risk": 0.2,
                    "risk_reduction": 0.102,
                    "candidate_evaluations": [],
                    "raw_action_type": "KEEP_LEFT",
                    "final_action_type": "KEEP_KEEP",
                    "lane_change_involved": True,
                    "ego_lane_id": "1",
                    "ego_lane_index": 1,
                    "ego_speed": 18.0,
                    "ttc": 3.0,
                    "min_distance": 6.0,
                    "collision": False,
                    "constraint_reason": "",
                    "replacement_margin": 0.102,
                }
            ],
        },
    )
    pipeline._write_json(
        legacy_dir / "d1trace_summary.json",
        {
            "variant_name": "D1",
            "seeds": [42],
            "effective_shield_config": {
                "risk_threshold": 0.30,
                "uncertainty_threshold": 0.45,
                "replacement_min_risk_margin": 0.10,
                "raw_passthrough_risk_threshold": 0.24,
                "effective_raw_passthrough_threshold": 0.24,
                "merge_override_margin": 0.12,
            },
            "regression_pair_count": 0,
            "blocked_by_margin_count": 3,
            "raw_passthrough_count": 1,
            "merge_lateral_guard_block_count": 2,
            "candidate_selected_count": 12,
            "pair_files": [str(pair_path)],
        },
    )

    payload = pipeline._write_shield_trace_tuning_summary()
    assert payload is not None
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    by_name = {entry["variant_name"]: entry for entry in summary["variants"]}
    assert by_name["D1"]["trace_summary_path"].endswith("d1trace_summary.json")
    assert by_name["D1"]["pair_count"] == 1
    assert by_name["D1"]["blocked_by_margin_count"] == 3
    assert by_name["D1"]["raw_passthrough_count"] == 1
    assert by_name["D1"]["candidate_selected_count"] == 12
    assert by_name["D1"]["mean_intervention_count"] == 12.0
    assert Path(by_name["D1"]["margin_analysis_path"]).exists()
    assert by_name["D1"]["replacement_margin_mean"] == pytest.approx(0.102)
    assert by_name["D1"]["unique_margin_count"] == 1

    margin_summary_path = Path(summary["shield_margin_analysis_summary_path"])
    assert margin_summary_path.exists()
    margin_summary = json.loads(margin_summary_path.read_text(encoding="utf-8"))
    margin_by_name = {entry["variant_name"]: entry for entry in margin_summary["variants"]}
    assert margin_by_name["D1"]["replacement_step_count"] == 1
    assert margin_by_name["D1"]["replacement_margin_mean"] == pytest.approx(0.102)
    assert margin_by_name["D1"]["unique_margin_count"] == 1


def test_shield_trace_tuning_summary_prefers_pair_scalar_summaries_for_large_pairs():
    config = _tiny_config()
    config.shield_trace.enabled = True
    config.shield_trace.trace_dir_name = "shield_trace_pair_bootstrap"

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_trace_pair_scalars_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.reports_dir is not None
    trace_dir = pipeline.reports_dir / "shield_trace_pair_bootstrap"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_path = trace_dir / "pair_00_seed_1000.json"
    pair_path.write_text("{not-valid-json", encoding="utf-8")
    pair_scalar_path = trace_dir / "pair_scalar_summaries.json"
    pipeline._write_json(
        pair_scalar_path,
        {
            "pairs": [
                {
                    "pair_index": 0,
                    "seed": 1000,
                    "baseline_collision": False,
                    "shielded_collision": False,
                    "baseline_reward": 0.3,
                    "shielded_reward": 0.25,
                    "intervention_count": 8,
                    "replacement_count": 0,
                    "fallback_action_count": 0,
                    "mean_risk_reduction": 0.0,
                }
            ]
        },
    )
    pipeline._write_json(
        trace_dir / "trace_summary.json",
        {
            "variant_name": "PAIR_BOOTSTRAP",
            "seeds": [1000],
            "effective_shield_config": {
                "risk_threshold": 0.30,
                "uncertainty_threshold": 0.45,
                "replacement_min_risk_margin": 0.02,
                "raw_passthrough_risk_threshold": 0.24,
                "effective_raw_passthrough_threshold": 0.24,
                "merge_override_margin": 0.12,
            },
            "regression_pair_count": 0,
            "blocked_by_margin_count": 0,
            "raw_passthrough_count": 10,
            "merge_lateral_guard_block_count": 0,
            "candidate_selected_count": 0,
            "pair_files": [str(pair_path)],
            "pair_scalar_summaries_path": str(pair_scalar_path),
        },
    )

    payload = pipeline._write_shield_trace_tuning_summary()
    assert payload is not None
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    by_name = {entry["variant_name"]: entry for entry in summary["variants"]}
    assert by_name["PAIR_BOOTSTRAP"]["pair_count"] == 1
    assert by_name["PAIR_BOOTSTRAP"]["mean_intervention_count"] == 8.0
    assert by_name["PAIR_BOOTSTRAP"]["mean_reward_gap_to_baseline_policy"] == pytest.approx(-0.05)
    assert by_name["PAIR_BOOTSTRAP"]["margin_analysis_path"] == ""


def test_shield_trace_tuning_summary_supports_d_variants_in_order():
    config = _tiny_config()
    config.shield_trace.enabled = True
    config.shield_trace.trace_dir_name = "shield_trace_d2"

    pipeline = SafeRLPipeline(config)
    run_id = f"ut_trace_d_order_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.reports_dir is not None
    variants = [
        ("shield_trace", "C_baseline"),
        ("shield_trace_pair_bootstrap", "PAIR_BOOTSTRAP"),
        ("shield_trace_g1", "G1"),
        ("shield_trace_g2", "G2"),
        ("shield_trace_g3", "G3"),
        ("shield_trace_g4", "G4"),
        ("shield_trace_g5", "G5"),
        ("shield_trace_c1", "C1"),
        ("shield_trace_c2", "C2"),
        ("shield_trace_d1", "D1"),
        ("shield_trace_e2", "E2"),
        ("shield_trace_f1", "F1"),
        ("shield_trace_holdout_c1", "HOLDOUT_C1"),
        ("shield_trace_f2", "F2"),
        ("shield_trace_f3", "F3"),
        ("shield_trace_e1", "E1"),
        ("shield_trace_e3", "E3"),
        ("shield_trace_d2", "D2"),
        ("shield_trace_d3", "D3"),
        ("shield_trace_c_strong", "C_strong"),
    ]

    for trace_dir_name, variant_name in variants:
        trace_dir = pipeline.reports_dir / trace_dir_name
        trace_dir.mkdir(parents=True, exist_ok=True)
        pair_path = trace_dir / "pair_00_seed_42.json"
        pipeline._write_json(
            pair_path,
            {
                "seed": 42,
                "baseline_collision": False,
                "shielded_collision": False,
                "intervention_count": 10,
                "replacement_count": 10,
                "fallback_action_count": 0,
                "mean_risk_reduction": 0.01,
                "baseline_reward": 0.2,
                "shielded_reward": 0.19,
                "blocked_by_margin_count": 1,
                "raw_passthrough_count": 2,
                "merge_lateral_guard_block_count": 3,
                "candidate_selected_count": 4,
            },
        )
        pipeline._write_json(
            trace_dir / "trace_summary.json",
            {
                "variant_name": variant_name,
                "seeds": [42],
                "effective_shield_config": {
                    "risk_threshold": 0.30,
                    "uncertainty_threshold": 0.45,
                    "replacement_min_risk_margin": 0.11,
                    "raw_passthrough_risk_threshold": 0.25,
                    "effective_raw_passthrough_threshold": 0.25,
                    "merge_override_margin": 0.12,
                },
                "regression_pair_count": 0,
                "blocked_by_margin_count": 1,
                "raw_passthrough_count": 2,
                "merge_lateral_guard_block_count": 3,
                "candidate_selected_count": 4,
                "pair_files": [str(pair_path)],
            },
        )

    payload = pipeline._write_shield_trace_tuning_summary()
    assert payload is not None
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    assert [entry["variant_name"] for entry in summary["variants"]] == [
        "C_baseline",
        "PAIR_BOOTSTRAP",
        "G1",
        "G2",
        "G3",
        "G4",
        "G5",
        "C1",
        "C2",
        "D1",
        "E2",
        "F1",
        "HOLDOUT_C1",
        "F2",
        "F3",
        "E1",
        "E3",
        "D2",
        "D3",
        "C_strong",
    ]
    by_name = {entry["variant_name"]: entry for entry in summary["variants"]}
    assert by_name["PAIR_BOOTSTRAP"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.11)
    assert by_name["G1"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.11)
    assert by_name["G2"]["candidate_selected_count"] == 4
    assert by_name["G3"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.11)
    assert by_name["D1"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.11)
    assert by_name["E2"]["candidate_selected_count"] == 4
    assert by_name["F1"]["effective_shield_config"]["replacement_min_risk_margin"] == pytest.approx(0.11)
    assert by_name["F2"]["raw_passthrough_count"] == 2
    assert by_name["F3"]["merge_lateral_guard_block_count"] == 3
    assert by_name["E1"]["raw_passthrough_count"] == 2
    assert by_name["E3"]["merge_lateral_guard_block_count"] == 3
    assert by_name["D2"]["raw_passthrough_count"] == 2
    assert by_name["D3"]["candidate_selected_count"] == 4


def test_stage2_pair_dataset_builder_uses_stage5_trace_and_stage4_buffer():
    import json as _json

    from safe_rl.data.types import InterventionRecord, SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage2_pairs_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    trace_dir = pipeline.reports_dir / "shield_trace_d1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_path = trace_dir / "pair_00_seed_42.json"
    pair_payload = {
        "pair_index": 0,
        "seed": 42,
        "first_replacement_step": 0,
        "baseline_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 4,
                "history_scene": dataclass_to_dict(history),
                "collision": False,
                "ttc": 5.0,
                "min_distance": 10.0,
                "reward": 0.3,
            }
        ],
        "shielded_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 3,
                "history_scene": dataclass_to_dict(history),
                "collision": True,
                "ttc": 0.5,
                "min_distance": 0.5,
                "reward": -0.2,
            }
        ],
    }
    pipeline._write_json(pair_path, pair_payload)
    pipeline._write_json(trace_dir / "trace_summary.json", {"variant_name": "D1", "pair_files": [str(pair_path)], "seeds": [42]})

    buffer = InterventionBuffer()
    buffer.push(
        InterventionRecord(
            history_scene=history,
            raw_action=4,
            final_action=3,
            raw_risk=0.8,
            final_risk=0.2,
            reason="risk_threshold_exceeded",
            meta={"episode_id": "ep_0"},
        )
    )
    buffer.save(str(pipeline.buffer_path))

    payload = pipeline._build_pair_datasets_for_stage2()
    assert len(payload["stage5_pairs"]) == 1
    assert len(payload["stage4_pairs"]) == 1
    assert payload["stage5_pairs"][0].preferred_action == 4
    assert payload["stage5_pairs"][0].meta["hard_negative"] is True
    assert payload["stage5_pairs"][0].meta["trusted_for_spread"] is True
    assert payload["stage5_pairs"][0].meta["history_hash"]
    assert "neighbor_summary" in payload["stage5_pairs"][0].meta
    assert Path(payload["pair_dataset_paths"]["stage5"]).exists()
    assert Path(payload["pair_dataset_paths"]["stage4"]).exists()



def test_stage2_report_includes_pair_finetune_metadata(monkeypatch):
    import pickle

    from safe_rl.data.types import InterventionRecord, SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage2_report_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)
    pipeline.datasets_dir.mkdir(parents=True, exist_ok=True)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    for path in [pipeline.train_pkl, pipeline.val_pkl, pipeline.test_pkl]:
        with open(path, "wb") as f:
            pickle.dump([], f)

    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    buffer = InterventionBuffer()
    buffer.push(
        InterventionRecord(
            history_scene=history,
            raw_action=4,
            final_action=3,
            raw_risk=0.8,
            final_risk=0.2,
            reason="risk_threshold_exceeded",
        )
    )
    buffer.save(str(pipeline.buffer_path))

    trace_dir = pipeline.reports_dir / "shield_trace_d1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_path = trace_dir / "pair_00_seed_42.json"
    pipeline._write_json(
        pair_path,
        {
            "pair_index": 0,
            "seed": 42,
            "first_replacement_step": 0,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": dataclass_to_dict(history), "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": dataclass_to_dict(history), "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
    )
    pipeline._write_json(trace_dir / "trace_summary.json", {"variant_name": "D1", "pair_files": [str(pair_path)], "seeds": [42]})

    captured = {}

    class _DummyPredictor:
        def __init__(self):
            self.device = "cpu"

    class _DummyEvaluator:
        def __init__(self, cfg):
            _ = cfg

        def evaluate_world_model(self, world_predictor, samples):
            _ = (world_predictor, samples)
            return {"traj_ade": 0.0, "risk_acc": 1.0, "risk_mae": 0.0}

    def _fake_train_models(*args, stage5_pair_samples=None, stage4_pair_samples=None, **kwargs):
        captured["stage5_pairs"] = len(stage5_pair_samples or [])
        captured["stage4_pairs"] = len(stage4_pair_samples or [])
        return _DummyPredictor(), _DummyPredictor(), {
            "pair_finetune_applied": True,
            "light_pair_finetune_applied": True,
            "world_pair_finetune_applied": True,
            "world_pair_finetune_mode": "fallback_all_pairs",
            "stage5_requirement_met": True,
            "world_pair_gate_degraded": False,
            "world_pair_stage5_pairs_required": 1,
            "world_pair_stage5_pairs_available": 1,
            "world_pair_total_pairs_available": 2,
            "ranking_metrics": {"light": {"pair_ranking_accuracy": 1.0}, "world": {"pair_ranking_accuracy": 1.0}},
            "light_training": {"variant": "v2"},
            "world_training": {"variant": "v2"},
            "light_pair_ft": {"before_pair_metrics": {"pair_ranking_accuracy": 0.5, "same_state_score_gap": 0.01, "score_spread": 0.01, "hard_negative_accuracy": 0.5}, "after_pair_metrics": {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.04, "score_spread": 0.05, "hard_negative_accuracy": 0.75}},
            "world_pair_ft": {
                "before_pair_metrics": {"pair_ranking_accuracy": 0.4, "same_state_score_gap": 0.01, "score_spread": 0.01, "hard_negative_accuracy": 0.4},
                "after_pair_metrics": {"pair_ranking_accuracy": 0.7, "same_state_score_gap": 0.03, "score_spread": 0.04, "hard_negative_accuracy": 0.7},
                "world_pair_ft_frozen_modules": ["traj_decoder"],
                "world_pair_ft_trainable_modules": ["fusion", "risk_score_head"],
                "stage5_pair_ranking_accuracy_before_after": {"before": 0.55, "after": 0.75},
                "stage4_pair_ranking_accuracy_before_after": {"before": 0.45, "after": 0.65},
                "stage5_same_state_score_gap_before_after": {"before": 0.02, "after": 0.05},
                "stage4_same_state_score_gap_before_after": {"before": 0.01, "after": 0.03},
                "stage5_score_spread_before_after": {"before": 0.02, "after": 0.06},
                "stage4_score_spread_before_after": {"before": 0.01, "after": 0.04},
                "stage5_spread_eligible_pair_count": 2,
                "stage4_spread_eligible_pair_count": 0,
                "world_pair_ft_best_epoch": 1,
                "world_pair_ft_best_metrics": {"pair_ranking_accuracy": 0.75, "same_state_score_gap": 0.05},
                "world_pair_ft_restored_best": True,
            },
            "world_pair_ft_source_mix": {"stage5_steps": 3, "stage4_steps": 1, "stage5_pairs_seen": 3, "stage4_pairs_seen": 1, "stage5_pair_seen_counts": {"p0": 2, "p1": 1}, "stage5_pair_cap": 8, "stage5_cap_reached_pairs": 0},
        }

    monkeypatch.setattr("safe_rl.pipeline.pipeline.SafeRLEvaluator", _DummyEvaluator)
    pipeline.train_models = _fake_train_models

    result = pipeline.run(stage="stage2", run_id=run_id)["stage2"]
    report = json.loads(Path(result["training_report"]).read_text(encoding="utf-8"))

    assert captured["stage5_pairs"] == 1
    assert captured["stage4_pairs"] == 1
    assert result["stage5_pairs_created"] == 1
    assert result["stage4_pairs_created"] == 1
    assert report["stage5_pairs_created"] == 1
    assert report["stage4_pairs_created"] == 1
    assert report["pair_finetune_applied"] is True
    assert report["light_pair_finetune_applied"] is True
    assert report["world_pair_finetune_applied"] is True
    assert report["light_model_variant"] == "v2"
    assert report["world_model_variant"] == "v2"
    assert report["pair_source_counts"]["stage5_trace_first_replacement"] == 1
    assert report["pair_source_counts"]["stage4_buffer"] == 1
    assert report["pair_source_weights"]["stage4_buffer"] == pytest.approx(0.2)
    assert "base_train_metrics" in report
    assert "pair_finetune_metrics" in report
    assert report["world_pair_ft_frozen_modules"] == ["traj_decoder"]
    assert report["world_pair_ft_trainable_modules"] == ["fusion", "risk_score_head"]
    assert report["world_pair_ft_source_mix"]["stage5_steps"] == 3
    assert report["world_pair_ft_source_mix"]["stage4_steps"] == 1
    assert report["stage2_pair_source_health"]["status"] == "healthy"
    risk_v2_summary = json.loads(Path(pipeline.risk_v2_eval_summary_path).read_text(encoding="utf-8"))
    assert risk_v2_summary["pair_finetune_applied"] is True
    assert risk_v2_summary["light_pair_finetune_applied"] is True
    assert risk_v2_summary["world_pair_finetune_applied"] is True
    assert risk_v2_summary["world_pair_ft_source_mix"]["stage5_steps"] == 3
    assert "stage2_snapshot" in risk_v2_summary
    assert "pair_source_consistency" in risk_v2_summary
    assert risk_v2_summary["after_trace_metrics"] == {"ANCHOR": None, "BOUNDARY": None, "CONSERVATIVE": None, "HOLDOUT": None}
    assert risk_v2_summary["after_trace_metrics_complete"] is False
    assert risk_v2_summary["score_spread_before_after"]["light"]["before"]["score_spread"] == pytest.approx(0.01)
    assert risk_v2_summary["score_spread_before_after"]["world"]["after"]["score_spread"] == pytest.approx(0.04)
    assert report["stage5_pair_ranking_accuracy_before_after"]["after"] == pytest.approx(0.75)
    assert report["stage4_pair_ranking_accuracy_before_after"]["after"] == pytest.approx(0.65)
    assert report["stage5_spread_eligible_pair_count"] == 2
    assert report["stage4_spread_eligible_pair_count"] == 0
    assert report["world_pair_ft_best_epoch"] == 1
    assert report["world_pair_ft_restored_best"] is True
    assert report["world_pair_finetune_mode"] == "fallback_all_pairs"
    assert report["stage5_requirement_met"] is True
    assert report["world_pair_gate_degraded"] is False
    assert risk_v2_summary["stage5_pair_ranking_accuracy_before_after"]["after"] == pytest.approx(0.75)
    assert risk_v2_summary["stage4_pair_ranking_accuracy_before_after"]["after"] == pytest.approx(0.65)


def test_train_models_world_pair_gate_fallback_all_pairs_applies_when_stage5_missing(monkeypatch):
    from safe_rl.data.types import RiskPairSample, SceneState, VehicleState
    pytest.importorskip("torch")

    config = _tiny_config()
    config.world_model.pair_finetune_gate_mode = "fallback_all_pairs"
    config.world_model.min_stage5_pairs_for_world_ft = 3
    pipeline = SafeRLPipeline(config)
    model_dir = _local_test_tmp_dir("ut_world_pair_gate_fallback")
    pipeline.light_model_path = model_dir / "light.pt"
    pipeline.world_model_path = model_dir / "world.pt"

    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    stage1_pair = RiskPairSample(
        history_scene=history,
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
    )

    class _DummyLightTrainer:
        def __init__(self, cfg, seed=0):
            _ = (cfg, seed)
            self.last_pair_ft_report = {"enabled": False, "pair_count": 0}
            self.last_train_report = {}

        def fit(self, train_samples, val_samples, tb_writer=None):
            _ = (train_samples, val_samples, tb_writer)
            return SimpleNamespace(device="cpu")

        def fine_tune_pairs(self, pair_samples, replay_samples=None, tb_writer=None):
            _ = (pair_samples, replay_samples, tb_writer)
            return {}

        def save(self, path: str):
            _ = path

    class _DummyWorldTrainer:
        def __init__(self, config, history_steps, seed=0):
            _ = (config, history_steps, seed)
            self.last_pair_ft_report = {}
            self.last_train_report = {}

        def fit(self, train_samples, val_samples, tb_writer=None):
            _ = (train_samples, val_samples, tb_writer)
            return SimpleNamespace(device="cpu")

        def _select_pair_ft_eval_samples(self, train_samples):
            _ = train_samples
            return []

        def evaluate_pairs(self, pair_samples):
            _ = pair_samples
            return {
                "pair_count": float(len(pair_samples)),
                "pair_ranking_accuracy": 0.5 if pair_samples else 0.0,
                "same_state_score_gap": 0.1 if pair_samples else 0.0,
                "score_spread": 0.1 if pair_samples else 0.0,
                "hard_negative_accuracy": 0.5 if pair_samples else 0.0,
            }

        def _evaluate_risk_only_samples(self, eval_replay_samples):
            _ = eval_replay_samples
            return {}

        def _spread_eligible_pair_count(self, pair_samples):
            return len(pair_samples)

        def fine_tune_pairs(
            self,
            pair_samples,
            replay_samples=None,
            tb_writer=None,
            stage5_pair_samples=None,
            stage1_probe_pair_samples=None,
            stage4_pair_samples=None,
        ):
            _ = (replay_samples, tb_writer, stage4_pair_samples)
            self.last_pair_ft_report = {
                "enabled": True,
                "pair_count": int(len(pair_samples)),
                "world_pair_ft_source_mix": {
                    "stage5_pair_count": int(len(stage5_pair_samples or [])),
                    "stage1_probe_pair_count": int(len(stage1_probe_pair_samples or [])),
                },
                "stage5_pair_ranking_accuracy_before_after": {"before": 0.0, "after": 0.0},
                "stage1_probe_pair_ranking_accuracy_before_after": {"before": 0.5, "after": 0.6},
                "stage4_pair_ranking_accuracy_before_after": {"before": 0.0, "after": 0.0},
                "stage5_same_state_score_gap_before_after": {"before": 0.0, "after": 0.0},
                "stage1_probe_same_state_score_gap_before_after": {"before": 0.1, "after": 0.12},
                "stage4_same_state_score_gap_before_after": {"before": 0.0, "after": 0.0},
                "stage5_score_spread_before_after": {"before": 0.0, "after": 0.0},
                "stage1_probe_score_spread_before_after": {"before": 0.1, "after": 0.12},
                "stage4_score_spread_before_after": {"before": 0.0, "after": 0.0},
                "stage5_spread_eligible_pair_count": 0,
                "stage1_probe_spread_eligible_pair_count": int(len(stage1_probe_pair_samples or [])),
                "stage4_spread_eligible_pair_count": 0,
                "world_pair_ft_best_epoch": 0,
                "world_pair_ft_best_metrics": {"pair_ranking_accuracy": 0.6},
                "world_pair_ft_restored_best": True,
            }
            return {"pair_ranking_accuracy": 0.6}

        def save(self, path: str):
            _ = path

    monkeypatch.setattr("safe_rl.models.light_risk_model.LightRiskTrainer", _DummyLightTrainer)
    monkeypatch.setattr("safe_rl.models.world_model.WorldModelTrainer", _DummyWorldTrainer)

    _, _, training_meta = pipeline.train_models(
        train_samples=[],
        val_samples=[],
        model_dir=model_dir,
        stage5_pair_samples=[],
        stage1_probe_pair_samples=[stage1_pair],
        stage4_pair_samples=[],
    )

    assert training_meta["world_pair_finetune_applied"] is True
    assert training_meta["world_pair_finetune_mode"] == "fallback_all_pairs"
    assert training_meta["stage5_requirement_met"] is False
    assert training_meta["world_pair_gate_degraded"] is True
    assert training_meta["world_pair_stage5_pairs_available"] == 0
    assert training_meta["world_pair_total_pairs_available"] == 1


def test_train_models_world_pair_gate_strict_skips_when_stage5_missing(monkeypatch):
    from safe_rl.data.types import RiskPairSample, SceneState, VehicleState
    pytest.importorskip("torch")

    config = _tiny_config()
    config.world_model.pair_finetune_gate_mode = "strict"
    config.world_model.min_stage5_pairs_for_world_ft = 2
    pipeline = SafeRLPipeline(config)
    model_dir = _local_test_tmp_dir("ut_world_pair_gate_strict")
    pipeline.light_model_path = model_dir / "light.pt"
    pipeline.world_model_path = model_dir / "world.pt"

    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    stage1_pair = RiskPairSample(
        history_scene=history,
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
    )

    class _DummyLightTrainer:
        def __init__(self, cfg, seed=0):
            _ = (cfg, seed)
            self.last_pair_ft_report = {"enabled": False, "pair_count": 0}
            self.last_train_report = {}

        def fit(self, train_samples, val_samples, tb_writer=None):
            _ = (train_samples, val_samples, tb_writer)
            return SimpleNamespace(device="cpu")

        def fine_tune_pairs(self, pair_samples, replay_samples=None, tb_writer=None):
            _ = (pair_samples, replay_samples, tb_writer)
            return {}

        def save(self, path: str):
            _ = path

    class _DummyWorldTrainer:
        def __init__(self, config, history_steps, seed=0):
            _ = (config, history_steps, seed)
            self.last_pair_ft_report = {}
            self.last_train_report = {}

        def fit(self, train_samples, val_samples, tb_writer=None):
            _ = (train_samples, val_samples, tb_writer)
            return SimpleNamespace(device="cpu")

        def _select_pair_ft_eval_samples(self, train_samples):
            _ = train_samples
            return []

        def evaluate_pairs(self, pair_samples):
            return {
                "pair_count": float(len(pair_samples)),
                "pair_ranking_accuracy": 0.0,
                "same_state_score_gap": 0.0,
                "score_spread": 0.0,
                "hard_negative_accuracy": 0.0,
            }

        def _evaluate_risk_only_samples(self, eval_replay_samples):
            _ = eval_replay_samples
            return {}

        def _spread_eligible_pair_count(self, pair_samples):
            _ = pair_samples
            return 0

        def fine_tune_pairs(self, *args, **kwargs):
            raise AssertionError("fine_tune_pairs should not be called in strict mode without stage5 pairs")

        def save(self, path: str):
            _ = path

    monkeypatch.setattr("safe_rl.models.light_risk_model.LightRiskTrainer", _DummyLightTrainer)
    monkeypatch.setattr("safe_rl.models.world_model.WorldModelTrainer", _DummyWorldTrainer)

    _, _, training_meta = pipeline.train_models(
        train_samples=[],
        val_samples=[],
        model_dir=model_dir,
        stage5_pair_samples=[],
        stage1_probe_pair_samples=[stage1_pair],
        stage4_pair_samples=[],
    )

    assert training_meta["world_pair_finetune_applied"] is False
    assert training_meta["world_pair_finetune_mode"] == "strict"
    assert training_meta["stage5_requirement_met"] is False
    assert training_meta["world_pair_finetune_skipped_reason"] == "insufficient_stage5_pairs"


def test_stage5_pair_mining_supports_non_shield_trace_prefix_directories():
    from safe_rl.data.types import SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_dir_discovery_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 1.0, 2.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    trace_dir = pipeline.reports_dir / "balance_scan_custom"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_path = trace_dir / "pair_00_seed_42.json"
    pipeline._write_json(
        pair_path,
        {
            "pair_index": 0,
            "seed": 42,
            "first_replacement_step": 0,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": dataclass_to_dict(history), "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": dataclass_to_dict(history), "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
    )
    pipeline._write_json(trace_dir / "trace_summary.json", {"variant_name": "BALANCE_SCAN_CUSTOM", "pair_files": [str(pair_path)], "seeds": [42]})

    stage5_pairs, summary = pipeline._build_stage5_pair_samples()
    assert len(stage5_pairs) == 1
    assert summary["trace_dirs_seen"] >= 1
    assert "balance_scan_custom" in summary["trace_dir_names"]
    assert summary["discovery"]["selected_dirs"] >= 1



def test_stage5_pair_from_payload_uses_history_fallback_and_top_level_same_state_proof():
    from safe_rl.data.types import SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_pair_fallback_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    history = [
        SceneState(
            timestamp=0.0,
            ego_id="ego",
            vehicles=[VehicleState("ego", 1.0, 2.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)],
        )
    ] * 2
    history_payload = dataclass_to_dict(history)
    proof = pipeline._build_same_state_proof(history_payload, {"ego_lane_id": "1"})
    payload = {
        "pair_index": 0,
        "seed": 42,
        "first_replacement_step": 0,
        **proof,
        "baseline_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 4,
                "history_scene": history_payload,
                "collision": False,
                "ttc": 5.0,
                "min_distance": 10.0,
                "reward": 0.3,
            }
        ],
        "shielded_steps": [
            {
                "step_index": 0,
                "raw_action": 4,
                "final_action": 3,
                "collision": True,
                "ttc": 0.5,
                "min_distance": 0.5,
                "reward": -0.2,
            }
        ],
    }

    sample, reason = pipeline._stage5_pair_from_payload(payload)

    assert reason == ""
    assert sample is not None
    assert sample.preferred_action == 4
    assert sample.meta["history_hash"] == proof["history_hash"]
    assert sample.meta["trusted_for_spread"] is True



def test_stage2_pair_dataset_builder_reports_detailed_stage5_skip_reasons():
    from safe_rl.data.types import SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage2_pair_reasons_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    history = [
        SceneState(
            timestamp=0.0,
            ego_id="ego",
            vehicles=[VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)],
        )
    ] * 2
    history_payload = dataclass_to_dict(history)
    proof = pipeline._build_same_state_proof(history_payload, {"ego_lane_id": "1"})

    trace_dir = pipeline.reports_dir / "shield_trace_d1"
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_payloads = {
        "d1pair_00_seed_42.json": {
            "pair_index": 0,
            "seed": 42,
            "first_replacement_step": 0,
            **proof,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
        "d1pair_01_seed_43.json": {
            "pair_index": 1,
            "seed": 43,
            "first_replacement_step": 0,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
        "d1pair_02_seed_44.json": {
            "pair_index": 2,
            "seed": 44,
            "first_replacement_step": 2,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": history_payload, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
        "d1pair_03_seed_45.json": {
            "pair_index": 3,
            "seed": 45,
            "first_replacement_step": 0,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "history_scene": history_payload, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
        "d1pair_04_seed_46.json": {
            "pair_index": 4,
            "seed": 46,
            "first_replacement_step": 0,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
        },
        "d1pair_05_seed_47.json": {
            "pair_index": 5,
            "seed": 47,
            "first_replacement_step": -1,
            "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
            "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": history_payload, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
        },
    }
    pair_files = []
    for filename, payload in pair_payloads.items():
        pair_path = trace_dir / filename
        pipeline._write_json(pair_path, payload)
        pair_files.append(str(pair_path))
    pipeline._write_json(trace_dir / "trace_summary.json", {"variant_name": "D1", "pair_files": pair_files, "seeds": [42]})

    payload = pipeline._build_pair_datasets_for_stage2()
    stage5_summary = payload["generation_summary"]["stage5"]

    assert len(payload["stage5_pairs"]) == 1
    assert stage5_summary["pairs_created"] == 1
    assert stage5_summary["skipped_missing_history"] == 1
    assert stage5_summary["skipped_invalid_first_replacement_step"] == 1
    assert stage5_summary["skipped_missing_actions"] == 1
    assert stage5_summary["skipped_no_preference"] == 1
    assert stage5_summary["skipped_no_replacement"] == 1



def test_stage5_pair_from_payload_reports_missing_same_state_proof(monkeypatch):
    from safe_rl.data.types import SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_pair_proof_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    history = [
        SceneState(
            timestamp=0.0,
            ego_id="ego",
            vehicles=[VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)],
        )
    ] * 2
    history_payload = dataclass_to_dict(history)
    payload = {
        "pair_index": 0,
        "seed": 42,
        "first_replacement_step": 0,
        "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": history_payload, "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
        "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": history_payload, "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
    }
    monkeypatch.setattr(pipeline, "_resolve_same_state_proof", lambda *_args, **_kwargs: {})

    sample, reason = pipeline._stage5_pair_from_payload(payload)

    assert sample is None
    assert reason == "skipped_missing_same_state_proof"




def test_stage5_pair_miner_ignores_pre_v2_and_holdout_trace_dirs():
    from safe_rl.data.types import SceneState, VehicleState, dataclass_to_dict

    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage5_pair_filter_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)

    assert pipeline.reports_dir is not None
    included = pipeline.reports_dir / "shield_trace_d1"
    excluded_pre = pipeline.reports_dir / "shield_trace_d1_pre_v2"
    excluded_holdout = pipeline.reports_dir / "shield_trace_holdout_c1"
    history = [SceneState(timestamp=0.0, ego_id="ego", vehicles=[VehicleState("ego", 1.0, 2.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)])] * 2
    pair_payload = {
        "pair_index": 0,
        "seed": 42,
        "first_replacement_step": 0,
        "baseline_steps": [{"step_index": 0, "raw_action": 4, "final_action": 4, "history_scene": dataclass_to_dict(history), "collision": False, "ttc": 5.0, "min_distance": 10.0, "reward": 0.3}],
        "shielded_steps": [{"step_index": 0, "raw_action": 4, "final_action": 3, "history_scene": dataclass_to_dict(history), "collision": True, "ttc": 0.5, "min_distance": 0.5, "reward": -0.2}],
    }
    for path in [included, excluded_pre, excluded_holdout]:
        path.mkdir(parents=True, exist_ok=True)
        pair_path = path / "pair_00_seed_42.json"
        pipeline._write_json(pair_path, pair_payload)
        pipeline._write_json(path / "trace_summary.json", {"variant_name": path.name, "pair_files": [str(pair_path)], "seeds": [42]})

    payload = pipeline._build_pair_datasets_for_stage2()

    assert payload["generation_summary"]["stage5"]["trace_dirs_seen"] == 1
    assert pipeline._include_trace_dir_for_stage5_pair_mining(included) is True
    assert pipeline._include_trace_dir_for_stage5_pair_mining(excluded_pre) is False
    assert pipeline._include_trace_dir_for_stage5_pair_mining(excluded_holdout) is False

def test_stage2_base_only_report_marks_pair_finetune_skipped(monkeypatch):
    import pickle

    config = _tiny_config()
    config.light_risk.pair_finetune = False
    config.world_model.pair_finetune = False
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_stage2_base_only_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage2", run_id=run_id)
    pipeline.datasets_dir.mkdir(parents=True, exist_ok=True)
    pipeline.models_dir.mkdir(parents=True, exist_ok=True)
    for path in [pipeline.train_pkl, pipeline.val_pkl, pipeline.test_pkl]:
        with open(path, "wb") as f:
            pickle.dump([], f)

    class _DummyPredictor:
        def __init__(self):
            self.device = "cpu"

    class _DummyEvaluator:
        def __init__(self, cfg):
            _ = cfg

        def evaluate_world_model(self, world_predictor, samples):
            _ = (world_predictor, samples)
            return {"traj_ade": 1.23, "risk_acc": 0.45, "risk_mae": 0.67}

    def _fake_train_models(*args, **kwargs):
        return _DummyPredictor(), _DummyPredictor(), {
            "pair_finetune_applied": False,
            "ranking_metrics": {"light": {}, "world": {}},
            "light_training": {"variant": "v2"},
            "world_training": {"variant": "v2"},
            "light_pair_ft": {},
            "world_pair_ft": {},
        }

    monkeypatch.setattr("safe_rl.pipeline.pipeline.SafeRLEvaluator", _DummyEvaluator)
    pipeline.train_models = _fake_train_models

    result = pipeline.run(stage="stage2", run_id=run_id)["stage2"]
    report = json.loads(Path(result["training_report"]).read_text(encoding="utf-8"))

    assert result["pair_finetune_applied"] is False
    assert report["pair_finetune_applied"] is False
    assert report["world_eval_metrics"]["traj_ade"] == pytest.approx(1.23)
    assert report["pair_finetune_metrics"] == {"light": {}, "world": {}}


def test_risk_v2_summary_tracks_after_trace_metrics_by_role():
    config = _tiny_config()
    pipeline = SafeRLPipeline(config)
    run_id = f"ut_risk_v2_after_trace_{uuid.uuid4().hex[:8]}"
    pipeline._prepare_run_context(stage="stage5", run_id=run_id)

    assert pipeline.shield_trace_tuning_summary_path is not None
    pipeline._write_json(
        pipeline.shield_trace_tuning_summary_path,
        {
            "variants": [
                {
                    "variant_name": "PAIR_BOOTSTRAP",
                    "trace_summary_path": "pair_bootstrap/trace_summary.json",
                    "margin_analysis_path": "pair_bootstrap/margin_analysis.json",
                    "candidate_selected_count": 12,
                    "mean_intervention_count": 4.0,
                    "mean_risk_reduction": 0.05,
                    "mean_reward_gap_to_baseline_policy": -0.03,
                    "margin_near_threshold_band_ratio": 0.15,
                    "best_margin_near_threshold_band_ratio": 0.25,
                    "effective_shield_config": {"replacement_min_risk_margin": 0.02},
                },
                {
                    "variant_name": "G4",
                    "trace_summary_path": "g4/trace_summary.json",
                    "margin_analysis_path": "g4/margin_analysis.json",
                    "candidate_selected_count": 7,
                    "mean_intervention_count": 1.5,
                    "mean_risk_reduction": 0.01,
                    "mean_reward_gap_to_baseline_policy": -0.01,
                    "margin_near_threshold_band_ratio": 0.44,
                    "best_margin_near_threshold_band_ratio": 0.54,
                    "effective_shield_config": {"replacement_min_risk_margin": 0.08},
                },
                {
                    "variant_name": "G5",
                    "trace_summary_path": "g5/trace_summary.json",
                    "margin_analysis_path": "g5/margin_analysis.json",
                    "candidate_selected_count": 5,
                    "mean_intervention_count": 1.0,
                    "mean_risk_reduction": 0.01,
                    "mean_reward_gap_to_baseline_policy": -0.01,
                    "margin_near_threshold_band_ratio": 0.55,
                    "best_margin_near_threshold_band_ratio": 0.65,
                    "effective_shield_config": {"replacement_min_risk_margin": 0.09},
                },
                {
                    "variant_name": "F1",
                    "trace_summary_path": "f1/trace_summary.json",
                    "margin_analysis_path": "f1/margin_analysis.json",
                    "candidate_selected_count": 0,
                    "mean_intervention_count": 0.0,
                    "mean_risk_reduction": 0.0,
                    "mean_reward_gap_to_baseline_policy": 0.0,
                    "margin_near_threshold_band_ratio": 0.66,
                    "best_margin_near_threshold_band_ratio": 0.76,
                    "effective_shield_config": {"replacement_min_risk_margin": 0.103},
                },
                {
                    "variant_name": "HOLDOUT_C1",
                    "trace_summary_path": "holdout/trace_summary.json",
                    "margin_analysis_path": "holdout/margin_analysis.json",
                    "candidate_selected_count": 0,
                    "mean_intervention_count": 0.0,
                    "mean_risk_reduction": 0.0,
                    "mean_reward_gap_to_baseline_policy": 0.0,
                    "margin_near_threshold_band_ratio": 0.77,
                    "best_margin_near_threshold_band_ratio": 0.87,
                    "effective_shield_config": {"replacement_min_risk_margin": 0.08},
                },
            ]
        },
    )

    stage2_report = {
        "light_model_variant": "v2",
        "world_model_variant": "v2",
        "pair_finetune_applied": True,
        "light_pair_finetune_applied": False,
        "world_pair_finetune_applied": True,
        "pair_source_weights": {"stage5_trace_first_replacement": 1.0, "stage1_probe_same_state": 0.7, "stage4_buffer": 0.2},
        "world_pair_ft_source_mix": {"stage5_steps": 3, "stage1_probe_steps": 2, "stage4_steps": 1},
        "base_train_metrics": {},
        "pair_finetune_metrics": {
            "light": {"before_pair_metrics": {}, "after_pair_metrics": {}},
            "world": {"before_pair_metrics": {}, "after_pair_metrics": {}},
        },
        "world_pair_ft_frozen_modules": ["traj_decoder"],
        "world_pair_ft_trainable_modules": ["fusion", "risk_score_head"],
    }
    pipeline._write_risk_v2_eval_summary_from_stage2(stage2_report)
    summary = pipeline._write_risk_v2_eval_summary_from_stage5({})

    assert summary is not None
    assert "stage2_snapshot" in summary
    assert "stage5_snapshot" in summary
    assert summary["stage2_snapshot"]["world_pair_finetune_applied"] is True
    assert summary["after_trace_metrics_complete"] is True
    assert summary["after_trace_metrics"]["ANCHOR"]["variant_name"] == "PAIR_BOOTSTRAP"
    assert summary["after_trace_metrics"]["BOUNDARY"]["variant_name"] == "G5"
    assert summary["after_trace_metrics"]["CONSERVATIVE"]["variant_name"] == "F1"
    assert summary["after_trace_metrics"]["HOLDOUT"]["variant_name"] == "HOLDOUT_C1"
    assert summary["margin_near_threshold_band_ratio_before_after"]["ANCHOR"]["after"] == pytest.approx(0.15)
    assert summary["margin_near_threshold_band_ratio_before_after"]["BOUNDARY"]["after"] == pytest.approx(0.55)
    assert summary["margin_near_threshold_band_ratio_before_after"]["CONSERVATIVE"]["after"] == pytest.approx(0.66)
    assert summary["margin_near_threshold_band_ratio_before_after"]["HOLDOUT"]["after"] == pytest.approx(0.77)
    assert summary["stage5_snapshot"]["after_trace_metrics_complete"] is True
