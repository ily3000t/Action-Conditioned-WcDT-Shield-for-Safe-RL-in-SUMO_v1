import json
import uuid
from types import SimpleNamespace
from pathlib import Path

import pytest

from safe_rl.config.config import SafeRLConfig
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
    assert result["policy_source"].endswith("policy_meta.json")
    assert result["shield_contribution_validated"] is True
