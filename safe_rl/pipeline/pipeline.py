import copy
import datetime as dt
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from safe_rl.buffer import InterventionBuffer
from safe_rl.config import SafeRLConfig, load_safe_rl_config
from safe_rl.data.collector import SumoDataCollector
from safe_rl.data.dataset_builder import ActionConditionedDatasetBuilder
from safe_rl.data.types import InterventionRecord
from safe_rl.eval import SafeRLEvaluator
from safe_rl.pipeline.tensorboard_logger import TensorboardManager
from safe_rl.shield import SafetyShield
from safe_rl.sim import create_backend


STAGE_ORDER = ("stage1", "stage2", "stage3", "stage4", "stage5")


@dataclass
class PolicyArtifactMeta:
    policy_type: str
    path: str
    algo: str
    created_at: str


class SafeRLPipeline:
    def __init__(self, config: SafeRLConfig):
        self.config = config
        self.output_root = Path("safe_rl_output")
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.run_id: Optional[str] = None
        self.run_root: Optional[Path] = None
        self.raw_dir: Optional[Path] = None
        self.datasets_dir: Optional[Path] = None
        self.models_dir: Optional[Path] = None
        self.policies_dir: Optional[Path] = None
        self.buffers_dir: Optional[Path] = None
        self.reports_dir: Optional[Path] = None
        self.sumo_logs_dir: Optional[Path] = None
        self.tensorboard_root: Optional[Path] = None
        self.manifest_path: Optional[Path] = None

        self.train_pkl: Optional[Path] = None
        self.val_pkl: Optional[Path] = None
        self.test_pkl: Optional[Path] = None
        self.light_model_path: Optional[Path] = None
        self.world_model_path: Optional[Path] = None
        self.policy_meta_path: Optional[Path] = None
        self.policy_sb3_path: Optional[Path] = None
        self.buffer_path: Optional[Path] = None
        self.report_path: Optional[Path] = None
        self.collector_failure_report_path: Optional[Path] = None
        self.warning_summary_report_path: Optional[Path] = None

        self.manifest: Dict = {}

    def run(self, stage: str = "all", run_id: Optional[str] = None) -> Dict:
        stage = (stage or "all").strip().lower()
        if stage not in ("all",) + STAGE_ORDER:
            raise ValueError(f"Unsupported stage: {stage}. Expected one of all, {', '.join(STAGE_ORDER)}")

        self._prepare_run_context(stage=stage, run_id=run_id)

        stages_to_run = list(STAGE_ORDER) if stage == "all" else [stage]
        stage_results: Dict[str, Dict] = {}
        stage_durations: Dict[str, float] = {}

        for current_stage in stages_to_run:
            stage_t0 = time.time()
            print(f"[Pipeline] {current_stage}: start", flush=True)

            tb_manager = self._create_tb_manager(current_stage)
            if self.config.tensorboard.enabled and not tb_manager.is_enabled():
                print(f"[TensorBoard] unavailable during {current_stage}, fallback to no-op logging", flush=True)

            try:
                if current_stage == "stage1":
                    result = self._run_stage1(tb_manager)
                elif current_stage == "stage2":
                    result = self._run_stage2(tb_manager)
                elif current_stage == "stage3":
                    result = self._run_stage3(tb_manager)
                elif current_stage == "stage4":
                    result = self._run_stage4(tb_manager)
                elif current_stage == "stage5":
                    result = self._run_stage5(tb_manager)
                else:
                    raise RuntimeError(f"Unexpected stage: {current_stage}")

                elapsed = time.time() - stage_t0
                stage_durations[current_stage] = elapsed
                tb_manager.add_scalar("eval", "runtime/stage_seconds", elapsed, 0)

                self._mark_stage_done(current_stage)
                self._save_manifest()

                stage_results[current_stage] = result
                print(f"[Pipeline] {current_stage}: done in {elapsed:.1f}s", flush=True)
            finally:
                tb_manager.close()

        final_result: Dict = {
            "run_id": self.run_id,
            "run_root": str(self.run_root),
            "manifest_path": str(self.manifest_path),
            "stage": stage,
            "stage_durations": stage_durations,
        }

        if stage == "all":
            final_result.update(stage_results.get("stage5", {}))
        else:
            final_result[stage] = stage_results.get(stage, {})

        return final_result

    def _prepare_run_context(self, stage: str, run_id: Optional[str]):
        if stage != "all" and not run_id:
            raise ValueError("--run-id is required when --stage is not all")

        self.run_id = run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_root = self.output_root / "runs" / self.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.run_root / "raw"
        self.datasets_dir = self.run_root / "datasets"
        self.models_dir = self.run_root / "models"
        self.policies_dir = self.run_root / "policies"
        self.buffers_dir = self.run_root / "buffers"
        self.reports_dir = self.run_root / "reports"
        self.sumo_logs_dir = self.run_root / "sumo_logs"
        self.tensorboard_root = self.run_root / "tensorboard"
        self.manifest_path = self.run_root / "manifest.json"

        self.train_pkl = self.datasets_dir / "train.pkl"
        self.val_pkl = self.datasets_dir / "val.pkl"
        self.test_pkl = self.datasets_dir / "test.pkl"
        self.light_model_path = self.models_dir / "light_risk.pt"
        self.world_model_path = self.models_dir / "world_model.pt"
        self.policy_meta_path = self.policies_dir / "policy_meta.json"
        self.policy_sb3_path = self.policies_dir / "ppo_sb3.zip"
        self.buffer_path = self.buffers_dir / "intervention_buffer.pkl"
        self.report_path = self.reports_dir / "pipeline_report.json"
        self.collector_failure_report_path = self.reports_dir / "collector_failures.json"
        self.warning_summary_report_path = self.reports_dir / "warning_summary.json"

        if self.manifest_path.exists():
            with self.manifest_path.open("r", encoding="utf-8") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {
                "run_id": self.run_id,
                "config_snapshot": asdict(self.config),
                "stage_done": {s: False for s in STAGE_ORDER},
                "artifact_paths": {},
                "timestamps": {},
            }

        artifact_paths = self.manifest.setdefault("artifact_paths", {})
        artifact_paths.update(
            {
                "raw_dir": str(self.raw_dir),
                "train_pkl": str(self.train_pkl),
                "val_pkl": str(self.val_pkl),
                "test_pkl": str(self.test_pkl),
                "light_model": str(self.light_model_path),
                "world_model": str(self.world_model_path),
                "policy_meta": str(self.policy_meta_path),
                "policy_sb3": str(self.policy_sb3_path),
                "buffer": str(self.buffer_path),
                "report": str(self.report_path),
                "collector_failure_report": str(self.collector_failure_report_path),
                "warning_summary_report": str(self.warning_summary_report_path),
                "sumo_logs_dir": str(self.sumo_logs_dir),
                "tensorboard_root": str(self.tensorboard_root),
            }
        )
        self.manifest.setdefault("stage_done", {s: False for s in STAGE_ORDER})
        self.manifest.setdefault("timestamps", {})
        self._save_manifest()

    def _save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def _mark_stage_done(self, stage: str):
        self.manifest.setdefault("stage_done", {})[stage] = True
        self.manifest.setdefault("timestamps", {})[stage] = dt.datetime.now().isoformat(timespec="seconds")

    def _create_tb_manager(self, stage: str) -> TensorboardManager:
        tb_cfg = copy.deepcopy(self.config.tensorboard)
        tb_cfg.root_dir = str(self.tensorboard_root)
        return TensorboardManager(tb_cfg, stage_prefix=stage)

    def _require_files(self, stage: str, files: Sequence[Path]):
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{stage} requires existing artifacts, but missing: {', '.join(missing)}"
            )

    def _config_with_run_paths(self) -> SafeRLConfig:
        cfg = copy.deepcopy(self.config)
        cfg.dataset.raw_log_dir = str(self.raw_dir)
        cfg.dataset.dataset_dir = str(self.datasets_dir)
        cfg.sim.runtime_log_dir = str(self.sumo_logs_dir)
        return cfg

    def _load_dataset_splits(self):
        self._require_files("dataset_load", [self.train_pkl, self.val_pkl, self.test_pkl])
        train_samples = ActionConditionedDatasetBuilder.load(str(self.train_pkl))
        val_samples = ActionConditionedDatasetBuilder.load(str(self.val_pkl))
        test_samples = ActionConditionedDatasetBuilder.load(str(self.test_pkl))
        return train_samples, val_samples, test_samples

    def _build_predictors_from_saved_models(self):
        self._require_files("model_load", [self.light_model_path, self.world_model_path])

        try:
            from safe_rl.models.light_risk_model import LightRiskPredictor, LightRiskTrainer
            from safe_rl.models.world_model import WorldModelPredictor, WorldModelTrainer
        except Exception as exc:
            raise RuntimeError(f"Failed to import model modules for loading: {exc}") from exc

        light_trainer = LightRiskTrainer(self.config.light_risk, seed=self.config.sim.random_seed)
        light_trainer.load(str(self.light_model_path))
        light_predictor = LightRiskPredictor(model=light_trainer.model, device=light_trainer.device)

        world_trainer = WorldModelTrainer(
            config=self.config.world_model,
            history_steps=self.config.sim.history_steps,
            seed=self.config.sim.random_seed,
        )
        world_trainer.load(str(self.world_model_path))
        world_predictor = WorldModelPredictor(
            model=world_trainer.model,
            tensorizer=world_trainer.tensorizer,
            device=world_trainer.device,
        )
        return light_predictor, world_predictor

    def _save_policy_artifact(self, policy) -> Dict:
        from safe_rl.rl.ppo import HeuristicPolicy, SB3PolicyAdapter

        self.policies_dir.mkdir(parents=True, exist_ok=True)

        created_at = dt.datetime.now().isoformat(timespec="seconds")
        if isinstance(policy, SB3PolicyAdapter):
            policy.model.save(str(self.policy_sb3_path))
            meta = PolicyArtifactMeta(
                policy_type="sb3",
                path="policies/ppo_sb3.zip",
                algo="PPO",
                created_at=created_at,
            )
        elif isinstance(policy, HeuristicPolicy):
            meta = PolicyArtifactMeta(
                policy_type="heuristic",
                path="",
                algo="HeuristicPolicy",
                created_at=created_at,
            )
        else:
            raise TypeError(f"Unsupported policy type for serialization: {type(policy).__name__}")

        payload = asdict(meta)
        with self.policy_meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    def _load_policy_artifact(self):
        from safe_rl.rl.ppo import HeuristicPolicy, SB3PolicyAdapter

        self._require_files("policy_load", [self.policy_meta_path])
        with self.policy_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        policy_type = str(meta.get("policy_type", "")).strip().lower()
        path_value = str(meta.get("path", "")).strip()

        if policy_type == "heuristic":
            return HeuristicPolicy()

        if policy_type == "sb3":
            if not path_value:
                raise FileNotFoundError("policy_meta.json indicates sb3 policy but path is empty")

            policy_path = Path(path_value)
            if not policy_path.is_absolute():
                policy_path = self.run_root / policy_path
            self._require_files("policy_load", [policy_path])

            try:
                from stable_baselines3 import PPO  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"stable_baselines3 is required to load sb3 policy: {exc}") from exc

            model = PPO.load(str(policy_path))
            return SB3PolicyAdapter(model=model)

        raise ValueError(f"Unsupported policy_type in policy_meta.json: {policy_type}")

    def _run_stage1(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage1: collect data and build dataset", flush=True)
        stage_config = self._config_with_run_paths()

        backend = create_backend(stage_config.sim)
        backend.start()
        collector = SumoDataCollector(backend=backend, config=stage_config)
        episodes = collector.collect()
        collector.save_raw_logs(episodes)
        backend.close()
        collector.save_failure_report(str(self.collector_failure_report_path))
        collector.save_warning_summary(str(self.warning_summary_report_path))
        failure_report = collector.failure_report()
        warning_report = collector.warning_summary()

        builder = ActionConditionedDatasetBuilder(sim_config=stage_config.sim, dataset_config=stage_config.dataset)
        samples = builder.build_samples(episodes)
        train_samples, val_samples, test_samples = builder.split_dataset(samples, seed=stage_config.sim.random_seed)
        builder.save_splits(train_samples, val_samples, test_samples)

        eval_writer = tb_manager.get_writer("eval")
        if eval_writer is not None:
            eval_writer.add_scalar("stage1/episodes", float(len(episodes)), 0)
            eval_writer.add_scalar("stage1/failed_episodes", float(failure_report["failed_episodes"]), 0)
            eval_writer.add_scalar("stage1/failure_rate", float(failure_report["failure_rate"]), 0)
            eval_writer.add_scalar("stage1/samples_total", float(len(samples)), 0)
            eval_writer.add_scalar("stage1/warnings_illegal_lane_index", float(warning_report["overall"]["illegal_lane_index"]["count"]), 0)
            eval_writer.add_scalar("stage1/warnings_no_connection_next_edge", float(warning_report["overall"]["no_connection_next_edge"]["count"]), 0)
            eval_writer.add_scalar("stage1/warning_acceptance_passed", float(bool(warning_report["acceptance"]["passed"])), 0)
            eval_writer.add_scalar("stage1/traci_command_errors", float(warning_report["overall"]["totals"]["traci_command_errors"]["count"]), 0)
            eval_writer.add_scalar("stage1/samples_train", float(len(train_samples)), 0)
            eval_writer.add_scalar("stage1/samples_val", float(len(val_samples)), 0)
            eval_writer.add_scalar("stage1/samples_test", float(len(test_samples)), 0)

        return {
            "episodes": len(episodes),
            "successful_episodes": failure_report["successful_episodes"],
            "failed_episodes": failure_report["failed_episodes"],
            "collector_failure_report": str(self.collector_failure_report_path),
            "warning_summary_report": str(self.warning_summary_report_path),
            "warning_acceptance_passed": bool(warning_report["acceptance"]["passed"]),
            "warning_acceptance": warning_report["acceptance"],
            "samples_total": len(samples),
            "samples_train": len(train_samples),
            "samples_val": len(val_samples),
            "samples_test": len(test_samples),
        }

    def _run_stage2(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage2: train light risk and world model", flush=True)
        self._require_files("stage2", [self.train_pkl, self.val_pkl, self.test_pkl])
        train_samples, val_samples, _ = self._load_dataset_splits()

        light_predictor, world_predictor = self.train_models(
            train_samples,
            val_samples,
            model_dir=self.models_dir,
            tb_light_writer=tb_manager.get_writer("light_risk"),
            tb_world_writer=tb_manager.get_writer("world_model"),
        )

        eval_writer = tb_manager.get_writer("eval")
        if eval_writer is not None:
            eval_writer.add_scalar("stage2/light_device_cuda", float("cuda" in str(light_predictor.device)), 0)
            eval_writer.add_scalar("stage2/world_device_cuda", float("cuda" in str(world_predictor.device)), 0)

        return {
            "light_model": str(self.light_model_path),
            "world_model": str(self.world_model_path),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        }

    def _run_stage3(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage3: train online policy with shield", flush=True)
        self._require_files("stage3", [self.light_model_path, self.world_model_path])

        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        policy = self.train_online_policy(shield, tb_writer=tb_manager.get_writer("ppo"))
        policy_meta = self._save_policy_artifact(policy)

        eval_writer = tb_manager.get_writer("eval")
        if eval_writer is not None:
            eval_writer.add_scalar("stage3/policy_is_sb3", float(policy_meta["policy_type"] == "sb3"), 0)

        return {
            "policy_meta": policy_meta,
            "policy_meta_path": str(self.policy_meta_path),
        }

    def _run_stage4(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage4: collect intervention buffer", flush=True)
        self._require_files("stage4", [self.light_model_path, self.world_model_path, self.policy_meta_path])

        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        policy = self._load_policy_artifact()

        intervention_buffer = self.collect_interventions(policy, shield, save_path=self.buffer_path)
        stats = intervention_buffer.stats()

        distill_writer = tb_manager.get_writer("distill")
        if distill_writer is not None:
            distill_writer.add_scalar("status/buffer_size", float(len(intervention_buffer)), 0)
            distill_writer.add_scalar("status/trigger_buffer_size", float(self.config.distill.trigger_buffer_size), 0)

        return {
            "buffer_path": str(self.buffer_path),
            "buffer_stats": stats,
        }

    def _run_stage5(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage5: distill and evaluate", flush=True)
        self._require_files(
            "stage5",
            [self.test_pkl, self.light_model_path, self.world_model_path, self.policy_meta_path, self.buffer_path],
        )

        _, _, test_samples = self._load_dataset_splits()
        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        shielded_policy = self._load_policy_artifact()

        buffer = InterventionBuffer(capacity=max(10000, self.config.distill.trigger_buffer_size * 4))
        buffer.load(str(self.buffer_path))

        from safe_rl.rl import PolicyDistiller

        distilled_policy = None
        distill_writer = tb_manager.get_writer("distill")
        if distill_writer is not None:
            distill_writer.add_scalar("status/buffer_size", float(len(buffer)), 0)
            distill_writer.add_scalar("status/trigger_buffer_size", float(self.config.distill.trigger_buffer_size), 0)

        if PolicyDistiller is not None:
            distiller = PolicyDistiller(config=self.config.distill)
            can_distill = distiller.should_distill(buffer)
            if distill_writer is not None:
                distill_writer.add_scalar("status/triggered", float(can_distill), 0)
                distill_writer.add_scalar("status/skipped", float(not can_distill), 0)
            if can_distill:
                distilled_policy = distiller.distill(buffer, tb_writer=distill_writer)

        evaluation = self.evaluate(
            shield=shield,
            shielded_policy=shielded_policy,
            world_predictor=world_predictor,
            test_samples=test_samples,
            distilled_policy=distilled_policy,
            tb_writer=tb_manager.get_writer("eval"),
        )
        evaluation["intervention_buffer"] = buffer.stats()
        self._save_report(evaluation)
        return evaluation

    def train_models(self, train_samples, val_samples, model_dir: Path, tb_light_writer=None, tb_world_writer=None):
        try:
            from safe_rl.models.light_risk_model import LightRiskTrainer
            from safe_rl.models.world_model import WorldModelTrainer
        except Exception as exc:
            raise RuntimeError(
                "Safe-RL model import failed. Expected PyTorch to be installed and no Waymo dependency required. "
                f"Original import error: {exc}"
            ) from exc

        model_dir.mkdir(parents=True, exist_ok=True)

        light_trainer = LightRiskTrainer(self.config.light_risk, seed=self.config.sim.random_seed)
        light_predictor = light_trainer.fit(train_samples, val_samples, tb_writer=tb_light_writer)
        light_trainer.save(str(self.light_model_path))

        world_trainer = WorldModelTrainer(
            config=self.config.world_model,
            history_steps=self.config.sim.history_steps,
            seed=self.config.sim.random_seed,
        )
        world_predictor = world_trainer.fit(train_samples, val_samples, tb_writer=tb_world_writer)
        world_trainer.save(str(self.world_model_path))
        return light_predictor, world_predictor

    def train_online_policy(self, shield, tb_writer=None):
        from safe_rl.rl.env import create_env
        from safe_rl.rl.ppo import SafePPOTrainer

        backend = create_backend(self.config.sim)
        backend.start()
        env = create_env(
            backend=backend,
            sim_config=self.config.sim,
            ppo_config=self.config.ppo,
            shield=shield,
        )
        trainer = SafePPOTrainer(self.config.ppo)
        policy = trainer.train(env, tb_writer=tb_writer)
        env.close()
        return policy

    def collect_interventions(self, policy, shield, save_path: Optional[Path] = None) -> InterventionBuffer:
        from safe_rl.rl.env import create_env

        backend = create_backend(self.config.sim)
        backend.start()
        env = create_env(
            backend=backend,
            sim_config=self.config.sim,
            ppo_config=self.config.ppo,
            shield=shield,
        )
        buffer = InterventionBuffer(capacity=max(10000, self.config.distill.trigger_buffer_size * 4))

        for _ in range(self.config.eval.eval_episodes):
            obs, _ = env.reset(options={"risky_mode": True})
            done = False
            while not done:
                action = int(policy.predict(obs, deterministic=True))
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info.get("intervened", False) and env.last_transition is not None:
                    decision = env.last_transition["decision"]
                    shield_meta = info.get("shield_meta", {})
                    record = InterventionRecord(
                        history_scene=env.last_transition["history_scene"],
                        raw_action=env.last_transition["raw_action"],
                        final_action=env.last_transition["final_action"],
                        raw_risk=float(info.get("risk_raw", decision.risk_raw)),
                        final_risk=float(info.get("risk_final", decision.risk_final)),
                        reason=str(info.get("intervention_reason", decision.reason)),
                        raw_future=shield_meta.get("raw_world_prediction"),
                        final_future=shield_meta.get("final_world_prediction"),
                        meta={"episode_step": env.step_count},
                    )
                    buffer.push(record)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            buffer.save(str(save_path))

        env.close()
        return buffer

    def evaluate(
        self,
        shield,
        shielded_policy,
        world_predictor,
        test_samples,
        distilled_policy=None,
        tb_writer=None,
    ) -> Dict:
        from safe_rl.rl.env import create_env
        from safe_rl.rl.ppo import HeuristicPolicy

        evaluator = SafeRLEvaluator(self.config.eval)

        baseline_backend = create_backend(self.config.sim)
        baseline_backend.start()
        baseline_env = create_env(baseline_backend, self.config.sim, self.config.ppo, shield=None)
        baseline_policy = HeuristicPolicy()
        baseline_metrics = evaluator.evaluate_policy(
            env=baseline_env,
            policy=baseline_policy,
            episodes=self.config.eval.eval_episodes,
            risky_mode=True,
            tb_writer=tb_writer,
            tb_prefix="baseline",
        )
        baseline_env.close()

        shield_backend = create_backend(self.config.sim)
        shield_backend.start()
        shield_env = create_env(shield_backend, self.config.sim, self.config.ppo, shield=shield)
        shielded_metrics = evaluator.evaluate_policy(
            env=shield_env,
            policy=shielded_policy,
            episodes=self.config.eval.eval_episodes,
            risky_mode=True,
            tb_writer=tb_writer,
            tb_prefix="shielded",
        )
        shield_env.close()

        delta = evaluator.compare_baseline_and_shielded(baseline_metrics, shielded_metrics)
        acceptance = evaluator.evaluate_acceptance(delta)

        world_metrics = evaluator.evaluate_world_model(world_predictor, test_samples[: min(200, len(test_samples))])

        result = {
            "world_model": world_metrics,
            "system_baseline": baseline_metrics,
            "system_shielded": shielded_metrics,
            "delta": delta,
            "acceptance_passed": acceptance,
        }

        if tb_writer is not None:
            tb_writer.add_scalar("summary/collision_reduction", float(delta.get("collision_reduction", 0.0)), 0)
            tb_writer.add_scalar("summary/efficiency_drop", float(delta.get("efficiency_drop", 0.0)), 0)
            tb_writer.add_scalar("summary/acceptance_passed", float(bool(acceptance)), 0)
            tb_writer.add_scalar("world_model/traj_ade", float(world_metrics.get("traj_ade", 0.0)), 0)
            tb_writer.add_scalar("world_model/risk_acc", float(world_metrics.get("risk_acc", 0.0)), 0)
            tb_writer.add_scalar("world_model/risk_mae", float(world_metrics.get("risk_mae", 0.0)), 0)

        if distilled_policy is not None:
            distill_backend = create_backend(self.config.sim)
            distill_backend.start()
            distill_env = create_env(distill_backend, self.config.sim, self.config.ppo, shield=shield)
            distilled_metrics = evaluator.evaluate_policy(
                env=distill_env,
                policy=distilled_policy,
                episodes=self.config.eval.eval_episodes,
                risky_mode=True,
                tb_writer=tb_writer,
                tb_prefix="distilled",
            )
            distill_env.close()
            result["system_distilled"] = distilled_metrics

        return result

    def _save_report(self, report: Dict):
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


def run_safe_rl_pipeline(config_path: Optional[str] = None, stage: str = "all", run_id: Optional[str] = None) -> Dict:
    config = load_safe_rl_config(config_path)
    pipeline = SafeRLPipeline(config)
    return pipeline.run(stage=stage, run_id=run_id)








