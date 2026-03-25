import copy
import hashlib
import inspect
import math
import datetime as dt
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from safe_rl.buffer import InterventionBuffer
from safe_rl.config import SafeRLConfig, load_safe_rl_config
from safe_rl.data.collector import SumoDataCollector
from safe_rl.data.dataset_builder import ActionConditionedDatasetBuilder
from safe_rl.data.pair_dataset import save_risk_pairs, summarize_pair_sources
from safe_rl.data.types import InterventionRecord, RiskPairSample, dataclass_to_dict, scene_state_list_from_dicts
from safe_rl.eval import SafeRLEvaluator
from safe_rl.pipeline.session_event_logger import IncrementalSessionEventLogger
from safe_rl.pipeline.telemetry import BufferTelemetryTracker, Stage3TelemetryTracker
from safe_rl.pipeline.tensorboard_logger import TensorboardManager
from safe_rl.shield import SafetyShield
from safe_rl.sim import create_backend


STAGE_ORDER = ("stage1", "stage2", "stage3", "stage4", "stage5")
TRACE_TRUST_TTC_DELTA = 0.75
TRACE_TRUST_DISTANCE_DELTA = 2.0
TRACE_TRUST_REWARD_DELTA = 0.10
TRACE_TRUST_TTC_SUPPORT = 0.50
TRACE_TRUST_DISTANCE_SUPPORT = 1.50


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
        self.stage2_training_report_path: Optional[Path] = None
        self.pairs_stage4_path: Optional[Path] = None
        self.pairs_stage5_path: Optional[Path] = None
        self.stage3_runtime_config_path: Optional[Path] = None
        self.stage3_session_events_path: Optional[Path] = None
        self.stage4_buffer_report_path: Optional[Path] = None
        self.stage5_paired_episode_results_path: Optional[Path] = None
        self.risk_v2_eval_summary_path: Optional[Path] = None
        self.shield_sweep_summary_path: Optional[Path] = None
        self.shield_trace_dir: Optional[Path] = None
        self.shield_trace_summary_path: Optional[Path] = None
        self.shield_trace_tuning_summary_path: Optional[Path] = None
        self.shield_margin_analysis_summary_path: Optional[Path] = None

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
        self.stage2_training_report_path = self.reports_dir / "stage2_training_report.json"
        self.pairs_stage4_path = self.datasets_dir / "pairs_stage4.pkl"
        self.pairs_stage5_path = self.datasets_dir / "pairs_stage5.pkl"
        self.stage3_runtime_config_path = self.reports_dir / "stage3_runtime_config.json"
        self.stage3_session_events_path = self.reports_dir / "stage3_session_events.json"
        self.stage4_buffer_report_path = self.reports_dir / "stage4_buffer_report.json"
        self.stage5_paired_episode_results_path = self.reports_dir / "stage5_paired_episode_results.json"
        self.risk_v2_eval_summary_path = self.reports_dir / "risk_v2_eval_summary.json"
        self.shield_sweep_summary_path = self.reports_dir / "shield_sweep_summary.json"
        self.shield_trace_dir = self.reports_dir / str(self.config.shield_trace.trace_dir_name)
        self.shield_trace_summary_path = self.shield_trace_dir / "trace_summary.json"
        self.shield_trace_tuning_summary_path = self.reports_dir / "shield_trace_tuning_summary.json"
        self.shield_margin_analysis_summary_path = self.reports_dir / "shield_margin_analysis_summary.json"

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
                "stage2_training_report": str(self.stage2_training_report_path),
                "pairs_stage4": str(self.pairs_stage4_path),
                "pairs_stage5": str(self.pairs_stage5_path),
                "stage3_runtime_config_report": str(self.stage3_runtime_config_path),
                "stage3_session_events_report": str(self.stage3_session_events_path),
                "stage4_buffer_report": str(self.stage4_buffer_report_path),
                "stage5_paired_episode_results": str(self.stage5_paired_episode_results_path),
                "risk_v2_eval_summary_report": str(self.risk_v2_eval_summary_path),
                "shield_sweep_summary_report": str(self.shield_sweep_summary_path),
                "shield_trace_summary_report": str(self.shield_trace_summary_path),
                "shield_trace_tuning_summary_report": str(self.shield_trace_tuning_summary_path),
                "shield_margin_analysis_summary_report": str(self.shield_margin_analysis_summary_path),
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

    def _write_json(self, path: Path, payload: Dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    def _build_training_runtime_report(self, stage: str, stage_config: SafeRLConfig, backend, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "stage": stage,
            "run_id": self.run_id,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "sim_config": asdict(stage_config.sim),
            "backend": backend.get_runtime_diagnostics(),
        }
        if extra:
            payload.update(extra)
        return payload

    def _build_training_session_report(
        self,
        stage: str,
        backend,
        env,
        error: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        payload = {
            "stage": stage,
            "run_id": self.run_id,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "training_error": error,
            "env_episodes": env.get_session_records() if env is not None else [],
            "backend_session_events": backend.get_session_events(clear=False) if backend is not None else [],
            "backend_final": backend.get_runtime_diagnostics() if backend is not None else {},
        }
        if extra:
            payload.update(extra)
        return payload

    def _load_dataset_splits(self):
        self._require_files("dataset_load", [self.train_pkl, self.val_pkl, self.test_pkl])
        train_samples = ActionConditionedDatasetBuilder.load(str(self.train_pkl))
        val_samples = ActionConditionedDatasetBuilder.load(str(self.val_pkl))
        test_samples = ActionConditionedDatasetBuilder.load(str(self.test_pkl))
        return train_samples, val_samples, test_samples

    def _build_pair_datasets_for_stage2(self) -> Dict[str, Any]:
        stage5_pairs, stage5_summary = self._build_stage5_pair_samples()
        stage4_pairs, stage4_summary = self._build_stage4_pair_samples()
        if stage5_pairs:
            save_risk_pairs(str(self.pairs_stage5_path), stage5_pairs)
        if stage4_pairs:
            save_risk_pairs(str(self.pairs_stage4_path), stage4_pairs)
        all_pairs = list(stage5_pairs) + list(stage4_pairs)
        return {
            "stage5_pairs": stage5_pairs,
            "stage4_pairs": stage4_pairs,
            "pair_source_counts": summarize_pair_sources(all_pairs),
            "pair_source_weights": {
                "stage5_trace_first_replacement": float(self.config.light_risk.stage5_pair_weight),
                "stage4_buffer": float(self.config.light_risk.stage4_pair_weight),
            },
            "pair_dataset_paths": {
                "stage5": str(self.pairs_stage5_path),
                "stage4": str(self.pairs_stage4_path),
            },
            "generation_summary": {
                "stage5": stage5_summary,
                "stage4": stage4_summary,
            },
        }

    def _build_stage5_pair_samples(self):
        summary = {
            "trace_dirs_seen": 0,
            "pair_files_seen": 0,
            "pairs_created": 0,
            "skipped_missing_same_state_proof": 0,
            "skipped_missing_history": 0,
            "skipped_invalid_first_replacement_step": 0,
            "skipped_missing_actions": 0,
            "skipped_no_replacement": 0,
            "skipped_no_preference": 0,
        }
        pair_samples: List[RiskPairSample] = []
        if self.reports_dir is None:
            return pair_samples, summary

        trace_dirs = sorted(
            path for path in self.reports_dir.glob("shield_trace*")
            if path.is_dir() and self._find_trace_summary_path(path) is not None
        )
        summary["trace_dirs_seen"] = len(trace_dirs)
        for trace_dir in trace_dirs:
            for pair_path in self._find_trace_pair_paths(trace_dir):
                summary["pair_files_seen"] += 1
                payload = self._read_json(pair_path)
                sample, reason = self._stage5_pair_from_payload(payload)
                if sample is None:
                    if reason in summary:
                        summary[reason] += 1
                    continue
                pair_samples.append(sample)
                summary["pairs_created"] += 1
        return pair_samples, summary

    def _stage5_pair_from_payload(self, payload: Dict[str, Any]):
        first_replacement_step = int(payload.get("first_replacement_step", -1))
        if first_replacement_step < 0:
            return None, "skipped_no_replacement"

        shielded_steps = list(payload.get("shielded_steps", []) or [])
        baseline_steps = list(payload.get("baseline_steps", []) or [])
        aligned_steps = list(payload.get("aligned_steps", []) or [])
        aligned_step = self._trace_payload_step_at(aligned_steps, first_replacement_step)
        selected_step = self._trace_payload_step_at(shielded_steps, first_replacement_step)
        aligned_shielded_step = dict(aligned_step.get("shielded", {}) or {})
        aligned_baseline_step = dict(aligned_step.get("baseline", {}) or {})
        baseline_step = self._trace_payload_step_at(baseline_steps, first_replacement_step)

        if not selected_step:
            selected_step = aligned_shielded_step
        if not selected_step:
            return None, "skipped_invalid_first_replacement_step"

        raw_action = self._first_valid_int(
            selected_step.get("raw_action", None),
            aligned_baseline_step.get("raw_action", None),
            baseline_step.get("raw_action", None),
        )
        replaced_action = self._first_valid_int(
            selected_step.get("final_action", None),
            aligned_shielded_step.get("final_action", None),
        )
        if raw_action < 0 or replaced_action < 0:
            return None, "skipped_missing_actions"

        history_scene_payload = self._first_nonempty_list(
            selected_step.get("history_scene", None),
            aligned_shielded_step.get("history_scene", None),
            baseline_step.get("history_scene", None),
            aligned_baseline_step.get("history_scene", None),
            payload.get("history_scene", None),
        )
        if not history_scene_payload:
            return None, "skipped_missing_history"

        baseline_suffix_steps = list(baseline_steps[first_replacement_step:]) if first_replacement_step < len(baseline_steps) else ([baseline_step] if baseline_step else [])
        shielded_suffix_steps = list(shielded_steps[first_replacement_step:]) if first_replacement_step < len(shielded_steps) else ([selected_step] if selected_step else [])
        preferred_action = self._preferred_action_from_trace_suffix(
            baseline_steps=baseline_suffix_steps,
            shielded_steps=shielded_suffix_steps,
            raw_action=raw_action,
            replaced_action=replaced_action,
        )
        if preferred_action is None:
            return None, "skipped_no_preference"

        history_scene = scene_state_list_from_dicts(history_scene_payload)
        baseline_suffix_target = self._trace_suffix_target(baseline_suffix_steps)
        shielded_suffix_target = self._trace_suffix_target(shielded_suffix_steps)
        baseline_collision = any(bool(step.get("collision", False)) for step in baseline_suffix_steps)
        shielded_collision = any(bool(step.get("collision", False)) for step in shielded_suffix_steps)
        baseline_min_ttc = self._trace_suffix_min_metric(baseline_suffix_steps, "ttc", fallback=1e6, prefer_lower=True)
        shielded_min_ttc = self._trace_suffix_min_metric(shielded_suffix_steps, "ttc", fallback=1e6, prefer_lower=True)
        baseline_min_distance = self._trace_suffix_min_metric(baseline_suffix_steps, "min_distance", fallback=1e6, prefer_lower=True)
        shielded_min_distance = self._trace_suffix_min_metric(shielded_suffix_steps, "min_distance", fallback=1e6, prefer_lower=True)
        reward_gap = float(self._trace_suffix_reward(shielded_suffix_steps) - self._trace_suffix_reward(baseline_suffix_steps))
        hard_negative = bool((not baseline_collision and shielded_collision) or reward_gap < -0.05)
        trusted_for_spread = self._is_trusted_stage5_pair(
            baseline_collision=baseline_collision,
            shielded_collision=shielded_collision,
            baseline_min_ttc=baseline_min_ttc,
            shielded_min_ttc=shielded_min_ttc,
            baseline_min_distance=baseline_min_distance,
            shielded_min_distance=shielded_min_distance,
            reward_gap=reward_gap,
            hard_negative=hard_negative,
        )
        proof = self._resolve_same_state_proof(payload, history_scene_payload, selected_step)
        if not self._has_same_state_proof(proof):
            return None, "skipped_missing_same_state_proof"
        sample = RiskPairSample(
            history_scene=history_scene,
            action_a=raw_action,
            action_b=replaced_action,
            preferred_action=int(preferred_action),
            source="stage5_trace_first_replacement",
            weight=float(self.config.light_risk.stage5_pair_weight) * (2.0 if hard_negative else 1.0),
            meta={
                "pair_index": int(payload.get("pair_index", -1)),
                "seed": int(payload.get("seed", -1)),
                "first_replacement_step": int(first_replacement_step),
                "target_risk_a": float(baseline_suffix_target),
                "target_risk_b": float(shielded_suffix_target),
                "hard_negative": hard_negative,
                "trusted_for_spread": bool(trusted_for_spread),
                "baseline_collision": bool(baseline_collision),
                "shielded_collision": bool(shielded_collision),
                "baseline_min_ttc": float(baseline_min_ttc),
                "shielded_min_ttc": float(shielded_min_ttc),
                "baseline_min_distance": float(baseline_min_distance),
                "shielded_min_distance": float(shielded_min_distance),
                "reward_gap": reward_gap,
                "source_path": str(payload.get("pair_path", payload.get("source_path", ""))),
                **proof,
            },
        )
        return sample, ""

    def _trace_payload_step_at(self, steps: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(steps):
            return {}
        return dict(steps[index] or {})

    def _first_nonempty_list(self, *candidates: Any) -> List[Dict[str, Any]]:
        for candidate in candidates:
            payload = list(candidate or [])
            if payload:
                return payload
        return []

    def _first_valid_int(self, *candidates: Any) -> int:
        for candidate in candidates:
            try:
                value = int(candidate)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return value
        return -1

    def _resolve_same_state_proof(
        self,
        payload: Dict[str, Any],
        history_scene_payload: Sequence[Dict[str, Any]],
        reference_step: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        proof = {
            "history_hash": str(payload.get("history_hash", "")),
            "ego_lane_id": str(payload.get("ego_lane_id", "")),
            "ego_x": float(payload.get("ego_x", 0.0) or 0.0),
            "ego_y": float(payload.get("ego_y", 0.0) or 0.0),
            "ego_speed": float(payload.get("ego_speed", 0.0) or 0.0),
            "neighbor_summary": list(payload.get("neighbor_summary", []) or []),
        }
        if self._has_same_state_proof(proof):
            return proof
        return self._build_same_state_proof(history_scene_payload, reference_step)

    def _has_same_state_proof(self, proof: Optional[Dict[str, Any]]) -> bool:
        item = dict(proof or {})
        if str(item.get("history_hash", "")):
            return True
        if str(item.get("ego_lane_id", "")):
            return True
        if list(item.get("neighbor_summary", []) or []):
            return True
        return any(abs(float(item.get(key, 0.0) or 0.0)) > 1e-6 for key in ("ego_x", "ego_y", "ego_speed"))

    def _build_stage4_pair_samples(self):
        summary = {
            "records_seen": 0,
            "pairs_created": 0,
            "skipped_same_action": 0,
            "skipped_equal_risk": 0,
        }
        pair_samples: List[RiskPairSample] = []
        if self.buffer_path is None or not self.buffer_path.exists():
            return pair_samples, summary

        buffer = InterventionBuffer(capacity=max(10000, self.config.distill.trigger_buffer_size * 4))
        buffer.load(str(self.buffer_path))
        for record in buffer.all_records():
            summary["records_seen"] += 1
            if int(record.raw_action) == int(record.final_action):
                summary["skipped_same_action"] += 1
                continue
            if abs(float(record.raw_risk) - float(record.final_risk)) <= 1e-8:
                summary["skipped_equal_risk"] += 1
                continue
            preferred_action = int(record.final_action if float(record.final_risk) < float(record.raw_risk) else record.raw_action)
            proof = self._build_same_state_proof([dataclass_to_dict(scene) for scene in list(record.history_scene)], {})
            sample = RiskPairSample(
                history_scene=list(record.history_scene),
                action_a=int(record.raw_action),
                action_b=int(record.final_action),
                preferred_action=preferred_action,
                source="stage4_buffer",
                weight=float(self.config.light_risk.stage4_pair_weight),
                meta={
                    "target_risk_a": float(record.raw_risk),
                    "target_risk_b": float(record.final_risk),
                    "reason": str(record.reason),
                    "episode_id": str(dict(record.meta or {}).get("episode_id", "")),
                    "hard_negative": False,
                    "trusted_for_spread": False,
                    **proof,
                },
            )
            pair_samples.append(sample)
            summary["pairs_created"] += 1
        return pair_samples, summary

    def _preferred_action_from_trace_suffix(
        self,
        baseline_steps: Sequence[Dict[str, Any]],
        shielded_steps: Sequence[Dict[str, Any]],
        raw_action: int,
        replaced_action: int,
    ) -> Optional[int]:
        baseline_collision = any(bool(step.get("collision", False)) for step in baseline_steps)
        shielded_collision = any(bool(step.get("collision", False)) for step in shielded_steps)
        if baseline_collision != shielded_collision:
            return int(raw_action if not baseline_collision else replaced_action)

        baseline_min_ttc = self._trace_suffix_min_metric(baseline_steps, "ttc", fallback=1e6, prefer_lower=False)
        shielded_min_ttc = self._trace_suffix_min_metric(shielded_steps, "ttc", fallback=1e6, prefer_lower=False)
        if abs(baseline_min_ttc - shielded_min_ttc) > 1e-6:
            return int(raw_action if baseline_min_ttc > shielded_min_ttc else replaced_action)

        baseline_min_distance = self._trace_suffix_min_metric(baseline_steps, "min_distance", fallback=1e6, prefer_lower=False)
        shielded_min_distance = self._trace_suffix_min_metric(shielded_steps, "min_distance", fallback=1e6, prefer_lower=False)
        if abs(baseline_min_distance - shielded_min_distance) > 1e-6:
            return int(raw_action if baseline_min_distance > shielded_min_distance else replaced_action)

        baseline_reward = self._trace_suffix_reward(baseline_steps)
        shielded_reward = self._trace_suffix_reward(shielded_steps)
        if abs(baseline_reward - shielded_reward) > 1e-6:
            return int(raw_action if baseline_reward > shielded_reward else replaced_action)
        return None

    def _trace_suffix_target(self, steps: Sequence[Dict[str, Any]]) -> float:
        if not steps:
            return 0.0
        min_distance = self._trace_suffix_min_metric(steps, "min_distance", fallback=30.0, prefer_lower=True)
        min_ttc = self._trace_suffix_min_metric(steps, "ttc", fallback=8.0, prefer_lower=True)
        collision = any(bool(step.get("collision", False)) for step in steps)
        if collision:
            return 1.0
        distance_term = 1.0 if min_distance < 3.0 else max(0.0, 1.0 - min_distance / 30.0)
        ttc_term = 1.0 if min_ttc < 1.5 else max(0.0, 1.0 - min_ttc / 8.0)
        return float(max(distance_term, ttc_term))

    def _trace_suffix_reward(self, steps: Sequence[Dict[str, Any]]) -> float:
        if not steps:
            return 0.0
        values = [float(step.get("reward", 0.0)) for step in steps]
        return float(sum(values) / max(1, len(values)))

    def _trace_suffix_min_metric(self, steps: Sequence[Dict[str, Any]], key: str, fallback: float, prefer_lower: bool) -> float:
        values = [float(step.get(key, fallback)) for step in steps if step.get(key) is not None]
        if not values:
            return float(fallback)
        return float(min(values) if prefer_lower else max(values))

    def _is_trusted_stage5_pair(
        self,
        baseline_collision: bool,
        shielded_collision: bool,
        baseline_min_ttc: float,
        shielded_min_ttc: float,
        baseline_min_distance: float,
        shielded_min_distance: float,
        reward_gap: float,
        hard_negative: bool,
    ) -> bool:
        if hard_negative or baseline_collision != shielded_collision:
            return True
        ttc_diff = abs(float(baseline_min_ttc) - float(shielded_min_ttc))
        distance_diff = abs(float(baseline_min_distance) - float(shielded_min_distance))
        if ttc_diff >= TRACE_TRUST_TTC_DELTA:
            return True
        if distance_diff >= TRACE_TRUST_DISTANCE_DELTA:
            return True
        if abs(float(reward_gap)) >= TRACE_TRUST_REWARD_DELTA and (
            ttc_diff >= TRACE_TRUST_TTC_SUPPORT or distance_diff >= TRACE_TRUST_DISTANCE_SUPPORT
        ):
            return True
        return False

    def _build_same_state_proof(self, history_scene_payload: Sequence[Dict[str, Any]], reference_step: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = list(history_scene_payload or [])
        if not payload:
            return {
                "history_hash": "",
                "ego_lane_id": str((reference_step or {}).get("ego_lane_id", "")),
                "ego_x": 0.0,
                "ego_y": 0.0,
                "ego_speed": 0.0,
                "neighbor_summary": [],
            }

        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        history_hash = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        last_scene = dict(payload[-1] or {})
        vehicles = list(last_scene.get("vehicles", []) or [])
        ego_id = str(last_scene.get("ego_id", ""))
        ego_vehicle = None
        for vehicle in vehicles:
            if str(dict(vehicle or {}).get("vehicle_id", "")) == ego_id:
                ego_vehicle = dict(vehicle or {})
                break
        if ego_vehicle is None and vehicles:
            ego_vehicle = dict(vehicles[0] or {})
        ego_vehicle = ego_vehicle or {}
        ego_x = float(ego_vehicle.get("x", 0.0))
        ego_y = float(ego_vehicle.get("y", 0.0))
        ego_speed = float(((float(ego_vehicle.get("vx", 0.0)) ** 2) + (float(ego_vehicle.get("vy", 0.0)) ** 2)) ** 0.5)
        ego_lane_id = str((reference_step or {}).get("ego_lane_id", ego_vehicle.get("lane_id", "")))

        neighbors = []
        for vehicle in vehicles:
            item = dict(vehicle or {})
            vehicle_id = str(item.get("vehicle_id", ""))
            if vehicle_id == ego_id:
                continue
            rel_x = float(item.get("x", 0.0)) - ego_x
            rel_y = float(item.get("y", 0.0)) - ego_y
            rel_speed = float((((float(item.get("vx", 0.0)) - float(ego_vehicle.get("vx", 0.0))) ** 2) + ((float(item.get("vy", 0.0)) - float(ego_vehicle.get("vy", 0.0))) ** 2)) ** 0.5)
            neighbors.append(
                {
                    "vehicle_id": vehicle_id,
                    "rel_x": round(rel_x, 3),
                    "rel_y": round(rel_y, 3),
                    "rel_speed": round(rel_speed, 3),
                }
            )
        neighbors.sort(key=lambda item: (abs(float(item["rel_x"])) + abs(float(item["rel_y"])), str(item["vehicle_id"])))
        return {
            "history_hash": history_hash,
            "ego_lane_id": ego_lane_id,
            "ego_x": round(ego_x, 3),
            "ego_y": round(ego_y, 3),
            "ego_speed": round(ego_speed, 3),
            "neighbor_summary": neighbors[:4],
        }

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
        train_samples, val_samples, test_samples = self._load_dataset_splits()
        pair_payload = self._build_pair_datasets_for_stage2()
        stage5_pairs_created = int(dict(pair_payload.get("generation_summary", {}).get("stage5", {})).get("pairs_created", 0))
        stage4_pairs_created = int(dict(pair_payload.get("generation_summary", {}).get("stage4", {})).get("pairs_created", 0))
        print(
            f"[Pipeline] stage2: pair datasets stage5_pairs_created={stage5_pairs_created}, stage4_pairs_created={stage4_pairs_created}",
            flush=True,
        )

        light_predictor, world_predictor, training_meta = self.train_models(
            train_samples,
            val_samples,
            model_dir=self.models_dir,
            tb_light_base_writer=tb_manager.get_writer("light_risk_base"),
            tb_light_pair_writer=tb_manager.get_writer("light_risk_pair_ft"),
            tb_world_base_writer=tb_manager.get_writer("world_model_base"),
            tb_world_pair_writer=tb_manager.get_writer("world_model_pair_ft"),
            stage5_pair_samples=pair_payload["stage5_pairs"],
            stage4_pair_samples=pair_payload["stage4_pairs"],
        )
        pair_finetune_applied = bool(training_meta.get("pair_finetune_applied", False))
        print(f"[Pipeline] stage2: pair_finetune_applied={pair_finetune_applied}", flush=True)

        evaluator = SafeRLEvaluator(self.config.eval)
        world_eval_metrics = evaluator.evaluate_world_model(world_predictor, test_samples)
        stage2_report = {
            "stage": "stage2",
            "run_id": self.run_id,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "light_model": str(self.light_model_path),
            "world_model": str(self.world_model_path),
            "light_device": str(light_predictor.device),
            "world_device": str(world_predictor.device),
            "light_model_variant": "v2",
            "world_model_variant": "v2",
            "stage5_pairs_created": stage5_pairs_created,
            "stage4_pairs_created": stage4_pairs_created,
            "pair_finetune_applied": pair_finetune_applied,
            "pair_source_counts": dict(pair_payload.get("pair_source_counts", {})),
            "pair_source_weights": dict(pair_payload.get("pair_source_weights", {})),
            "pair_dataset_paths": dict(pair_payload.get("pair_dataset_paths", {})),
            "ranking_metrics": dict(training_meta.get("ranking_metrics", {})),
            "world_eval_metrics": world_eval_metrics,
            "pair_generation": dict(pair_payload.get("generation_summary", {})),
            "light_training": dict(training_meta.get("light_training", {})),
            "world_training": dict(training_meta.get("world_training", {})),
            "base_train_metrics": {
                "light": dict(training_meta.get("light_training", {})),
                "world": dict(training_meta.get("world_training", {})),
            },
            "pair_finetune_metrics": {
                "light": dict(training_meta.get("light_pair_ft", {})),
                "world": dict(training_meta.get("world_pair_ft", {})),
            },
            "world_pair_ft_frozen_modules": list(dict(training_meta.get("world_pair_ft", {})).get("world_pair_ft_frozen_modules", [])),
            "world_pair_ft_trainable_modules": list(dict(training_meta.get("world_pair_ft", {})).get("world_pair_ft_trainable_modules", [])),
        }
        self._write_json(self.stage2_training_report_path, stage2_report)
        self._write_risk_v2_eval_summary_from_stage2(stage2_report)

        return {
            "light_model": str(self.light_model_path),
            "world_model": str(self.world_model_path),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "stage5_pairs_created": stage5_pairs_created,
            "stage4_pairs_created": stage4_pairs_created,
            "pair_finetune_applied": pair_finetune_applied,
            "training_report": str(self.stage2_training_report_path),
        }

    def _run_stage3(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage3: train online policy with shield", flush=True)
        self._require_files("stage3", [self.light_model_path, self.world_model_path])

        stage_config = self._config_with_run_paths()
        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        policy = self.train_online_policy(stage_config, shield, tb_writer=tb_manager.get_writer("ppo"))
        policy_meta = self._save_policy_artifact(policy)

        return {
            "policy_meta": policy_meta,
            "policy_meta_path": str(self.policy_meta_path),
            "runtime_report": str(self.stage3_runtime_config_path),
            "session_events_report": str(self.stage3_session_events_path),
        }

    def _run_stage4(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage4: collect intervention buffer", flush=True)
        self._require_files("stage4", [self.light_model_path, self.world_model_path, self.policy_meta_path])

        stage_config = self._config_with_run_paths()
        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        policy = self._load_policy_artifact()
        policy_meta = self._read_policy_artifact_meta()

        if self._shield_sweep_enabled():
            return self._run_stage4_sweep(
                tb_manager=tb_manager,
                stage_config=stage_config,
                light_predictor=light_predictor,
                world_predictor=world_predictor,
                policy=policy,
                policy_meta=policy_meta,
            )

        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        intervention_buffer = self.collect_interventions(
            stage_config,
            policy,
            shield,
            save_path=self.buffer_path,
            tb_writer=tb_manager.get_writer("buffer"),
        )
        stats = intervention_buffer.stats()
        buffer_metadata = self._build_stage4_buffer_metadata(stage_config, policy_meta)
        stage4_report = {
            **buffer_metadata,
            "buffer_path": str(self.buffer_path),
            "buffer_stats": stats,
        }
        self._write_json(self.stage4_buffer_report_path, stage4_report)

        return stage4_report

    def _run_stage5(self, tb_manager: TensorboardManager) -> Dict:
        print("[Pipeline] stage5: distill and evaluate", flush=True)
        stage_config = self._config_with_run_paths()

        if self._shield_sweep_enabled():
            self._require_files(
                "stage5",
                [self.test_pkl, self.light_model_path, self.world_model_path, self.policy_meta_path],
            )
            return self._run_stage5_sweep(tb_manager=tb_manager, stage_config=stage_config)

        self._require_files(
            "stage5",
            [self.test_pkl, self.light_model_path, self.world_model_path, self.policy_meta_path, self.buffer_path],
        )

        _, _, test_samples = self._load_dataset_splits()
        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shield = SafetyShield(config=self.config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
        shielded_policy = self._load_policy_artifact()
        policy_meta = self._read_policy_artifact_meta()

        buffer = InterventionBuffer(capacity=max(10000, self.config.distill.trigger_buffer_size * 4))
        buffer.load(str(self.buffer_path))

        from safe_rl.rl import PolicyDistiller

        distilled_policy = None
        distill_writer = tb_manager.get_writer("distill")
        if distill_writer is not None:
            distill_writer.add_scalar("status/buffer_size", float(len(buffer)), 0)

        if PolicyDistiller is not None:
            distiller = PolicyDistiller(config=self.config.distill)
            can_distill = distiller.should_distill(buffer)
            if distill_writer is not None:
                distill_writer.add_scalar("status/triggered", float(can_distill), 0)
                distill_writer.add_scalar("status/skipped", float(not can_distill), 0)
            if can_distill:
                distilled_policy = distiller.distill(buffer, tb_writer=distill_writer)

        evaluation = self.evaluate(
            stage_config=stage_config,
            shield=shield,
            shielded_policy=shielded_policy,
            world_predictor=world_predictor,
            test_samples=test_samples,
            distilled_policy=distilled_policy,
            tb_writer=tb_manager.get_writer("eval"),
        )
        buffer_stats = buffer.stats()
        buffer_metadata = self._load_stage4_buffer_report() or self._build_stage4_buffer_metadata(stage_config, policy_meta)
        evaluation["intervention_buffer"] = buffer_stats
        evaluation.update(buffer_metadata)
        evaluation["performance_passed"] = bool(evaluation.get("acceptance_passed", False))
        evaluation["acceptance_passed"] = bool(evaluation["performance_passed"])
        evaluation["attribution_passed"] = self._compute_attribution_passed(evaluation.get("system_shielded", {}))
        evaluation["sanity_check_passed"] = self._compute_sanity_check_passed(
            shielded_metrics=evaluation.get("system_shielded", {}),
            buffer_stats=buffer_stats,
        )
        evaluation["shield_contribution_validated"] = bool(evaluation["attribution_passed"])
        evaluation["conclusion_text"] = self._build_evaluation_conclusion(evaluation)
        self._save_report(evaluation)
        return evaluation

    def train_models(
        self,
        train_samples,
        val_samples,
        model_dir: Path,
        tb_light_base_writer=None,
        tb_light_pair_writer=None,
        tb_world_base_writer=None,
        tb_world_pair_writer=None,
        stage5_pair_samples: Optional[Sequence[RiskPairSample]] = None,
        stage4_pair_samples: Optional[Sequence[RiskPairSample]] = None,
    ):
        try:
            from safe_rl.models.light_risk_model import LightRiskTrainer
            from safe_rl.models.world_model import WorldModelTrainer
        except Exception as exc:
            raise RuntimeError(
                "Safe-RL model import failed. Expected PyTorch to be installed and no Waymo dependency required. "
                f"Original import error: {exc}"
            ) from exc

        model_dir.mkdir(parents=True, exist_ok=True)
        stage5_pair_samples = list(stage5_pair_samples or [])
        stage4_pair_samples = list(stage4_pair_samples or [])
        all_pair_samples = stage5_pair_samples + stage4_pair_samples

        light_trainer = LightRiskTrainer(self.config.light_risk, seed=self.config.sim.random_seed)
        light_predictor = light_trainer.fit(train_samples, val_samples, tb_writer=tb_light_base_writer)
        light_pair_metrics = light_trainer.fine_tune_pairs(all_pair_samples, replay_samples=train_samples, tb_writer=tb_light_pair_writer)
        light_trainer.save(str(self.light_model_path))

        world_trainer = WorldModelTrainer(
            config=self.config.world_model,
            history_steps=self.config.sim.history_steps,
            seed=self.config.sim.random_seed,
        )
        world_predictor = world_trainer.fit(train_samples, val_samples, tb_writer=tb_world_base_writer)
        world_pair_metrics = world_trainer.fine_tune_pairs(all_pair_samples, replay_samples=train_samples, tb_writer=tb_world_pair_writer)
        world_trainer.save(str(self.world_model_path))

        training_meta = {
            "pair_finetune_applied": bool(all_pair_samples and (self.config.light_risk.pair_finetune or self.config.world_model.pair_finetune)),
            "ranking_metrics": {
                "light": light_pair_metrics,
                "world": world_pair_metrics,
            },
            "light_training": dict(getattr(light_trainer, "last_train_report", {})),
            "world_training": dict(getattr(world_trainer, "last_train_report", {})),
            "light_pair_ft": dict(getattr(light_trainer, "last_pair_ft_report", {})),
            "world_pair_ft": dict(getattr(world_trainer, "last_pair_ft_report", {})),
        }
        return light_predictor, world_predictor, training_meta

    def train_online_policy(self, stage_config: SafeRLConfig, shield, tb_writer=None):
        from safe_rl.rl.env import create_env
        from safe_rl.rl.ppo import SafePPOTrainer

        backend = create_backend(stage_config.sim)
        env = None
        training_error: Optional[Dict[str, Any]] = None
        close_error: Optional[Dict[str, str]] = None
        session_logger = IncrementalSessionEventLogger(
            path=str(self.stage3_session_events_path),
            stage="stage3",
            run_id=str(self.run_id or ""),
            metadata={
                "runtime_report_path": str(self.stage3_runtime_config_path),
                "sim_config": asdict(stage_config.sim),
                "backend_requested": str(stage_config.sim.backend),
            },
        )
        telemetry = Stage3TelemetryTracker(tb_writer)

        def session_event_sink(event: Dict[str, Any]):
            session_logger.append_event(event)
            telemetry.handle_session_event(event)

        backend.set_session_event_sink(session_event_sink)

        try:
            backend.start()
            runtime_report = self._build_training_runtime_report("stage3", stage_config, backend)
            self._write_json(self.stage3_runtime_config_path, runtime_report)
            session_logger.set_metadata(backend_start=backend.get_runtime_diagnostics(), runtime_report=runtime_report)

            env = create_env(
                backend=backend,
                sim_config=stage_config.sim,
                ppo_config=stage_config.ppo,
                shield=shield,
                episode_prefix="stage3_train",
                session_event_sink=session_event_sink,
            )
            trainer = SafePPOTrainer(stage_config.ppo)
            policy = trainer.train(env, tb_writer=tb_writer, telemetry=telemetry)
            session_logger.set_metadata(training_completed=True)
            return policy
        except Exception as exc:
            training_error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
            if hasattr(exc, "to_dict"):
                try:
                    training_error["details"] = exc.to_dict()
                except Exception:
                    pass
            session_logger.set_metadata(training_error=training_error)
            raise
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    close_error = {"type": exc.__class__.__name__, "message": str(exc)}
            elif backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    close_error = {"type": exc.__class__.__name__, "message": str(exc)}

            runtime_report = self._build_training_runtime_report(
                "stage3",
                stage_config,
                backend,
                extra={"training_error": training_error, "close_error": close_error},
            )
            self._write_json(self.stage3_runtime_config_path, runtime_report)
            session_logger.set_metadata(
                training_error=training_error,
                close_error=close_error,
                backend_final=backend.get_runtime_diagnostics() if backend is not None else {},
                env_episode_count=len(env.get_session_records()) if env is not None else 0,
                env_episodes=env.get_session_records() if env is not None else [],
                runtime_report=runtime_report,
            )

    def collect_interventions(
        self,
        stage_config: SafeRLConfig,
        policy,
        shield,
        save_path: Optional[Path] = None,
        tb_writer=None,
    ) -> InterventionBuffer:
        from safe_rl.rl.env import create_env

        backend = create_backend(stage_config.sim)
        backend.start()
        env = create_env(
            backend=backend,
            sim_config=stage_config.sim,
            ppo_config=stage_config.ppo,
            shield=shield,
            episode_prefix="stage4_buffer",
        )
        telemetry = BufferTelemetryTracker(tb_writer)
        buffer = InterventionBuffer(capacity=max(10000, stage_config.distill.trigger_buffer_size * 4))

        eval_seeds = self._resolve_eval_seeds(stage_config.eval.eval_episodes)
        for episode_index in range(stage_config.eval.eval_episodes):
            obs, _ = env.reset(seed=eval_seeds[episode_index], options={"risky_mode": True})
            done = False
            last_episode_id = ""
            while not done:
                action = int(policy.predict(obs, deterministic=True))
                obs, _, terminated, truncated, info = env.step(action)
                telemetry.on_step(info)
                done = terminated or truncated
                last_episode_id = str(info.get("episode_id", "") or last_episode_id)
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
                        meta={"episode_step": env.step_count, "episode_id": info.get("episode_id", "")},
                    )
                    buffer.push(record)
                    telemetry.on_push(record, buffer.stats())
            telemetry.on_episode_end(last_episode_id)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            buffer.save(str(save_path))

        env.close()
        return buffer

    def evaluate(
        self,
        stage_config: SafeRLConfig,
        shield,
        shielded_policy,
        world_predictor,
        test_samples,
        distilled_policy=None,
        tb_writer=None,
        paired_results_path: Optional[Path] = None,
    ) -> Dict:
        from safe_rl.rl.env import create_env

        evaluator = SafeRLEvaluator(stage_config.eval)
        trace_enabled = self._shield_trace_enabled(stage_config)
        eval_seeds = self._resolve_stage5_eval_seeds(stage_config)

        baseline_backend = create_backend(stage_config.sim)
        baseline_backend.start()
        baseline_env = create_env(
            baseline_backend,
            stage_config.sim,
            stage_config.ppo,
            shield=None,
            episode_prefix="stage5_eval_baseline",
        )
        baseline_metrics = self._evaluate_policy_with_trace_option(
            evaluator=evaluator,
            env=baseline_env,
            policy=shielded_policy,
            episodes=stage_config.eval.eval_episodes,
            risky_mode=True,
            tb_writer=tb_writer,
            tb_prefix="baseline",
            seeds=eval_seeds,
            collect_step_traces=trace_enabled,
        )
        baseline_env.close()

        shield_backend = create_backend(stage_config.sim)
        shield_backend.start()
        shield_env = create_env(
            shield_backend,
            stage_config.sim,
            stage_config.ppo,
            shield=shield,
            episode_prefix="stage5_eval_shielded",
        )
        shielded_metrics = self._evaluate_policy_with_trace_option(
            evaluator=evaluator,
            env=shield_env,
            policy=shielded_policy,
            episodes=stage_config.eval.eval_episodes,
            risky_mode=True,
            tb_writer=tb_writer,
            tb_prefix="shielded",
            seeds=eval_seeds,
            collect_step_traces=trace_enabled,
        )
        shield_env.close()

        delta = evaluator.compare_baseline_and_shielded(baseline_metrics, shielded_metrics)
        acceptance = evaluator.evaluate_acceptance(delta)

        world_metrics = evaluator.evaluate_world_model(world_predictor, test_samples[: min(200, len(test_samples))])

        paired_episode_results = self._build_paired_episode_results(
            baseline_metrics.get("episode_details", []),
            shielded_metrics.get("episode_details", []),
            scenario_source=str(stage_config.sim.sumo_cfg),
            risky_mode=True,
        )
        target_paired_results_path = paired_results_path or self.stage5_paired_episode_results_path
        self._write_json(target_paired_results_path, {"pairs": paired_episode_results})

        trace_payload = self._write_shield_trace_outputs(
            stage_config=stage_config,
            baseline_details=baseline_metrics.get("episode_details", []),
            shielded_details=shielded_metrics.get("episode_details", []),
        ) if trace_enabled else None

        result = {
            "comparison_mode": "same_policy_shield_off_vs_on",
            "policy_source": str(self.policy_meta_path),
            "paired_eval": True,
            "paired_risky_mode": True,
            "paired_scenario_source": str(stage_config.sim.sumo_cfg),
            "evaluation_seeds": eval_seeds,
            "effective_shield_config": self._effective_shield_config(stage_config),
            "world_model": world_metrics,
            "system_baseline": baseline_metrics,
            "system_shielded": shielded_metrics,
            "delta": delta,
            "acceptance_passed": acceptance,
            "performance_passed": bool(acceptance),
            "attribution_passed": self._compute_attribution_passed(shielded_metrics),
            "shield_contribution_validated": self._compute_attribution_passed(shielded_metrics),
            "stage5_paired_episode_results_path": str(target_paired_results_path),
        }
        if trace_payload is not None:
            result["shield_trace_summary_path"] = str(trace_payload["trace_summary_path"])
            result["shield_trace"] = trace_payload["summary"]
            tuning_payload = self._write_shield_trace_tuning_summary()
            if tuning_payload is not None:
                result["shield_trace_tuning_summary_path"] = str(tuning_payload["summary_path"])
                result["shield_trace_tuning_summary"] = tuning_payload["summary"]
            margin_payload = self._write_shield_margin_analysis_summary()
            if margin_payload is not None:
                result["shield_margin_analysis_summary_path"] = str(margin_payload["summary_path"])
                result["shield_margin_analysis_summary"] = margin_payload["summary"]

        if tb_writer is not None:
            tb_writer.add_scalar("summary/collision_reduction", float(delta.get("collision_reduction", 0.0)), 0)
            tb_writer.add_scalar("summary/efficiency_drop", float(delta.get("efficiency_drop", 0.0)), 0)
            tb_writer.add_scalar("summary/acceptance_passed", float(bool(acceptance)), 0)
            tb_writer.add_scalar("summary/shield_contribution_validated", float(result["shield_contribution_validated"]), 0)
            tb_writer.add_scalar("world_model/traj_ade", float(world_metrics.get("traj_ade", 0.0)), 0)
            tb_writer.add_scalar("world_model/risk_acc", float(world_metrics.get("risk_acc", 0.0)), 0)
            tb_writer.add_scalar("world_model/risk_mae", float(world_metrics.get("risk_mae", 0.0)), 0)

        if distilled_policy is not None:
            distill_backend = create_backend(stage_config.sim)
            distill_backend.start()
            distill_env = create_env(
                distill_backend,
                stage_config.sim,
                stage_config.ppo,
                shield=shield,
                episode_prefix="stage5_eval_distilled",
            )
            distilled_metrics = self._evaluate_policy_with_trace_option(
                evaluator=evaluator,
                env=distill_env,
                policy=distilled_policy,
                episodes=stage_config.eval.eval_episodes,
                risky_mode=True,
                tb_writer=tb_writer,
                tb_prefix="distilled",
                seeds=eval_seeds,
                collect_step_traces=False,
            )
            distill_env.close()
            result["system_distilled"] = distilled_metrics

        self._write_risk_v2_eval_summary_from_stage5(result)
        return result

    def _shield_sweep_enabled(self) -> bool:
        return bool(self.config.shield_sweep.enabled and self.config.shield_sweep.variants)

    def _sanitize_artifact_key(self, value: str) -> str:
        text = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in str(value or "item"))
        return text.strip("_") or "item"

    def _iter_shield_sweep_variants(self):
        seen = set()
        variants = []
        for index, variant in enumerate(self.config.shield_sweep.variants):
            name = str(variant.name or f"variant_{index + 1}")
            key = self._sanitize_artifact_key(name)
            if key in seen:
                raise ValueError(f"Duplicate shield_sweep variant name: {name}")
            seen.add(key)
            variants.append(variant)
        return variants

    def _shield_sweep_reports_dir(self) -> Path:
        return self.reports_dir / "shield_sweep"

    def _shield_sweep_buffers_dir(self) -> Path:
        return self.buffers_dir / "shield_sweep"

    def _shield_sweep_variant_report_dir(self, variant_name: str) -> Path:
        return self._shield_sweep_reports_dir() / self._sanitize_artifact_key(variant_name)

    def _shield_sweep_variant_buffer_dir(self, variant_name: str) -> Path:
        return self._shield_sweep_buffers_dir() / self._sanitize_artifact_key(variant_name)

    def _shield_sweep_variant_buffer_path(self, variant_name: str) -> Path:
        return self._shield_sweep_variant_buffer_dir(variant_name) / "intervention_buffer.pkl"

    def _shield_sweep_variant_stage4_report_path(self, variant_name: str) -> Path:
        return self._shield_sweep_variant_report_dir(variant_name) / "stage4_buffer_report.json"

    def _shield_sweep_variant_stage5_report_path(self, variant_name: str) -> Path:
        return self._shield_sweep_variant_report_dir(variant_name) / "pipeline_report.json"

    def _shield_sweep_variant_stage5_paired_results_path(self, variant_name: str) -> Path:
        return self._shield_sweep_variant_report_dir(variant_name) / "stage5_paired_episode_results.json"

    def _register_shield_sweep_artifacts(
        self,
        variant_name: str,
        buffer_path: Optional[Path] = None,
        stage4_report_path: Optional[Path] = None,
        stage5_report_path: Optional[Path] = None,
        stage5_paired_results_path: Optional[Path] = None,
    ):
        key = self._sanitize_artifact_key(variant_name)
        artifact_paths = self.manifest.setdefault("artifact_paths", {})
        if buffer_path is not None:
            artifact_paths[f"shield_sweep_{key}_buffer"] = str(buffer_path)
        if stage4_report_path is not None:
            artifact_paths[f"shield_sweep_{key}_stage4_report"] = str(stage4_report_path)
        if stage5_report_path is not None:
            artifact_paths[f"shield_sweep_{key}_stage5_report"] = str(stage5_report_path)
        if stage5_paired_results_path is not None:
            artifact_paths[f"shield_sweep_{key}_stage5_paired_results"] = str(stage5_paired_results_path)
        artifact_paths["shield_sweep_summary_report"] = str(self.shield_sweep_summary_path)

    def _config_for_shield_variant(self, stage_config: SafeRLConfig, variant) -> SafeRLConfig:
        variant_config = copy.deepcopy(stage_config)
        variant_config.shield.risk_threshold = float(variant.risk_threshold)
        variant_config.shield.uncertainty_threshold = float(variant.uncertainty_threshold)
        variant_config.shield.coarse_top_k = int(variant.coarse_top_k)
        return variant_config

    def _run_stage4_sweep(
        self,
        tb_manager: TensorboardManager,
        stage_config: SafeRLConfig,
        light_predictor,
        world_predictor,
        policy,
        policy_meta: Dict[str, Any],
    ) -> Dict:
        base_metadata = self._build_stage4_buffer_metadata(stage_config, policy_meta)
        variant_reports = []

        for variant in self._iter_shield_sweep_variants():
            variant_name = self._sanitize_artifact_key(str(variant.name or "variant"))
            variant_config = self._config_for_shield_variant(stage_config, variant)
            shield = SafetyShield(config=variant_config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
            buffer_path = self._shield_sweep_variant_buffer_path(variant_name)
            report_path = self._shield_sweep_variant_stage4_report_path(variant_name)

            intervention_buffer = self.collect_interventions(
                variant_config,
                policy,
                shield,
                save_path=buffer_path,
                tb_writer=tb_manager.get_writer(f"buffer_sweep_{variant_name}"),
            )
            stats = intervention_buffer.stats()
            variant_report = {
                "mode": "shield_sweep_variant",
                "variant_name": variant_name,
                "risk_threshold": float(variant.risk_threshold),
                "uncertainty_threshold": float(variant.uncertainty_threshold),
                "coarse_top_k": int(variant.coarse_top_k),
                **base_metadata,
                "buffer_path": str(buffer_path),
                "buffer_stats": stats,
            }
            self._write_json(report_path, variant_report)
            self._register_shield_sweep_artifacts(
                variant_name=variant_name,
                buffer_path=buffer_path,
                stage4_report_path=report_path,
            )
            variant_reports.append(
                {
                    **variant_report,
                    "stage4_buffer_report_path": str(report_path),
                }
            )

        overview = {
            "mode": "shield_sweep",
            **base_metadata,
            "variants": variant_reports,
        }
        self._write_json(self.stage4_buffer_report_path, overview)
        return overview

    def _run_stage5_sweep(self, tb_manager: TensorboardManager, stage_config: SafeRLConfig) -> Dict:
        _, _, test_samples = self._load_dataset_splits()
        light_predictor, world_predictor = self._build_predictors_from_saved_models()
        shielded_policy = self._load_policy_artifact()
        policy_meta = self._read_policy_artifact_meta()

        from safe_rl.rl import PolicyDistiller

        summary_entries = []
        eval_seeds = self._resolve_eval_seeds(stage_config.eval.eval_episodes, eval_config=stage_config.eval)

        for variant in self._iter_shield_sweep_variants():
            variant_name = self._sanitize_artifact_key(str(variant.name or "variant"))
            variant_config = self._config_for_shield_variant(stage_config, variant)
            buffer_path = self._shield_sweep_variant_buffer_path(variant_name)
            stage4_report_path = self._shield_sweep_variant_stage4_report_path(variant_name)
            paired_results_path = self._shield_sweep_variant_stage5_paired_results_path(variant_name)
            variant_report_path = self._shield_sweep_variant_stage5_report_path(variant_name)

            self._require_files(
                f"stage5_sweep_{variant_name}",
                [buffer_path, stage4_report_path],
            )

            shield = SafetyShield(config=variant_config.shield, light_predictor=light_predictor, world_predictor=world_predictor)
            buffer = InterventionBuffer(capacity=max(10000, self.config.distill.trigger_buffer_size * 4))
            buffer.load(str(buffer_path))

            distilled_policy = None
            distill_writer = tb_manager.get_writer(f"distill_sweep_{variant_name}")
            if distill_writer is not None:
                distill_writer.add_scalar("status/buffer_size", float(len(buffer)), 0)

            if PolicyDistiller is not None:
                distiller = PolicyDistiller(config=self.config.distill)
                can_distill = distiller.should_distill(buffer)
                if distill_writer is not None:
                    distill_writer.add_scalar("status/triggered", float(can_distill), 0)
                    distill_writer.add_scalar("status/skipped", float(not can_distill), 0)
                if can_distill:
                    distilled_policy = distiller.distill(buffer, tb_writer=distill_writer)

            evaluation = self.evaluate(
                stage_config=variant_config,
                shield=shield,
                shielded_policy=shielded_policy,
                world_predictor=world_predictor,
                test_samples=test_samples,
                distilled_policy=distilled_policy,
                tb_writer=tb_manager.get_writer(f"eval_sweep_{variant_name}"),
                paired_results_path=paired_results_path,
            )
            buffer_stats = buffer.stats()
            buffer_metadata = self._load_stage4_buffer_report(stage4_report_path) or self._build_stage4_buffer_metadata(variant_config, policy_meta)
            evaluation["intervention_buffer"] = buffer_stats
            evaluation.update(buffer_metadata)
            evaluation["mode"] = "shield_sweep_variant"
            evaluation["variant_name"] = variant_name
            evaluation["risk_threshold"] = float(variant.risk_threshold)
            evaluation["uncertainty_threshold"] = float(variant.uncertainty_threshold)
            evaluation["coarse_top_k"] = int(variant.coarse_top_k)
            evaluation["performance_passed"] = bool(evaluation.get("acceptance_passed", False))
            evaluation["acceptance_passed"] = bool(evaluation["performance_passed"])
            evaluation["attribution_passed"] = self._compute_attribution_passed(evaluation.get("system_shielded", {}))
            evaluation["sanity_check_passed"] = self._compute_sanity_check_passed(
                shielded_metrics=evaluation.get("system_shielded", {}),
                buffer_stats=buffer_stats,
            )
            evaluation["shield_contribution_validated"] = bool(evaluation["attribution_passed"])
            evaluation["conclusion_text"] = self._build_evaluation_conclusion(evaluation)
            self._save_report(evaluation, path=variant_report_path)
            self._register_shield_sweep_artifacts(
                variant_name=variant_name,
                buffer_path=buffer_path,
                stage4_report_path=stage4_report_path,
                stage5_report_path=variant_report_path,
                stage5_paired_results_path=paired_results_path,
            )
            summary_entries.append(
                self._build_shield_sweep_summary_entry(
                    variant=variant,
                    variant_name=variant_name,
                    stage5_report=evaluation,
                    buffer_stats=buffer_stats,
                    buffer_path=buffer_path,
                    stage4_report_path=stage4_report_path,
                    stage5_report_path=variant_report_path,
                    paired_results_path=paired_results_path,
                )
            )

        sweep_summary = {
            "mode": "shield_sweep",
            "policy_source": str(self.policy_meta_path),
            "paired_eval": True,
            "paired_risky_mode": True,
            "paired_scenario_source": str(stage_config.sim.sumo_cfg),
            "evaluation_seeds": eval_seeds,
            "variants": summary_entries,
        }
        self._write_json(self.shield_sweep_summary_path, sweep_summary)
        self.manifest.setdefault("artifact_paths", {})["shield_sweep_summary_report"] = str(self.shield_sweep_summary_path)

        overview = {
            "mode": "shield_sweep",
            "policy_source": str(self.policy_meta_path),
            "paired_eval": True,
            "paired_risky_mode": True,
            "paired_scenario_source": str(stage_config.sim.sumo_cfg),
            "evaluation_seeds": eval_seeds,
            "variants": summary_entries,
            "shield_sweep_summary_path": str(self.shield_sweep_summary_path),
        }
        self._save_report(overview)
        return overview

    def _build_shield_sweep_summary_entry(
        self,
        variant,
        variant_name: str,
        stage5_report: Dict[str, Any],
        buffer_stats: Dict[str, Any],
        buffer_path: Path,
        stage4_report_path: Path,
        stage5_report_path: Path,
        paired_results_path: Path,
    ) -> Dict[str, Any]:
        shielded = dict(stage5_report.get("system_shielded", {}))
        intervention_rate = float(shielded.get("intervention_rate", 0.0))
        replacement_count = float(shielded.get("replacement_count", 0.0))
        fallback_action_count = float(shielded.get("fallback_action_count", 0.0))
        mean_risk_reduction = float(shielded.get("mean_risk_reduction", 0.0))
        avg_speed = float(shielded.get("avg_speed", 0.0))
        min_band = float(self.config.shield_sweep.target_intervention_min)
        max_band = float(self.config.shield_sweep.target_intervention_max)
        min_avg_speed = float(self.config.shield_sweep.min_avg_speed)

        return {
            "variant_name": variant_name,
            "risk_threshold": float(variant.risk_threshold),
            "uncertainty_threshold": float(variant.uncertainty_threshold),
            "coarse_top_k": int(variant.coarse_top_k),
            "buffer_size": float(buffer_stats.get("size", 0.0)),
            "intervention_rate": intervention_rate,
            "replacement_count": replacement_count,
            "replacement_same_as_raw_count": float(shielded.get("replacement_same_as_raw_count", 0.0)),
            "fallback_action_count": fallback_action_count,
            "mean_raw_risk": float(shielded.get("mean_raw_risk", 0.0)),
            "mean_final_risk": float(shielded.get("mean_final_risk", 0.0)),
            "mean_risk_reduction": mean_risk_reduction,
            "avg_speed": avg_speed,
            "mean_reward": float(shielded.get("mean_reward", 0.0)),
            "collision_rate": float(shielded.get("collision_rate", 0.0)),
            "success_rate": float(shielded.get("success_rate", 0.0)),
            "performance_passed": bool(stage5_report.get("performance_passed", False)),
            "attribution_passed": bool(stage5_report.get("attribution_passed", False)),
            "intervention_band_passed": min_band <= intervention_rate <= max_band,
            "efficiency_guard_passed": avg_speed >= min_avg_speed,
            "risk_reduction_passed": mean_risk_reduction > 0.0,
            "replacement_passed": replacement_count > 0.0,
            "fallback_dominant": fallback_action_count >= 0.9 * max(1.0, replacement_count),
            "buffer_path": str(buffer_path),
            "stage4_buffer_report_path": str(stage4_report_path),
            "stage5_report_path": str(stage5_report_path),
            "stage5_paired_episode_results_path": str(paired_results_path),
        }

    def _resolve_eval_seeds(self, episodes: int, eval_config: Optional[Any] = None) -> List[int]:
        eval_cfg = eval_config or self.config.eval
        configured = [int(seed) for seed in getattr(eval_cfg, "seed_list", [])]
        if not configured:
            base_seed = int(self.config.sim.random_seed)
            return [base_seed + i for i in range(max(0, episodes))]
        return [configured[i % len(configured)] for i in range(max(0, episodes))]

    def _compute_attribution_passed(self, shielded_metrics: Dict[str, Any]) -> bool:
        intervention_rate = float(shielded_metrics.get("intervention_rate", 0.0))
        mean_risk_reduction = float(shielded_metrics.get("mean_risk_reduction", 0.0))
        replacement_count = float(shielded_metrics.get("replacement_count", 0.0))
        return intervention_rate > 0.0 and mean_risk_reduction > 0.0 and replacement_count > 0.0

    def _compute_shield_contribution_validated(self, shielded_metrics: Dict[str, Any]) -> bool:
        return self._compute_attribution_passed(shielded_metrics)

    def _compute_sanity_check_passed(self, shielded_metrics: Dict[str, Any], buffer_stats: Dict[str, Any]) -> bool:
        intervention_rate = float(shielded_metrics.get("intervention_rate", 0.0))
        mean_raw_risk = float(shielded_metrics.get("mean_raw_risk", 0.0))
        mean_final_risk = float(shielded_metrics.get("mean_final_risk", 0.0))
        buffer_size = float(buffer_stats.get("size", 0.0))
        replacement_count = float(shielded_metrics.get("replacement_count", 0.0))
        return intervention_rate > 0.0 and mean_final_risk < mean_raw_risk and buffer_size > 0.0 and replacement_count > 0.0

    def _read_policy_artifact_meta(self) -> Dict[str, Any]:
        self._require_files("policy_meta", [self.policy_meta_path])
        with self.policy_meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_stage4_buffer_metadata(self, stage_config: SafeRLConfig, policy_meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "buffer_policy_source": str(self.policy_meta_path),
            "buffer_policy_type": str(policy_meta.get("policy_type", "")),
            "buffer_eval_seeds": self._resolve_eval_seeds(stage_config.eval.eval_episodes, eval_config=stage_config.eval),
            "buffer_risky_mode": True,
            "buffer_scenario_source": str(stage_config.sim.sumo_cfg),
        }

    def _load_stage4_buffer_report(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        target_path = path or self.stage4_buffer_report_path
        if target_path is None or not target_path.exists():
            return None
        with target_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_paired_episode_results(
        self,
        baseline_details: Sequence[Dict[str, Any]],
        shielded_details: Sequence[Dict[str, Any]],
        scenario_source: str,
        risky_mode: bool,
    ) -> List[Dict[str, Any]]:
        pair_count = min(len(baseline_details), len(shielded_details))
        results: List[Dict[str, Any]] = []
        for idx in range(pair_count):
            baseline = dict(baseline_details[idx])
            shielded = dict(shielded_details[idx])
            results.append(
                {
                    "pair_index": idx,
                    "seed": shielded.get("seed", baseline.get("seed")),
                    "risky_mode": bool(shielded.get("risky_mode", baseline.get("risky_mode", risky_mode))),
                    "scenario_source": str(shielded.get("scenario_source", baseline.get("scenario_source", scenario_source))),
                    "baseline_episode_id": str(baseline.get("episode_id", "")),
                    "shielded_episode_id": str(shielded.get("episode_id", "")),
                    "baseline_collision": bool(int(baseline.get("collisions", 0)) > 0),
                    "shielded_collision": bool(int(shielded.get("collisions", 0)) > 0),
                    "baseline_reward": float(baseline.get("mean_reward", 0.0)),
                    "shielded_reward": float(shielded.get("mean_reward", 0.0)),
                    "baseline_raw_risk": float(baseline.get("mean_raw_risk", 0.0)),
                    "shielded_raw_risk": float(shielded.get("mean_raw_risk", 0.0)),
                    "shielded_final_risk": float(shielded.get("mean_final_risk", 0.0)),
                    "intervention_count": int(shielded.get("interventions", 0)),
                    "replacement_count": int(shielded.get("replacement_count", 0)),
                    "replacement_same_as_raw_count": int(shielded.get("replacement_same_as_raw_count", 0)),
                    "fallback_action_count": int(shielded.get("fallback_action_count", 0)),
                    "mean_risk_reduction": float(shielded.get("mean_risk_reduction", 0.0)),
                }
            )
        return results

    def _build_evaluation_conclusion(self, evaluation: Dict[str, Any]) -> str:
        performance_passed = bool(evaluation.get("performance_passed", False))
        attribution_passed = bool(evaluation.get("attribution_passed", False))
        if performance_passed and attribution_passed:
            return (
                "The engineering loop and the shield attribution checks both passed; "
                "the current run shows a measurable safety contribution from shield action replacement."
            )
        if performance_passed and not attribution_passed:
            return (
                "The system-level performance passed, but shield attribution is still not independently validated; "
                "current gains are more safely attributed to the learned policy than to confirmed shield replacements."
            )
        return (
            "Neither the performance gate nor the shield attribution gate fully passed; "
            "the next step should focus on shield thresholds, replacement execution, and intervention logging."
        )

    def _save_report(self, report: Dict, path: Optional[Path] = None):
        target_path = path or self.report_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    def _build_pair_before_after(self, before_metrics: Dict[str, Any], after_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "before": dict(before_metrics or {}),
            "after": dict(after_metrics or {}),
        }

    def _select_focus_trace_variant(self, tuning_summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not tuning_summary:
            return None
        variants = list(tuning_summary.get("variants", []) or [])
        if not variants:
            return None
        preferred = ["D1", "E2", "F1", "F2", "F3", "C_baseline"]
        by_name = {str(item.get("variant_name", "")): item for item in variants}
        for name in preferred:
            if name in by_name:
                return dict(by_name[name])
        return dict(variants[0])

    def _read_trace_focus_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.shield_trace_tuning_summary_path is None or not self.shield_trace_tuning_summary_path.exists():
            return None
        tuning_summary = self._read_json(self.shield_trace_tuning_summary_path)
        focus = self._select_focus_trace_variant(tuning_summary)
        if focus is None:
            return None
        return {
            "variant_name": str(focus.get("variant_name", "")),
            "trace_summary_path": str(focus.get("trace_summary_path", "")),
            "margin_analysis_path": str(focus.get("margin_analysis_path", "")),
            "candidate_selected_count": int(focus.get("candidate_selected_count", 0)),
            "mean_intervention_count": float(focus.get("mean_intervention_count", 0.0)),
            "mean_risk_reduction": float(focus.get("mean_risk_reduction", 0.0)),
            "mean_reward_gap_to_baseline_policy": float(focus.get("mean_reward_gap_to_baseline_policy", 0.0)),
            "margin_near_threshold_band_ratio": float(focus.get("margin_near_threshold_band_ratio", 0.0)),
            "effective_shield_config": dict(focus.get("effective_shield_config", {})),
        }

    def _write_risk_v2_eval_summary_from_stage2(self, stage2_report: Dict[str, Any]):
        if self.risk_v2_eval_summary_path is None:
            return None
        light_pair_ft = dict(stage2_report.get("pair_finetune_metrics", {}).get("light", {}))
        world_pair_ft = dict(stage2_report.get("pair_finetune_metrics", {}).get("world", {}))
        before_trace = self._read_trace_focus_snapshot()
        summary = {
            "run_id": str(self.run_id or ""),
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "stage2_training_report_path": str(self.stage2_training_report_path),
            "model_variants": {
                "light": str(stage2_report.get("light_model_variant", "")),
                "world": str(stage2_report.get("world_model_variant", "")),
            },
            "pair_finetune_applied": bool(stage2_report.get("pair_finetune_applied", False)),
            "pair_source_weights": dict(stage2_report.get("pair_source_weights", {})),
            "base_train_metrics": dict(stage2_report.get("base_train_metrics", {})),
            "pair_finetune_metrics": dict(stage2_report.get("pair_finetune_metrics", {})),
            "world_pair_ft_frozen_modules": list(stage2_report.get("world_pair_ft_frozen_modules", [])),
            "world_pair_ft_trainable_modules": list(stage2_report.get("world_pair_ft_trainable_modules", [])),
            "before_trace_metrics": before_trace,
            "after_trace_metrics": None,
            "score_spread_before_after": {
                "light": self._build_pair_before_after(light_pair_ft.get("before_pair_metrics", {}), light_pair_ft.get("after_pair_metrics", {})),
                "world": self._build_pair_before_after(world_pair_ft.get("before_pair_metrics", {}), world_pair_ft.get("after_pair_metrics", {})),
            },
            "same_state_score_gap_before_after": {
                "light": self._build_pair_before_after(light_pair_ft.get("before_pair_metrics", {}), light_pair_ft.get("after_pair_metrics", {})),
                "world": self._build_pair_before_after(world_pair_ft.get("before_pair_metrics", {}), world_pair_ft.get("after_pair_metrics", {})),
            },
            "pair_ranking_accuracy_before_after": {
                "light": self._build_pair_before_after(light_pair_ft.get("before_pair_metrics", {}), light_pair_ft.get("after_pair_metrics", {})),
                "world": self._build_pair_before_after(world_pair_ft.get("before_pair_metrics", {}), world_pair_ft.get("after_pair_metrics", {})),
            },
            "hard_negative_accuracy_before_after": {
                "light": self._build_pair_before_after(light_pair_ft.get("before_pair_metrics", {}), light_pair_ft.get("after_pair_metrics", {})),
                "world": self._build_pair_before_after(world_pair_ft.get("before_pair_metrics", {}), world_pair_ft.get("after_pair_metrics", {})),
            },
            "margin_near_threshold_band_ratio_before_after": {
                "before": None if before_trace is None else float(before_trace.get("margin_near_threshold_band_ratio", 0.0)),
                "after": None,
            },
        }
        self._write_json(self.risk_v2_eval_summary_path, summary)
        self.manifest.setdefault("artifact_paths", {})["risk_v2_eval_summary_report"] = str(self.risk_v2_eval_summary_path)
        return summary

    def _write_risk_v2_eval_summary_from_stage5(self, evaluation: Dict[str, Any]):
        if self.risk_v2_eval_summary_path is None:
            return None
        if self.risk_v2_eval_summary_path.exists():
            summary = self._read_json(self.risk_v2_eval_summary_path)
        else:
            summary = {"run_id": str(self.run_id or "")}
        after_trace = self._read_trace_focus_snapshot()
        summary["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
        summary["stage5_report_path"] = str(self.report_path)
        summary["after_trace_metrics"] = after_trace
        summary["stage5_effective_shield_config"] = dict(evaluation.get("effective_shield_config", {}))
        summary["stage5_paired_episode_results_path"] = str(evaluation.get("stage5_paired_episode_results_path", ""))
        summary["stage5_trace_summary_path"] = str(evaluation.get("shield_trace_summary_path", ""))
        summary["stage5_trace_tuning_summary_path"] = str(evaluation.get("shield_trace_tuning_summary_path", ""))
        summary["stage5_margin_analysis_summary_path"] = str(evaluation.get("shield_margin_analysis_summary_path", ""))
        before_trace = summary.get("before_trace_metrics")
        summary["margin_near_threshold_band_ratio_before_after"] = {
            "before": None if before_trace is None else float(dict(before_trace).get("margin_near_threshold_band_ratio", 0.0)),
            "after": None if after_trace is None else float(after_trace.get("margin_near_threshold_band_ratio", 0.0)),
        }
        self._write_json(self.risk_v2_eval_summary_path, summary)
        self.manifest.setdefault("artifact_paths", {})["risk_v2_eval_summary_report"] = str(self.risk_v2_eval_summary_path)
        return summary

    def _evaluate_policy_with_trace_option(
        self,
        evaluator,
        env,
        policy,
        episodes: int,
        risky_mode: bool,
        tb_writer,
        tb_prefix: str,
        seeds: Sequence[int],
        collect_step_traces: bool,
    ) -> Dict[str, Any]:
        kwargs = {
            "env": env,
            "policy": policy,
            "episodes": episodes,
            "risky_mode": risky_mode,
            "tb_writer": tb_writer,
            "tb_prefix": tb_prefix,
            "seeds": seeds,
        }
        try:
            signature = inspect.signature(evaluator.evaluate_policy)
        except (TypeError, ValueError):
            signature = None
        if signature is None or "collect_step_traces" in signature.parameters:
            kwargs["collect_step_traces"] = collect_step_traces
        return evaluator.evaluate_policy(**kwargs)

    def _shield_trace_enabled(self, stage_config: Optional[SafeRLConfig] = None) -> bool:
        cfg = stage_config or self.config
        return bool(getattr(cfg.shield_trace, "enabled", False))

    def _resolve_stage5_eval_seeds(self, stage_config: SafeRLConfig) -> List[int]:
        if self._shield_trace_enabled(stage_config) and getattr(stage_config.shield_trace, "seed_list", []):
            configured = [int(seed) for seed in stage_config.shield_trace.seed_list]
            return [configured[i % len(configured)] for i in range(max(0, stage_config.eval.eval_episodes))]
        return self._resolve_eval_seeds(stage_config.eval.eval_episodes, eval_config=stage_config.eval)

    def _write_shield_trace_outputs(
        self,
        stage_config: SafeRLConfig,
        baseline_details: Sequence[Dict[str, Any]],
        shielded_details: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trace_dir = self.shield_trace_dir or (self.reports_dir / str(stage_config.shield_trace.trace_dir_name))
        trace_dir.mkdir(parents=True, exist_ok=True)
        pair_count = min(len(baseline_details), len(shielded_details))
        summary = {
            "variant_name": self._shield_trace_variant_name(str(stage_config.shield_trace.trace_dir_name)),
            "run_id": str(self.run_id or ""),
            "seeds": [],
            "effective_shield_config": self._effective_shield_config(stage_config),
            "regression_pair_count": 0,
            "pairs_with_lane_change_replacement": 0,
            "pairs_with_merge_guard_triggered": 0,
            "pairs_with_fallback": 0,
            "pairs_with_collision_after_replacement": 0,
            "blocked_by_margin_count": 0,
            "raw_passthrough_count": 0,
            "merge_lateral_guard_block_count": 0,
            "candidate_selected_count": 0,
            "pair_files": [],
        }

        save_pair_traces = bool(getattr(stage_config.shield_trace, "save_pair_traces", True))
        pair_payloads: List[Dict[str, Any]] = []
        for pair_index in range(pair_count):
            baseline = dict(baseline_details[pair_index])
            shielded = dict(shielded_details[pair_index])
            pair_payload = self._build_trace_pair_payload(
                pair_index=pair_index,
                baseline=baseline,
                shielded=shielded,
                scenario_source=str(stage_config.sim.sumo_cfg),
                risky_mode=True,
            )
            seed_value = pair_payload["seed"]
            pair_path = trace_dir / f"pair_{pair_index:02d}_seed_{seed_value}.json"
            pair_payloads.append(pair_payload)
            if save_pair_traces:
                self._write_json(pair_path, pair_payload)
                summary["pair_files"].append(str(pair_path))
            summary["seeds"].append(seed_value)
            if pair_payload["regression_pair"]:
                summary["regression_pair_count"] += 1
            if pair_payload["has_lane_change_replacement"]:
                summary["pairs_with_lane_change_replacement"] += 1
            if pair_payload["merge_guard_triggered"]:
                summary["pairs_with_merge_guard_triggered"] += 1
            if pair_payload["has_fallback"]:
                summary["pairs_with_fallback"] += 1
            if pair_payload["collision_after_replacement"]:
                summary["pairs_with_collision_after_replacement"] += 1
            summary["blocked_by_margin_count"] += int(pair_payload.get("blocked_by_margin_count", 0))
            summary["raw_passthrough_count"] += int(pair_payload.get("raw_passthrough_count", 0))
            summary["merge_lateral_guard_block_count"] += int(pair_payload.get("merge_lateral_guard_block_count", 0))
            summary["candidate_selected_count"] += int(pair_payload.get("candidate_selected_count", 0))

        margin_analysis = self._ensure_trace_margin_analysis(
            trace_dir=trace_dir,
            trace_summary=summary,
            pair_payloads=pair_payloads,
        )
        if margin_analysis is not None:
            summary["margin_analysis_path"] = str(margin_analysis["analysis_path"])
            summary["replacement_margin_stats"] = dict(margin_analysis["analysis"].get("replacement_margin_stats", {}))
            summary["unique_margin_count"] = int(margin_analysis["analysis"].get("unique_margin_count", 0))
            summary["margin_range"] = float(margin_analysis["analysis"].get("margin_range", 0.0))
            summary["margin_near_threshold_band_count"] = int(margin_analysis["analysis"].get("margin_near_threshold_band_count", 0))
            summary["margin_near_threshold_band_ratio"] = float(margin_analysis["analysis"].get("margin_near_threshold_band_ratio", 0.0))

        self._write_json(self.shield_trace_summary_path, summary)
        artifact_paths = self.manifest.setdefault("artifact_paths", {})
        artifact_paths["shield_trace_summary_report"] = str(self.shield_trace_summary_path)
        if margin_analysis is not None:
            artifact_paths[f"shield_margin_analysis_{self._sanitize_artifact_key(summary['variant_name'])}"] = str(margin_analysis["analysis_path"])
        return {"trace_summary_path": self.shield_trace_summary_path, "summary": summary}

    def _write_shield_trace_tuning_summary(self) -> Optional[Dict[str, Any]]:
        if self.reports_dir is None or self.shield_trace_tuning_summary_path is None:
            return None

        trace_dirs = sorted(
            path for path in self.reports_dir.glob("shield_trace*")
            if path.is_dir() and self._find_trace_summary_path(path) is not None
        )
        if not trace_dirs:
            return None

        variants: List[Dict[str, Any]] = []
        for trace_dir in trace_dirs:
            entry = self._build_shield_trace_tuning_entry(trace_dir)
            if entry is not None:
                variants.append(entry)

        if not variants:
            return None

        margin_summary_payload = self._write_shield_margin_analysis_summary(trace_dirs=trace_dirs, tuning_variants=variants)
        margin_summary_path = margin_summary_payload["summary_path"] if margin_summary_payload is not None else None

        order = {"C_baseline": 0, "C1": 1, "C2": 2, "D1": 3, "E2": 4, "F1": 5, "F2": 6, "F3": 7, "E1": 8, "E3": 9, "D2": 10, "D3": 11, "C_strong": 12}
        variants.sort(key=lambda item: (order.get(str(item.get("variant_name", "")), 99), str(item.get("variant_name", ""))))

        summary = {
            "run_id": str(self.run_id or ""),
            "baseline_available": any(item["variant_name"] == "C_baseline" for item in variants),
            "shield_margin_analysis_summary_path": str(margin_summary_path) if margin_summary_path is not None else "",
            "variants": variants,
        }
        self._write_json(self.shield_trace_tuning_summary_path, summary)
        self.manifest.setdefault("artifact_paths", {})["shield_trace_tuning_summary_report"] = str(self.shield_trace_tuning_summary_path)
        return {"summary_path": self.shield_trace_tuning_summary_path, "summary": summary}

    def _build_shield_trace_tuning_entry(self, trace_dir: Path) -> Optional[Dict[str, Any]]:
        trace_summary_path = self._find_trace_summary_path(trace_dir)
        if trace_summary_path is None:
            return None

        trace_summary = self._read_json(trace_summary_path)
        pair_paths = [Path(path) for path in list(trace_summary.get("pair_files", []) or []) if Path(path).exists()]
        if not pair_paths:
            pair_paths = self._find_trace_pair_paths(trace_dir)

        pair_payloads = [self._read_json(path) for path in pair_paths if path.exists()]
        if pair_payloads:
            mean_intervention_count = sum(int(item.get("intervention_count", 0)) for item in pair_payloads) / len(pair_payloads)
            mean_risk_reduction = sum(float(item.get("mean_risk_reduction", 0.0)) for item in pair_payloads) / len(pair_payloads)
            reward_deltas = [
                float(item.get("shielded_reward", 0.0)) - float(item.get("baseline_reward", 0.0))
                for item in pair_payloads
            ]
            mean_reward_gap = sum(reward_deltas) / len(reward_deltas)
            mean_replacement_count = sum(int(item.get("replacement_count", 0)) for item in pair_payloads) / len(pair_payloads)
            mean_fallback_count = sum(int(item.get("fallback_action_count", 0)) for item in pair_payloads) / len(pair_payloads)
            all_pairs_collision_free = all(
                not bool(item.get("baseline_collision", False)) and not bool(item.get("shielded_collision", False))
                for item in pair_payloads
            )
        else:
            mean_intervention_count = 0.0
            mean_risk_reduction = 0.0
            reward_deltas = []
            mean_reward_gap = 0.0
            mean_replacement_count = 0.0
            mean_fallback_count = 0.0
            all_pairs_collision_free = False

        margin_analysis = self._ensure_trace_margin_analysis(
            trace_dir=trace_dir,
            trace_summary=trace_summary,
            pair_payloads=pair_payloads,
            trace_summary_path=trace_summary_path,
        )
        margin_analysis_path = margin_analysis["analysis_path"] if margin_analysis is not None else None
        margin_analysis_payload = margin_analysis["analysis"] if margin_analysis is not None else {}
        replacement_margin_stats = dict(margin_analysis_payload.get("replacement_margin_stats", {}))

        return {
            "variant_name": self._shield_trace_variant_name(trace_dir.name),
            "trace_dir_name": trace_dir.name,
            "trace_summary_path": str(trace_summary_path),
            "margin_analysis_path": str(margin_analysis_path) if margin_analysis_path is not None else "",
            "pair_count": len(pair_payloads),
            "seeds": list(trace_summary.get("seeds", [])),
            "effective_shield_config": dict(trace_summary.get("effective_shield_config", {})),
            "regression_pair_count": int(trace_summary.get("regression_pair_count", 0)),
            "blocked_by_margin_count": int(trace_summary.get("blocked_by_margin_count", 0)),
            "raw_passthrough_count": int(trace_summary.get("raw_passthrough_count", 0)),
            "merge_lateral_guard_block_count": int(trace_summary.get("merge_lateral_guard_block_count", 0)),
            "candidate_selected_count": int(trace_summary.get("candidate_selected_count", 0)),
            "mean_intervention_count": float(mean_intervention_count),
            "mean_risk_reduction": float(mean_risk_reduction),
            "mean_reward_gap_to_baseline_policy": float(mean_reward_gap),
            "mean_replacement_count": float(mean_replacement_count),
            "mean_fallback_action_count": float(mean_fallback_count),
            "all_pairs_collision_free": bool(all_pairs_collision_free),
            "all_pairs_reward_delta": reward_deltas,
            "replacement_margin_mean": replacement_margin_stats.get("mean"),
            "replacement_margin_stdev": replacement_margin_stats.get("stdev"),
            "replacement_margin_min": replacement_margin_stats.get("min"),
            "replacement_margin_max": replacement_margin_stats.get("max"),
            "unique_margin_count": int(margin_analysis_payload.get("unique_margin_count", 0)),
            "margin_near_threshold_band_ratio": float(margin_analysis_payload.get("margin_near_threshold_band_ratio", 0.0)),
        }

    def _write_shield_margin_analysis_summary(
        self,
        trace_dirs: Optional[Sequence[Path]] = None,
        tuning_variants: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.reports_dir is None or self.shield_margin_analysis_summary_path is None:
            return None

        if trace_dirs is None:
            trace_dirs = sorted(
                path for path in self.reports_dir.glob("shield_trace*")
                if path.is_dir() and self._find_trace_summary_path(path) is not None
            )
        if not trace_dirs:
            return None

        tuning_by_name = {str(item.get("variant_name", "")): item for item in list(tuning_variants or [])}
        entries: List[Dict[str, Any]] = []
        for trace_dir in trace_dirs:
            tuning_entry = tuning_by_name.get(self._shield_trace_variant_name(trace_dir.name))
            if tuning_entry is None:
                tuning_entry = self._build_shield_trace_tuning_entry(trace_dir)
            if tuning_entry is None:
                continue
            analysis_path = Path(str(tuning_entry.get("margin_analysis_path", ""))) if tuning_entry.get("margin_analysis_path") else None
            if analysis_path is None or not analysis_path.exists():
                continue
            analysis = self._read_json(analysis_path)
            stats = dict(analysis.get("replacement_margin_stats", {}))
            entries.append({
                "variant_name": str(tuning_entry.get("variant_name", self._shield_trace_variant_name(trace_dir.name))),
                "trace_dir_name": str(tuning_entry.get("trace_dir_name", trace_dir.name)),
                "margin_analysis_path": str(analysis_path),
                "replacement_min_risk_margin": float(dict(tuning_entry.get("effective_shield_config", {})).get("replacement_min_risk_margin", 0.0)),
                "replacement_step_count": int(analysis.get("replacement_step_count", 0)),
                "replacement_margin_mean": stats.get("mean"),
                "replacement_margin_stdev": stats.get("stdev"),
                "replacement_margin_min": stats.get("min"),
                "replacement_margin_max": stats.get("max"),
                "unique_margin_count": int(analysis.get("unique_margin_count", 0)),
                "margin_near_threshold_band_ratio": float(analysis.get("margin_near_threshold_band_ratio", 0.0)),
                "candidate_selected_count": int(tuning_entry.get("candidate_selected_count", 0)),
                "mean_intervention_count": float(tuning_entry.get("mean_intervention_count", 0.0)),
                "mean_risk_reduction": float(tuning_entry.get("mean_risk_reduction", 0.0)),
                "mean_reward_gap_to_baseline_policy": float(tuning_entry.get("mean_reward_gap_to_baseline_policy", 0.0)),
            })

        if not entries:
            return None

        order = {"D1": 0, "E2": 1, "F1": 2, "F2": 3, "F3": 4}
        entries.sort(key=lambda item: (order.get(str(item.get("variant_name", "")), 99), str(item.get("variant_name", ""))))
        summary = {
            "run_id": str(self.run_id or ""),
            "focus_variants": ["D1", "E2", "F1", "F2", "F3"],
            "variants": entries,
        }
        self._write_json(self.shield_margin_analysis_summary_path, summary)
        self.manifest.setdefault("artifact_paths", {})["shield_margin_analysis_summary_report"] = str(self.shield_margin_analysis_summary_path)
        return {"summary_path": self.shield_margin_analysis_summary_path, "summary": summary}

    def _ensure_trace_margin_analysis(
        self,
        trace_dir: Path,
        trace_summary: Dict[str, Any],
        pair_payloads: Sequence[Dict[str, Any]],
        trace_summary_path: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        if trace_summary_path is None:
            trace_summary_path = self._find_trace_summary_path(trace_dir)
        if trace_summary_path is None:
            return None

        analysis = self._build_trace_margin_analysis(
            variant_name=self._shield_trace_variant_name(trace_dir.name),
            seeds=list(trace_summary.get("seeds", [])),
            effective_shield_config=dict(trace_summary.get("effective_shield_config", {})),
            pair_payloads=pair_payloads,
        )
        analysis_path = trace_dir / "margin_analysis.json"
        self._write_json(analysis_path, analysis)

        trace_summary["margin_analysis_path"] = str(analysis_path)
        trace_summary["replacement_margin_stats"] = dict(analysis.get("replacement_margin_stats", {}))
        trace_summary["unique_margin_count"] = int(analysis.get("unique_margin_count", 0))
        trace_summary["margin_range"] = float(analysis.get("margin_range", 0.0))
        trace_summary["margin_near_threshold_band_count"] = int(analysis.get("margin_near_threshold_band_count", 0))
        trace_summary["margin_near_threshold_band_ratio"] = float(analysis.get("margin_near_threshold_band_ratio", 0.0))
        self._write_json(trace_summary_path, trace_summary)
        self.manifest.setdefault("artifact_paths", {})[f"shield_margin_analysis_{self._sanitize_artifact_key(trace_dir.name)}"] = str(analysis_path)
        return {"analysis_path": analysis_path, "analysis": analysis}

    def _build_trace_margin_analysis(
        self,
        variant_name: str,
        seeds: Sequence[Any],
        effective_shield_config: Dict[str, Any],
        pair_payloads: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        replacement_steps: List[Dict[str, Any]] = []
        for pair_payload in pair_payloads:
            for step in list(pair_payload.get("shielded_steps", []) or []):
                normalized = self._normalize_trace_step(step)
                if bool(normalized.get("replacement_happened", False)) and not bool(normalized.get("fallback_used", False)):
                    replacement_steps.append(normalized)

        margins = [float(step.get("replacement_margin", 0.0)) for step in replacement_steps]
        raw_risks = [float(step.get("raw_risk", 0.0)) for step in replacement_steps]
        final_risks = [float(step.get("final_risk", 0.0)) for step in replacement_steps]
        improvements = [float(step.get("risk_reduction", 0.0)) for step in replacement_steps]
        raw_action_types = Counter(str(step.get("raw_action_type", "")) for step in replacement_steps if str(step.get("raw_action_type", "")))
        selected_action_types = Counter(str(step.get("final_action_type", "")) for step in replacement_steps if str(step.get("final_action_type", "")))
        chosen_rank_counts = Counter(str(int(step.get("chosen_candidate_rank_by_risk", -1))) for step in replacement_steps)

        threshold = float(dict(effective_shield_config).get("replacement_min_risk_margin", 0.0))
        margin_near_threshold_band_count = sum(1 for value in margins if abs(float(value) - threshold) <= 0.003)
        unique_margin_count = len({round(float(value), 9) for value in margins}) if margins else 0
        margin_stats = self._numeric_stats(margins)

        return {
            "variant_name": str(variant_name),
            "run_id": str(self.run_id or ""),
            "seeds": [int(seed) for seed in seeds],
            "replacement_step_count": len(replacement_steps),
            "replacement_margin_stats": margin_stats,
            "selected_candidate_fine_risk_stats": self._numeric_stats(final_risks),
            "raw_action_fine_risk_stats": self._numeric_stats(raw_risks),
            "risk_improvement_stats": self._numeric_stats(improvements),
            "selected_action_type_counts": dict(sorted(selected_action_types.items())),
            "raw_action_type_counts": dict(sorted(raw_action_types.items())),
            "chosen_candidate_rank_counts": dict(sorted(chosen_rank_counts.items(), key=lambda item: int(item[0]))),
            "unique_margin_count": int(unique_margin_count),
            "margin_range": float((max(margins) - min(margins)) if margins else 0.0),
            "margin_near_threshold_band_count": int(margin_near_threshold_band_count),
            "margin_near_threshold_band_ratio": float(margin_near_threshold_band_count / len(margins)) if margins else 0.0,
        }

    def _numeric_stats(self, values: Sequence[float]) -> Dict[str, Any]:
        cleaned = [float(value) for value in values]
        if not cleaned:
            return {
                "count": 0,
                "min": None,
                "p10": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "max": None,
                "mean": None,
                "stdev": 0.0,
            }

        ordered = sorted(cleaned)
        mean_value = sum(ordered) / len(ordered)
        variance = sum((value - mean_value) ** 2 for value in ordered) / len(ordered)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "p10": self._quantile(ordered, 0.10),
            "p25": self._quantile(ordered, 0.25),
            "p50": self._quantile(ordered, 0.50),
            "p75": self._quantile(ordered, 0.75),
            "p90": self._quantile(ordered, 0.90),
            "p95": self._quantile(ordered, 0.95),
            "max": ordered[-1],
            "mean": mean_value,
            "stdev": math.sqrt(variance),
        }

    def _quantile(self, ordered_values: Sequence[float], q: float) -> Optional[float]:
        if not ordered_values:
            return None
        if len(ordered_values) == 1:
            return float(ordered_values[0])
        index = (len(ordered_values) - 1) * float(q)
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return float(ordered_values[lower])
        weight = index - lower
        return float(ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight)

    def _find_trace_summary_path(self, trace_dir: Path) -> Optional[Path]:
        canonical = trace_dir / "trace_summary.json"
        if canonical.exists():
            return canonical
        legacy_candidates = sorted(trace_dir.glob("*trace_summary.json"))
        if legacy_candidates:
            return legacy_candidates[0]
        return None

    def _find_trace_pair_paths(self, trace_dir: Path) -> List[Path]:
        canonical = sorted(trace_dir.glob("pair_*_seed_*.json"))
        if canonical:
            return canonical
        legacy = sorted(trace_dir.glob("*pair_*seed_*.json"))
        if legacy:
            return legacy
        return sorted(trace_dir.glob("*pair_*.json"))

    def _shield_trace_variant_name(self, trace_dir_name: str) -> str:
        normalized = str(trace_dir_name or "").strip().lower()
        if normalized == "shield_trace":
            return "C_baseline"
        if normalized == "shield_trace_c1":
            return "C1"
        if normalized == "shield_trace_c2":
            return "C2"
        if normalized == "shield_trace_d1":
            return "D1"
        if normalized == "shield_trace_e1":
            return "E1"
        if normalized == "shield_trace_e2":
            return "E2"
        if normalized == "shield_trace_f1":
            return "F1"
        if normalized == "shield_trace_f2":
            return "F2"
        if normalized == "shield_trace_f3":
            return "F3"
        if normalized == "shield_trace_e3":
            return "E3"
        if normalized == "shield_trace_d2":
            return "D2"
        if normalized == "shield_trace_d3":
            return "D3"
        if normalized == "shield_trace_c_strong":
            return "C_strong"
        return str(trace_dir_name or "trace")

    def _effective_shield_config(self, stage_config: SafeRLConfig) -> Dict[str, float]:
        return {
            "risk_threshold": float(stage_config.shield.risk_threshold),
            "uncertainty_threshold": float(stage_config.shield.uncertainty_threshold),
            "replacement_min_risk_margin": float(stage_config.shield.replacement_min_risk_margin),
            "raw_passthrough_risk_threshold": float(stage_config.shield.raw_passthrough_risk_threshold),
            "effective_raw_passthrough_threshold": float(min(float(stage_config.shield.risk_threshold), float(stage_config.shield.raw_passthrough_risk_threshold))),
            "merge_override_margin": float(stage_config.shield.merge_override_margin),
        }

    def _read_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_trace_pair_payload(
        self,
        pair_index: int,
        baseline: Dict[str, Any],
        shielded: Dict[str, Any],
        scenario_source: str,
        risky_mode: bool,
    ) -> Dict[str, Any]:
        baseline_steps = [self._normalize_trace_step(item) for item in list(baseline.get("step_trace", []) or [])]
        shielded_steps = [self._normalize_trace_step(item) for item in list(shielded.get("step_trace", []) or [])]
        aligned_steps = self._align_trace_steps(baseline_steps, shielded_steps)
        first_replacement_step = self._first_matching_step(shielded_steps, lambda item: bool(item.get("replacement_happened", False)))
        collision_step_baseline = self._first_matching_step(baseline_steps, lambda item: bool(item.get("collision", False)))
        collision_step_shielded = self._first_matching_step(shielded_steps, lambda item: bool(item.get("collision", False)))
        has_lane_change_replacement = any(bool(item.get("replacement_happened", False) and item.get("lane_change_involved", False)) for item in shielded_steps)
        merge_guard_triggered = any(str(item.get("constraint_reason", "")) == "merge_lateral_guard" for item in shielded_steps)
        has_fallback = any(bool(item.get("fallback_used", False)) for item in shielded_steps)
        collision_after_replacement = collision_step_shielded >= 0 and any(
            int(item.get("step_index", -1)) <= collision_step_shielded and bool(item.get("replacement_happened", False))
            for item in shielded_steps
        )
        regression_pair = bool(not bool(int(baseline.get("collisions", 0)) > 0) and bool(int(shielded.get("collisions", 0)) > 0))
        blocked_by_margin_count = sum(
            1
            for item in shielded_steps
            if str(item.get("constraint_reason", "")) == "blocked_by_margin"
            or any(str(candidate.get("constraint_reason", "")) == "blocked_by_margin" for candidate in list(item.get("candidate_evaluations", []) or []))
        )
        raw_passthrough_count = sum(
            1
            for item in shielded_steps
            if str(item.get("constraint_reason", "")) == "raw_passthrough" and not bool(item.get("replacement_happened", False))
        )
        merge_lateral_guard_block_count = sum(
            1
            for item in shielded_steps
            if str(item.get("constraint_reason", "")) == "merge_lateral_guard"
            or any(str(candidate.get("constraint_reason", "")) == "merge_lateral_guard" for candidate in list(item.get("candidate_evaluations", []) or []))
        )
        candidate_selected_count = sum(
            1
            for item in shielded_steps
            if bool(item.get("replacement_happened", False)) and not bool(item.get("fallback_used", False))
        )
        proof_step = shielded_steps[first_replacement_step] if 0 <= first_replacement_step < len(shielded_steps) else {}
        same_state_proof = self._build_same_state_proof(list(proof_step.get("history_scene", []) or []), proof_step)

        return {
            "pair_index": int(pair_index),
            "seed": int(shielded.get("seed", baseline.get("seed", -1))),
            "risky_mode": bool(shielded.get("risky_mode", baseline.get("risky_mode", risky_mode))),
            "scenario_source": str(shielded.get("scenario_source", baseline.get("scenario_source", scenario_source))),
            "baseline_episode_id": str(baseline.get("episode_id", "")),
            "shielded_episode_id": str(shielded.get("episode_id", "")),
            "baseline_collision": bool(int(baseline.get("collisions", 0)) > 0),
            "shielded_collision": bool(int(shielded.get("collisions", 0)) > 0),
            "baseline_reward": float(baseline.get("mean_reward", 0.0)),
            "shielded_reward": float(shielded.get("mean_reward", 0.0)),
            "baseline_raw_risk": float(baseline.get("mean_raw_risk", 0.0)),
            "shielded_raw_risk": float(shielded.get("mean_raw_risk", 0.0)),
            "shielded_final_risk": float(shielded.get("mean_final_risk", 0.0)),
            "intervention_count": int(shielded.get("interventions", 0)),
            "replacement_count": int(shielded.get("replacement_count", 0)),
            "replacement_same_as_raw_count": int(shielded.get("replacement_same_as_raw_count", 0)),
            "fallback_action_count": int(shielded.get("fallback_action_count", 0)),
            "mean_risk_reduction": float(shielded.get("mean_risk_reduction", 0.0)),
            "blocked_by_margin_count": int(blocked_by_margin_count),
            "raw_passthrough_count": int(raw_passthrough_count),
            "merge_lateral_guard_block_count": int(merge_lateral_guard_block_count),
            "candidate_selected_count": int(candidate_selected_count),
            "first_replacement_step": int(first_replacement_step),
            "collision_step_baseline": int(collision_step_baseline),
            "collision_step_shielded": int(collision_step_shielded),
            "regression_pair": regression_pair,
            "has_lane_change_replacement": bool(has_lane_change_replacement),
            "merge_guard_triggered": bool(merge_guard_triggered),
            "has_fallback": bool(has_fallback),
            "collision_after_replacement": bool(collision_after_replacement),
            **same_state_proof,
            "baseline_steps": baseline_steps,
            "shielded_steps": shielded_steps,
            "aligned_steps": aligned_steps,
        }

    def _normalize_trace_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(step)
        normalized.setdefault("step_index", 0)
        normalized.setdefault("raw_action", -1)
        normalized.setdefault("final_action", -1)
        normalized.setdefault("executed_action", normalized.get("final_action", -1))
        normalized.setdefault("replacement_happened", False)
        normalized.setdefault("fallback_used", False)
        normalized.setdefault("chosen_candidate_index", -1)
        normalized.setdefault("chosen_candidate_rank_by_risk", -1)
        normalized.setdefault("raw_risk", 0.0)
        normalized.setdefault("final_risk", normalized.get("raw_risk", 0.0))
        normalized.setdefault("risk_reduction", float(normalized.get("raw_risk", 0.0)) - float(normalized.get("final_risk", 0.0)))
        normalized.setdefault("candidate_evaluations", [])
        normalized.setdefault("raw_action_type", "")
        normalized.setdefault("final_action_type", "")
        normalized.setdefault("lane_change_involved", False)
        normalized.setdefault("ego_lane_id", "")
        normalized.setdefault("ego_lane_index", 0)
        normalized.setdefault("ego_speed", 0.0)
        normalized.setdefault("ttc", 0.0)
        normalized.setdefault("min_distance", 0.0)
        normalized.setdefault("collision", False)
        normalized.setdefault("constraint_reason", "")
        normalized.setdefault("replacement_margin", 0.0)
        normalized.setdefault("reward", 0.0)
        normalized.setdefault("task_reward", 0.0)
        return normalized

    def _align_trace_steps(self, baseline_steps: Sequence[Dict[str, Any]], shielded_steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        aligned: List[Dict[str, Any]] = []
        max_steps = max(len(baseline_steps), len(shielded_steps))
        for idx in range(max_steps):
            aligned.append(
                {
                    "step_index": idx,
                    "baseline": baseline_steps[idx] if idx < len(baseline_steps) else None,
                    "shielded": shielded_steps[idx] if idx < len(shielded_steps) else None,
                }
            )
        return aligned

    def _first_matching_step(self, steps: Sequence[Dict[str, Any]], predicate) -> int:
        for item in steps:
            if predicate(item):
                return int(item.get("step_index", -1))
        return -1


def run_safe_rl_pipeline(config_path: Optional[str] = None, stage: str = "all", run_id: Optional[str] = None) -> Dict:
    config = load_safe_rl_config(config_path)
    pipeline = SafeRLPipeline(config)
    return pipeline.run(stage=stage, run_id=run_id)
























