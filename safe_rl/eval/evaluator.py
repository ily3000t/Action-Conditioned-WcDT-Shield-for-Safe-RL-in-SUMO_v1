from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from safe_rl.config.config import EvalConfig
from safe_rl.data.risk import get_ego_vehicle
from safe_rl.data.types import ActionConditionedSample, dataclass_to_dict
from safe_rl.eval.metrics import (
    acceptance_passed,
    aggregate_episode_summaries,
    compare_system_metrics,
    summarize_episode,
    summary_to_dict,
)


class SafeRLEvaluator:
    def __init__(self, eval_config: EvalConfig):
        self.eval_config = eval_config

    def evaluate_world_model(self, world_predictor, samples: Sequence[ActionConditionedSample]) -> Dict[str, float]:
        if not samples:
            return {
                "traj_ade": 0.0,
                "risk_acc": 0.0,
                "risk_mae": 0.0,
            }

        ade_list = []
        risk_correct = 0
        risk_mae = []

        for sample in samples:
            prediction = world_predictor.predict(sample.history_scene, sample.candidate_action)
            pred_traj = prediction.multimodal_future
            if isinstance(pred_traj, np.ndarray) and pred_traj.ndim >= 3:
                target_xy = np.array(
                    [[get_ego_vehicle(s).x, get_ego_vehicle(s).y] for s in sample.future_scene],
                    dtype=np.float32,
                )
                target_len = min(target_xy.shape[0], pred_traj.shape[1])
                if target_len > 0:
                    target_xy = target_xy[:target_len]
                    modal_errors = []
                    for m in range(pred_traj.shape[0]):
                        pred_xy = pred_traj[m, :target_len, :2]
                        modal_errors.append(np.mean(np.linalg.norm(pred_xy - target_xy, axis=-1)))
                    ade_list.append(float(np.min(modal_errors)))

            pred_binary = 1.0 if prediction.aggregated_risk >= 0.5 else 0.0
            gt_binary = float(sample.risk_labels.overall_risk >= 0.5)
            risk_correct += int(pred_binary == gt_binary)
            risk_mae.append(abs(float(prediction.aggregated_risk) - float(sample.risk_labels.overall_risk)))

        return {
            "traj_ade": float(np.mean(ade_list)) if ade_list else 0.0,
            "risk_acc": float(risk_correct / len(samples)),
            "risk_mae": float(np.mean(risk_mae)) if risk_mae else 0.0,
        }

    def evaluate_policy(
        self,
        env,
        policy,
        episodes: int,
        risky_mode: bool = True,
        tb_writer=None,
        tb_prefix: str = "",
        seeds: Optional[Sequence[int]] = None,
        collect_step_traces: bool = False,
    ) -> Dict[str, float]:
        summaries = []
        episode_details = []
        prefix = (tb_prefix or "policy").strip("/")
        is_baseline = prefix == "baseline"
        for i in range(episodes):
            seed = None
            if seeds is not None and i < len(seeds):
                seed = int(seeds[i])
            obs, reset_info = env.reset(seed=seed, options={"risky_mode": risky_mode})
            step_infos = []
            step_history_scenes = []
            rewards = []
            done = False
            while not done:
                action = int(policy.predict(obs, deterministic=True))
                obs, reward, terminated, truncated, info = env.step(action)
                normalized_info = self._normalize_risk_info(info, is_baseline=is_baseline)
                step_infos.append(normalized_info)
                history_scene = []
                if hasattr(env, "last_transition") and getattr(env, "last_transition", None) is not None:
                    history_scene = env.last_transition.get("history_scene", [])
                step_history_scenes.append(history_scene)
                rewards.append(float(reward))
                done = terminated or truncated

            episode_id = str(reset_info.get("episode_id", f"eval_{i:04d}"))
            summary = summarize_episode(episode_id, step_infos, rewards)
            summaries.append(summary)
            detail = {
                **summary_to_dict(summary),
                "seed": None if seed is None else int(seed),
                "risky_mode": bool(reset_info.get("risky_mode", risky_mode)),
                "scenario_source": str(reset_info.get("scenario_source", "")),
                "episode_id": episode_id,
            }
            if collect_step_traces:
                step_trace = []
                for step_index, item in enumerate(step_infos):
                    history_scene = step_history_scenes[step_index] if step_index < len(step_history_scenes) else None
                    step_trace.append(self._build_step_trace(step_index, item, history_scene=history_scene))
                detail["step_trace"] = step_trace
            episode_details.append(detail)

            if tb_writer is not None:
                tb_writer.add_scalar(f"{prefix}/episode_reward", float(summary.mean_reward), i)
                tb_writer.add_scalar(f"{prefix}/episode_task_reward", float(summary.mean_task_reward), i)
                tb_writer.add_scalar(f"{prefix}/episode_collision", float(summary.collisions > 0), i)
                tb_writer.add_scalar(f"{prefix}/episode_interventions", float(summary.interventions), i)
                tb_writer.add_scalar(f"{prefix}/episode_success", float(summary.success), i)
                tb_writer.add_scalar(f"{prefix}/episode_avg_speed", float(summary.avg_speed), i)
                tb_writer.add_scalar(f"{prefix}/episode_steps", float(summary.steps), i)
                tb_writer.add_scalar(f"{prefix}/episode_min_ttc", float(summary.min_ttc), i)
                tb_writer.add_scalar(f"{prefix}/episode_min_distance", float(summary.min_distance), i)
                tb_writer.add_scalar(f"{prefix}/episode_near_risk_step_rate", float(summary.near_risk_step_rate), i)
                tb_writer.add_scalar(f"{prefix}/episode_mean_raw_risk", float(summary.mean_raw_risk), i)
                tb_writer.add_scalar(f"{prefix}/episode_mean_final_risk", float(summary.mean_final_risk), i)
                tb_writer.add_scalar(f"{prefix}/episode_mean_risk_reduction", float(summary.mean_risk_reduction), i)
                tb_writer.add_scalar(f"{prefix}/episode_replacement_count", float(summary.replacement_count), i)
                tb_writer.add_scalar(f"{prefix}/episode_fallback_action_count", float(summary.fallback_action_count), i)

        aggregated = aggregate_episode_summaries(summaries)
        aggregated["episode_details"] = episode_details
        if tb_writer is not None:
            tb_writer.add_scalar(f"{prefix}/summary_collision_rate", float(aggregated.get("collision_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_intervention_rate", float(aggregated.get("intervention_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_success_rate", float(aggregated.get("success_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_avg_speed", float(aggregated.get("avg_speed", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_reward", float(aggregated.get("mean_reward", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_task_reward", float(aggregated.get("mean_task_reward", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_min_ttc", float(aggregated.get("min_ttc", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_min_distance", float(aggregated.get("min_distance", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_near_risk_step_rate", float(aggregated.get("near_risk_step_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_raw_risk", float(aggregated.get("mean_raw_risk", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_final_risk", float(aggregated.get("mean_final_risk", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_risk_reduction", float(aggregated.get("mean_risk_reduction", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_replacement_count", float(aggregated.get("replacement_count", 0.0)), 0)
        return aggregated

    def compare_baseline_and_shielded(self, baseline: Dict[str, float], shielded: Dict[str, float]) -> Dict[str, float]:
        return compare_system_metrics(baseline, shielded)

    def evaluate_acceptance(self, delta_metrics: Dict[str, float]) -> bool:
        return acceptance_passed(delta_metrics, self.eval_config)

    def _normalize_risk_info(self, info: Dict[str, float], is_baseline: bool) -> Dict[str, float]:
        normalized = dict(info)
        raw = float(normalized.get("risk_raw", 0.0))
        final = float(normalized.get("risk_final", 0.0))

        if is_baseline:
            proxy = self._proxy_risk(normalized)
            raw = proxy
            final = proxy
        normalized["risk_raw"] = raw
        normalized["risk_final"] = final
        return normalized

    def _proxy_risk(self, info: Dict[str, float]) -> float:
        min_distance = float(info.get("min_distance", 30.0))
        min_ttc = float(info.get("ttc", 8.0))
        distance_term = 1.0 if min_distance < 3.0 else max(0.0, 1.0 - min_distance / 30.0)
        ttc_term = 1.0 if min_ttc < 1.5 else max(0.0, 1.0 - min_ttc / 8.0)
        return float(max(distance_term, ttc_term))


    def _build_step_trace(self, step_index: int, info: Dict[str, Any], history_scene=None) -> Dict[str, Any]:
        candidate_evaluations = []
        for item in list(info.get("candidate_evaluations", []) or []):
            candidate_evaluations.append(
                {
                    "action_id": int(item.get("action_id", -1)),
                    "action_type": str(item.get("action_type", "")),
                    "distance_to_raw": int(item.get("distance_to_raw", 0)),
                    "coarse_risk": float(item.get("coarse_risk", 0.0)),
                    "fine_risk": None if item.get("fine_risk") is None else float(item.get("fine_risk", 0.0)),
                    "uncertainty": None if item.get("uncertainty") is None else float(item.get("uncertainty", 0.0)),
                    "selected": bool(item.get("selected", False)),
                    "safe_under_threshold": bool(item.get("safe_under_threshold", False)),
                    "evaluated": bool(item.get("evaluated", False)),
                    "constraint_reason": str(item.get("constraint_reason", "")),
                }
            )

        history_scene_payload = dataclass_to_dict(history_scene) if history_scene is not None else []
        return {
            "step_index": int(step_index),
            "history_scene": history_scene_payload,
            "raw_action": int(info.get("raw_action", -1)),
            "final_action": int(info.get("final_action", -1)),
            "executed_action": int(info.get("executed_action", info.get("final_action", -1))),
            "replacement_happened": bool(info.get("replacement_happened", False)),
            "fallback_used": bool(info.get("fallback_used", False)),
            "chosen_candidate_index": int(info.get("chosen_candidate_index", -1)),
            "chosen_candidate_rank_by_risk": int(info.get("chosen_candidate_rank_by_risk", -1)),
            "raw_risk": float(info.get("risk_raw", 0.0)),
            "final_risk": float(info.get("risk_final", 0.0)),
            "risk_reduction": float(info.get("risk_reduction", float(info.get("risk_raw", 0.0)) - float(info.get("risk_final", 0.0)))),
            "candidate_evaluations": candidate_evaluations,
            "raw_action_type": str(info.get("raw_action_type", "")),
            "final_action_type": str(info.get("final_action_type", "")),
            "lane_change_involved": bool(info.get("lane_change_involved", False)),
            "ego_lane_id": str(info.get("ego_lane_id", "")),
            "ego_lane_index": int(info.get("ego_lane_index", 0)),
            "ego_speed": float(info.get("ego_speed", 0.0)),
            "ttc": float(info.get("ttc", 0.0)),
            "min_distance": float(info.get("min_distance", 0.0)),
            "collision": bool(info.get("collision", False)),
            "constraint_reason": str(info.get("constraint_reason", "")),
            "replacement_margin": float(info.get("replacement_margin", 0.0)),
            "best_candidate_action": int(info.get("best_candidate_action", -1)),
            "best_candidate_fine_risk": None if info.get("best_candidate_fine_risk") is None else float(info.get("best_candidate_fine_risk", 0.0)),
            "raw_action_fine_risk": float(info.get("raw_action_fine_risk", info.get("risk_raw", 0.0))),
            "best_margin": None if info.get("best_margin") is None else float(info.get("best_margin", 0.0)),
            "no_safe_candidate": bool(info.get("no_safe_candidate", False)),
            "raw_already_best": bool(info.get("raw_already_best", False)),
            "primary_nonreplacement_reason": str(info.get("primary_nonreplacement_reason", "")),
            "reward": float(info.get("reward", 0.0)),
            "task_reward": float(info.get("task_reward", 0.0)),
        }
