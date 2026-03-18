from typing import Dict, Sequence

import numpy as np

from safe_rl.config.config import EvalConfig
from safe_rl.data.risk import get_ego_vehicle
from safe_rl.data.types import ActionConditionedSample
from safe_rl.eval.metrics import (
    acceptance_passed,
    aggregate_episode_summaries,
    compare_system_metrics,
    summarize_episode,
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
                # [M, T, 5]
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

    def evaluate_policy(self, env, policy, episodes: int, risky_mode: bool = True, tb_writer=None, tb_prefix: str = "") -> Dict[str, float]:
        summaries = []
        prefix = (tb_prefix or "policy").strip("/")
        for i in range(episodes):
            obs, _ = env.reset(options={"risky_mode": risky_mode})
            step_infos = []
            rewards = []
            done = False
            while not done:
                action = int(policy.predict(obs, deterministic=True))
                obs, reward, terminated, truncated, info = env.step(action)
                step_infos.append(info)
                rewards.append(float(reward))
                done = terminated or truncated

            summary = summarize_episode(f"eval_{i:04d}", step_infos, rewards)
            summaries.append(summary)

            if tb_writer is not None:
                tb_writer.add_scalar(f"{prefix}/episode_reward", float(summary.mean_reward), i)
                tb_writer.add_scalar(f"{prefix}/episode_collision", float(summary.collisions > 0), i)
                tb_writer.add_scalar(f"{prefix}/episode_interventions", float(summary.interventions), i)
                tb_writer.add_scalar(f"{prefix}/episode_success", float(summary.success), i)
                tb_writer.add_scalar(f"{prefix}/episode_avg_speed", float(summary.avg_speed), i)
                tb_writer.add_scalar(f"{prefix}/episode_steps", float(summary.steps), i)

        aggregated = aggregate_episode_summaries(summaries)
        if tb_writer is not None:
            tb_writer.add_scalar(f"{prefix}/summary_collision_rate", float(aggregated.get("collision_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_intervention_rate", float(aggregated.get("intervention_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_success_rate", float(aggregated.get("success_rate", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_avg_speed", float(aggregated.get("avg_speed", 0.0)), 0)
            tb_writer.add_scalar(f"{prefix}/summary_mean_reward", float(aggregated.get("mean_reward", 0.0)), 0)
        return aggregated

    def compare_baseline_and_shielded(self, baseline: Dict[str, float], shielded: Dict[str, float]) -> Dict[str, float]:
        return compare_system_metrics(baseline, shielded)

    def evaluate_acceptance(self, delta_metrics: Dict[str, float]) -> bool:
        return acceptance_passed(delta_metrics, self.eval_config)
