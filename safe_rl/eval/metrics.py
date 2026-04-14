from dataclasses import asdict
from typing import Dict, List

import numpy as np

from safe_rl.config.config import EvalConfig
from safe_rl.data.types import EpisodeSummary


NEAR_RISK_TTC_THRESHOLD = 2.0
NEAR_RISK_MIN_DISTANCE_THRESHOLD = 6.0


def summarize_episode(
    episode_id: str,
    step_infos: List[dict],
    rewards: List[float],
    low_speed_threshold_mps: float = 2.0,
) -> EpisodeSummary:
    steps = len(step_infos)
    collisions = sum(1 for info in step_infos if bool(info.get("collision", False)))
    interventions = sum(1 for info in step_infos if bool(info.get("intervened", False)))
    avg_speed = float(np.mean([float(info.get("ego_speed", 0.0)) for info in step_infos])) if step_infos else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    mean_task_reward = float(
        np.mean([float(info.get("task_reward", info.get("reward", 0.0))) for info in step_infos])
    ) if step_infos else 0.0
    ttc_values = [float(info.get("ttc", 1e6)) for info in step_infos]
    min_distance_values = [float(info.get("min_distance", 1e6)) for info in step_infos]
    min_ttc = float(min(ttc_values)) if ttc_values else 0.0
    min_distance = float(min(min_distance_values)) if min_distance_values else 0.0
    near_risk_step_count = sum(
        int(
            float(info.get("ttc", 1e6)) <= NEAR_RISK_TTC_THRESHOLD
            or float(info.get("min_distance", 1e6)) <= NEAR_RISK_MIN_DISTANCE_THRESHOLD
        )
        for info in step_infos
    )
    near_risk_step_rate = float(near_risk_step_count / max(1, steps))
    low_speed_step_count = sum(
        int(float(info.get("ego_speed", 0.0)) < float(low_speed_threshold_mps))
        for info in step_infos
    )
    low_speed_step_rate = float(low_speed_step_count / max(1, steps))
    mean_raw_risk = float(np.mean([float(info.get("risk_raw", 0.0)) for info in step_infos])) if step_infos else 0.0
    mean_final_risk = float(np.mean([float(info.get("risk_final", 0.0)) for info in step_infos])) if step_infos else 0.0
    mean_risk_reduction = float(
        np.mean([float(info.get("risk_raw", 0.0)) - float(info.get("risk_final", 0.0)) for info in step_infos])
    ) if step_infos else 0.0
    replacement_count = sum(int(bool(info.get("replacement_happened", False))) for info in step_infos)
    replacement_same_as_raw_count = sum(int(info.get("replacement_same_as_raw_count", 0)) for info in step_infos)
    fallback_action_count = sum(int(info.get("fallback_action_count", 0)) for info in step_infos)
    shield_called_steps = sum(int(info.get("shield_called_steps", 0)) for info in step_infos)
    shield_candidate_evaluated_steps = sum(int(info.get("shield_candidate_evaluated_steps", 0)) for info in step_infos)
    shield_blocked_steps = sum(int(info.get("shield_blocked_steps", 0)) for info in step_infos)
    shield_replaced_steps = sum(int(info.get("shield_replaced_steps", 0)) for info in step_infos)
    success = collisions == 0 and steps > 0
    return EpisodeSummary(
        episode_id=episode_id,
        steps=steps,
        collisions=collisions,
        interventions=interventions,
        avg_speed=avg_speed,
        mean_reward=mean_reward,
        success=success,
        mean_task_reward=mean_task_reward,
        min_ttc=min_ttc,
        min_distance=min_distance,
        near_risk_step_count=near_risk_step_count,
        near_risk_step_rate=near_risk_step_rate,
        low_speed_step_count=low_speed_step_count,
        low_speed_step_rate=low_speed_step_rate,
        mean_raw_risk=mean_raw_risk,
        mean_final_risk=mean_final_risk,
        mean_risk_reduction=mean_risk_reduction,
        replacement_count=replacement_count,
        replacement_same_as_raw_count=replacement_same_as_raw_count,
        fallback_action_count=fallback_action_count,
        shield_called_steps=shield_called_steps,
        shield_candidate_evaluated_steps=shield_candidate_evaluated_steps,
        shield_blocked_steps=shield_blocked_steps,
        shield_replaced_steps=shield_replaced_steps,
    )


def aggregate_episode_summaries(summaries: List[EpisodeSummary]) -> Dict[str, float]:
    if not summaries:
        return {
            "episodes": 0.0,
            "collision_rate": 0.0,
            "intervention_rate": 0.0,
            "success_rate": 0.0,
            "avg_speed": 0.0,
            "mean_reward": 0.0,
            "mean_task_reward": 0.0,
            "min_ttc": 0.0,
            "min_distance": 0.0,
            "near_risk_step_count": 0.0,
            "near_risk_step_rate": 0.0,
            "near_risk_episode_rate": 0.0,
            "low_speed_step_count": 0.0,
            "low_speed_step_rate": 0.0,
            "mean_raw_risk": 0.0,
            "mean_final_risk": 0.0,
            "mean_risk_reduction": 0.0,
            "replacement_count": 0.0,
            "replacement_same_as_raw_count": 0.0,
            "fallback_action_count": 0.0,
            "shield_called_steps": 0.0,
            "shield_candidate_evaluated_steps": 0.0,
            "shield_blocked_steps": 0.0,
            "shield_replaced_steps": 0.0,
        }

    episodes = len(summaries)
    collision_rate = float(sum(1 for s in summaries if s.collisions > 0) / episodes)
    intervention_rate = float(sum(s.interventions for s in summaries) / max(1, sum(s.steps for s in summaries)))
    success_rate = float(sum(1 for s in summaries if s.success) / episodes)
    avg_speed = float(np.mean([s.avg_speed for s in summaries]))
    mean_reward = float(np.mean([s.mean_reward for s in summaries]))
    mean_task_reward = float(np.mean([s.mean_task_reward for s in summaries]))
    min_ttc = float(np.min([s.min_ttc for s in summaries]))
    min_distance = float(np.min([s.min_distance for s in summaries]))
    near_risk_step_count = float(sum(s.near_risk_step_count for s in summaries))
    near_risk_step_rate = float(near_risk_step_count / max(1, sum(s.steps for s in summaries)))
    near_risk_episode_rate = float(sum(1 for s in summaries if s.near_risk_step_count > 0) / episodes)
    low_speed_step_count = float(sum(s.low_speed_step_count for s in summaries))
    low_speed_step_rate = float(low_speed_step_count / max(1, sum(s.steps for s in summaries)))
    mean_raw_risk = float(np.mean([s.mean_raw_risk for s in summaries]))
    mean_final_risk = float(np.mean([s.mean_final_risk for s in summaries]))
    mean_risk_reduction = float(np.mean([s.mean_risk_reduction for s in summaries]))
    replacement_count = float(sum(s.replacement_count for s in summaries))
    replacement_same_as_raw_count = float(sum(s.replacement_same_as_raw_count for s in summaries))
    fallback_action_count = float(sum(s.fallback_action_count for s in summaries))
    shield_called_steps = float(sum(s.shield_called_steps for s in summaries))
    shield_candidate_evaluated_steps = float(sum(s.shield_candidate_evaluated_steps for s in summaries))
    shield_blocked_steps = float(sum(s.shield_blocked_steps for s in summaries))
    shield_replaced_steps = float(sum(s.shield_replaced_steps for s in summaries))
    return {
        "episodes": float(episodes),
        "collision_rate": collision_rate,
        "intervention_rate": intervention_rate,
        "success_rate": success_rate,
        "avg_speed": avg_speed,
        "mean_reward": mean_reward,
        "mean_task_reward": mean_task_reward,
        "min_ttc": min_ttc,
        "min_distance": min_distance,
        "near_risk_step_count": near_risk_step_count,
        "near_risk_step_rate": near_risk_step_rate,
        "near_risk_episode_rate": near_risk_episode_rate,
        "low_speed_step_count": low_speed_step_count,
        "low_speed_step_rate": low_speed_step_rate,
        "mean_raw_risk": mean_raw_risk,
        "mean_final_risk": mean_final_risk,
        "mean_risk_reduction": mean_risk_reduction,
        "replacement_count": replacement_count,
        "replacement_same_as_raw_count": replacement_same_as_raw_count,
        "fallback_action_count": fallback_action_count,
        "shield_called_steps": shield_called_steps,
        "shield_candidate_evaluated_steps": shield_candidate_evaluated_steps,
        "shield_blocked_steps": shield_blocked_steps,
        "shield_replaced_steps": shield_replaced_steps,
    }


def compare_system_metrics(baseline: Dict[str, float], shielded: Dict[str, float]) -> Dict[str, float]:
    baseline_collision = float(baseline.get("collision_rate", 0.0))
    shielded_collision = float(shielded.get("collision_rate", 0.0))
    if baseline_collision <= 1e-8:
        collision_reduction = 0.0
    else:
        collision_reduction = (baseline_collision - shielded_collision) / baseline_collision

    baseline_speed = float(baseline.get("avg_speed", 0.0))
    shielded_speed = float(shielded.get("avg_speed", 0.0))
    if baseline_speed <= 1e-8:
        efficiency_drop = 0.0
    else:
        efficiency_drop = (baseline_speed - shielded_speed) / baseline_speed

    return {
        "collision_reduction": float(collision_reduction),
        "efficiency_drop": float(efficiency_drop),
    }


def acceptance_passed(metrics_delta: Dict[str, float], eval_config: EvalConfig) -> bool:
    return (
        metrics_delta.get("collision_reduction", 0.0) >= eval_config.target_collision_reduction
        and metrics_delta.get("efficiency_drop", 1.0) <= eval_config.max_efficiency_drop
    )


def summary_to_dict(summary: EpisodeSummary) -> Dict:
    return asdict(summary)
