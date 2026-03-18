from dataclasses import asdict
from typing import Dict, List

import numpy as np

from safe_rl.config.config import EvalConfig
from safe_rl.data.types import EpisodeSummary


def summarize_episode(episode_id: str, step_infos: List[dict], rewards: List[float]) -> EpisodeSummary:
    steps = len(step_infos)
    collisions = sum(1 for info in step_infos if bool(info.get("collision", False)))
    interventions = sum(1 for info in step_infos if bool(info.get("intervened", False)))
    avg_speed = float(np.mean([float(info.get("ego_speed", 0.0)) for info in step_infos])) if step_infos else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    success = collisions == 0 and steps > 0
    return EpisodeSummary(
        episode_id=episode_id,
        steps=steps,
        collisions=collisions,
        interventions=interventions,
        avg_speed=avg_speed,
        mean_reward=mean_reward,
        success=success,
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
        }

    episodes = len(summaries)
    collision_rate = float(sum(1 for s in summaries if s.collisions > 0) / episodes)
    intervention_rate = float(sum(s.interventions for s in summaries) / max(1, sum(s.steps for s in summaries)))
    success_rate = float(sum(1 for s in summaries if s.success) / episodes)
    avg_speed = float(np.mean([s.avg_speed for s in summaries]))
    mean_reward = float(np.mean([s.mean_reward for s in summaries]))
    return {
        "episodes": float(episodes),
        "collision_rate": collision_rate,
        "intervention_rate": intervention_rate,
        "success_rate": success_rate,
        "avg_speed": avg_speed,
        "mean_reward": mean_reward,
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
