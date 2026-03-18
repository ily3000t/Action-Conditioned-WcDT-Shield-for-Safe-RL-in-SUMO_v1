import json
import random
import time
from pathlib import Path
from typing import List

from safe_rl.config.config import SafeRLConfig
from safe_rl.data.risk import aggregate_future_risk, compute_min_distance, compute_min_ttc, detect_collision
from safe_rl.data.types import EpisodeLog, RiskLabels, ShieldDecision, StepLog, dataclass_to_dict
from safe_rl.sim.actions import all_action_ids
from safe_rl.sim.backend_interface import ISumoBackend


class SumoDataCollector:
    def __init__(self, backend: ISumoBackend, config: SafeRLConfig):
        self.backend = backend
        self.config = config
        self._rng = random.Random(config.sim.random_seed)

    def backend_mode(self) -> str:
        # Backends expose _use_mock internally; introspection keeps interface stable.
        use_mock = bool(getattr(self.backend, "_use_mock", True))
        if use_mock:
            return "mock"
        return "sumo-real"

    def collect(self) -> List[EpisodeLog]:
        episodes: List[EpisodeLog] = []
        total = self.config.sim.normal_episodes + self.config.sim.risky_episodes
        print(
            f"[Collector] start: backend={self.backend_mode()}, total_episodes={total}, "
            f"episode_steps={self.config.sim.episode_steps}",
            flush=True,
        )

        t0 = time.time()
        log_interval = max(1, total // 20)

        for i in range(total):
            risky_mode = i >= self.config.sim.normal_episodes
            episode_id = f"ep_{i:05d}"
            episodes.append(self.collect_episode(episode_id, risky_mode=risky_mode))

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                eps_per_sec = (i + 1) / max(1e-6, elapsed)
                remaining = (total - (i + 1)) / max(1e-6, eps_per_sec)
                print(
                    f"[Collector] progress: {i + 1}/{total}, elapsed={elapsed:.1f}s, "
                    f"eta={remaining:.1f}s",
                    flush=True,
                )

        print(f"[Collector] done: episodes={len(episodes)}, total_time={time.time() - t0:.1f}s", flush=True)
        return episodes

    def collect_episode(self, episode_id: str, risky_mode: bool) -> EpisodeLog:
        steps: List[StepLog] = []
        self.backend.reset(seed=self._rng.randint(1, 1_000_000))

        if risky_mode:
            self.backend.inject_risk_event()

        for step_idx in range(self.config.sim.episode_steps):
            raw_action = self._rng.choice(all_action_ids())
            result = self.backend.step(raw_action)

            if risky_mode and self._rng.random() < self.config.sim.risk_event_prob:
                self.backend.inject_risk_event()

            labels = RiskLabels(
                collision=bool(result.info.get("collision", detect_collision(result.scene))),
                ttc_risk=compute_min_ttc(result.scene) < self.config.dataset.ttc_threshold,
                lane_violation=bool(result.info.get("lane_violation", False)),
                overall_risk=0.0,
                min_ttc=compute_min_ttc(result.scene),
                min_distance=compute_min_distance(result.scene),
            )
            labels.overall_risk = max(
                1.0 if labels.collision else 0.0,
                0.7 if labels.ttc_risk else 0.0,
                0.5 if labels.lane_violation else 0.0,
            )

            shield_decision = ShieldDecision(
                raw_action=raw_action,
                final_action=raw_action,
                intervened=False,
                reason="collector_no_shield",
                risk_raw=labels.overall_risk,
                risk_final=labels.overall_risk,
                candidate_risks={raw_action: labels.overall_risk},
            )
            steps.append(
                StepLog(
                    step_index=step_idx,
                    scene=result.scene,
                    raw_action=raw_action,
                    final_action=raw_action,
                    shield_decision=shield_decision,
                    task_reward=float(result.task_reward),
                    final_reward=float(result.task_reward),
                    done=bool(result.done),
                    risk_labels=labels,
                )
            )
            if result.done:
                break

        if len(steps) >= self.config.sim.future_steps:
            future_risk = aggregate_future_risk(
                (step.scene for step in steps[-self.config.sim.future_steps:]),
                ttc_threshold=self.config.dataset.ttc_threshold,
                lane_violation=steps[-1].risk_labels.lane_violation,
            )
            steps[-1].risk_labels = future_risk

        return EpisodeLog(episode_id=episode_id, risky_mode=risky_mode, steps=steps)

    def save_raw_logs(self, episodes: List[EpisodeLog]):
        output_dir = Path(self.config.dataset.raw_log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for index, episode in enumerate(episodes):
            file_path = output_dir / f"{episode.episode_id}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(dataclass_to_dict(episode), f, ensure_ascii=False)
            if (index + 1) % max(1, len(episodes) // 10) == 0 or (index + 1) == len(episodes):
                print(f"[Collector] save_raw_logs: {index + 1}/{len(episodes)}", flush=True)
        print(f"[Collector] raw logs saved in {time.time() - t0:.1f}s -> {output_dir}", flush=True)
