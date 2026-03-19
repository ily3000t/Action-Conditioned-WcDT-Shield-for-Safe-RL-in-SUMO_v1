import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        self.failure_records: List[Dict] = []
        self.successful_episodes: int = 0
        self.failed_episodes: int = 0

    def backend_mode(self) -> str:
        use_mock = bool(getattr(self.backend, "_use_mock", True))
        if use_mock:
            return "mock"
        return "sumo-real"

    def collect(self) -> List[EpisodeLog]:
        episodes: List[EpisodeLog] = []
        self.failure_records = []
        self.successful_episodes = 0
        self.failed_episodes = 0

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

            try:
                episode = self.collect_episode(episode_id, risky_mode=risky_mode)
            except Exception as exc:
                self._record_failure(
                    episode_id=episode_id,
                    risky_mode=risky_mode,
                    exception_type=type(exc).__name__,
                    exception_text=str(exc),
                    reason="episode_exception",
                )
                self._reset_backend_after_failure()
                episode = None

            if episode is not None:
                if episode.steps:
                    episodes.append(episode)
                    self.successful_episodes += 1
                else:
                    self._record_failure(
                        episode_id=episode_id,
                        risky_mode=risky_mode,
                        exception_type="EmptyEpisodeError",
                        exception_text="Episode ended before collecting any steps.",
                        reason="empty_episode",
                    )

            if (i + 1) % log_interval == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                eps_per_sec = (i + 1) / max(1e-6, elapsed)
                remaining = (total - (i + 1)) / max(1e-6, eps_per_sec)
                print(
                    f"[Collector] progress: {i + 1}/{total}, elapsed={elapsed:.1f}s, "
                    f"eta={remaining:.1f}s",
                    flush=True,
                )

        self.failed_episodes = len(self.failure_records)
        print(
            f"[Collector] done: episodes={len(episodes)}, failed={self.failed_episodes}, "
            f"total_time={time.time() - t0:.1f}s",
            flush=True,
        )
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

    def failure_report(self) -> Dict:
        total_attempts = self.successful_episodes + self.failed_episodes
        failure_rate = float(self.failed_episodes / total_attempts) if total_attempts > 0 else 0.0
        return {
            "successful_episodes": int(self.successful_episodes),
            "failed_episodes": int(self.failed_episodes),
            "failure_rate": failure_rate,
            "failures": list(self.failure_records),
        }

    def save_failure_report(self, path: str):
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(self.failure_report(), f, ensure_ascii=False, indent=2)
        print(f"[Collector] failure report saved -> {report_path}", flush=True)

    def _record_failure(
        self,
        episode_id: str,
        risky_mode: bool,
        exception_type: str,
        exception_text: str,
        reason: str,
    ):
        runtime_log_path = str(getattr(self.backend, "runtime_log_path", "") or "")
        record = {
            "episode_id": episode_id,
            "risky_mode": bool(risky_mode),
            "exception_type": exception_type,
            "exception_text": exception_text,
            "reason": reason,
            "sumo_log_path": runtime_log_path,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.failure_records.append(record)
        print(
            f"[Collector] episode failed: id={episode_id}, reason={reason}, "
            f"exception={exception_type}, sumo_log={runtime_log_path}",
            flush=True,
        )

    def _reset_backend_after_failure(self):
        try:
            self.backend.close()
        except Exception:
            pass
