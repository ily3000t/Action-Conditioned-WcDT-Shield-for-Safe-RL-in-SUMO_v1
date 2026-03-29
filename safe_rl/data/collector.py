import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from safe_rl.config.config import SafeRLConfig
from safe_rl.data.stage1_probe import Stage1ProbeRunner
from safe_rl.data.types import EpisodeLog, ShieldDecision, StepLog, dataclass_to_dict
from safe_rl.data.warning_summary import aggregate_warning_records, summarize_episode_warnings
from safe_rl.sim.actions import all_action_ids
from safe_rl.sim.backend_interface import ISumoBackend
from safe_rl.sim.mock_core import RISK_EVENTS


class SumoDataCollector:
    def __init__(self, backend: ISumoBackend, config: SafeRLConfig, probe_backend: Optional[ISumoBackend] = None):
        self.backend = backend
        self.config = config
        self.probe_backend = probe_backend
        self._rng = random.Random(config.sim.random_seed)
        self.failure_records: List[Dict] = []
        self.warning_records: List[Dict] = []
        self.successful_episodes: int = 0
        self.failed_episodes: int = 0
        self.probe_runner = Stage1ProbeRunner(config=config, probe_backend=probe_backend)
        self.probe_pairs = self.probe_runner.pairs
        self.probe_events = self.probe_runner.events
        self.probe_summary = self.probe_runner.summary
        self.bucket_summary_payload = self.probe_runner.bucket_summary
        self._warning_episode_records: Dict[str, Dict[str, Any]] = {}

    def backend_mode(self) -> str:
        use_mock = bool(getattr(self.backend, '_use_mock', True))
        if use_mock:
            return 'mock'
        return 'sumo-real'

    def collect(self) -> List[EpisodeLog]:
        episodes: List[EpisodeLog] = []
        self.failure_records = []
        self.warning_records = []
        self.successful_episodes = 0
        self.failed_episodes = 0
        self.probe_runner.reset()
        self.probe_pairs = self.probe_runner.pairs
        self.probe_events = self.probe_runner.events
        self.probe_summary = self.probe_runner.summary
        self.bucket_summary_payload = self.probe_runner.bucket_summary
        self._warning_episode_records = {}

        total = self.config.sim.normal_episodes + self.config.sim.risky_episodes
        print(
            f"[Collector] start: backend={self.backend_mode()}, total_episodes={total}, "
            f"episode_steps={self.config.sim.episode_steps}",
            flush=True,
        )

        probe_backend_started = False
        if bool(getattr(self.config.stage1_collection, 'probe_enabled', False)) and self.probe_backend is not None:
            try:
                self.probe_backend.start()
                probe_backend_started = True
            except Exception as exc:
                self.probe_runner.summary['probe_backend_available'] = False
                self.probe_runner.summary['probe_backend_error'] = str(exc)
                self.probe_backend = None
                self.probe_runner.probe_backend = None

        t0 = time.time()
        log_interval = max(1, total // 20)
        try:
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
                        reason='episode_exception',
                    )
                    self._record_warning_snapshot(episode_id, risky_mode)
                    self._reset_backend_after_failure()
                    episode = None
                else:
                    self._record_warning_snapshot(episode_id, risky_mode)

                if episode is not None:
                    if episode.steps:
                        episodes.append(episode)
                        self.successful_episodes += 1
                    else:
                        self._record_failure(
                            episode_id=episode_id,
                            risky_mode=risky_mode,
                            exception_type='EmptyEpisodeError',
                            exception_text='Episode ended before collecting any steps.',
                            reason='empty_episode',
                        )

                if (i + 1) % log_interval == 0 or (i + 1) == total:
                    elapsed = time.time() - t0
                    eps_per_sec = (i + 1) / max(1e-6, elapsed)
                    remaining = (total - (i + 1)) / max(1e-6, eps_per_sec)
                    print(
                        f"[Collector] progress: {i + 1}/{total}, elapsed={elapsed:.1f}s, eta={remaining:.1f}s",
                        flush=True,
                    )
        finally:
            if probe_backend_started and self.probe_backend is not None:
                try:
                    self.probe_backend.close()
                except Exception:
                    pass

        self._warning_episode_records = self._build_warning_episode_records()
        self.bucket_summary_payload = self.probe_runner.assign_collection_buckets(episodes, self._warning_episode_records)
        self.failed_episodes = len(self.failure_records)
        print(
            f"[Collector] done: episodes={len(episodes)}, failed={self.failed_episodes}, total_time={time.time() - t0:.1f}s",
            flush=True,
        )
        return episodes

    def collect_episode(self, episode_id: str, risky_mode: bool) -> EpisodeLog:
        steps: List[StepLog] = []
        self.backend.set_episode_context(episode_id=episode_id, risky_mode=risky_mode)
        episode_seed = self._rng.randint(1, 1_000_000)
        self.backend.reset(seed=episode_seed)

        risk_event_schedule: List[Dict[str, Any]] = []
        raw_action_prefix: List[int] = []
        if risky_mode:
            requested_event = str(self._rng.choice(RISK_EVENTS))
            self.backend.inject_risk_event(requested_event)
            risk_event_schedule.append({'before_step': 0, 'event_type': requested_event})

        for step_idx in range(self.config.sim.episode_steps):
            raw_action = self._rng.choice(all_action_ids())
            result = self.backend.step(raw_action)
            raw_action_prefix.append(int(raw_action))

            next_event_type = ''
            if risky_mode and (step_idx + 1) < self.config.sim.episode_steps and self._rng.random() < self.config.sim.risk_event_prob:
                next_event_type = str(self._rng.choice(RISK_EVENTS))
                self.backend.inject_risk_event(next_event_type)
                risk_event_schedule.append({'before_step': step_idx + 1, 'event_type': next_event_type})

            labels = self.probe_runner.compute_risk_labels(result.scene, result.info)
            shield_decision = ShieldDecision(
                raw_action=raw_action,
                final_action=raw_action,
                intervened=False,
                reason='collector_no_shield',
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
                    meta={
                        'risk_event_next_type': next_event_type,
                        'probe_triggered': False,
                        'probe_bucket': '',
                        'same_state_proof': self.probe_runner.same_state_proof_from_scene(result.scene, history_scene=[]),
                        'teleport_flag': bool(result.info.get('teleport', False) or result.info.get('sim_teleport', False)),
                    },
                )
            )
            if result.done:
                break

        episode = EpisodeLog(
            episode_id=episode_id,
            risky_mode=risky_mode,
            steps=steps,
            meta={
                'episode_seed': int(episode_seed),
                'risk_event_schedule': list(risk_event_schedule),
                'raw_action_prefix': list(raw_action_prefix),
                'collection_bucket': '',
                'structural_warning_counts': {},
            },
        )
        self.probe_runner.probe_episode(episode)
        return episode

    def save_raw_logs(self, episodes: List[EpisodeLog]):
        output_dir = Path(self.config.dataset.raw_log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for index, episode in enumerate(episodes):
            file_path = output_dir / f"{episode.episode_id}.json"
            with file_path.open('w', encoding='utf-8') as f:
                json.dump(dataclass_to_dict(episode), f, ensure_ascii=False)
            if (index + 1) % max(1, len(episodes) // 10) == 0 or (index + 1) == len(episodes):
                print(f"[Collector] save_raw_logs: {index + 1}/{len(episodes)}", flush=True)
        print(f"[Collector] raw logs saved in {time.time() - t0:.1f}s -> {output_dir}", flush=True)

    def failure_report(self) -> Dict:
        total_attempts = self.successful_episodes + self.failed_episodes
        failure_rate = float(self.failed_episodes / total_attempts) if total_attempts > 0 else 0.0
        return {
            'successful_episodes': int(self.successful_episodes),
            'failed_episodes': int(self.failed_episodes),
            'failure_rate': failure_rate,
            'failures': list(self.failure_records),
        }

    def warning_summary(self) -> Dict:
        return aggregate_warning_records(list(self._build_warning_episode_records().values()))

    def bucket_summary(self) -> Dict[str, Any]:
        return dict(self.bucket_summary_payload)

    def save_failure_report(self, path: str):
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w', encoding='utf-8') as f:
            json.dump(self.failure_report(), f, ensure_ascii=False, indent=2)
        print(f"[Collector] failure report saved -> {report_path}", flush=True)

    def save_warning_summary(self, path: str):
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open('w', encoding='utf-8') as f:
            json.dump(self.warning_summary(), f, ensure_ascii=False, indent=2)
        print(f"[Collector] warning summary saved -> {report_path}", flush=True)

    def _build_warning_episode_records(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for record in self.warning_records:
            records[str(record['episode_id'])] = summarize_episode_warnings(
                episode_id=str(record['episode_id']),
                risky_mode=bool(record['risky_mode']),
                log_path=str(record['sumo_log_path']),
            )
        return records

    def _record_failure(
        self,
        episode_id: str,
        risky_mode: bool,
        exception_type: str,
        exception_text: str,
        reason: str,
    ):
        runtime_log_path = str(getattr(self.backend, 'runtime_log_path', '') or '')
        record = {
            'episode_id': episode_id,
            'risky_mode': bool(risky_mode),
            'exception_type': exception_type,
            'exception_text': exception_text,
            'reason': reason,
            'sumo_log_path': runtime_log_path,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        self.failure_records.append(record)
        print(
            f"[Collector] episode failed: id={episode_id}, reason={reason}, exception={exception_type}, sumo_log={runtime_log_path}",
            flush=True,
        )

    def _record_warning_snapshot(self, episode_id: str, risky_mode: bool):
        runtime_log_path = str(getattr(self.backend, 'runtime_log_path', '') or '')
        self.warning_records.append(
            {
                'episode_id': str(episode_id),
                'risky_mode': bool(risky_mode),
                'sumo_log_path': runtime_log_path,
            }
        )

    def _reset_backend_after_failure(self):
        try:
            self.backend.close()
        except Exception:
            pass
