from dataclasses import dataclass
from typing import Any, Dict, Optional

from safe_rl.data.types import InterventionRecord


@dataclass
class _EpisodeRiskStats:
    episode_index: int
    step_count: int = 0
    raw_sum: float = 0.0
    final_sum: float = 0.0
    reduction_sum: float = 0.0
    intervened_count: int = 0
    push_count: int = 0
    restart_count: int = 0
    had_restart: int = 0
    had_reset_error: int = 0
    had_fatal_step: int = 0
    finalized: bool = False


class Stage3TelemetryTracker:
    def __init__(self, writer=None):
        self.writer = writer
        self._event_index = 0
        self._episode_counter = 0
        self._step_index = 0
        self._counters: Dict[str, int] = {
            "reset_started_count": 0,
            "reset_failed_count": 0,
            "load_failed_count": 0,
            "restart_count": 0,
            "fatal_step_count": 0,
        }
        self._episodes: Dict[str, _EpisodeRiskStats] = {}

    def handle_session_event(self, event: Dict[str, Any]):
        name = str(event.get("event", ""))
        episode_id = str(event.get("episode_id", ""))
        episode = self._get_episode(episode_id)
        self._event_index += 1

        if name == "episode_reset_started":
            self._counters["reset_started_count"] += 1
            self._log_counter("stability/reset_started_count", self._counters["reset_started_count"])
        elif name == "episode_reset_failed":
            self._counters["reset_failed_count"] += 1
            episode.had_reset_error = 1
            self._log_counter("stability/reset_failed_count", self._counters["reset_failed_count"])
            self._log_episode_stability(episode)
            episode.finalized = True
        elif name == "reset_load_failed":
            self._counters["load_failed_count"] += 1
            self._log_counter("stability/load_failed_count", self._counters["load_failed_count"])
        elif name == "restart_real_session":
            self._counters["restart_count"] += 1
            episode.restart_count += 1
            episode.had_restart = 1
            self._log_counter("stability/restart_count", self._counters["restart_count"])
        elif name == "fatal_step":
            self._counters["fatal_step_count"] += 1
            episode.had_fatal_step = 1
            self._log_counter("stability/fatal_step_count", self._counters["fatal_step_count"])
        elif name == "episode_completed":
            self._log_episode_risk(episode)
            self._log_episode_stability(episode)
            episode.finalized = True

    def on_step(self, step_index: int, info: Dict[str, Any]):
        episode_id = str(info.get("episode_id", ""))
        episode = self._get_episode(episode_id)
        raw = float(info.get("risk_raw", 0.0))
        final = float(info.get("risk_final", 0.0))
        reduction = float(raw - final)
        intervened = float(bool(info.get("intervened", False)))

        self._step_index = max(self._step_index, int(step_index))
        self._add_scalar("risk/step_raw", raw, self._step_index)
        self._add_scalar("risk/step_final", final, self._step_index)
        self._add_scalar("risk/step_reduction", reduction, self._step_index)
        self._add_scalar("risk/step_intervened", intervened, self._step_index)

        episode.step_count += 1
        episode.raw_sum += raw
        episode.final_sum += final
        episode.reduction_sum += reduction
        episode.intervened_count += int(intervened > 0.0)

    def _get_episode(self, episode_id: str) -> _EpisodeRiskStats:
        key = episode_id or f"__episode_{self._episode_counter + 1:06d}"
        if key not in self._episodes:
            self._episode_counter += 1
            self._episodes[key] = _EpisodeRiskStats(episode_index=self._episode_counter)
        return self._episodes[key]

    def _log_counter(self, tag: str, value: int):
        self._add_scalar(tag, float(value), self._event_index)

    def _log_episode_stability(self, episode: _EpisodeRiskStats):
        step = int(episode.episode_index)
        self._add_scalar("stability/episode_restart_count", float(episode.restart_count), step)
        self._add_scalar("stability/episode_had_restart", float(episode.had_restart), step)
        self._add_scalar("stability/episode_had_reset_error", float(episode.had_reset_error), step)
        self._add_scalar("stability/episode_had_fatal_step", float(episode.had_fatal_step), step)

    def _log_episode_risk(self, episode: _EpisodeRiskStats):
        if episode.step_count <= 0:
            return
        step = int(episode.episode_index)
        denom = float(max(1, episode.step_count))
        self._add_scalar("risk/episode_mean_raw", episode.raw_sum / denom, step)
        self._add_scalar("risk/episode_mean_final", episode.final_sum / denom, step)
        self._add_scalar("risk/episode_mean_reduction", episode.reduction_sum / denom, step)
        self._add_scalar("risk/episode_intervention_rate", float(episode.intervened_count) / denom, step)

    def _add_scalar(self, tag: str, value: float, step: int):
        if self.writer is None:
            return
        self.writer.add_scalar(tag, float(value), int(step))


class BufferTelemetryTracker:
    def __init__(self, writer=None):
        self.writer = writer
        self._global_step = 0
        self._push_count = 0
        self._episode_counter = 0
        self._episodes: Dict[str, _EpisodeRiskStats] = {}

    def on_step(self, info: Dict[str, Any]):
        self._global_step += 1
        episode_id = str(info.get("episode_id", ""))
        episode = self._get_episode(episode_id)
        raw = float(info.get("risk_raw", 0.0))
        final = float(info.get("risk_final", 0.0))
        reduction = float(raw - final)
        intervened = float(bool(info.get("intervened", False)))

        self._add_scalar("buffer/step_intervened", intervened, self._global_step)
        self._add_scalar("buffer/step_raw_risk", raw, self._global_step)
        self._add_scalar("buffer/step_final_risk", final, self._global_step)
        self._add_scalar("buffer/step_risk_reduction", reduction, self._global_step)

        episode.step_count += 1
        episode.raw_sum += raw
        episode.final_sum += final
        episode.reduction_sum += reduction
        episode.intervened_count += int(intervened > 0.0)

    def on_push(self, record: InterventionRecord, buffer_stats: Dict[str, float]):
        self._push_count += 1
        episode_id = str(record.meta.get("episode_id", "")) if isinstance(record.meta, dict) else ""
        episode = self._get_episode(episode_id)
        episode.push_count += 1

        reduction = float(record.raw_risk - record.final_risk)
        self._add_scalar("buffer/push_count", float(self._push_count), self._push_count)
        self._add_scalar("buffer/push_raw_risk", float(record.raw_risk), self._push_count)
        self._add_scalar("buffer/push_final_risk", float(record.final_risk), self._push_count)
        self._add_scalar("buffer/push_risk_reduction", reduction, self._push_count)
        self._add_scalar("buffer/size", float(buffer_stats.get("size", 0.0)), self._push_count)
        self._add_scalar("buffer/running_mean_raw_risk", float(buffer_stats.get("mean_raw_risk", 0.0)), self._push_count)
        self._add_scalar("buffer/running_mean_final_risk", float(buffer_stats.get("mean_final_risk", 0.0)), self._push_count)
        self._add_scalar("buffer/running_mean_risk_reduction", float(buffer_stats.get("mean_risk_reduction", 0.0)), self._push_count)
        push_rate = float(self._push_count) / max(1.0, float(self._global_step))
        self._add_scalar("buffer/push_rate", push_rate, self._push_count)

    def on_episode_end(self, episode_id: str):
        episode = self._get_episode(str(episode_id or ""))
        step = int(episode.episode_index)
        self._add_scalar("buffer/episode_pushes", float(episode.push_count), step)
        denom = float(max(1, episode.step_count))
        self._add_scalar("buffer/episode_intervention_rate", float(episode.intervened_count) / denom, step)
        self._add_scalar("buffer/episode_mean_raw_risk", episode.raw_sum / denom, step)
        self._add_scalar("buffer/episode_mean_final_risk", episode.final_sum / denom, step)
        self._add_scalar("buffer/episode_mean_risk_reduction", episode.reduction_sum / denom, step)
        episode.finalized = True

    def _get_episode(self, episode_id: str) -> _EpisodeRiskStats:
        key = episode_id or f"__buffer_episode_{self._episode_counter + 1:06d}"
        if key not in self._episodes:
            self._episode_counter += 1
            self._episodes[key] = _EpisodeRiskStats(episode_index=self._episode_counter)
        return self._episodes[key]

    def _add_scalar(self, tag: str, value: float, step: int):
        if self.writer is None:
            return
        self.writer.add_scalar(tag, float(value), int(step))
