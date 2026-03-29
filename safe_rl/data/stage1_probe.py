import hashlib
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from safe_rl.config.config import SafeRLConfig
from safe_rl.data.risk import compute_min_distance, compute_min_ttc, detect_collision, get_ego_vehicle
from safe_rl.data.types import EpisodeLog, RiskLabels, RiskPairSample, SceneState
from safe_rl.data.warning_summary import STRUCTURAL_BUCKETS
from safe_rl.sim.actions import all_action_ids, encode_action
from safe_rl.sim.backend_interface import ISumoBackend


KEEP_KEEP_ACTION = encode_action(0, 0)
STRUCTURAL_SKIP_REASONS = {
    'invalid_lane_index',
    'no_route_connection',
    'no_connection_next_edge',
    'emergency_stop_no_connection',
}


class Stage1ProbeRunner:
    def __init__(self, config: SafeRLConfig, probe_backend: Optional[ISumoBackend] = None):
        self.config = config
        self.probe_backend = probe_backend
        self.pairs: List[RiskPairSample] = []
        self.events: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        self.bucket_summary: Dict[str, Any] = {}
        self.reset()

    def reset(self):
        self.pairs = []
        self.events = []
        self.summary = {
            'enabled': bool(getattr(self.config.stage1_collection, 'probe_enabled', False)),
            'probe_backend_available': bool(self.probe_backend is not None),
            'episodes_seen': 0,
            'episodes_probed': 0,
            'probe_steps_considered': 0,
            'probe_steps_selected': 0,
            'candidate_rollouts': 0,
            'pairs_created': 0,
            'skipped_missing_probe_backend': 0,
            'skipped_short_episode': 0,
            'skipped_replay_failed': 0,
        }
        self.bucket_summary = {}

    def compute_risk_labels(self, scene: SceneState, info: Dict[str, Any]) -> RiskLabels:
        collision = bool(info.get('collision', detect_collision(scene)))
        min_ttc = compute_min_ttc(scene)
        min_distance = compute_min_distance(scene)
        lane_violation = bool(info.get('lane_violation', False))
        distance_term = 1.0 if min_distance < 3.0 else max(0.0, 1.0 - float(min_distance) / 30.0)
        ttc_term = 1.0 if min_ttc < 1.5 else max(0.0, 1.0 - float(min_ttc) / 8.0)
        overall_risk = max(
            1.0 if collision else 0.0,
            0.7 if min_ttc < self.config.dataset.ttc_threshold else 0.0,
            0.5 if lane_violation else 0.0,
            distance_term,
            ttc_term,
        )
        return RiskLabels(
            collision=collision,
            ttc_risk=min_ttc < self.config.dataset.ttc_threshold,
            lane_violation=lane_violation,
            overall_risk=float(overall_risk),
            min_ttc=float(min_ttc),
            min_distance=float(min_distance),
        )

    def assign_collection_buckets(self, episodes: Sequence[EpisodeLog], warning_by_episode: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        episode_bucket_counts = Counter()
        step_bucket_counts = Counter()
        structural_excluded = 0
        for episode in episodes:
            warning_record = dict(warning_by_episode.get(episode.episode_id, {}))
            structural_counts = {
                bucket: int(dict(warning_record.get('buckets', {})).get(bucket, 0))
                for bucket in STRUCTURAL_BUCKETS
            }
            has_structural_warning = any(int(value) > 0 for value in structural_counts.values())
            has_teleport = any(bool(dict(step.meta or {}).get('teleport_flag', False)) for step in episode.steps)
            has_fatal_failure = bool(dict(episode.meta or {}).get('fatal_reset_failure', False))
            if has_structural_warning or has_teleport or has_fatal_failure:
                bucket = 'structural_failure'
            elif self._episode_is_clean_risky(episode):
                bucket = 'clean_risky'
            else:
                bucket = 'clean_normal'
            episode.meta['collection_bucket'] = bucket
            episode.meta['structural_warning_counts'] = dict(structural_counts)
            if bucket == 'structural_failure' and bool(getattr(self.config.stage1_collection, 'exclude_structural_from_main', True)):
                structural_excluded += 1
            episode_bucket_counts[bucket] += 1
            step_bucket_counts[bucket] += len(episode.steps)
            for step in episode.steps:
                step.meta['probe_bucket'] = bucket
        self.bucket_summary = {
            'episodes_by_bucket': dict(sorted(episode_bucket_counts.items())),
            'steps_by_bucket': dict(sorted(step_bucket_counts.items())),
            'structural_excluded_episodes': int(structural_excluded),
            'exclude_structural_from_main': bool(getattr(self.config.stage1_collection, 'exclude_structural_from_main', True)),
        }
        return dict(self.bucket_summary)

    def probe_episode(self, episode: EpisodeLog):
        self.summary['episodes_seen'] += 1
        if not bool(getattr(self.config.stage1_collection, 'probe_enabled', False)):
            return
        if self.probe_backend is None:
            self.summary['skipped_missing_probe_backend'] += 1
            return
        if len(episode.steps) <= int(self.config.sim.history_steps):
            self.summary['skipped_short_episode'] += 1
            return

        selected_steps = self._select_probe_steps(episode)
        self.summary['probe_steps_considered'] += max(0, len(episode.steps) - int(self.config.sim.history_steps))
        self.summary['probe_steps_selected'] += len(selected_steps)
        if selected_steps:
            self.summary['episodes_probed'] += 1
        for step_index in selected_steps:
            episode.steps[step_index].meta['probe_triggered'] = True
            event = self._probe_step_candidates(episode, step_index)
            self.events.append(event)

    def _episode_is_clean_risky(self, episode: EpisodeLog) -> bool:
        if bool(episode.risky_mode):
            return True
        for step in episode.steps:
            labels = step.risk_labels
            if bool(labels.collision):
                return True
            if float(labels.min_ttc) <= float(self.config.stage1_collection.probe_trigger_ttc_threshold):
                return True
            if float(labels.min_distance) <= float(self.config.stage1_collection.probe_trigger_min_distance):
                return True
        return False

    def _select_probe_steps(self, episode: EpisodeLog) -> List[int]:
        candidates: List[Tuple[Tuple[float, float, float, int], int]] = []
        for step in episode.steps:
            if int(step.step_index) < int(self.config.sim.history_steps):
                continue
            if bool(step.done):
                continue
            labels = step.risk_labels
            should_probe = (
                bool(labels.collision)
                or float(labels.min_ttc) <= float(self.config.stage1_collection.probe_trigger_ttc_threshold)
                or float(labels.min_distance) <= float(self.config.stage1_collection.probe_trigger_min_distance)
                or bool(episode.risky_mode)
            )
            if not should_probe:
                continue
            priority = (
                -float(labels.overall_risk),
                float(labels.min_ttc),
                float(labels.min_distance),
                int(step.step_index),
            )
            candidates.append((priority, int(step.step_index)))
        candidates.sort(key=lambda item: item[0])
        max_steps = max(0, int(getattr(self.config.stage1_collection, 'probe_max_steps_per_episode', 0) or 0))
        return [step_index for _, step_index in candidates[:max_steps]]

    def _probe_step_candidates(self, episode: EpisodeLog, step_index: int) -> Dict[str, Any]:
        history_steps = int(self.config.sim.history_steps)
        history_scene = [episode.steps[idx].scene for idx in range(max(0, step_index - history_steps), step_index)]
        replay_state = self._replay_state_to_step(episode, step_index)
        if replay_state is None:
            self.summary['skipped_replay_failed'] += 1
            return {
                'episode_id': episode.episode_id,
                'step_index': int(step_index),
                'status': 'replay_failed',
                'pairs_created': 0,
            }

        same_state_proof = self.same_state_proof_from_scene(replay_state, history_scene=history_scene)
        episode.steps[step_index].meta['same_state_proof'] = dict(same_state_proof)
        candidate_records: List[Dict[str, Any]] = []
        for action_id in self._probe_action_ids():
            candidate_records.append(self._rollout_probe_candidate(episode, step_index, action_id, same_state_proof))
        self.summary['candidate_rollouts'] += len(candidate_records)
        pair_samples = self._probe_pairs_from_candidates(episode, step_index, history_scene, same_state_proof, candidate_records)
        self.pairs.extend(pair_samples)
        self.summary['pairs_created'] += len(pair_samples)
        return {
            'episode_id': episode.episode_id,
            'step_index': int(step_index),
            'status': 'ok',
            'pairs_created': int(len(pair_samples)),
            'candidate_count': int(len(candidate_records)),
            'same_state_proof': dict(same_state_proof),
            'candidates': candidate_records,
        }

    def _probe_action_ids(self) -> List[int]:
        if str(getattr(self.config.stage1_collection, 'probe_action_set', 'all_9')).strip().lower() == 'all_9':
            return all_action_ids()
        return all_action_ids()

    def _replay_state_to_step(self, episode: EpisodeLog, step_index: int) -> Optional[SceneState]:
        try:
            self.probe_backend.set_episode_context(f"{episode.episode_id}_probe", episode.risky_mode)
            self.probe_backend.reset(seed=int(episode.meta.get('episode_seed', self.config.sim.random_seed)))
            raw_action_prefix = list(episode.meta.get('raw_action_prefix', []))
            for prefix_step in range(int(step_index)):
                self._apply_scheduled_events_before_step(episode, prefix_step)
                prefix_action = int(raw_action_prefix[prefix_step]) if prefix_step < len(raw_action_prefix) else KEEP_KEEP_ACTION
                self.probe_backend.step(prefix_action)
            self._apply_scheduled_events_before_step(episode, int(step_index))
            return self.probe_backend.get_state()
        except Exception:
            return None

    def _apply_scheduled_events_before_step(self, episode: EpisodeLog, step_index: int):
        for item in list(episode.meta.get('risk_event_schedule', [])):
            if int(item.get('before_step', -1)) == int(step_index):
                self.probe_backend.inject_risk_event(str(item.get('event_type', '')) or None)

    def _rollout_probe_candidate(
        self,
        episode: EpisodeLog,
        step_index: int,
        candidate_action: int,
        same_state_proof: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_action_prefix = list(episode.meta.get('raw_action_prefix', []))
        future_scenes: List[SceneState] = []
        infos: List[Dict[str, Any]] = []
        replay_ok = True
        try:
            self.probe_backend.set_episode_context(f"{episode.episode_id}_probe", episode.risky_mode)
            self.probe_backend.reset(seed=int(episode.meta.get('episode_seed', self.config.sim.random_seed)))
            for prefix_step in range(int(step_index)):
                self._apply_scheduled_events_before_step(episode, prefix_step)
                prefix_action = int(raw_action_prefix[prefix_step]) if prefix_step < len(raw_action_prefix) else KEEP_KEEP_ACTION
                self.probe_backend.step(prefix_action)
            self._apply_scheduled_events_before_step(episode, int(step_index))
            result = self.probe_backend.step(int(candidate_action))
            future_scenes.append(result.scene)
            infos.append(dict(result.info or {}))
            for offset in range(1, max(1, int(self.config.stage1_collection.probe_horizon_steps))):
                next_step_index = int(step_index) + offset
                if next_step_index >= int(self.config.sim.episode_steps):
                    break
                self._apply_scheduled_events_before_step(episode, next_step_index)
                follow_action = int(raw_action_prefix[next_step_index]) if next_step_index < len(raw_action_prefix) else KEEP_KEEP_ACTION
                result = self.probe_backend.step(follow_action)
                future_scenes.append(result.scene)
                infos.append(dict(result.info or {}))
                if bool(result.done):
                    break
        except Exception:
            replay_ok = False

        collision = any(bool(info.get('collision', False)) for info in infos) or any(detect_collision(scene) for scene in future_scenes)
        min_ttc = min((compute_min_ttc(scene) for scene in future_scenes), default=1e6)
        min_distance = min((compute_min_distance(scene) for scene in future_scenes), default=1e6)
        lane_conflict = any(bool(info.get('lane_violation', False)) or str(info.get('lane_change_skipped_reason', '')) for info in infos)
        route_structural_flag = any(
            str(info.get('lane_change_skipped_reason', '') or info.get('risk_skipped_reason', '')).strip().lower() in STRUCTURAL_SKIP_REASONS
            for info in infos
        )
        teleport_flag = any(bool(info.get('teleport', False) or info.get('sim_teleport', False)) for info in infos)
        distance_term = 1.0 if min_distance < 3.0 else max(0.0, 1.0 - float(min_distance) / 30.0)
        ttc_term = 1.0 if min_ttc < 1.5 else max(0.0, 1.0 - float(min_ttc) / 8.0)
        overall_proxy_risk = max(
            1.0 if collision else 0.0,
            0.7 if min_ttc < self.config.dataset.ttc_threshold else 0.0,
            0.5 if lane_conflict else 0.0,
            1.0 if route_structural_flag or teleport_flag else 0.0,
            distance_term,
            ttc_term,
        )
        return {
            'candidate_action': int(candidate_action),
            'collision': bool(collision),
            'min_ttc': float(min_ttc),
            'min_distance': float(min_distance),
            'lane_conflict': bool(lane_conflict),
            'route_structural_flag': bool(route_structural_flag),
            'teleport_flag': bool(teleport_flag),
            'overall_proxy_risk': float(overall_proxy_risk),
            'replay_ok': bool(replay_ok),
            'history_hash': str(same_state_proof.get('history_hash', '')),
        }

    def _probe_pairs_from_candidates(
        self,
        episode: EpisodeLog,
        step_index: int,
        history_scene: Sequence[SceneState],
        same_state_proof: Dict[str, Any],
        candidate_records: Sequence[Dict[str, Any]],
    ) -> List[RiskPairSample]:
        pairs: List[RiskPairSample] = []
        ordered = sorted(list(candidate_records), key=lambda item: int(item.get('candidate_action', -1)))
        for idx in range(len(ordered)):
            for jdx in range(idx + 1, len(ordered)):
                left = ordered[idx]
                right = ordered[jdx]
                preferred = self._preferred_probe_action(left, right)
                if preferred is None:
                    continue
                target_gap = abs(float(left.get('overall_proxy_risk', 0.0)) - float(right.get('overall_proxy_risk', 0.0)))
                trusted_for_spread = self._probe_pair_trusted_for_spread(left, right)
                pairs.append(
                    RiskPairSample(
                        history_scene=list(history_scene),
                        action_a=int(left['candidate_action']),
                        action_b=int(right['candidate_action']),
                        preferred_action=int(preferred),
                        source='stage1_probe_same_state',
                        weight=0.7,
                        meta={
                            'episode_id': episode.episode_id,
                            'step_index': int(step_index),
                            'target_risk_a': float(left.get('overall_proxy_risk', 0.0)),
                            'target_risk_b': float(right.get('overall_proxy_risk', 0.0)),
                            'hard_negative': bool(left.get('collision', False) != right.get('collision', False)),
                            'trusted_for_spread': bool(trusted_for_spread),
                            'target_gap': float(target_gap),
                            'clear_dominance': bool(trusted_for_spread),
                            **same_state_proof,
                        },
                    )
                )
        return pairs

    def _preferred_probe_action(self, left: Dict[str, Any], right: Dict[str, Any]) -> Optional[int]:
        left_key = self._probe_candidate_sort_key(left)
        right_key = self._probe_candidate_sort_key(right)
        if left_key == right_key:
            return None
        if left_key < right_key:
            return int(left['candidate_action'])
        return int(right['candidate_action'])

    def _probe_candidate_sort_key(self, candidate: Dict[str, Any]) -> Tuple[float, float, float, float, int]:
        return (
            1.0 if bool(candidate.get('collision', False)) else 0.0,
            1.0 if bool(candidate.get('route_structural_flag', False) or candidate.get('teleport_flag', False)) else 0.0,
            float(candidate.get('overall_proxy_risk', 0.0)),
            -float(candidate.get('min_distance', 0.0)),
            int(candidate.get('candidate_action', -1)),
        )

    def _probe_pair_trusted_for_spread(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if bool(left.get('collision', False)) != bool(right.get('collision', False)):
            return True
        if abs(float(left.get('min_ttc', 0.0)) - float(right.get('min_ttc', 0.0))) >= 0.75:
            return True
        if abs(float(left.get('min_distance', 0.0)) - float(right.get('min_distance', 0.0))) >= 2.0:
            return True
        if abs(float(left.get('overall_proxy_risk', 0.0)) - float(right.get('overall_proxy_risk', 0.0))) >= 0.15:
            return True
        return False

    def same_state_proof_from_scene(self, scene: SceneState, history_scene: Sequence[SceneState]) -> Dict[str, Any]:
        ego = get_ego_vehicle(scene)
        neighbors = []
        for vehicle in scene.vehicles:
            if vehicle.vehicle_id == ego.vehicle_id:
                continue
            neighbors.append({
                'vehicle_id': str(vehicle.vehicle_id),
                'dx': round(float(vehicle.x - ego.x), 3),
                'dy': round(float(vehicle.y - ego.y), 3),
                'dvx': round(float(vehicle.vx - ego.vx), 3),
            })
        neighbors.sort(key=lambda item: (abs(float(item['dx'])), str(item['vehicle_id'])))
        return {
            'history_hash': self._history_hash(history_scene if history_scene else [scene]),
            'ego_lane_id': str(ego.lane_id),
            'ego_x': float(ego.x),
            'ego_y': float(ego.y),
            'ego_speed': float(ego.vx),
            'neighbor_summary': neighbors[:4],
        }

    def _history_hash(self, history_scene: Sequence[SceneState]) -> str:
        payload = []
        for scene in history_scene:
            ego = get_ego_vehicle(scene)
            payload.append({
                't': round(float(scene.timestamp), 3),
                'ego_x': round(float(ego.x), 3),
                'ego_y': round(float(ego.y), 3),
                'ego_vx': round(float(ego.vx), 3),
                'lane': str(ego.lane_id),
                'n': [
                    (
                        str(vehicle.vehicle_id),
                        round(float(vehicle.x - ego.x), 3),
                        round(float(vehicle.y - ego.y), 3),
                        round(float(vehicle.vx - ego.vx), 3),
                    )
                    for vehicle in sorted(scene.vehicles, key=lambda item: str(item.vehicle_id))
                    if vehicle.vehicle_id != ego.vehicle_id
                ],
            })
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        return digest[:16]
