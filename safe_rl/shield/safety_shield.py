from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from safe_rl.config.config import ShieldConfig
from safe_rl.data.risk import compute_min_distance, compute_min_ttc, get_ego_vehicle
from safe_rl.data.types import RiskPrediction, SceneState, ShieldDecision, WorldPrediction
from safe_rl.shield.candidate_generator import CandidateActionGenerator
from safe_rl.shield.risk_aggregator import aggregate_tail_risk
from safe_rl.sim.actions import action_distance, action_name, decode_action, fallback_action_id


MERGE_COORDINATION_X_MIN = 930.0
MERGE_COORDINATION_X_MAX = 1035.0
MERGE_COORDINATION_NEARBY_DISTANCE = 40.0


class LightRiskPredictorProtocol(Protocol):
    def predict(self, history_scene: List[SceneState], action_id: int) -> RiskPrediction:
        raise NotImplementedError


class WorldModelPredictorProtocol(Protocol):
    def predict(self, history_scene: List[SceneState], action_id: int) -> WorldPrediction:
        raise NotImplementedError


@dataclass
class CandidateEvaluation:
    action_id: int
    action_type: str
    distance_to_raw: int
    coarse_risk: float
    fine_risk: Optional[float] = None
    uncertainty: Optional[float] = None
    world_prediction: Optional[WorldPrediction] = None
    evaluated: bool = False
    safe_under_threshold: bool = False
    selected: bool = False
    constraint_reason: str = ""


class SafetyShield:
    def __init__(
        self,
        config: ShieldConfig,
        light_predictor: Optional[LightRiskPredictorProtocol],
        world_predictor: Optional[WorldModelPredictorProtocol],
    ):
        self.config = config
        self.light_predictor = light_predictor
        self.world_predictor = world_predictor
        self.generator = CandidateActionGenerator(candidate_count=config.candidate_count)

    def select_action(self, history_scene: List[SceneState], policy_action: int) -> ShieldDecision:
        candidates = self.generator.generate(policy_action)
        coarse_risk = {c: self._coarse_risk(history_scene, c) for c in candidates}
        evaluations: Dict[int, CandidateEvaluation] = {
            action_id: CandidateEvaluation(
                action_id=action_id,
                action_type=action_name(action_id),
                distance_to_raw=action_distance(policy_action, action_id),
                coarse_risk=float(coarse_risk[action_id]),
            )
            for action_id in candidates
        }

        coarse_sorted = sorted(candidates, key=lambda c: coarse_risk[c])
        refined = coarse_sorted[: max(1, self.config.coarse_top_k)]
        if policy_action not in refined:
            refined.append(policy_action)

        for action_id in refined:
            self._evaluate_candidate(history_scene, evaluations, action_id)

        raw_eval = evaluations[policy_action]
        raw_threshold = min(float(self.config.risk_threshold), float(self.config.raw_passthrough_risk_threshold))
        shield_blocked = bool(
            (raw_eval.fine_risk is not None and raw_eval.fine_risk > self.config.risk_threshold)
            or (raw_eval.uncertainty is not None and raw_eval.uncertainty > self.config.uncertainty_threshold)
        )
        merge_phase_active = self._is_merge_coordination_phase(history_scene)

        if not shield_blocked:
            raw_eval.selected = True
            constraint_reason = "raw_passthrough" if self._is_raw_passthrough(raw_eval, raw_threshold) else ""
            return self._build_decision(
                candidates=candidates,
                evaluations=evaluations,
                raw_eval=raw_eval,
                selected_eval=raw_eval,
                policy_action=policy_action,
                intervened=False,
                reason="raw_action_safe",
                fallback_used=False,
                replacement_margin=0.0,
                constraint_reason=constraint_reason,
                merge_phase_active=merge_phase_active,
            )

        safe_candidates = [
            evaluations[action_id]
            for action_id in refined
            if evaluations[action_id].safe_under_threshold
        ]
        safe_candidates.sort(key=lambda ev: (ev.distance_to_raw, float(ev.fine_risk or 0.0), float(ev.uncertainty or 0.0)))

        blocked_reasons: List[str] = []
        for candidate in safe_candidates:
            if candidate.action_id == policy_action:
                continue
            margin = float(raw_eval.fine_risk or 0.0) - float(candidate.fine_risk or 0.0)
            if margin < float(self.config.replacement_min_risk_margin):
                candidate.constraint_reason = "blocked_by_margin"
                blocked_reasons.append(candidate.constraint_reason)
                continue
            if self._should_block_merge_lateral(history_scene, policy_action, candidate.action_id, margin):
                candidate.constraint_reason = "merge_lateral_guard"
                blocked_reasons.append(candidate.constraint_reason)
                continue

            candidate.selected = True
            return self._build_decision(
                candidates=candidates,
                evaluations=evaluations,
                raw_eval=raw_eval,
                selected_eval=candidate,
                policy_action=policy_action,
                intervened=True,
                reason="risk_threshold_exceeded",
                fallback_used=False,
                replacement_margin=margin,
                constraint_reason="",
                merge_phase_active=merge_phase_active,
            )

        if safe_candidates:
            raw_eval.selected = True
            overall_reason = blocked_reasons[0] if blocked_reasons else "blocked_by_margin"
            return self._build_decision(
                candidates=candidates,
                evaluations=evaluations,
                raw_eval=raw_eval,
                selected_eval=raw_eval,
                policy_action=policy_action,
                intervened=False,
                reason="replacement_blocked_by_constraint",
                fallback_used=False,
                replacement_margin=0.0,
                constraint_reason=overall_reason,
                merge_phase_active=merge_phase_active,
            )

        fallback = fallback_action_id()
        if fallback not in evaluations:
            evaluations[fallback] = CandidateEvaluation(
                action_id=fallback,
                action_type=action_name(fallback),
                distance_to_raw=action_distance(policy_action, fallback),
                coarse_risk=self._coarse_risk(history_scene, fallback),
            )
            candidates.append(fallback)
        fallback_eval = self._evaluate_candidate(history_scene, evaluations, fallback)
        fallback_eval.selected = True
        return self._build_decision(
            candidates=candidates,
            evaluations=evaluations,
            raw_eval=raw_eval,
            selected_eval=fallback_eval,
            policy_action=policy_action,
            intervened=True,
            reason="all_candidates_high_risk_or_uncertain",
            fallback_used=True,
            replacement_margin=max(0.0, float(raw_eval.fine_risk or 0.0) - float(fallback_eval.fine_risk or 0.0)),
            constraint_reason="fallback_due_to_no_safe_candidate",
            merge_phase_active=merge_phase_active,
        )

    def _evaluate_candidate(
        self,
        history_scene: List[SceneState],
        evaluations: Dict[int, CandidateEvaluation],
        action_id: int,
    ) -> CandidateEvaluation:
        evaluation = evaluations[action_id]
        if evaluation.evaluated:
            return evaluation
        fine_risk, uncertainty, world_prediction = self._fine_risk(history_scene, action_id)
        evaluation.fine_risk = float(fine_risk)
        evaluation.uncertainty = float(uncertainty)
        evaluation.world_prediction = world_prediction
        evaluation.evaluated = True
        evaluation.safe_under_threshold = bool(
            evaluation.fine_risk <= self.config.risk_threshold
            and evaluation.uncertainty <= self.config.uncertainty_threshold
        )
        return evaluation

    def _build_decision(
        self,
        candidates: List[int],
        evaluations: Dict[int, CandidateEvaluation],
        raw_eval: CandidateEvaluation,
        selected_eval: CandidateEvaluation,
        policy_action: int,
        intervened: bool,
        reason: str,
        fallback_used: bool,
        replacement_margin: float,
        constraint_reason: str,
        merge_phase_active: bool,
    ) -> ShieldDecision:
        selected_eval.selected = True
        final_action = selected_eval.action_id
        final_action_type = action_name(final_action)
        raw_action_type = action_name(policy_action)
        replacement_happened = final_action != policy_action
        lane_change_involved = bool(decode_action(policy_action).lateral != 0 or decode_action(final_action).lateral != 0)
        meta = {
            "candidate_count": int(len(candidates)),
            "evaluated_candidate_count": int(sum(1 for ev in evaluations.values() if ev.evaluated)),
            "shield_blocked": bool(reason != "raw_action_safe"),
            "raw_world_prediction": raw_eval.world_prediction,
            "fallback_used": bool(fallback_used),
            "replacement_happened": bool(replacement_happened),
            "final_world_prediction": selected_eval.world_prediction,
            "chosen_candidate_index": int(candidates.index(final_action)) if final_action in candidates else -1,
            "chosen_candidate_rank_by_risk": int(self._candidate_risk_rank(final_action, evaluations)),
            "raw_action_type": raw_action_type,
            "final_action_type": final_action_type,
            "lane_change_involved": lane_change_involved,
            "constraint_reason": str(constraint_reason),
            "replacement_margin": float(replacement_margin),
            "candidate_evaluations": self._serialize_candidate_evaluations(candidates, evaluations),
            "merge_phase_active": bool(merge_phase_active),
        }
        return ShieldDecision(
            raw_action=policy_action,
            final_action=final_action,
            intervened=bool(intervened),
            reason=reason,
            risk_raw=float(raw_eval.fine_risk or 0.0),
            risk_final=float(selected_eval.fine_risk or 0.0),
            candidate_risks={k: float(v.fine_risk) for k, v in evaluations.items() if v.evaluated and v.fine_risk is not None},
            meta=meta,
        )

    def _serialize_candidate_evaluations(
        self,
        candidates: List[int],
        evaluations: Dict[int, CandidateEvaluation],
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for action_id in candidates:
            ev = evaluations[action_id]
            payload.append(
                {
                    "action_id": int(ev.action_id),
                    "action_type": str(ev.action_type),
                    "distance_to_raw": int(ev.distance_to_raw),
                    "coarse_risk": float(ev.coarse_risk),
                    "fine_risk": None if ev.fine_risk is None else float(ev.fine_risk),
                    "uncertainty": None if ev.uncertainty is None else float(ev.uncertainty),
                    "selected": bool(ev.selected),
                    "safe_under_threshold": bool(ev.safe_under_threshold),
                    "evaluated": bool(ev.evaluated),
                    "constraint_reason": str(ev.constraint_reason),
                }
            )
        return payload

    def _candidate_risk_rank(self, action_id: int, evaluations: Dict[int, CandidateEvaluation]) -> int:
        ranked = [ev for ev in evaluations.values() if ev.evaluated and ev.fine_risk is not None]
        ranked.sort(key=lambda ev: (float(ev.fine_risk), float(ev.uncertainty or 0.0), ev.distance_to_raw, ev.action_id))
        for index, evaluation in enumerate(ranked):
            if evaluation.action_id == action_id:
                return index
        return -1

    def _is_raw_passthrough(self, raw_eval: CandidateEvaluation, raw_threshold: float) -> bool:
        if raw_eval.fine_risk is None or raw_eval.uncertainty is None:
            return False
        return raw_eval.fine_risk <= raw_threshold and raw_eval.uncertainty <= self.config.uncertainty_threshold

    def _should_block_merge_lateral(
        self,
        history_scene: List[SceneState],
        raw_action: int,
        candidate_action: int,
        margin: float,
    ) -> bool:
        if not self.config.protect_merge_lateral_decisions:
            return False
        if not self._is_merge_coordination_phase(history_scene):
            return False
        raw_lateral = decode_action(raw_action).lateral
        candidate_lateral = decode_action(candidate_action).lateral
        if raw_lateral == candidate_lateral:
            return False
        return margin < float(self.config.merge_override_margin)

    def _is_merge_coordination_phase(self, history_scene: List[SceneState]) -> bool:
        if not history_scene:
            return False
        current = history_scene[-1]
        try:
            ego = get_ego_vehicle(current)
        except Exception:
            return False
        if ego is None:
            return False
        if not (MERGE_COORDINATION_X_MIN <= float(ego.x) <= MERGE_COORDINATION_X_MAX):
            return False
        for vehicle in current.vehicles:
            if vehicle.vehicle_id == current.ego_id:
                continue
            if abs(float(vehicle.x) - float(ego.x)) <= MERGE_COORDINATION_NEARBY_DISTANCE:
                return True
        return False

    def _coarse_risk(self, history_scene: List[SceneState], action_id: int) -> float:
        if self.light_predictor is None:
            return self._heuristic_risk(history_scene)
        prediction = self.light_predictor.predict(history_scene, action_id)
        return float(prediction.p_overall)

    def _fine_risk(self, history_scene: List[SceneState], action_id: int) -> Tuple[float, float, Optional[WorldPrediction]]:
        if self.world_predictor is None:
            risk = self._heuristic_risk(history_scene)
            return risk, 0.0, None

        prediction = self.world_predictor.predict(history_scene, action_id)
        modality_risks = [item.p_overall for item in prediction.modality_risk]
        tail_risk = aggregate_tail_risk(
            modality_risks,
            quantile=self.config.tail_quantile,
            uncertainty=prediction.uncertainty,
            uncertainty_weight=self.config.uncertainty_weight,
        )
        return float(tail_risk), float(prediction.uncertainty), prediction

    @staticmethod
    def _heuristic_risk(history_scene: List[SceneState]) -> float:
        current = history_scene[-1]
        min_distance = compute_min_distance(current)
        min_ttc = compute_min_ttc(current)
        distance_term = 1.0 if min_distance < 3.0 else max(0.0, 1.0 - min_distance / 30.0)
        ttc_term = 1.0 if min_ttc < 1.5 else max(0.0, 1.0 - min_ttc / 8.0)
        return max(distance_term, ttc_term)
