from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

from safe_rl.config.config import ShieldConfig
from safe_rl.data.risk import compute_min_distance, compute_min_ttc
from safe_rl.data.types import RiskPrediction, SceneState, ShieldDecision, WorldPrediction
from safe_rl.shield.candidate_generator import CandidateActionGenerator
from safe_rl.shield.risk_aggregator import aggregate_tail_risk
from safe_rl.sim.actions import action_distance, fallback_action_id


class LightRiskPredictorProtocol(Protocol):
    def predict(self, history_scene: List[SceneState], action_id: int) -> RiskPrediction:
        raise NotImplementedError


class WorldModelPredictorProtocol(Protocol):
    def predict(self, history_scene: List[SceneState], action_id: int) -> WorldPrediction:
        raise NotImplementedError


@dataclass
class CandidateEvaluation:
    action_id: int
    coarse_risk: float
    fine_risk: float
    uncertainty: float
    world_prediction: Optional[WorldPrediction]


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

        coarse_sorted = sorted(candidates, key=lambda c: coarse_risk[c])
        refined = coarse_sorted[: max(1, self.config.coarse_top_k)]
        if policy_action not in refined:
            refined.append(policy_action)

        evaluations: Dict[int, CandidateEvaluation] = {}
        for action in refined:
            fine_risk, uncertainty, world_prediction = self._fine_risk(history_scene, action)
            evaluations[action] = CandidateEvaluation(
                action_id=action,
                coarse_risk=coarse_risk[action],
                fine_risk=fine_risk,
                uncertainty=uncertainty,
                world_prediction=world_prediction,
            )

        raw_eval = evaluations[policy_action]
        shield_blocked = raw_eval.fine_risk > self.config.risk_threshold or raw_eval.uncertainty > self.config.uncertainty_threshold
        common_meta = {
            "candidate_count": int(len(candidates)),
            "evaluated_candidate_count": int(len(evaluations)),
            "shield_blocked": bool(shield_blocked),
            "raw_world_prediction": raw_eval.world_prediction,
        }

        if not shield_blocked:
            return ShieldDecision(
                raw_action=policy_action,
                final_action=policy_action,
                intervened=False,
                reason="raw_action_safe",
                risk_raw=raw_eval.fine_risk,
                risk_final=raw_eval.fine_risk,
                candidate_risks={k: v.fine_risk for k, v in evaluations.items()},
                meta={
                    **common_meta,
                    "fallback_used": False,
                    "replacement_happened": False,
                    "final_world_prediction": raw_eval.world_prediction,
                },
            )

        safe_candidates = [
            ev for ev in evaluations.values()
            if ev.fine_risk <= self.config.risk_threshold and ev.uncertainty <= self.config.uncertainty_threshold
        ]

        if safe_candidates:
            safe_candidates.sort(key=lambda ev: (action_distance(policy_action, ev.action_id), ev.fine_risk))
            selected = safe_candidates[0]
            replacement_happened = selected.action_id != policy_action
            return ShieldDecision(
                raw_action=policy_action,
                final_action=selected.action_id,
                intervened=replacement_happened,
                reason="risk_threshold_exceeded",
                risk_raw=raw_eval.fine_risk,
                risk_final=selected.fine_risk,
                candidate_risks={k: v.fine_risk for k, v in evaluations.items()},
                meta={
                    **common_meta,
                    "fallback_used": False,
                    "replacement_happened": bool(replacement_happened),
                    "final_world_prediction": selected.world_prediction,
                },
            )

        fallback = fallback_action_id()
        if fallback in evaluations:
            fallback_eval = evaluations[fallback]
        else:
            fallback_eval = sorted(evaluations.values(), key=lambda ev: ev.fine_risk)[0]
            fallback = fallback_eval.action_id

        replacement_happened = fallback != policy_action
        return ShieldDecision(
            raw_action=policy_action,
            final_action=fallback,
            intervened=True,
            reason="all_candidates_high_risk_or_uncertain",
            risk_raw=raw_eval.fine_risk,
            risk_final=fallback_eval.fine_risk,
            candidate_risks={k: v.fine_risk for k, v in evaluations.items()},
            meta={
                **common_meta,
                "fallback_used": True,
                "replacement_happened": bool(replacement_happened),
                "final_world_prediction": fallback_eval.world_prediction,
            },
        )

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
