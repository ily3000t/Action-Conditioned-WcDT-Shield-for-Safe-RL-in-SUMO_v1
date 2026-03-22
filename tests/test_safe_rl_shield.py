from safe_rl.config.config import ShieldConfig
from safe_rl.data.types import RiskPrediction, SceneState, VehicleState, WorldPrediction
from safe_rl.shield.risk_aggregator import aggregate_tail_risk
from safe_rl.shield.safety_shield import SafetyShield


class DummyLightPredictor:
    def predict(self, history_scene, action_id):
        return RiskPrediction(0.1, 0.1, 0.1, 0.1, 0.05)


class DummyWorldPredictor:
    def __init__(self, risk_by_action):
        self.risk_by_action = risk_by_action

    def predict(self, history_scene, action_id):
        risk = self.risk_by_action.get(action_id, 0.8)
        modality = [RiskPrediction(0.1, 0.1, 0.1, risk, 0.05) for _ in range(6)]
        return WorldPrediction(multimodal_future=None, modality_risk=modality, aggregated_risk=risk, uncertainty=0.05)


def _history_scene():
    ego = VehicleState("ego", 0.0, 4.0, 22.0, 0.0, 0.0, 0.0, 0.0, 1)
    lead = VehicleState("lead", 12.0, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1)
    scene = SceneState(timestamp=0.0, ego_id="ego", vehicles=[ego, lead])
    return [scene] * 10


def _merge_history_scene():
    ego = VehicleState("ego", 980.0, 4.0, 22.0, 0.0, 0.0, 0.0, 0.0, 1)
    lead = VehicleState("lead", 992.0, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1)
    ramp = VehicleState("ramp", 975.0, 0.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0)
    scene = SceneState(timestamp=0.0, ego_id="ego", vehicles=[ego, lead, ramp])
    return [scene] * 10


def test_tail_risk_aggregation_with_uncertainty():
    risk = aggregate_tail_risk([0.1, 0.2, 0.8, 0.9], quantile=0.9, uncertainty=0.5, uncertainty_weight=0.2)
    assert risk > 0.8


def test_shield_replaces_high_risk_action():
    config = ShieldConfig(risk_threshold=0.4, uncertainty_threshold=0.3, candidate_count=7, coarse_top_k=4)
    predictor = DummyWorldPredictor({4: 0.9, 1: 0.2, 3: 0.3})
    shield = SafetyShield(config=config, light_predictor=DummyLightPredictor(), world_predictor=predictor)

    decision = shield.select_action(_history_scene(), 4)
    assert decision.intervened is True
    assert decision.final_action != 4
    assert decision.risk_final <= decision.risk_raw
    assert decision.meta["candidate_count"] == 7
    assert decision.meta["evaluated_candidate_count"] >= 1
    assert decision.meta["replacement_happened"] is True
    assert decision.meta["fallback_used"] is False
    assert decision.meta["chosen_candidate_index"] >= 0
    assert decision.meta["chosen_candidate_rank_by_risk"] >= 0
    assert isinstance(decision.meta["candidate_evaluations"], list)


def test_shield_passes_safe_action():
    config = ShieldConfig(risk_threshold=0.6, uncertainty_threshold=0.3, candidate_count=7, coarse_top_k=4)
    predictor = DummyWorldPredictor({4: 0.2})
    shield = SafetyShield(config=config, light_predictor=DummyLightPredictor(), world_predictor=predictor)

    decision = shield.select_action(_history_scene(), 4)
    assert decision.intervened is False
    assert decision.final_action == 4
    assert decision.meta["replacement_happened"] is False
    assert decision.meta["fallback_used"] is False


def test_shield_blocks_replacement_without_margin():
    config = ShieldConfig(
        risk_threshold=0.40,
        uncertainty_threshold=0.30,
        candidate_count=7,
        coarse_top_k=4,
        replacement_min_risk_margin=0.05,
    )
    predictor = DummyWorldPredictor({4: 0.43, 1: 0.39})
    shield = SafetyShield(config=config, light_predictor=DummyLightPredictor(), world_predictor=predictor)

    decision = shield.select_action(_history_scene(), 4)
    assert decision.final_action == 4
    assert decision.intervened is False
    assert decision.reason == "replacement_blocked_by_constraint"
    assert decision.meta["constraint_reason"] == "blocked_by_margin"


def test_shield_marks_raw_passthrough_for_low_risk_action():
    config = ShieldConfig(risk_threshold=0.60, uncertainty_threshold=0.30, candidate_count=7, coarse_top_k=4)
    predictor = DummyWorldPredictor({4: 0.18, 1: 0.10})
    shield = SafetyShield(config=config, light_predictor=DummyLightPredictor(), world_predictor=predictor)

    decision = shield.select_action(_history_scene(), 4)
    assert decision.final_action == 4
    assert decision.intervened is False
    assert decision.meta["constraint_reason"] == "raw_passthrough"


def test_shield_merge_guard_blocks_cross_lateral_replacement():
    config = ShieldConfig(
        risk_threshold=0.40,
        uncertainty_threshold=0.30,
        candidate_count=7,
        coarse_top_k=6,
        replacement_min_risk_margin=0.05,
        protect_merge_lateral_decisions=True,
        merge_override_margin=0.12,
    )
    predictor = DummyWorldPredictor({3: 0.50, 4: 0.39})
    shield = SafetyShield(config=config, light_predictor=DummyLightPredictor(), world_predictor=predictor)

    decision = shield.select_action(_merge_history_scene(), 3)
    assert decision.final_action == 3
    assert decision.intervened is False
    assert decision.reason == "replacement_blocked_by_constraint"
    assert decision.meta["constraint_reason"] == "merge_lateral_guard"
