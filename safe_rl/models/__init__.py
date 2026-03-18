from safe_rl.models.features import ACTION_DIM, BASE_FEATURE_DIM, encode_history, history_action_feature

# Torch-dependent imports are optional at import-time to keep module discovery robust.
try:
    from safe_rl.models.action_encoder import ActionEncoder
    from safe_rl.models.light_risk_model import (
        LightRiskMLP,
        LightRiskPredictor,
        LightRiskTrainer,
        create_untrained_light_predictor,
    )
    from safe_rl.models.world_model import (
        ActionConditionedWorldModel,
        SceneTensorizer,
        WorldModelPredictor,
        WorldModelTrainer,
        create_untrained_world_predictor,
    )
except Exception:
    ActionEncoder = None
    LightRiskMLP = None
    LightRiskPredictor = None
    LightRiskTrainer = None
    create_untrained_light_predictor = None
    ActionConditionedWorldModel = None
    SceneTensorizer = None
    WorldModelPredictor = None
    WorldModelTrainer = None
    create_untrained_world_predictor = None

__all__ = [
    "ActionEncoder",
    "ACTION_DIM",
    "BASE_FEATURE_DIM",
    "encode_history",
    "history_action_feature",
    "LightRiskMLP",
    "LightRiskPredictor",
    "LightRiskTrainer",
    "create_untrained_light_predictor",
    "ActionConditionedWorldModel",
    "SceneTensorizer",
    "WorldModelPredictor",
    "WorldModelTrainer",
    "create_untrained_world_predictor",
]
