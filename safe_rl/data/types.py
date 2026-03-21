from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    heading: float
    lane_id: int
    length: float = 4.8
    width: float = 2.0


@dataclass
class TrafficLightState:
    light_id: str
    x: float
    y: float
    state: str


@dataclass
class SceneState:
    timestamp: float
    ego_id: str
    vehicles: List[VehicleState]
    traffic_lights: List[TrafficLightState] = field(default_factory=list)
    lane_polylines: List[List[List[float]]] = field(default_factory=list)


@dataclass
class RiskLabels:
    collision: bool
    ttc_risk: bool
    lane_violation: bool
    overall_risk: float
    min_ttc: float
    min_distance: float


@dataclass
class ActionConditionedSample:
    history_scene: List[SceneState]
    candidate_action: int
    future_scene: List[SceneState]
    risk_labels: RiskLabels
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskPrediction:
    p_collision: float
    p_ttc: float
    p_lane_violation: float
    p_overall: float
    uncertainty: float


@dataclass
class WorldPrediction:
    multimodal_future: Any
    modality_risk: List[RiskPrediction]
    aggregated_risk: float
    uncertainty: float


@dataclass
class ShieldDecision:
    raw_action: int
    final_action: int
    intervened: bool
    reason: str
    risk_raw: float
    risk_final: float
    candidate_risks: Dict[int, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    history_scene: List[SceneState]
    raw_action: int
    final_action: int
    raw_risk: float
    final_risk: float
    reason: str
    raw_future: Optional[Any] = None
    final_future: Optional[Any] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepLog:
    step_index: int
    scene: SceneState
    raw_action: int
    final_action: int
    shield_decision: ShieldDecision
    task_reward: float
    final_reward: float
    done: bool
    risk_labels: RiskLabels


@dataclass
class EpisodeLog:
    episode_id: str
    risky_mode: bool
    steps: List[StepLog]


@dataclass
class EpisodeSummary:
    episode_id: str
    steps: int
    collisions: int
    interventions: int
    avg_speed: float
    mean_reward: float
    success: bool
    mean_raw_risk: float = 0.0
    mean_final_risk: float = 0.0
    mean_risk_reduction: float = 0.0
    replacement_count: int = 0
    replacement_same_as_raw_count: int = 0
    fallback_action_count: int = 0
    shield_called_steps: int = 0
    shield_candidate_evaluated_steps: int = 0
    shield_blocked_steps: int = 0
    shield_replaced_steps: int = 0


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    return value
