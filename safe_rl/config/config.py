from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SimConfig:
    backend: str = "traci"
    sumo_cfg: str = "scenarios/highway_merge/highway_merge.sumocfg"
    scenario_variant: str = ""
    net_file: str = ""
    route_file: str = ""
    runtime_log_dir: str = "safe_rl_output/sumo_logs"
    sumo_home: str = ""
    sumo_bin: str = ""
    sumo_gui_bin: str = ""
    netconvert_bin: str = ""
    auto_build_network: bool = True
    force_mock: bool = False
    use_gui: bool = False
    step_length: float = 0.1
    episode_steps: int = 300
    history_steps: int = 10
    future_steps: int = 20
    normal_episodes: int = 200
    risky_episodes: int = 200
    risk_event_prob: float = 0.35
    random_seed: int = 42
    ego_vehicle_id: str = "ego"
    collision_action: str = "teleport"
    collision_stoptime: float = 1.0
    collision_check_junctions: bool = True


@dataclass
class DatasetConfig:
    raw_log_dir: str = "safe_rl_output/raw_logs"
    dataset_dir: str = "safe_rl_output/datasets"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    ttc_threshold: float = 2.0
    lane_violation_margin: float = 0.2


@dataclass
class Stage1CollectionConfig:
    probe_enabled: bool = True
    probe_horizon_steps: int = 8
    probe_max_steps_per_episode: int = 4
    probe_action_set: str = "all_9"
    probe_trigger_ttc_threshold: float = 3.0
    probe_trigger_min_distance: float = 12.0
    probe_warmup_steps: int = 12
    initial_risk_event_step: int = 12
    min_gap_between_risk_events: int = 8
    probe_pair_min_target_gap: float = 0.01
    probe_pair_max_pairs_per_step: int = 12
    probe_pair_boundary_gap_floor: float = 0.005
    probe_pair_boundary_keep_per_risky_step: int = 1
    stage4_candidate_min_target_gap: float = 0.01
    exclude_structural_from_main: bool = True


@dataclass
class LightRiskConfig:
    enable_v2: bool = True
    pair_finetune: bool = True
    pointwise_replay_weight: float = 1.0
    ranking_loss_weight: float = 0.3
    spread_loss_weight: float = 0.05
    stage5_pair_weight: float = 1.0
    stage4_pair_weight: float = 0.2
    pair_finetune_epochs: int = 3
    pair_ft_eval_max_samples: int = 2048
    hidden_dim: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20


@dataclass
class WorldModelConfig:
    enable_v2: bool = True
    pair_finetune: bool = True
    pointwise_replay_weight: float = 1.0
    ranking_loss_weight: float = 0.3
    spread_loss_weight: float = 0.05
    stage5_pair_weight: float = 1.0
    stage4_pair_weight: float = 0.2
    pair_finetune_epochs: int = 6
    pair_ft_freeze_traj_decoder: bool = True
    pair_ft_freeze_backbone: str = "partial"
    pair_ft_eval_max_samples: int = 2048
    stage5_pair_max_seen_per_epoch: int = 32
    pair_ft_patience: int = 2
    pair_ft_tie_gap_epsilon: float = 0.01
    pair_ft_min_score_spread_floor: float = 0.008
    pair_ft_min_same_state_gap_floor: float = 0.008
    pair_ft_min_unique_score_floor: int = 12
    pair_ft_selection_accuracy_tie_epsilon: float = 1e-4
    pair_ft_resolution_loss_weight: float = 0.02
    pair_ft_resolution_min_score_gap: float = 0.03
    pair_ft_resolution_min_logit_gap: float = 0.14
    min_spread_eligible_pairs_for_gate_source: int = 128
    stage4_aux_min_high_gap_pairs: int = 128
    stage4_aux_unique_floor: int = 12
    stage4_aux_target_gap_threshold: float = 0.068
    min_stage5_pairs_for_world_ft: int = 50
    pair_finetune_gate_mode: str = "fallback_all_pairs"
    multimodal: int = 6
    future_steps: int = 20
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 20
    uncertainty_weight: float = 0.1


@dataclass
class ShieldConfig:
    candidate_count: int = 7
    profile: str = "legacy"
    coarse_top_k: int = 4
    tail_quantile: float = 0.9
    risk_threshold: float = 0.45
    uncertainty_threshold: float = 0.35
    uncertainty_weight: float = 0.2
    fallback_action: str = "DECEL_KEEP"
    replacement_min_risk_margin: float = 0.05
    replacement_min_risk_margin_blocked: Optional[float] = None
    blocked_distance_margin_slope: float = 0.0
    raw_passthrough_risk_threshold: float = 0.20
    legacy_replacement_min_risk_margin: float = 0.05
    legacy_raw_passthrough_risk_threshold: float = 0.20
    balanced_replacement_min_risk_margin: float = 0.104
    balanced_raw_passthrough_risk_threshold: float = 0.193
    protect_merge_lateral_decisions: bool = True
    merge_override_margin: float = 0.12


@dataclass
class ShieldSweepVariant:
    name: str = ""
    risk_threshold: float = 0.45
    uncertainty_threshold: float = 0.35
    coarse_top_k: int = 4


@dataclass
class ShieldSweepConfig:
    enabled: bool = False
    variants: List[ShieldSweepVariant] = field(default_factory=list)
    target_intervention_min: float = 0.05
    target_intervention_max: float = 0.30
    min_avg_speed: float = 10.0


@dataclass
class ShieldTraceConfig:
    enabled: bool = False
    seed_list: List[int] = field(default_factory=list)
    save_pair_traces: bool = True
    trace_dir_name: str = "shield_trace"


@dataclass
class PPOConfig:
    use_sb3: bool = True
    total_timesteps: int = 50000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    intervene_penalty: float = 0.1
    device: str = "cpu"


@dataclass
class DistillConfig:
    trigger_buffer_size: int = 500
    batch_size: int = 64
    learning_rate: float = 5e-4
    epochs: int = 10
    interval_steps: int = 10000


@dataclass
class EvalConfig:
    eval_episodes: int = 30
    seed_list: List[int] = field(default_factory=lambda: [42, 123, 2024])
    target_collision_reduction: float = 0.4
    max_efficiency_drop: float = 0.1
    low_speed_threshold_mps: float = 2.0
    min_avg_speed_guard: float = 10.0
    min_avg_speed_ratio_guard: float = 0.6
    max_low_speed_step_rate_guard: float = 0.15


@dataclass
class TensorboardConfig:
    enabled: bool = True
    root_dir: str = "safe_rl_output/tensorboard"
    run_name: str = ""
    flush_secs: int = 10


@dataclass
class SafeRLConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    stage1_collection: Stage1CollectionConfig = field(default_factory=Stage1CollectionConfig)
    light_risk: LightRiskConfig = field(default_factory=LightRiskConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    shield_sweep: ShieldSweepConfig = field(default_factory=ShieldSweepConfig)
    shield_trace: ShieldTraceConfig = field(default_factory=ShieldTraceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)


def _update_dataclass(instance, values: dict):
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(instance, key, value)


def _apply_config_data(config: SafeRLConfig, data: dict):
    if "sim" in data:
        _update_dataclass(config.sim, data["sim"])
    if "dataset" in data:
        _update_dataclass(config.dataset, data["dataset"])
    if "stage1_collection" in data:
        _update_dataclass(config.stage1_collection, data["stage1_collection"])
    if "light_risk" in data:
        _update_dataclass(config.light_risk, data["light_risk"])
    if "world_model" in data:
        _update_dataclass(config.world_model, data["world_model"])
    if "shield" in data:
        _update_dataclass(config.shield, data["shield"])
    if "shield_sweep" in data:
        shield_sweep_data = dict(data["shield_sweep"] or {})
        variants_data = list(shield_sweep_data.pop("variants", []) or [])
        _update_dataclass(config.shield_sweep, shield_sweep_data)
        config.shield_sweep.variants = []
        for item in variants_data:
            variant = ShieldSweepVariant()
            _update_dataclass(variant, dict(item or {}))
            config.shield_sweep.variants.append(variant)
    if "shield_trace" in data:
        _update_dataclass(config.shield_trace, data["shield_trace"])
    if "ppo" in data:
        _update_dataclass(config.ppo, data["ppo"])
    if "distill" in data:
        _update_dataclass(config.distill, data["distill"])
    if "eval" in data:
        _update_dataclass(config.eval, data["eval"])
    if "tensorboard" in data:
        _update_dataclass(config.tensorboard, data["tensorboard"])


def _load_yaml_dict(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(path: Optional[str], default_path: Path) -> Path:
    if path is None:
        return default_path
    requested = Path(path)
    direct = requested if requested.is_absolute() else Path(path)
    if direct.exists():
        return direct.resolve()
    raise FileNotFoundError(
        f"Config path not found: {path}. "
        "Use explicit config paths under safe_rl/config/{default,advanced,visualization,experiments,debug}."
    )


def _resolve_shield_profile(config: SafeRLConfig, shield_data: Optional[dict] = None):
    payload = dict(shield_data or {})
    profile = str(getattr(config.shield, "profile", "legacy") or "legacy").strip().lower()
    if profile not in ("legacy", "balanced"):
        raise ValueError(f"Unknown shield.profile: {profile}. Expected one of: legacy, balanced")

    explicit_threshold_keys = {
        "replacement_min_risk_margin": "replacement_min_risk_margin",
        "raw_passthrough_risk_threshold": "raw_passthrough_risk_threshold",
    }
    explicit_replacement = explicit_threshold_keys["replacement_min_risk_margin"] in payload
    explicit_passthrough = explicit_threshold_keys["raw_passthrough_risk_threshold"] in payload

    if profile == "balanced":
        profile_replacement = float(config.shield.balanced_replacement_min_risk_margin)
        profile_passthrough = float(config.shield.balanced_raw_passthrough_risk_threshold)
    else:
        profile_replacement = float(config.shield.legacy_replacement_min_risk_margin)
        profile_passthrough = float(config.shield.legacy_raw_passthrough_risk_threshold)

    if not explicit_replacement:
        config.shield.replacement_min_risk_margin = profile_replacement
    if not explicit_passthrough:
        config.shield.raw_passthrough_risk_threshold = profile_passthrough


def load_safe_rl_config(path: Optional[str] = None) -> SafeRLConfig:
    config = SafeRLConfig()
    default_path = (Path(__file__).resolve().parent / "default_safe_rl.yaml").resolve()
    target_path = _resolve_config_path(path, default_path)
    default_data = _load_yaml_dict(default_path)

    if target_path != default_path:
        _apply_config_data(config, default_data)
        _resolve_shield_profile(config, shield_data=dict(default_data.get("shield", {}) or {}))

    target_data = default_data if target_path == default_path else _load_yaml_dict(target_path)
    _apply_config_data(config, target_data)
    _resolve_shield_profile(config, shield_data=dict(target_data.get("shield", {}) or {}))
    return config
