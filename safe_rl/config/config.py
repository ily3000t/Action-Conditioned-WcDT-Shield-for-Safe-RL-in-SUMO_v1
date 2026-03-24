from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class SimConfig:
    backend: str = "traci"
    sumo_cfg: str = "scenarios/highway_merge/highway_merge.sumocfg"
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
class LightRiskConfig:
    enable_v2: bool = True
    pair_finetune: bool = True
    pointwise_replay_weight: float = 1.0
    ranking_loss_weight: float = 0.3
    spread_loss_weight: float = 0.05
    stage5_pair_weight: float = 1.0
    stage4_pair_weight: float = 0.2
    pair_finetune_epochs: int = 3
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
    pair_finetune_epochs: int = 3
    pair_ft_freeze_traj_decoder: bool = True
    pair_ft_freeze_backbone: str = "partial"
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
    coarse_top_k: int = 4
    tail_quantile: float = 0.9
    risk_threshold: float = 0.45
    uncertainty_threshold: float = 0.35
    uncertainty_weight: float = 0.2
    fallback_action: str = "DECEL_KEEP"
    replacement_min_risk_margin: float = 0.05
    raw_passthrough_risk_threshold: float = 0.20
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


def load_safe_rl_config(path: Optional[str] = None) -> SafeRLConfig:
    config = SafeRLConfig()
    if path is None:
        path = str(Path(__file__).resolve().parent / "default_safe_rl.yaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "sim" in data:
        _update_dataclass(config.sim, data["sim"])
    if "dataset" in data:
        _update_dataclass(config.dataset, data["dataset"])
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
    return config
