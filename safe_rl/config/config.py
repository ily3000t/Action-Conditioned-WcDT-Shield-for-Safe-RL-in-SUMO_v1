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
    hidden_dim: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20


@dataclass
class WorldModelConfig:
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
    if "ppo" in data:
        _update_dataclass(config.ppo, data["ppo"])
    if "distill" in data:
        _update_dataclass(config.distill, data["distill"])
    if "eval" in data:
        _update_dataclass(config.eval, data["eval"])
    if "tensorboard" in data:
        _update_dataclass(config.tensorboard, data["tensorboard"])
    return config
