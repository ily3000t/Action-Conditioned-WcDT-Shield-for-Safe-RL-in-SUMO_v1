from pathlib import Path

from safe_rl.config.config import load_safe_rl_config


def test_default_config_loads():
    config = load_safe_rl_config()
    assert config.sim.history_steps > 0
    assert config.shield.candidate_count == 7
    assert config.ppo.total_timesteps > 0
    assert config.tensorboard.enabled is True
    assert config.tensorboard.root_dir
    assert config.sim.ego_vehicle_id == "ego"
    assert config.sim.runtime_log_dir
    assert config.sim.collision_action == "teleport"
    assert config.sim.collision_check_junctions is True
    assert config.shield.replacement_min_risk_margin == 0.05
    assert config.shield.protect_merge_lateral_decisions is True


def test_tensorboard_config_override():
    temp_dir = Path("safe_rl_output/test_artifacts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = temp_dir / "tb_override.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "tensorboard:",
                "  enabled: true",
                "  root_dir: safe_rl_output/tb_test",
                "  run_name: ci_quick",
                "  flush_secs: 3",
            ]
        ),
        encoding="utf-8",
    )

    config = load_safe_rl_config(str(yaml_path))
    assert config.tensorboard.enabled is True
    assert config.tensorboard.root_dir == "safe_rl_output/tb_test"
    assert config.tensorboard.run_name == "ci_quick"
    assert config.tensorboard.flush_secs == 3


def test_debug_real_config_uses_warn_collision_action():
    config = load_safe_rl_config("safe_rl/config/debug_real_sumo.yaml")
    assert config.sim.collision_action == "warn"
    assert config.sim.collision_stoptime == 1.0


def test_debug_stage3_config_is_real_debug_profile():
    config = load_safe_rl_config("safe_rl/config/debug_stage3_sumo.yaml")
    assert config.sim.collision_action == "warn"
    assert config.ppo.use_sb3 is True
    assert config.ppo.total_timesteps == 1024
    assert config.ppo.n_steps == 128
    assert config.eval.eval_episodes == 2


def test_shield_sanity_config_uses_aggressive_thresholds():
    config = load_safe_rl_config("safe_rl/config/shield_sanity.yaml")
    assert config.shield.risk_threshold == 0.05
    assert config.shield.uncertainty_threshold == 1.0
    assert config.shield.coarse_top_k == 7
    assert config.eval.eval_episodes == 10



def test_shield_sweep_config_loads_default_variants():
    config = load_safe_rl_config("safe_rl/config/shield_sweep.yaml")
    assert config.shield_sweep.enabled is True
    assert config.shield_sweep.target_intervention_min == 0.05
    assert config.shield_sweep.target_intervention_max == 0.30
    assert config.shield_sweep.min_avg_speed == 10.0
    assert [(variant.name, variant.risk_threshold, variant.uncertainty_threshold, variant.coarse_top_k) for variant in config.shield_sweep.variants] == [
        ("A", 0.20, 0.60, 7),
        ("B", 0.25, 0.50, 6),
        ("C", 0.30, 0.45, 5),
    ]


def test_shield_trace_c_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_c.yaml")
    assert config.shield.risk_threshold == 0.30
    assert config.shield.uncertainty_threshold == 0.45
    assert config.shield.coarse_top_k == 5
    assert config.shield_trace.enabled is True
    assert config.shield_trace.seed_list == [42, 123, 2024]
    assert config.eval.eval_episodes == 3

def test_shield_trace_c1_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_c1.yaml")
    assert config.shield.replacement_min_risk_margin == 0.08
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_c1"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_c2_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_c2.yaml")
    assert config.shield.replacement_min_risk_margin == 0.10
    assert config.shield.raw_passthrough_risk_threshold == 0.25
    assert config.shield_trace.trace_dir_name == "shield_trace_c2"
    assert config.eval.seed_list == [42, 123, 2024]

def test_shield_trace_c_strong_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_c_strong.yaml")
    assert config.shield.replacement_min_risk_margin == 0.15
    assert config.shield.raw_passthrough_risk_threshold == 0.30
    assert config.shield_trace.trace_dir_name == "shield_trace_c_strong"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_d1_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_d1.yaml")
    assert config.shield.replacement_min_risk_margin == 0.10
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_d1"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_d2_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_d2.yaml")
    assert config.shield.replacement_min_risk_margin == 0.12
    assert config.shield.raw_passthrough_risk_threshold == 0.26
    assert config.shield_trace.trace_dir_name == "shield_trace_d2"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_d3_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_d3.yaml")
    assert config.shield.replacement_min_risk_margin == 0.13
    assert config.shield.raw_passthrough_risk_threshold == 0.28
    assert config.shield_trace.trace_dir_name == "shield_trace_d3"
    assert config.eval.seed_list == [42, 123, 2024]



def test_shield_trace_e1_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_e1.yaml")
    assert config.shield.replacement_min_risk_margin == 0.11
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_e1"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_e2_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_e2.yaml")
    assert config.shield.replacement_min_risk_margin == 0.10
    assert config.shield.raw_passthrough_risk_threshold == 0.25
    assert config.shield_trace.trace_dir_name == "shield_trace_e2"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_e3_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_e3.yaml")
    assert config.shield.replacement_min_risk_margin == 0.11
    assert config.shield.raw_passthrough_risk_threshold == 0.25
    assert config.shield_trace.trace_dir_name == "shield_trace_e3"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_f1_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_f1.yaml")
    assert config.shield.replacement_min_risk_margin == 0.103
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_f1"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_f2_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_f2.yaml")
    assert config.shield.replacement_min_risk_margin == 0.106
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_f2"
    assert config.eval.seed_list == [42, 123, 2024]


def test_shield_trace_f3_config_loads():
    config = load_safe_rl_config("safe_rl/config/shield_trace_f3.yaml")
    assert config.shield.replacement_min_risk_margin == 0.108
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_f3"
    assert config.eval.seed_list == [42, 123, 2024]

def test_risk_model_v2_defaults_enabled():
    config = load_safe_rl_config()
    assert config.light_risk.enable_v2 is True
    assert config.light_risk.pair_finetune is True
    assert config.light_risk.ranking_loss_weight == 1.0
    assert config.world_model.enable_v2 is True
    assert config.world_model.pair_finetune is True
    assert config.world_model.stage5_pair_weight == 1.0
    assert config.world_model.stage4_pair_weight == 0.5
