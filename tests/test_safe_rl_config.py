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
