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
