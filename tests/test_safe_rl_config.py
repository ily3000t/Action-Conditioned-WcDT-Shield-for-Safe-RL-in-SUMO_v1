from safe_rl.config.config import load_safe_rl_config


def test_default_config_loads():
    config = load_safe_rl_config()
    assert config.sim.history_steps > 0
    assert config.shield.candidate_count == 7
    assert config.ppo.total_timesteps > 0
    assert config.tensorboard.enabled is True
    assert config.tensorboard.root_dir


def test_tensorboard_config_override(tmp_path):
    yaml_path = tmp_path / "tb_override.yaml"
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
