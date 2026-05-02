from pathlib import Path

import pytest

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
    assert config.sim.scenario_variant == "compact_midhigh_202604"
    assert config.sim.collision_action == "teleport"
    assert config.sim.collision_check_junctions is True
    assert config.shield.replacement_min_risk_margin == 0.104
    assert config.shield.protect_merge_lateral_decisions is True
    assert config.stage1_collection.probe_enabled is True
    assert config.stage1_collection.probe_horizon_steps == 8
    assert config.stage1_collection.probe_action_set == "all_9"
    assert config.stage1_collection.probe_warmup_steps == 12
    assert config.stage1_collection.initial_risk_event_step == 12
    assert config.stage1_collection.min_gap_between_risk_events == 8
    assert config.stage1_collection.probe_pair_min_target_gap == 0.01
    assert config.stage1_collection.probe_pair_max_pairs_per_step == 12
    assert config.stage1_collection.probe_pair_boundary_gap_floor == 0.005
    assert config.stage1_collection.probe_pair_boundary_keep_per_risky_step == 1
    assert config.stage1_collection.stage4_candidate_min_target_gap == 0.01
    assert config.stage1_collection.exclude_structural_from_main is True
    assert config.world_model.min_spread_eligible_pairs_for_gate_source == 128
    assert config.world_model.pair_ft_min_unique_score_floor == 12
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.0001
    assert config.world_model.stage4_aux_min_high_gap_pairs == 128
    assert config.world_model.stage4_aux_unique_floor == 12
    assert config.world_model.stage4_aux_target_gap_threshold == 0.068
    assert config.world_model.min_stage5_pairs_for_world_ft == 50
    assert config.world_model.pair_finetune_gate_mode == "fallback_all_pairs"
    assert config.shield.profile == "balanced"
    assert config.shield.legacy_raw_passthrough_risk_threshold == 0.20
    assert config.shield.balanced_raw_passthrough_risk_threshold == 0.193
    assert config.shield.raw_passthrough_risk_threshold == 0.193
    assert config.shield.replacement_min_risk_margin_blocked == 0.02
    assert config.shield.blocked_distance_margin_slope == 0.0
    assert config.eval.low_speed_threshold_mps == 2.0
    assert config.eval.min_avg_speed_guard == 10.0
    assert config.eval.min_avg_speed_ratio_guard == 0.6
    assert config.eval.max_low_speed_step_rate_guard == 0.15


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
    config = load_safe_rl_config("safe_rl/config/debug/debug_real_sumo.yaml")
    assert config.sim.collision_action == "warn"
    assert config.sim.collision_stoptime == 1.0


def test_debug_stage3_config_is_real_debug_profile():
    config = load_safe_rl_config("safe_rl/config/debug/debug_stage3_sumo.yaml")
    assert config.sim.collision_action == "warn"
    assert config.ppo.use_sb3 is True
    assert config.ppo.total_timesteps == 1024
    assert config.ppo.n_steps == 128
    assert config.eval.eval_episodes == 2


def test_shield_sanity_config_uses_aggressive_thresholds():
    config = load_safe_rl_config("safe_rl/config/debug/shield_sanity.yaml")
    assert config.shield.risk_threshold == 0.05
    assert config.shield.uncertainty_threshold == 1.0
    assert config.shield.coarse_top_k == 7
    assert config.eval.eval_episodes == 10



def test_shield_sweep_config_loads_default_variants():
    config = load_safe_rl_config("safe_rl/config/experiments/shield_sweep.yaml")
    assert config.shield_sweep.enabled is True
    assert config.shield_sweep.target_intervention_min == 0.05
    assert config.shield_sweep.target_intervention_max == 0.30
    assert config.shield_sweep.min_avg_speed == 10.0
    assert [(variant.name, variant.risk_threshold, variant.uncertainty_threshold, variant.coarse_top_k) for variant in config.shield_sweep.variants] == [
        ("A", 0.20, 0.60, 7),
        ("B", 0.25, 0.50, 6),
        ("C", 0.30, 0.45, 5),
    ]


@pytest.mark.parametrize(
    "legacy_path",
    [
        "safe_rl/config/deprecated/shield_trace_c.yaml",
        "safe_rl/config/deprecated/shield_trace_c_strong.yaml",
        "safe_rl/config/deprecated/stage2_world_base_only.yaml",
        "safe_rl/config/deprecated/safe_rl_balanced_profile.yaml",
    ],
)
def test_removed_deprecated_config_paths_fail_fast(legacy_path: str):
    with pytest.raises(FileNotFoundError):
        load_safe_rl_config(legacy_path)


def test_stage2_v2_world_pair_focus_config_loads():
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_v2_world_pair_focus.yaml")
    assert config.light_risk.pair_finetune is False
    assert config.world_model.pair_finetune is True
    assert config.light_risk.epochs == 50
    assert config.world_model.epochs == 20
    assert config.world_model.stage5_pair_weight == 1.0
    assert config.world_model.stage4_pair_weight == 0.2
    assert config.world_model.stage5_pair_max_seen_per_epoch == 32
    assert config.world_model.pair_finetune_epochs == 6
    assert config.world_model.pair_ft_patience == 2
    assert config.tensorboard.run_name == "stage2_v2_world_pair_focus"


def test_shield_trace_holdout_c1_config_loads():
    config = load_safe_rl_config("safe_rl/config/experiments/shield_trace_holdout_c1.yaml")
    assert config.shield.replacement_min_risk_margin == 0.08
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_holdout_c1"
    assert config.shield_trace.seed_list == [11, 29, 47]
    assert config.eval.seed_list == [11, 29, 47]
    assert config.tensorboard.run_name == "shield_trace_holdout_c1"


def test_partial_trace_config_inherits_default_sim_settings():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/experiments/shield_trace_holdout_c1.yaml")
    assert config.sim.sumo_cfg == default_config.sim.sumo_cfg
    assert config.sim.sumo_home == default_config.sim.sumo_home
    assert config.sim.sumo_bin == default_config.sim.sumo_bin
    assert config.sim.runtime_log_dir == default_config.sim.runtime_log_dir


def test_stage5_pair_bootstrap_config_loads():
    config = load_safe_rl_config("safe_rl/config/advanced/stage5_pair_bootstrap.yaml")
    assert config.shield.risk_threshold == 0.30
    assert config.shield.replacement_min_risk_margin == 0.02
    assert config.shield.raw_passthrough_risk_threshold == 0.24
    assert config.shield_trace.trace_dir_name == "shield_trace_pair_bootstrap"
    assert config.shield_trace.seed_list[0] == 1000
    assert config.shield_trace.seed_list[-1] == 1049
    assert len(config.shield_trace.seed_list) == 50
    assert config.eval.eval_episodes == 50
    assert config.tensorboard.run_name == "shield_trace_pair_bootstrap"


def test_stage5_eval_hardening_config_loads():
    config = load_safe_rl_config("safe_rl/config/visualization/stage5_eval_hardening.yaml")
    assert config.eval.eval_episodes == 90
    assert len(config.eval.seed_list) >= 12
    assert config.tensorboard.run_name == "stage5_eval_hardening"


def test_stage45_cost_desensitize_config_loads():
    config = load_safe_rl_config("safe_rl/config/visualization/stage45_cost_desensitize.yaml")
    assert config.shield.blocked_distance_margin_slope == 0.015
    assert config.distill.learning_rate == 0.0003
    assert config.distill.epochs == 8
    assert config.tensorboard.run_name == "stage45_cost_desensitize"


def test_stage5_trace_capture_default_config_loads():
    config = load_safe_rl_config("safe_rl/config/visualization/stage5_trace_capture_default.yaml")
    assert config.shield_trace.enabled is True
    assert config.shield_trace.save_pair_traces is True
    assert config.shield_trace.trace_dir_name == "stage5_trace_capture_default"
    assert config.shield_trace.seed_list == [42, 123, 2024, 7, 11, 29, 47, 64]
    assert config.eval.eval_episodes == 8
    assert config.eval.eval_episodes == len(config.shield_trace.seed_list)
    assert config.tensorboard.run_name == "stage5_trace_capture_default"


def test_stage5_trace_capture_cost_config_loads():
    config = load_safe_rl_config("safe_rl/config/visualization/stage5_trace_capture_cost.yaml")
    assert config.shield_trace.enabled is True
    assert config.shield_trace.save_pair_traces is True
    assert config.shield_trace.trace_dir_name == "stage5_trace_capture_cost"
    assert config.shield_trace.seed_list == [42, 123, 2024, 7, 11, 29, 47, 64]
    assert config.eval.eval_episodes == 8
    assert config.eval.eval_episodes == len(config.shield_trace.seed_list)
    assert config.shield.blocked_distance_margin_slope == 0.015
    assert config.distill.learning_rate == 0.0003
    assert config.distill.epochs == 8
    assert config.tensorboard.run_name == "stage5_trace_capture_cost"


def test_risk_model_v2_defaults_enabled():
    config = load_safe_rl_config()
    assert config.light_risk.enable_v2 is True
    assert config.light_risk.pair_finetune is True
    assert config.light_risk.ranking_loss_weight == 0.3
    assert config.world_model.enable_v2 is True
    assert config.world_model.pair_finetune is True
    assert config.light_risk.pair_ft_eval_max_samples == 2048
    assert config.world_model.pair_ft_eval_max_samples == 2048
    assert config.light_risk.pointwise_replay_weight == 1.0
    assert config.light_risk.spread_loss_weight == 0.05
    assert config.world_model.ranking_loss_weight == 0.3
    assert config.world_model.stage5_pair_weight == 1.0
    assert config.world_model.stage4_pair_weight == 0.2
    assert config.world_model.stage5_pair_max_seen_per_epoch == 32
    assert config.world_model.pair_finetune_epochs == 6
    assert config.world_model.pair_ft_patience == 2
    assert config.world_model.pair_ft_tie_gap_epsilon == 0.01
    assert config.world_model.pair_ft_min_score_spread_floor == 0.008
    assert config.world_model.pair_ft_min_same_state_gap_floor == 0.008
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == 4
    assert config.world_model.pair_ft_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_resolution_min_score_gap == 0.03
    assert config.world_model.pair_ft_resolution_min_logit_gap == 0.14
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.0
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.015
    assert config.world_model.pair_ft_stage1_resolution_mode == "fixed"
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_tail_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_enabled is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_acc_tolerance == 0.01
    assert config.world_model.pair_ft_stage1_tail_acceptance_spread_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_acceptance_gap_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_sampling_mode == "with_replacement"
    assert config.world_model.pair_ft_stage1_tail_ranking_loss_weight is None
    assert config.world_model.pair_ft_stage1_tail_resolution_loss_weight is None
    assert config.world_model.pair_ft_stage1_tail_anticollapse_weight == 0.0
    assert config.world_model.pair_ft_stage1_tail_score_range_floor == 0.02
    assert config.world_model.pair_ft_stage1_tail_score_range_quantile_low == 0.10
    assert config.world_model.pair_ft_stage1_tail_score_range_quantile_high == 0.90
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is False
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.0)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_floor == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_apply_on == "priority_only"
    assert config.world_model.pair_ft_stage1_calibration_enabled is False
    assert config.world_model.pair_ft_stage1_calibration_scale_init == pytest.approx(1.0)
    assert config.world_model.pair_ft_stage1_calibration_bias_init == pytest.approx(0.0)
    assert config.world_model.pair_ft_stage1_calibration_train_scope == "pair_ft_only"
    assert config.world_model.pair_ft_stage1_softbin_loss_weight == pytest.approx(0.0)
    assert config.world_model.pair_ft_stage1_softbin_num_bins == 16
    assert config.world_model.pair_ft_stage1_softbin_temperature == pytest.approx(80.0)
    assert config.world_model.pair_ft_stage1_softbin_apply_on == "stage1_probe"
    assert config.world_model.pair_ft_stage1_softbin_apply_trusted_only is True
    assert config.world_model.pair_ft_freeze_traj_decoder is True
    assert config.world_model.pair_ft_freeze_backbone == "partial"



def test_explicit_shield_thresholds_override_profile_defaults():
    temp_dir = Path("safe_rl_output/test_artifacts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = temp_dir / "shield_profile_override.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "shield:",
                "  profile: balanced",
                "  raw_passthrough_risk_threshold: 0.22",
                "  replacement_min_risk_margin: 0.11",
            ]
        ),
        encoding="utf-8",
    )
    config = load_safe_rl_config(str(yaml_path))
    assert config.shield.profile == "balanced"
    assert config.shield.raw_passthrough_risk_threshold == 0.22
    assert config.shield.replacement_min_risk_margin == 0.11


def test_stage2_stage4_aux_push_config_loads():
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage4_aux_push.yaml")
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == 2
    assert config.world_model.pair_ft_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_resolution_min_score_gap == 0.03


def test_stage2_stage1_gate_resolution_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.01
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.015
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_resolution_min_score_gap == default_config.world_model.pair_ft_resolution_min_score_gap


def test_stage2_stage1_gate_resolution_w002_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution_w002.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.015
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_resolution_min_score_gap == default_config.world_model.pair_ft_resolution_min_score_gap
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience


def test_stage2_stage1_gate_resolution_w002_selectfix_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution_w002_selectfix.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.015
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_resolution_w003_selectfix_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution_w003_selectfix.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.03
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.015
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_resolution_w002_gap018_selectfix_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution_w002_gap018_selectfix.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_resolution_adaptive_margin_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_resolution_adaptive_margin.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_tail_calibration_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_tail_calibration.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 2
    assert config.world_model.pair_ft_stage1_tail_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_enabled is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_acc_tolerance == 0.01
    assert config.world_model.pair_ft_stage1_tail_acceptance_spread_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_acceptance_gap_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_sampling_mode == "with_replacement"
    assert config.world_model.pair_ft_stage1_tail_ranking_loss_weight is None
    assert config.world_model.pair_ft_stage1_tail_resolution_loss_weight is None
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_tail_calibration_noreplacement_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_gate_tail_calibration_noreplacement.yaml")
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 2
    assert config.world_model.pair_ft_stage1_tail_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_enabled is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_acc_tolerance == 0.01
    assert config.world_model.pair_ft_stage1_tail_acceptance_spread_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_acceptance_gap_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_sampling_mode == "without_replacement"
    assert config.world_model.pair_ft_stage1_tail_ranking_loss_weight is None
    assert config.world_model.pair_ft_stage1_tail_resolution_loss_weight is None
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_tail_calibration_noreplacement_reweight_balanced_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config(
        "safe_rl/config/advanced/stage2_stage1_gate_tail_calibration_noreplacement_reweight_balanced.yaml"
    )
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 2
    assert config.world_model.pair_ft_stage1_tail_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_enabled is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_acc_tolerance == 0.01
    assert config.world_model.pair_ft_stage1_tail_acceptance_spread_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_acceptance_gap_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_sampling_mode == "without_replacement"
    assert config.world_model.pair_ft_stage1_tail_ranking_loss_weight == pytest.approx(0.25)
    assert config.world_model.pair_ft_stage1_tail_resolution_loss_weight == pytest.approx(0.025)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_gate_tail_calibration_noreplacement_anticollapse_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config(
        "safe_rl/config/advanced/stage2_stage1_gate_tail_calibration_noreplacement_anticollapse.yaml"
    )
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == 0.02
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == 0.018
    assert config.world_model.pair_ft_stage1_resolution_alpha == 0.2
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == 0.05
    assert config.world_model.pair_ft_stage1_resolution_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 2
    assert config.world_model.pair_ft_stage1_tail_apply_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_enabled is True
    assert config.world_model.pair_ft_stage1_tail_acceptance_acc_tolerance == 0.01
    assert config.world_model.pair_ft_stage1_tail_acceptance_spread_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_acceptance_gap_tolerance == 0.001
    assert config.world_model.pair_ft_stage1_tail_sampling_mode == "without_replacement"
    assert config.world_model.pair_ft_stage1_tail_ranking_loss_weight is None
    assert config.world_model.pair_ft_stage1_tail_resolution_loss_weight == pytest.approx(0.025)
    assert config.world_model.pair_ft_stage1_tail_anticollapse_weight == pytest.approx(0.005)
    assert config.world_model.pair_ft_stage1_tail_score_range_floor == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_tail_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_tail_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == 0.01
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_priority_mix_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_priority_mix.yaml")
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_priority_mix_phaseb_anticollapse_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_priority_mix_phaseb_anticollapse.yaml")
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.003)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_floor == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_apply_on == "priority_only"
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_priority_mix_phaseb_anticollapse_f025_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_priority_mix_phaseb_anticollapse_f025.yaml")
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.003)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_floor == pytest.approx(0.025)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_apply_on == "priority_only"
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_priority_mix_phaseb_anticollapse_f025_w005_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_priority_mix_phaseb_anticollapse_f025_w005.yaml")
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.005)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_floor == pytest.approx(0.025)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_apply_on == "priority_only"
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_priority_mix_phaseb_anticollapse_f025_allstage1_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config(
        "safe_rl/config/advanced/stage2_stage1_priority_mix_phaseb_anticollapse_f025_allstage1.yaml"
    )
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.003)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_floor == pytest.approx(0.025)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_low == pytest.approx(0.10)
    assert config.world_model.pair_ft_stage1_phaseb_score_range_quantile_high == pytest.approx(0.90)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_apply_on == "all_stage1"
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage2_stage1_calibration_softbin_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage2_stage1_calibration_softbin.yaml")
    assert config.world_model.pair_ft_stage1_priority_mix_enabled is True
    assert config.world_model.pair_ft_stage1_priority_mix_fraction == pytest.approx(0.35)
    assert config.world_model.pair_ft_stage1_priority_trusted_only is True
    assert config.world_model.pair_ft_stage1_tail_epochs == 0
    assert config.world_model.pair_ft_stage1_resolution_loss_weight == pytest.approx(0.02)
    assert config.world_model.pair_ft_stage1_resolution_mode == "adaptive"
    assert config.world_model.pair_ft_stage1_resolution_min_score_gap == pytest.approx(0.018)
    assert config.world_model.pair_ft_stage1_resolution_alpha == pytest.approx(0.2)
    assert config.world_model.pair_ft_stage1_resolution_max_score_gap == pytest.approx(0.05)
    assert config.world_model.pair_ft_selection_accuracy_tie_epsilon == pytest.approx(0.01)
    assert config.world_model.pair_ft_stage1_phaseb_anticollapse_weight == pytest.approx(0.0)
    assert config.world_model.pair_ft_stage1_calibration_enabled is True
    assert config.world_model.pair_ft_stage1_calibration_scale_init == pytest.approx(1.0)
    assert config.world_model.pair_ft_stage1_calibration_bias_init == pytest.approx(0.0)
    assert config.world_model.pair_ft_stage1_calibration_train_scope == "pair_ft_only"
    assert config.world_model.pair_ft_stage1_softbin_loss_weight == pytest.approx(0.003)
    assert config.world_model.pair_ft_stage1_softbin_num_bins == 16
    assert config.world_model.pair_ft_stage1_softbin_temperature == pytest.approx(80.0)
    assert config.world_model.pair_ft_stage1_softbin_apply_on == "stage1_probe"
    assert config.world_model.pair_ft_stage1_softbin_apply_trusted_only is True
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_patience == default_config.world_model.pair_ft_patience
    assert config.world_model.pair_finetune_gate_mode == default_config.world_model.pair_finetune_gate_mode
    assert config.shield.profile == default_config.shield.profile
    assert config.shield.risk_threshold == default_config.shield.risk_threshold


def test_stage1_probe_recovery_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage1_probe_recovery.yaml")
    assert config.stage1_collection.probe_max_steps_per_episode == 6
    assert config.stage1_collection.probe_pair_boundary_keep_per_risky_step == 2
    assert config.stage1_collection.probe_pair_min_target_gap == default_config.stage1_collection.probe_pair_min_target_gap
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_resolution_min_score_gap == default_config.world_model.pair_ft_resolution_min_score_gap


def test_stage1_probe_recovery_gap008_config_loads():
    default_config = load_safe_rl_config("safe_rl/config/default_safe_rl.yaml")
    config = load_safe_rl_config("safe_rl/config/advanced/stage1_probe_recovery_gap008.yaml")
    assert config.stage1_collection.probe_max_steps_per_episode == 6
    assert config.stage1_collection.probe_pair_boundary_keep_per_risky_step == 2
    assert config.stage1_collection.probe_pair_min_target_gap == 0.008
    assert config.world_model.pair_ft_stage4_mix_every_n_steps == default_config.world_model.pair_ft_stage4_mix_every_n_steps
    assert config.world_model.pair_ft_resolution_loss_weight == default_config.world_model.pair_ft_resolution_loss_weight
    assert config.world_model.pair_ft_resolution_min_score_gap == default_config.world_model.pair_ft_resolution_min_score_gap
    assert config.shield.profile == default_config.shield.profile


