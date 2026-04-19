import pytest
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from safe_rl.config.config import LightRiskConfig, WorldModelConfig
from safe_rl.data.pair_dataset import RiskPairDataset, collate_risk_pairs
from safe_rl.data.types import RiskLabels, RiskPairSample, SceneState, VehicleState, ActionConditionedSample
from safe_rl.models.light_risk_model import LightRiskMLP, LightRiskTrainer
from safe_rl.models.world_model import ActionConditionedWorldModel, SceneTensorizer, WorldModelPredictor


def _history_scene():
    ego = VehicleState("ego", 0.0, 4.0, 20.0, 0.0, 0.0, 0.0, 0.0, 1)
    lead = VehicleState("lead", 12.0, 4.0, 18.0, 0.0, 0.0, 0.0, 0.0, 1)
    scene = SceneState(timestamp=0.0, ego_id="ego", vehicles=[ego, lead])
    return [scene] * 4


def test_risk_pair_dataset_batches():
    sample = RiskPairSample(
        history_scene=_history_scene(),
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=2.0,
        meta={"target_risk_a": 0.2, "target_risk_b": 0.8},
    )
    loader = DataLoader(RiskPairDataset([sample, sample]), batch_size=2, shuffle=False, collate_fn=collate_risk_pairs)
    batch = next(iter(loader))
    assert len(batch) == 2
    assert batch[0].preferred_action == 4


def test_light_risk_v2_outputs_shapes():
    model = LightRiskMLP(hidden_dim=32)
    x = torch.randn(5, 33)
    output = model(x)
    assert output["risk_type_logits"].shape == (5, 3)
    assert output["risk_score"].shape == (5,)
    assert output["uncertainty"].shape == (5,)


def test_world_model_v2_outputs_shapes():
    config = WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2)
    tensorizer = SceneTensorizer(history_steps=2, future_steps=2)
    sample = ActionConditionedSample(
        history_scene=_history_scene()[:2],
        candidate_action=4,
        future_scene=_history_scene()[:2],
        risk_labels=RiskLabels(False, False, False, 0.2, 5.0, 10.0),
        meta={},
    )
    batch = tensorizer.tensorize_batch([sample])
    model = ActionConditionedWorldModel(config=config, history_steps=2)
    output = model(batch)
    assert output["risk_type_logits"].shape == (1, 3)
    assert output["risk_score_logit"].shape == (1,)
    assert output["risk_score"].shape == (1,)
    assert output["uncertainty"].shape == (1,)


def test_light_risk_pair_metrics_compute():
    trainer = LightRiskTrainer(LightRiskConfig(hidden_dim=32, epochs=1, batch_size=2), device="cpu")
    pair = RiskPairSample(
        history_scene=_history_scene(),
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "hard_negative": True},
    )
    metrics = trainer.evaluate_pairs([pair])
    assert metrics["pair_count"] == 1.0
    assert "pair_ranking_accuracy" in metrics



def test_world_pair_tensorization_avoids_future_targets():
    from safe_rl.models.world_model import WorldModelTrainer

    config = WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2)
    trainer = WorldModelTrainer(config=config, history_steps=2, device="cpu")
    pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "trusted_for_spread": True},
    )
    batch_a, batch_b, *_ = trainer._tensorize_pair_batch([pair])
    assert "target_future" not in batch_a
    assert "target_future" not in batch_b


def test_light_pair_batch_tracks_trusted_for_spread():
    trainer = LightRiskTrainer(LightRiskConfig(hidden_dim=32, epochs=1, batch_size=2), device="cpu")
    pair = RiskPairSample(
        history_scene=_history_scene(),
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "trusted_for_spread": True},
    )
    *_, trusted = trainer._pair_batch_tensors([pair])
    assert bool(trusted[0].item()) is True


def test_world_pair_ft_freeze_policy_freezes_traj_decoder():
    from safe_rl.models.world_model import WorldModelTrainer

    config = WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2, pair_ft_freeze_traj_decoder=True, pair_ft_freeze_backbone="partial")
    trainer = WorldModelTrainer(config=config, history_steps=2, device="cpu")
    grad_state, frozen, trainable = trainer._apply_pair_ft_freeze_policy()
    try:
        assert "traj_decoder" in frozen
        assert "fusion" in trainable
        assert "risk_score_head" in trainable
    finally:
        trainer._restore_grad_state(grad_state)


def test_world_pair_spread_eligibility_respects_trust_and_gap():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu")
    trusted_large_gap = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "trusted_for_spread": True},
    )
    trusted_small_gap = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.2, "trusted_for_spread": True},
    )
    untrusted_large_gap = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage4_buffer",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "trusted_for_spread": False},
    )
    assert trainer._spread_eligible_pair_count([trusted_large_gap, trusted_small_gap, untrusted_large_gap]) == 1


def test_world_pair_stage5_sampling_respects_epoch_cap():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu", seed=123)
    pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage5_trace_first_replacement",
        weight=1.0,
        meta={"seed": 42, "pair_index": 0},
    )
    batch1, ids1 = trainer._sample_stage5_batch_with_cap([pair], ["p0"], batch_size=1, seen_counts={"p0": 0}, max_seen_per_epoch=1)
    assert len(batch1) == 1
    assert ids1 == ["p0"]
    batch2, ids2 = trainer._sample_stage5_batch_with_cap([pair], ["p0"], batch_size=1, seen_counts={"p0": 1}, max_seen_per_epoch=1)
    assert batch2 == []
    assert ids2 == []


def test_world_pair_best_metric_prefers_accuracy_then_gap():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu")
    assert trainer._is_better_pair_ft_metrics(
        {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.01},
        {"pair_ranking_accuracy": 0.7, "same_state_score_gap": 0.5},
    ) is True
    assert trainer._is_better_pair_ft_metrics(
        {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.06},
        {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.05},
    ) is True
    assert trainer._is_better_pair_ft_metrics(
        {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.04},
        {"pair_ranking_accuracy": 0.8, "same_state_score_gap": 0.05},
    ) is False


def test_world_pair_best_metric_rejects_collapsed_candidate_with_higher_accuracy():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu")
    assert trainer._is_better_pair_ft_metrics(
        {"pair_ranking_accuracy": 0.82, "same_state_score_gap": 0.002, "score_spread": 0.002},
        {"pair_ranking_accuracy": 0.75, "same_state_score_gap": 0.02, "score_spread": 0.02},
    ) is False
    assert trainer._is_better_pair_ft_metrics(
        {"pair_ranking_accuracy": 0.76, "same_state_score_gap": 0.02, "score_spread": 0.02},
        {"pair_ranking_accuracy": 0.75, "same_state_score_gap": 0.002, "score_spread": 0.002},
    ) is True


def test_world_pair_epoch_eligibility_requires_unique_and_gap_floors():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(
        config=WorldModelConfig(
            hidden_dim=64,
            future_steps=2,
            multimodal=2,
            pair_ft_min_unique_score_floor=12,
            pair_ft_min_score_spread_floor=0.008,
            pair_ft_min_same_state_gap_floor=0.008,
        ),
        history_steps=2,
        device="cpu",
    )
    assert trainer._is_epoch_metrics_eligible_for_unique_guard(
        {"unique_score_count": 12.0, "score_spread": 0.01, "same_state_score_gap": 0.01}
    )
    assert not trainer._is_epoch_metrics_eligible_for_unique_guard(
        {"unique_score_count": 11.0, "score_spread": 0.01, "same_state_score_gap": 0.01}
    )
    assert not trainer._is_epoch_metrics_eligible_for_unique_guard(
        {"unique_score_count": 12.0, "score_spread": 0.007, "same_state_score_gap": 0.01}
    )


def test_world_pair_selection_compare_uses_unique_gap_spread_when_accuracy_tied():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(
        config=WorldModelConfig(
            hidden_dim=64,
            future_steps=2,
            multimodal=2,
            pair_ft_selection_accuracy_tie_epsilon=1e-4,
        ),
        history_steps=2,
        device="cpu",
    )
    # ranking accuracy tied within epsilon: unique decides first.
    assert trainer._compare_pair_ft_metrics_for_selection(
        {
            "pair_ranking_accuracy": 0.80001,
            "unique_score_count": 13.0,
            "same_state_score_gap": 0.02,
            "score_spread": 0.02,
        },
        {
            "pair_ranking_accuracy": 0.80000,
            "unique_score_count": 12.0,
            "same_state_score_gap": 0.03,
            "score_spread": 0.03,
        },
    ) == 1
    # unique tied: same-state gap decides.
    assert trainer._compare_pair_ft_metrics_for_selection(
        {
            "pair_ranking_accuracy": 0.80000,
            "unique_score_count": 12.0,
            "same_state_score_gap": 0.021,
            "score_spread": 0.020,
        },
        {
            "pair_ranking_accuracy": 0.80000,
            "unique_score_count": 12.0,
            "same_state_score_gap": 0.020,
            "score_spread": 0.030,
        },
    ) == 1


def test_world_predictor_does_not_apply_confidence_risk_scaling():
    class _DummyTensorizer:
        def tensorize_inference(self, history_scene, action_id):
            _ = (history_scene, action_id)
            return {"candidate_action": torch.tensor([int(action_id)], dtype=torch.long)}

    class _DummyModel(torch.nn.Module):
        def forward(self, batch):
            _ = batch
            return {
                "traj": torch.zeros((1, 1, 1, 1, 5), dtype=torch.float32),
                "confidence": torch.tensor([[[2.0, 1.0, 0.5]]], dtype=torch.float32),
                "risk_type_logits": torch.zeros((1, 3), dtype=torch.float32),
                "risk_score_logit": torch.tensor([1.3862944], dtype=torch.float32),
                "risk_score": torch.tensor([0.8], dtype=torch.float32),
                "uncertainty": torch.tensor([0.1], dtype=torch.float32),
            }

    predictor = WorldModelPredictor(
        model=_DummyModel(),
        tensorizer=_DummyTensorizer(),
        device=torch.device("cpu"),
    )
    prediction = predictor.predict(history_scene=_history_scene()[:2], action_id=4)
    assert prediction.aggregated_risk == pytest.approx(0.8)
    assert all(item.p_overall == pytest.approx(0.8) for item in prediction.modality_risk)


def test_world_pair_losses_use_logit_space_when_available():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu")

    class _LogitOnlyModel(torch.nn.Module):
        def forward(self, batch):
            actions = batch["candidate_action"]
            action_float = actions.to(torch.float32)
            logits = torch.where(action_float == 4.0, torch.full_like(action_float, -0.2), torch.full_like(action_float, 0.3))
            probs = torch.full_like(action_float, 0.5)
            bs = int(actions.shape[0])
            return {
                "traj": torch.zeros((bs, 1, 1, 1, 5), dtype=torch.float32),
                "confidence": torch.zeros((bs, 1, 2), dtype=torch.float32),
                "risk_type_logits": torch.zeros((bs, 3), dtype=torch.float32),
                "risk_score_logit": logits,
                "risk_score": probs,
                "uncertainty": torch.zeros((bs,), dtype=torch.float32),
            }

    trainer.model = _LogitOnlyModel().to(trainer.device)
    pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "trusted_for_spread": True},
    )
    ranking_loss, spread_loss, resolution_loss, diagnostics = trainer._compute_pair_losses([pair], enable_resolution=False)
    assert float(ranking_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(spread_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(resolution_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["stage4_aux_active_pair_count"] == 0


def test_world_pair_losses_tie_aware_skips_small_target_gap_for_ranking():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2, pair_ft_tie_gap_epsilon=0.01), history_steps=2, device="cpu")

    class _LogitGapModel(torch.nn.Module):
        def forward(self, batch):
            actions = batch["candidate_action"]
            action_float = actions.to(torch.float32)
            logits = torch.where(action_float == 4.0, torch.full_like(action_float, -0.1), torch.full_like(action_float, -0.08))
            probs = torch.sigmoid(logits)
            bs = int(actions.shape[0])
            return {
                "traj": torch.zeros((bs, 1, 1, 1, 5), dtype=torch.float32),
                "confidence": torch.zeros((bs, 1, 2), dtype=torch.float32),
                "risk_type_logits": torch.zeros((bs, 3), dtype=torch.float32),
                "risk_score_logit": logits,
                "risk_score": probs,
                "uncertainty": torch.zeros((bs,), dtype=torch.float32),
            }

    trainer.model = _LogitGapModel().to(trainer.device)
    pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
        meta={"target_risk_a": 0.500, "target_risk_b": 0.505, "trusted_for_spread": False},
    )
    ranking_loss, spread_loss, resolution_loss, diagnostics = trainer._compute_pair_losses([pair], enable_resolution=False)
    assert float(ranking_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(spread_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert float(resolution_loss.item()) == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["stage4_aux_active_pair_count"] == 0


def test_world_resolution_loss_only_applies_to_stage4_aux_pairs():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(
        config=WorldModelConfig(
            hidden_dim=64,
            future_steps=2,
            multimodal=2,
            pair_ft_resolution_min_score_gap=0.03,
            pair_ft_resolution_min_logit_gap=0.14,
            pair_ft_resolution_loss_weight=0.02,
        ),
        history_steps=2,
        device="cpu",
    )

    class _NarrowGapModel(torch.nn.Module):
        def forward(self, batch):
            actions = batch["candidate_action"]
            action_float = actions.to(torch.float32)
            logits = torch.where(action_float == 4.0, torch.full_like(action_float, 0.00), torch.full_like(action_float, 0.05))
            probs = torch.sigmoid(logits)
            bs = int(actions.shape[0])
            return {
                "traj": torch.zeros((bs, 1, 1, 1, 5), dtype=torch.float32),
                "confidence": torch.zeros((bs, 1, 2), dtype=torch.float32),
                "risk_type_logits": torch.zeros((bs, 3), dtype=torch.float32),
                "risk_score_logit": logits,
                "risk_score": probs,
                "uncertainty": torch.zeros((bs,), dtype=torch.float32),
            }

    trainer.model = _NarrowGapModel().to(trainer.device)
    stage4_aux_pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage4_candidate_rank",
        weight=1.0,
        meta={
            "target_risk_a": 0.30,
            "target_risk_b": 0.80,
            "trusted_for_spread": False,
            "stage4_aux_candidate": True,
        },
    )
    stage1_pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
        meta={
            "target_risk_a": 0.30,
            "target_risk_b": 0.80,
            "trusted_for_spread": False,
            "stage4_aux_candidate": True,
        },
    )

    _, _, resolution_on, diag_on = trainer._compute_pair_losses([stage4_aux_pair], enable_resolution=True)
    _, _, resolution_off, diag_off = trainer._compute_pair_losses([stage1_pair], enable_resolution=True)
    _, _, resolution_disabled, _ = trainer._compute_pair_losses([stage4_aux_pair], enable_resolution=False)

    assert float(resolution_on.item()) > 0.0
    assert diag_on["stage4_aux_active_pair_count"] == 1
    assert diag_on["resolution_space"] == "score"
    assert diag_on["pair_ft_resolution_min_score_gap"] == pytest.approx(0.03, abs=1e-6)
    assert diag_on["ignored_legacy_logit_margin"] == pytest.approx(0.14, abs=1e-6)
    assert diag_on["stage4_aux_logit_gap_mean"] == pytest.approx(0.05, abs=1e-6)
    assert diag_on["stage4_aux_score_gap_mean"] == pytest.approx(0.012497, abs=1e-4)
    assert diag_on["stage4_aux_score_gap_p50"] == pytest.approx(diag_on["stage4_aux_score_gap_mean"], abs=1e-9)
    assert diag_on["stage4_aux_score_gap_p90"] == pytest.approx(diag_on["stage4_aux_score_gap_mean"], abs=1e-9)
    assert diag_on["stage4_aux_below_score_margin_count"] == 1
    assert diag_on["stage4_aux_below_score_margin_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert diag_on["stage4_aux_below_margin_count"] == 1
    assert diag_on["stage4_aux_below_margin_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert float(resolution_off.item()) == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_active_pair_count"] == 0
    assert diag_off["stage4_aux_logit_gap_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_score_gap_mean"] == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_score_gap_p50"] == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_score_gap_p90"] == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_below_score_margin_count"] == 0
    assert diag_off["stage4_aux_below_score_margin_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert diag_off["stage4_aux_below_margin_count"] == 0
    assert diag_off["stage4_aux_below_margin_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert float(resolution_disabled.item()) == pytest.approx(0.0, abs=1e-6)


def test_world_evaluate_pairs_restores_train_mode():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(config=WorldModelConfig(hidden_dim=64, future_steps=2, multimodal=2), history_steps=2, device="cpu")
    trainer.model.train()
    pair = RiskPairSample(
        history_scene=_history_scene()[:2],
        action_a=4,
        action_b=3,
        preferred_action=4,
        source="stage1_probe_same_state",
        weight=1.0,
        meta={"target_risk_a": 0.1, "target_risk_b": 0.8, "hard_negative": True},
    )
    _ = trainer.evaluate_pairs([pair])
    assert trainer.model.training is True


def test_world_pair_ft_source_mix_uses_configured_stage4_mix_frequency():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(
        config=WorldModelConfig(
            hidden_dim=64,
            future_steps=2,
            multimodal=2,
            pair_finetune=False,
            pair_ft_stage4_mix_every_n_steps=2,
        ),
        history_steps=2,
        device="cpu",
    )
    _ = trainer.fine_tune_pairs(pair_samples=[], replay_samples=[])
    report = dict(trainer.last_pair_ft_report or {})
    source_mix = dict(report.get("world_pair_ft_source_mix", {}) or {})
    assert source_mix.get("stage4_mix_every_n_steps") == 2


def test_world_pair_ft_source_mix_stage4_mix_frequency_is_clamped_to_one():
    from safe_rl.models.world_model import WorldModelTrainer

    trainer = WorldModelTrainer(
        config=WorldModelConfig(
            hidden_dim=64,
            future_steps=2,
            multimodal=2,
            pair_finetune=False,
            pair_ft_stage4_mix_every_n_steps=0,
        ),
        history_steps=2,
        device="cpu",
    )
    _ = trainer.fine_tune_pairs(pair_samples=[], replay_samples=[])
    report = dict(trainer.last_pair_ft_report or {})
    source_mix = dict(report.get("world_pair_ft_source_mix", {}) or {})
    assert source_mix.get("stage4_mix_every_n_steps") == 1
