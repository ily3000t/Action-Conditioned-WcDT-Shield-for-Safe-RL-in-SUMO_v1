import pytest
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from safe_rl.config.config import LightRiskConfig, WorldModelConfig
from safe_rl.data.pair_dataset import RiskPairDataset, collate_risk_pairs
from safe_rl.data.types import RiskLabels, RiskPairSample, SceneState, VehicleState, ActionConditionedSample
from safe_rl.models.light_risk_model import LightRiskMLP, LightRiskTrainer
from safe_rl.models.world_model import ActionConditionedWorldModel, SceneTensorizer


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
