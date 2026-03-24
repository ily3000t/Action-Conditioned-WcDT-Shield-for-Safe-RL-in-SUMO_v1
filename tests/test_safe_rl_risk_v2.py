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
