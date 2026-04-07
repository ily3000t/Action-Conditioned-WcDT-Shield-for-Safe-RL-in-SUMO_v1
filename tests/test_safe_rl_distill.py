import numpy as np
import pytest

from safe_rl.buffer import InterventionBuffer
from safe_rl.config.config import DistillConfig
from safe_rl.models.features import BASE_FEATURE_DIM


def _supervision_sample(feature_idx: int, raw_action: int, final_action: int, intervened: bool):
    feature = [0.0] * int(BASE_FEATURE_DIM)
    feature[feature_idx % int(BASE_FEATURE_DIM)] = 1.0
    return {
        "history_feature": feature,
        "raw_action": int(raw_action),
        "final_action": int(final_action),
        "intervened": bool(intervened),
        "raw_risk": 0.5,
        "final_risk": 0.2,
    }


def test_should_distill_uses_supervision_sample_count():
    pytest.importorskip("torch")
    from safe_rl.rl.distill import PolicyDistiller

    config = DistillConfig(trigger_buffer_size=5, batch_size=4, learning_rate=1e-3, epochs=1, interval_steps=100)
    distiller = PolicyDistiller(config=config, device="cpu")
    buffer = InterventionBuffer()
    samples = [_supervision_sample(i, raw_action=i % 9, final_action=i % 9, intervened=False) for i in range(5)]
    assert distiller.should_distill(buffer, supervision_samples=samples) is True
    assert distiller.should_distill(buffer, supervision_samples=samples[:4]) is False


def test_distill_mixed_supervision_avoids_single_action_collapse():
    torch = pytest.importorskip("torch")
    from safe_rl.rl.distill import PolicyDistiller

    np.random.seed(0)
    torch.manual_seed(0)

    config = DistillConfig(trigger_buffer_size=10, batch_size=32, learning_rate=1e-3, epochs=4, interval_steps=100)
    distiller = PolicyDistiller(config=config, device="cpu")
    buffer = InterventionBuffer()

    supervision_samples = []
    for idx in range(120):
        action = idx % 9
        supervision_samples.append(_supervision_sample(idx, raw_action=action, final_action=action, intervened=False))
    for idx in range(120, 180):
        supervision_samples.append(_supervision_sample(idx, raw_action=5, final_action=1, intervened=True))

    policy = distiller.distill(buffer, supervision_samples=supervision_samples, tb_writer=None)
    report = dict(distiller.last_training_report)

    assert policy is not None
    assert report["source"] == "stage4_supervision_dataset"
    assert report["sample_count"] == len(supervision_samples)
    assert report["non_intervened_sample_count"] > 0
    assert report["intervened_sample_count"] > 0
    assert report["label_entropy"] > 0.1
    assert report["pred_entropy"] > 0.05
    assert report["pred_top1_ratio"] < 0.99
