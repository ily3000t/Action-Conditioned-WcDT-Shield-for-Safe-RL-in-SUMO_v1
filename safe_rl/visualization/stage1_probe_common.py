import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from safe_rl.config import load_safe_rl_config
from safe_rl.data.types import RiskPairSample
from safe_rl.models.world_model import WorldModelTrainer

GAP_BIN_SPECS: Tuple[Tuple[float, float, str], ...] = (
    (0.008, 0.012, "gap_0.008_0.012"),
    (0.012, 0.020, "gap_0.012_0.020"),
    (0.020, float("inf"), "gap_0.020_inf"),
)


def resolve_run_dir(run_id: str, run_root: Path) -> Path:
    run_dir = Path(run_root) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def load_stage1_probe_pairs(run_dir: Path) -> List[RiskPairSample]:
    pair_path = Path(run_dir) / "datasets" / "pairs_stage1_probe.pkl"
    if not pair_path.exists():
        raise FileNotFoundError(f"Stage1 probe pairs not found: {pair_path}")
    with pair_path.open("rb") as f:
        payload = pickle.load(f)
    return list(payload or [])


def load_stage2_report(run_dir: Path) -> Dict[str, Any]:
    report_path = Path(run_dir) / "reports" / "stage2_training_report.json"
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as f:
        return dict(json.load(f) or {})


def _iter_chunks(values: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _gap_bin_label(target_gap: float) -> str:
    gap = float(target_gap)
    for low, high, label in GAP_BIN_SPECS:
        if gap >= low and gap < high:
            return label
    return "gap_out_of_range"


def _predict_scores_for_actions(
    trainer: WorldModelTrainer,
    action_items: Sequence[Tuple[List[Any], int]],
    batch_size: int,
) -> List[float]:
    results: List[float] = []
    model = trainer.model
    tensorizer = trainer.tensorizer
    device = trainer.device
    model.eval()
    with torch.no_grad():
        for chunk in _iter_chunks(list(action_items), max(1, int(batch_size))):
            batch = tensorizer.tensorize_state_action_batch(chunk)
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(batch)
            chunk_scores = output["risk_score"].detach().cpu().numpy().astype(float).tolist()
            results.extend(float(item) for item in chunk_scores)
    return results


def build_stage1_pair_records(
    run_dir: Path,
    config_path: str = "safe_rl/config/default_safe_rl.yaml",
    device: str = "cpu",
    batch_size: int = 256,
    max_pairs: Optional[int] = None,
) -> List[Dict[str, Any]]:
    config = load_safe_rl_config(config_path)
    pair_samples = load_stage1_probe_pairs(run_dir)
    if max_pairs is not None and int(max_pairs) > 0:
        pair_samples = pair_samples[: int(max_pairs)]
    model_path = Path(run_dir) / "models" / "world_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"World model checkpoint not found: {model_path}")

    trainer = WorldModelTrainer(
        config=config.world_model,
        history_steps=int(config.sim.history_steps),
        device=device,
    )
    trainer.load(str(model_path))

    actions_a = [(sample.history_scene, int(sample.action_a)) for sample in pair_samples]
    actions_b = [(sample.history_scene, int(sample.action_b)) for sample in pair_samples]
    scores_a = _predict_scores_for_actions(trainer, actions_a, batch_size=batch_size)
    scores_b = _predict_scores_for_actions(trainer, actions_b, batch_size=batch_size)

    records: List[Dict[str, Any]] = []
    for sample, score_a, score_b in zip(pair_samples, scores_a, scores_b):
        meta = dict(sample.meta or {})
        target_risk_a = _safe_float(meta.get("target_risk_a", 0.0), 0.0)
        target_risk_b = _safe_float(meta.get("target_risk_b", 0.0), 0.0)
        target_gap = _safe_float(meta.get("target_gap", abs(target_risk_a - target_risk_b)), abs(target_risk_a - target_risk_b))
        predicted_gap = abs(float(score_a) - float(score_b))
        risk_midpoint = 0.5 * (target_risk_a + target_risk_b)
        records.append(
            {
                "episode_id": str(meta.get("episode_id", "")),
                "step_index": int(meta.get("step_index", -1)),
                "history_hash": str(meta.get("history_hash", "")),
                "source": str(sample.source),
                "action_a": int(sample.action_a),
                "action_b": int(sample.action_b),
                "preferred_action": int(sample.preferred_action),
                "target_risk_a": float(target_risk_a),
                "target_risk_b": float(target_risk_b),
                "target_gap": float(target_gap),
                "risk_midpoint": float(risk_midpoint),
                "predicted_score_a": float(score_a),
                "predicted_score_b": float(score_b),
                "predicted_score_gap": float(predicted_gap),
                "trusted_for_spread": bool(meta.get("trusted_for_spread", False)),
                "boundary_pair": bool(meta.get("boundary_pair", False)),
                "target_gap_bin": _gap_bin_label(target_gap),
            }
        )
    return records


def summarize_records(records: Sequence[Dict[str, Any]], score_hist_bins: int = 20) -> Dict[str, Any]:
    score_bins = max(2, int(score_hist_bins))

    def _summary(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        values = list(items or [])
        count = len(values)
        if count == 0:
            return {
                "count": 0,
                "target_gap_p50": 0.0,
                "target_gap_p90": 0.0,
                "predicted_gap_p50": 0.0,
                "predicted_gap_p90": 0.0,
                "predicted_gap_mean": 0.0,
                "compression_mean": 0.0,
                "unique_score_count": 0,
                "score_hist_bin_occupancy": 0,
                "score_hist_bins_total": score_bins,
            }

        target_gaps = np.array([_safe_float(item.get("target_gap", 0.0)) for item in values], dtype=np.float32)
        pred_gaps = np.array([_safe_float(item.get("predicted_score_gap", 0.0)) for item in values], dtype=np.float32)
        score_values = []
        for item in values:
            score_values.append(_safe_float(item.get("predicted_score_a", 0.0)))
            score_values.append(_safe_float(item.get("predicted_score_b", 0.0)))
        rounded = np.round(np.array(score_values, dtype=np.float32), 6)
        unique_score_count = int(np.unique(rounded).shape[0])
        hist = np.histogram(np.clip(np.array(score_values, dtype=np.float32), 0.0, 1.0), bins=np.linspace(0.0, 1.0, score_bins + 1))[0]
        occupancy = int(np.sum(hist > 0))
        compression = target_gaps - pred_gaps
        return {
            "count": int(count),
            "target_gap_p50": float(np.quantile(target_gaps, 0.50)),
            "target_gap_p90": float(np.quantile(target_gaps, 0.90)),
            "predicted_gap_p50": float(np.quantile(pred_gaps, 0.50)),
            "predicted_gap_p90": float(np.quantile(pred_gaps, 0.90)),
            "predicted_gap_mean": float(np.mean(pred_gaps)),
            "compression_mean": float(np.mean(compression)),
            "unique_score_count": int(unique_score_count),
            "score_hist_bin_occupancy": int(occupancy),
            "score_hist_bins_total": int(score_bins),
        }

    all_records = list(records or [])
    trusted_records = [item for item in all_records if bool(item.get("trusted_for_spread", False))]
    by_gap_bin: Dict[str, Any] = {}
    for _, _, label in GAP_BIN_SPECS:
        bin_records = [item for item in all_records if str(item.get("target_gap_bin", "")) == label]
        by_gap_bin[label] = _summary(bin_records)

    return {
        "overall": _summary(all_records),
        "trusted_for_spread": _summary(trusted_records),
        "by_gap_bin": by_gap_bin,
    }
