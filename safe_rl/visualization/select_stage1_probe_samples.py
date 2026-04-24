import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from safe_rl.visualization.stage1_probe_common import (
    GAP_BIN_SPECS,
    build_stage1_pair_records,
    resolve_run_dir,
)


def _record_sort_key(item: Dict[str, Any]):
    compression = float(item.get("target_gap", 0.0)) - float(item.get("predicted_score_gap", 0.0))
    return (
        -compression,
        -float(item.get("target_gap", 0.0)),
        float(item.get("predicted_score_gap", 0.0)),
        str(item.get("episode_id", "")),
        int(item.get("step_index", -1)),
    )


def _record_id(item: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(item.get("episode_id", "")),
            str(item.get("step_index", -1)),
            str(item.get("action_a", -1)),
            str(item.get("action_b", -1)),
            str(item.get("history_hash", "")),
        ]
    )


def _select_top_k_by_gap_bins(records: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    all_records = list(records or [])
    if top_k <= 0 or not all_records:
        return []

    ordered_labels = [label for _, _, label in GAP_BIN_SPECS]
    quota_base = max(1, int(top_k) // max(1, len(ordered_labels)))
    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()

    for label in ordered_labels:
        subset = [item for item in all_records if str(item.get("target_gap_bin", "")) == label]
        subset = sorted(subset, key=_record_sort_key)
        for item in subset[:quota_base]:
            key = _record_id(item)
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selected.append(item)

    if len(selected) < int(top_k):
        remaining = sorted(all_records, key=_record_sort_key)
        for item in remaining:
            key = _record_id(item)
            if key in selected_ids:
                continue
            selected_ids.add(key)
            selected.append(item)
            if len(selected) >= int(top_k):
                break

    return selected[: int(top_k)]


def select_stage1_probe_samples(
    run_id: str,
    run_root: Path = Path("safe_rl_output/runs"),
    output_root: Path = Path("qualitative_results/stage1_probe_diagnostics"),
    config_path: str = "safe_rl/config/default_safe_rl.yaml",
    device: str = "cpu",
    batch_size: int = 256,
    max_pairs: Optional[int] = None,
    top_k: int = 30,
) -> Dict[str, Any]:
    run_dir = resolve_run_dir(run_id=run_id, run_root=run_root)
    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_stage1_pair_records(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
        max_pairs=max_pairs,
    )
    selected = _select_top_k_by_gap_bins(records=records, top_k=top_k)

    per_bin_count: Dict[str, int] = {}
    cases: List[Dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        label = str(item.get("target_gap_bin", "gap_out_of_range"))
        per_bin_count[label] = int(per_bin_count.get(label, 0) + 1)
        compression = float(item.get("target_gap", 0.0)) - float(item.get("predicted_score_gap", 0.0))
        case = {
            "rank": int(rank),
            "gap_bin": label,
            "episode_id": str(item.get("episode_id", "")),
            "step_index": int(item.get("step_index", -1)),
            "history_hash": str(item.get("history_hash", "")),
            "source": str(item.get("source", "")),
            "action_a": int(item.get("action_a", -1)),
            "action_b": int(item.get("action_b", -1)),
            "preferred_action": int(item.get("preferred_action", -1)),
            "target_risk_a": float(item.get("target_risk_a", 0.0)),
            "target_risk_b": float(item.get("target_risk_b", 0.0)),
            "target_gap": float(item.get("target_gap", 0.0)),
            "predicted_score_a": float(item.get("predicted_score_a", 0.0)),
            "predicted_score_b": float(item.get("predicted_score_b", 0.0)),
            "predicted_score_gap": float(item.get("predicted_score_gap", 0.0)),
            "compression_gap": float(compression),
            "risk_midpoint": float(item.get("risk_midpoint", 0.0)),
            "trusted_for_spread": bool(item.get("trusted_for_spread", False)),
            "boundary_pair": bool(item.get("boundary_pair", False)),
        }
        cases.append(case)

    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_pairs": int(max_pairs) if max_pairs is not None else None,
        "top_k_requested": int(top_k),
        "records_seen": int(len(records)),
        "selected_count": int(len(cases)),
        "gap_bins": [
            {"low": float(low), "high": (float(high) if high != float("inf") else "inf"), "label": str(label)}
            for low, high, label in GAP_BIN_SPECS
        ],
        "selected_count_by_gap_bin": per_bin_count,
        "cases": cases,
    }
    output_path = output_dir / "sample_cases.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select representative Stage1 probe samples by target gap bins.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--output-root", default="qualitative_results/stage1_probe_diagnostics", help="Output directory root")
    parser.add_argument("--config", default="safe_rl/config/default_safe_rl.yaml", help="Config path for world model reconstruction")
    parser.add_argument("--device", default="cpu", help="Torch device used for inference")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size for pair scoring")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for pair count (0 means all)")
    parser.add_argument("--top-k", type=int, default=30, help="Total number of sample cases to export")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = select_stage1_probe_samples(
        run_id=args.run_id,
        run_root=Path(args.run_root),
        output_root=Path(args.output_root),
        config_path=str(args.config),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_pairs=(None if int(args.max_pairs) <= 0 else int(args.max_pairs)),
        top_k=int(args.top_k),
    )
    print(f"[select_stage1_probe_samples] selected={payload['selected_count']} total={payload['records_seen']}")
    print(f"[select_stage1_probe_samples] output={payload['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
