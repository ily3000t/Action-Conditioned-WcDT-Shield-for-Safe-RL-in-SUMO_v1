import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_rl.visualization.stage1_probe_common import (
    GAP_BIN_SPECS,
    build_stage1_pair_records,
    load_stage2_report,
    resolve_run_dir,
    summarize_records,
)


def _plot_hist(values: Sequence[float], path: Path, title: str, xlabel: str, bins: int = 40):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if values:
        ax.hist(list(values), bins=max(2, int(bins)), color="#2f6cad", alpha=0.85, edgecolor="white", linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_scatter(
    xs: Sequence[float],
    ys: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if xs and ys:
        ax.scatter(xs, ys, s=12, alpha=0.45, color="#0c7c59", edgecolors="none")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_overlay_hist(
    current_values: Sequence[float],
    baseline_values: Sequence[float],
    path: Path,
    title: str,
    xlabel: str,
    bins: int = 40,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if baseline_values:
        ax.hist(
            list(baseline_values),
            bins=max(2, int(bins)),
            alpha=0.45,
            color="#b35c1e",
            edgecolor="white",
            linewidth=0.4,
            label="baseline",
        )
    if current_values:
        ax.hist(
            list(current_values),
            bins=max(2, int(bins)),
            alpha=0.45,
            color="#2f6cad",
            edgecolor="white",
            linewidth=0.4,
            label="current",
        )
    if not current_values and not baseline_values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _extract_stage2_snapshot(stage2_report: Dict[str, Any]) -> Dict[str, Any]:
    world_metrics = dict(((stage2_report or {}).get("pair_finetune_metrics", {}) or {}).get("world", {}) or {})
    health = dict(((stage2_report or {}).get("stage2_pair_source_health", {}) or {}).get("model_quality", {}) or {})
    gate = dict((stage2_report or {}).get("model_quality_gate_metrics", {}) or {})
    epoch_metrics = list(world_metrics.get("epoch_metrics", []) or [])
    return {
        "selection_path": str(world_metrics.get("selection_path", "")),
        "selection_reason": str(world_metrics.get("selection_reason", "")),
        "best_epoch_stage1_unique": float(world_metrics.get("best_epoch_stage1_unique", 0.0) or 0.0),
        "best_epoch_eval_unique": float(world_metrics.get("best_epoch_eval_unique", 0.0) or 0.0),
        "stage1_probe_unique_after": float(
            dict(world_metrics.get("stage1_probe_unique_score_count_before_after", {}) or {}).get("after", 0.0) or 0.0
        ),
        "world_unique_score_count": float(gate.get("world_unique_score_count", 0.0) or 0.0),
        "model_quality_status": str(health.get("status", "")),
        "model_quality_metric_source": str(health.get("metric_source", "")),
        "final_epoch_stage1_probe_below_score_margin_fraction": float(
            (epoch_metrics[-1] or {}).get("stage1_probe_below_score_margin_fraction", 0.0)
            if epoch_metrics
            else 0.0
        ),
    }


def _build_delta(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    def _delta(a: Any, b: Any) -> Optional[float]:
        try:
            return float(a) - float(b)
        except Exception:
            return None

    delta = {
        "overall": {},
        "trusted_for_spread": {},
        "by_gap_bin": {},
    }
    for group in ("overall", "trusted_for_spread"):
        cur = dict(current.get(group, {}) or {})
        base = dict(baseline.get(group, {}) or {})
        delta[group] = {key: _delta(cur.get(key), base.get(key)) for key in set(cur.keys()) | set(base.keys())}
    current_bins = dict(current.get("by_gap_bin", {}) or {})
    baseline_bins = dict(baseline.get("by_gap_bin", {}) or {})
    for label in set(current_bins.keys()) | set(baseline_bins.keys()):
        cur = dict(current_bins.get(label, {}) or {})
        base = dict(baseline_bins.get(label, {}) or {})
        delta["by_gap_bin"][label] = {key: _delta(cur.get(key), base.get(key)) for key in set(cur.keys()) | set(base.keys())}
    return delta


def _render_record_plots(records: Sequence[Dict[str, Any]], output_dir: Path, prefix: str) -> Dict[str, str]:
    saved: Dict[str, str] = {}
    values_target = [float(item.get("target_gap", 0.0)) for item in records]
    values_pred = [float(item.get("predicted_score_gap", 0.0)) for item in records]
    values_mid = [float(item.get("risk_midpoint", 0.0)) for item in records]
    values_scores = [
        float(value)
        for item in records
        for value in (item.get("predicted_score_a", 0.0), item.get("predicted_score_b", 0.0))
    ]

    overall_target_hist = output_dir / f"{prefix}_target_gap_hist_overall.png"
    _plot_hist(values_target, overall_target_hist, f"{prefix} target_gap distribution (overall)", "target_gap", bins=40)
    saved["target_gap_hist_overall"] = str(overall_target_hist)

    overall_target_vs_pred = output_dir / f"{prefix}_target_vs_pred_gap_overall.png"
    _plot_scatter(
        values_target,
        values_pred,
        overall_target_vs_pred,
        f"{prefix} target_gap vs predicted_score_gap (overall)",
        "target_gap",
        "predicted_score_gap",
    )
    saved["target_vs_pred_gap_overall"] = str(overall_target_vs_pred)

    overall_mid_vs_pred = output_dir / f"{prefix}_risk_midpoint_vs_pred_gap_overall.png"
    _plot_scatter(
        values_mid,
        values_pred,
        overall_mid_vs_pred,
        f"{prefix} risk_midpoint vs predicted_score_gap (overall)",
        "risk_midpoint(target_risk_a,target_risk_b)",
        "predicted_score_gap",
    )
    saved["risk_midpoint_vs_pred_gap_overall"] = str(overall_mid_vs_pred)

    overall_score_hist = output_dir / f"{prefix}_risk_score_hist_overall.png"
    _plot_hist(values_scores, overall_score_hist, f"{prefix} risk_score distribution (overall)", "risk_score", bins=40)
    saved["risk_score_hist_overall"] = str(overall_score_hist)

    for _, _, label in GAP_BIN_SPECS:
        subset = [item for item in records if str(item.get("target_gap_bin", "")) == label]
        sub_target = [float(item.get("target_gap", 0.0)) for item in subset]
        sub_pred = [float(item.get("predicted_score_gap", 0.0)) for item in subset]
        sub_mid = [float(item.get("risk_midpoint", 0.0)) for item in subset]
        sub_scores = [
            float(value)
            for item in subset
            for value in (item.get("predicted_score_a", 0.0), item.get("predicted_score_b", 0.0))
        ]

        p_target = output_dir / f"{prefix}_target_gap_hist_{label}.png"
        _plot_hist(sub_target, p_target, f"{prefix} target_gap distribution ({label})", "target_gap", bins=30)
        saved[f"target_gap_hist_{label}"] = str(p_target)

        p_target_pred = output_dir / f"{prefix}_target_vs_pred_gap_{label}.png"
        _plot_scatter(
            sub_target,
            sub_pred,
            p_target_pred,
            f"{prefix} target_gap vs predicted_score_gap ({label})",
            "target_gap",
            "predicted_score_gap",
        )
        saved[f"target_vs_pred_gap_{label}"] = str(p_target_pred)

        p_mid_pred = output_dir / f"{prefix}_risk_midpoint_vs_pred_gap_{label}.png"
        _plot_scatter(
            sub_mid,
            sub_pred,
            p_mid_pred,
            f"{prefix} risk_midpoint vs predicted_score_gap ({label})",
            "risk_midpoint(target_risk_a,target_risk_b)",
            "predicted_score_gap",
        )
        saved[f"risk_midpoint_vs_pred_gap_{label}"] = str(p_mid_pred)

        p_score_hist = output_dir / f"{prefix}_risk_score_hist_{label}.png"
        _plot_hist(sub_scores, p_score_hist, f"{prefix} risk_score distribution ({label})", "risk_score", bins=30)
        saved[f"risk_score_hist_{label}"] = str(p_score_hist)

    return saved


def generate_stage1_probe_diagnostics(
    run_id: str,
    baseline_run_id: Optional[str] = None,
    run_root: Path = Path("safe_rl_output/runs"),
    output_root: Path = Path("qualitative_results/stage1_probe_diagnostics"),
    config_path: str = "safe_rl/config/default_safe_rl.yaml",
    device: str = "cpu",
    batch_size: int = 256,
    max_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    run_dir = resolve_run_dir(run_id=run_id, run_root=run_root)
    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_records = build_stage1_pair_records(
        run_dir=run_dir,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
        max_pairs=max_pairs,
    )
    current_summary = summarize_records(current_records)
    current_stage2_snapshot = _extract_stage2_snapshot(load_stage2_report(run_dir))
    current_plots = _render_record_plots(current_records, output_dir=output_dir, prefix="current")

    baseline_payload: Dict[str, Any] = {}
    comparison_delta: Dict[str, Any] = {}
    comparison_plots: Dict[str, str] = {}
    if baseline_run_id:
        baseline_dir = resolve_run_dir(run_id=baseline_run_id, run_root=run_root)
        baseline_records = build_stage1_pair_records(
            run_dir=baseline_dir,
            config_path=config_path,
            device=device,
            batch_size=batch_size,
            max_pairs=max_pairs,
        )
        baseline_summary = summarize_records(baseline_records)
        baseline_snapshot = _extract_stage2_snapshot(load_stage2_report(baseline_dir))
        baseline_plots = _render_record_plots(baseline_records, output_dir=output_dir, prefix="baseline")
        comparison_delta = _build_delta(current_summary, baseline_summary)
        baseline_pred_gaps = [float(item.get("predicted_score_gap", 0.0)) for item in baseline_records]
        current_pred_gaps = [float(item.get("predicted_score_gap", 0.0)) for item in current_records]
        baseline_scores = [
            float(value)
            for item in baseline_records
            for value in (item.get("predicted_score_a", 0.0), item.get("predicted_score_b", 0.0))
        ]
        current_scores = [
            float(value)
            for item in current_records
            for value in (item.get("predicted_score_a", 0.0), item.get("predicted_score_b", 0.0))
        ]
        overlay_pred = output_dir / "current_vs_baseline_predicted_gap_hist.png"
        _plot_overlay_hist(
            current_values=current_pred_gaps,
            baseline_values=baseline_pred_gaps,
            path=overlay_pred,
            title="predicted_score_gap distribution (current vs baseline)",
            xlabel="predicted_score_gap",
            bins=40,
        )
        overlay_score = output_dir / "current_vs_baseline_risk_score_hist.png"
        _plot_overlay_hist(
            current_values=current_scores,
            baseline_values=baseline_scores,
            path=overlay_score,
            title="risk_score distribution (current vs baseline)",
            xlabel="risk_score",
            bins=40,
        )
        comparison_plots = {
            "current_vs_baseline_predicted_gap_hist": str(overlay_pred),
            "current_vs_baseline_risk_score_hist": str(overlay_score),
        }
        baseline_payload = {
            "run_id": str(baseline_run_id),
            "record_count": int(len(baseline_records)),
            "summary": baseline_summary,
            "stage2_snapshot": baseline_snapshot,
            "plots": baseline_plots,
        }

    payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "baseline_run_id": str(baseline_run_id or ""),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_pairs": int(max_pairs) if max_pairs is not None else None,
        "gap_bins": [
            {"low": float(low), "high": (float(high) if np.isfinite(high) else "inf"), "label": str(label)}
            for low, high, label in GAP_BIN_SPECS
        ],
        "current": {
            "record_count": int(len(current_records)),
            "summary": current_summary,
            "stage2_snapshot": current_stage2_snapshot,
            "plots": current_plots,
        },
        "baseline": baseline_payload,
        "comparison_delta": comparison_delta,
        "comparison_plots": comparison_plots,
    }
    output_path = output_dir / "diagnostics_summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["output_path"] = str(output_path)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage1 probe target-vs-predicted gap diagnostics.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--baseline-run-id", default="", help="Optional baseline run id")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--output-root", default="qualitative_results/stage1_probe_diagnostics", help="Diagnostics output root")
    parser.add_argument("--config", default="safe_rl/config/default_safe_rl.yaml", help="Config path for world model reconstruction")
    parser.add_argument("--device", default="cpu", help="Torch device used for inference")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size for pair scoring")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for pair count (0 means all)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = generate_stage1_probe_diagnostics(
        run_id=args.run_id,
        baseline_run_id=(args.baseline_run_id or None),
        run_root=Path(args.run_root),
        output_root=Path(args.output_root),
        config_path=str(args.config),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_pairs=(None if int(args.max_pairs) <= 0 else int(args.max_pairs)),
    )
    print(f"[stage1_probe_diagnostics] run_id={args.run_id} records={result['current']['record_count']}")
    print(f"[stage1_probe_diagnostics] output={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
