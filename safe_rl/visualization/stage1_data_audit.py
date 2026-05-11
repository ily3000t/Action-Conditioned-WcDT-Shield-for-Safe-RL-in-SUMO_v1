import argparse
import datetime as dt
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from safe_rl.data.types import RiskPairSample


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _resolve_run_dir(run_id: str, run_root: Path) -> Path:
    run_dir = Path(run_root) / str(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def _load_stage1_pairs(path: Path) -> List[RiskPairSample]:
    if not path.exists():
        raise FileNotFoundError(f"Stage1 probe pair dataset not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    return list(payload or [])


def _load_stage1_probe_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Stage1 probe event report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = dict(json.load(f) or {})
    return list(payload.get("events", []) or [])


def _load_optional_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return dict(json.load(f) or {})


def _ecdf(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(list(values or []), dtype=np.float32))
    if arr.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    y = np.arange(1, arr.size + 1, dtype=np.float32) / float(arr.size)
    return arr, y


def _bin_metrics(values: Sequence[float], bins: int) -> Dict[str, Any]:
    bin_count = max(2, int(bins))
    arr = np.clip(np.asarray(list(values or []), dtype=np.float32), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bin_count + 1, dtype=np.float32)
    hist = np.histogram(arr, bins=edges)[0].astype(np.float32)
    total = float(np.sum(hist))
    occupancy = hist / total if total > 0 else np.zeros_like(hist)
    entropy = float(-(occupancy * np.log(np.clip(occupancy, 1e-12, 1.0))).sum()) if occupancy.size > 0 else 0.0
    effective = float(np.exp(entropy))
    return {
        "hist": hist.astype(int).tolist(),
        "occupancy": occupancy.astype(float).tolist(),
        "nonempty": int(np.sum(hist > 0)),
        "entropy": float(entropy),
        "effective": float(effective),
    }


def _bin_index(value: float, bins: int) -> int:
    k = max(2, int(bins))
    clipped = min(1.0, max(0.0, float(value)))
    return min(k - 1, int(np.floor(clipped * k)))


def _plot_target_risk_hist_ecdf(
    risk_all: Sequence[float],
    risk_trusted: Sequence[float],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    if risk_all:
        axes[0].hist(list(risk_all), bins=40, alpha=0.70, color="#2f6cad", label="pair_all")
    if risk_trusted:
        axes[0].hist(list(risk_trusted), bins=40, alpha=0.60, color="#d07c34", label="pair_trusted")
    if not risk_all and not risk_trusted:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("Stage1 target_risk histogram")
    axes[0].set_xlabel("target_risk")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.2, linestyle="--", linewidth=0.5)
    axes[0].legend(loc="best")

    x_all, y_all = _ecdf(risk_all)
    x_trusted, y_trusted = _ecdf(risk_trusted)
    if x_all.size > 0:
        axes[1].plot(x_all, y_all, color="#2f6cad", label="pair_all")
    if x_trusted.size > 0:
        axes[1].plot(x_trusted, y_trusted, color="#d07c34", label="pair_trusted")
    if x_all.size == 0 and x_trusted.size == 0:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("Stage1 target_risk ECDF")
    axes[1].set_xlabel("target_risk")
    axes[1].set_ylabel("cdf")
    axes[1].grid(alpha=0.2, linestyle="--", linewidth=0.5)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_bin_occupancy(occupancy: Sequence[float], output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    occ = list(occupancy or [])
    x = np.arange(len(occ))
    ax.bar(x, occ, color="#2f6cad", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("bin")
    ax.set_ylabel("mass")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_bin_occupancy_compare(
    occupancy_all: Sequence[float],
    occupancy_trusted: Sequence[float],
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    a = np.asarray(list(occupancy_all or []), dtype=np.float32)
    b = np.asarray(list(occupancy_trusted or []), dtype=np.float32)
    k = int(max(a.size, b.size))
    if k <= 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        if a.size < k:
            a = np.pad(a, (0, k - a.size))
        if b.size < k:
            b = np.pad(b, (0, k - b.size))
        x = np.arange(k)
        w = 0.42
        ax.bar(x - w / 2.0, a, width=w, color="#2f6cad", alpha=0.80, label="pair_all")
        ax.bar(x + w / 2.0, b, width=w, color="#d07c34", alpha=0.80, label="pair_trusted")
        ax.legend(loc="best")
    ax.set_title("target_risk bin occupancy: trusted vs all")
    ax.set_xlabel("bin")
    ax.set_ylabel("mass")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_hist(values: Sequence[float], output_path: Path, title: str, xlabel: str, bins: int = 40):
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
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_pair_bin_heatmap(bin_indices_a: Sequence[int], bin_indices_b: Sequence[int], bins: int, output_path: Path):
    k = max(2, int(bins))
    heat = np.zeros((k, k), dtype=np.float32)
    for a, b in zip(bin_indices_a, bin_indices_b):
        if 0 <= int(a) < k and 0 <= int(b) < k:
            heat[int(a), int(b)] += 1.0
    if float(np.sum(heat)) > 0.0:
        heat = heat / float(np.sum(heat))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heat, origin="lower", cmap="viridis")
    ax.set_title("pair bin heatmap (risk_a_bin x risk_b_bin)")
    ax.set_xlabel("risk_b_bin")
    ax.set_ylabel("risk_a_bin")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _safe_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.quantile(arr, float(q)))


def generate_stage1_data_audit(
    run_id: str,
    run_root: Path = Path("safe_rl_output/runs"),
    output_root: Path = Path("qualitative_results/stage1_data_audit"),
    bins: int = 16,
) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_id=run_id, run_root=run_root)
    pair_path = run_dir / "datasets" / "pairs_stage1_probe.pkl"
    events_path = run_dir / "reports" / "stage1_probe_events.json"
    summary_path = run_dir / "reports" / "stage1_probe_summary.json"

    pairs = _load_stage1_pairs(pair_path)
    events = _load_stage1_probe_events(events_path)
    stage1_summary = _load_optional_json(summary_path)

    risk_all: List[float] = []
    risk_trusted: List[float] = []
    risk_a_values: List[float] = []
    risk_b_values: List[float] = []
    pair_target_gaps: List[float] = []
    candidate_risks: List[float] = []
    missing_target_risk_count = 0
    preferred_a = 0
    preferred_b = 0
    tie_like = 0

    pair_bin_idx_a: List[int] = []
    pair_bin_idx_b: List[int] = []

    for sample in pairs:
        meta = dict(sample.meta or {})
        raw_a = meta.get("target_risk_a", None)
        raw_b = meta.get("target_risk_b", None)
        if raw_a is None or raw_b is None:
            missing_target_risk_count += 1
        risk_a = _safe_float(raw_a, 0.0)
        risk_b = _safe_float(raw_b, 0.0)
        risk_a_values.append(risk_a)
        risk_b_values.append(risk_b)
        risk_all.extend([risk_a, risk_b])
        if bool(meta.get("trusted_for_spread", False)):
            risk_trusted.extend([risk_a, risk_b])
        target_gap = _safe_float(meta.get("target_gap", abs(risk_a - risk_b)), abs(risk_a - risk_b))
        pair_target_gaps.append(target_gap)

        pair_bin_idx_a.append(_bin_index(risk_a, bins=bins))
        pair_bin_idx_b.append(_bin_index(risk_b, bins=bins))

        if int(sample.preferred_action) == int(sample.action_a):
            preferred_a += 1
        elif int(sample.preferred_action) == int(sample.action_b):
            preferred_b += 1
        else:
            tie_like += 1

    for event in events:
        for candidate in list(event.get("candidates", []) or []):
            candidate_risks.append(
                _safe_float(
                    candidate.get("target_proxy_risk", candidate.get("overall_proxy_risk", 0.0)),
                    0.0,
                )
            )

    risk_all = [min(1.0, max(0.0, float(v))) for v in risk_all]
    risk_trusted = [min(1.0, max(0.0, float(v))) for v in risk_trusted]
    candidate_risks = [min(1.0, max(0.0, float(v))) for v in candidate_risks]

    all_bin = _bin_metrics(risk_all, bins=bins)
    trusted_bin = _bin_metrics(risk_trusted, bins=bins)
    candidate_bin = _bin_metrics(candidate_risks, bins=bins)

    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "target_risk_hist_ecdf": str(output_dir / "target_risk_hist_ecdf.png"),
        "target_risk_bin_occupancy": str(output_dir / "target_risk_bin_occupancy.png"),
        "target_risk_trusted_vs_all_bin_occupancy": str(output_dir / "target_risk_trusted_vs_all_bin_occupancy.png"),
        "candidate_risk_hist": str(output_dir / "candidate_risk_hist.png"),
        "candidate_bin_occupancy": str(output_dir / "candidate_bin_occupancy.png"),
        "pair_target_gap_hist": str(output_dir / "pair_target_gap_hist.png"),
        "pair_bin_heatmap": str(output_dir / "pair_bin_heatmap.png"),
    }

    _plot_target_risk_hist_ecdf(
        risk_all=risk_all,
        risk_trusted=risk_trusted,
        output_path=Path(files["target_risk_hist_ecdf"]),
    )
    _plot_bin_occupancy(
        occupancy=all_bin["occupancy"],
        output_path=Path(files["target_risk_bin_occupancy"]),
        title="target_risk 16-bin occupancy (pair_all)",
    )
    _plot_bin_occupancy_compare(
        occupancy_all=all_bin["occupancy"],
        occupancy_trusted=trusted_bin["occupancy"],
        output_path=Path(files["target_risk_trusted_vs_all_bin_occupancy"]),
    )
    _plot_hist(
        values=candidate_risks,
        output_path=Path(files["candidate_risk_hist"]),
        title="Stage1 candidate overall_proxy_risk histogram",
        xlabel="overall_proxy_risk",
        bins=40,
    )
    _plot_bin_occupancy(
        occupancy=candidate_bin["occupancy"],
        output_path=Path(files["candidate_bin_occupancy"]),
        title="candidate overall_proxy_risk bin occupancy",
    )
    _plot_hist(
        values=pair_target_gaps,
        output_path=Path(files["pair_target_gap_hist"]),
        title="Stage1 pair target_gap histogram",
        xlabel="target_gap",
        bins=50,
    )
    _plot_pair_bin_heatmap(
        bin_indices_a=pair_bin_idx_a,
        bin_indices_b=pair_bin_idx_b,
        bins=bins,
        output_path=Path(files["pair_bin_heatmap"]),
    )

    summary: Dict[str, Any] = {
        "pair_count": int(len(pairs)),
        "candidate_count": int(len(candidate_risks)),
        "preferred_a": int(preferred_a),
        "preferred_b": int(preferred_b),
        "tie_like": int(tie_like),
        "missing_target_risk_count": int(missing_target_risk_count),
        "pairs_dropped_small_gap": int(stage1_summary.get("pairs_dropped_small_gap", 0)),
        "pairs_kept_strong_signal": int(stage1_summary.get("pairs_kept_strong_signal", 0)),
        "pairs_capped_by_budget": int(stage1_summary.get("pairs_capped_by_budget", 0)),
        "pairs_boundary_appended": int(stage1_summary.get("pairs_boundary_appended", 0)),
        "target_risk_q01": _safe_quantile(risk_all, 0.01),
        "target_risk_q10": _safe_quantile(risk_all, 0.10),
        "target_risk_q50": _safe_quantile(risk_all, 0.50),
        "target_risk_q90": _safe_quantile(risk_all, 0.90),
        "target_risk_q99": _safe_quantile(risk_all, 0.99),
        "pair_all_bin_nonempty": int(all_bin["nonempty"]),
        "pair_all_bin_effective": float(all_bin["effective"]),
        "pair_all_bin_entropy": float(all_bin["entropy"]),
        "pair_all_bin_occupancy": list(all_bin["occupancy"]),
        "pair_trusted_bin_nonempty": int(trusted_bin["nonempty"]),
        "pair_trusted_bin_effective": float(trusted_bin["effective"]),
        "pair_trusted_bin_entropy": float(trusted_bin["entropy"]),
        "pair_trusted_bin_occupancy": list(trusted_bin["occupancy"]),
        "candidate_bin_nonempty": int(candidate_bin["nonempty"]),
        "candidate_bin_effective": float(candidate_bin["effective"]),
        "candidate_bin_entropy": float(candidate_bin["entropy"]),
        "candidate_bin_occupancy": list(candidate_bin["occupancy"]),
        "target_gap_q50": _safe_quantile(pair_target_gaps, 0.50),
        "target_gap_q90": _safe_quantile(pair_target_gaps, 0.90),
    }

    summary_payload: Dict[str, Any] = {
        "run_id": str(run_id),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bins": int(max(2, int(bins))),
        "input_paths": {
            "pairs_stage1_probe": str(pair_path),
            "stage1_probe_events": str(events_path),
            "stage1_probe_summary": str(summary_path),
        },
        "files": files,
        **summary,
    }

    output_path = output_dir / "audit_summary.json"
    output_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": str(run_id),
        "output_dir": str(output_dir),
        "output_path": str(output_path),
        "files": files,
        "summary": summary_payload,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage1 probe data distribution and pair structure.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--output-root", default="qualitative_results/stage1_data_audit", help="Audit output root")
    parser.add_argument("--bins", type=int, default=16, help="Number of bins used for occupancy statistics")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = generate_stage1_data_audit(
        run_id=args.run_id,
        run_root=Path(args.run_root),
        output_root=Path(args.output_root),
        bins=int(args.bins),
    )
    print(f"[stage1_data_audit] run_id={args.run_id}")
    print(f"[stage1_data_audit] output={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
