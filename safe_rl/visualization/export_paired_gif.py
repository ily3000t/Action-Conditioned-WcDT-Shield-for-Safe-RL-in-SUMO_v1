import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from safe_rl.visualization.replay_episode import load_pair_payload, render_pair_gif
except Exception:  # pragma: no cover
    from replay_episode import load_pair_payload, render_pair_gif  # type: ignore


def export_paired_gifs(
    run_id: str,
    trace_dir: Optional[str] = None,
    run_root: Path = Path("safe_rl_output/runs"),
    anomaly_cases_path: Optional[Path] = None,
    top_k: int = 10,
    output_root: Path = Path("qualitative_results/stage5_replays"),
    fps: int = 6,
    mode: str = "auto",
) -> Dict[str, Any]:
    run_dir = Path(run_root) / str(run_id)
    reports_dir = run_dir / "reports"
    resolved_trace_dir = _resolve_trace_dir(reports_dir=reports_dir, trace_dir=trace_dir)
    candidate_pairs = _resolve_pair_files(trace_dir=resolved_trace_dir, anomaly_cases_path=anomaly_cases_path)
    selected_pairs = candidate_pairs[: max(0, int(top_k))]

    output_dir = Path(output_root) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for pair_path in selected_pairs:
        try:
            payload = load_pair_payload(pair_path)
            pair_index = int(payload.get("pair_index", -1))
            seed = int(payload.get("seed", -1))
            output_path = output_dir / f"pair_{pair_index:03d}_seed_{seed}.gif"
            render_pair_gif(payload, output_path=output_path, fps=fps, mode=mode)
            exported.append(
                {
                    "pair_file": str(pair_path),
                    "pair_index": pair_index,
                    "seed": seed,
                    "gif_path": str(output_path),
                    "distilled_unavailable": bool(payload.get("distilled_unavailable", True)),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive path for malformed traces
            failed.append({"pair_file": str(pair_path), "error": str(exc)})

    summary = {
        "run_id": str(run_id),
        "trace_dir": str(resolved_trace_dir),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_top_k": int(top_k),
        "candidate_pair_count": int(len(candidate_pairs)),
        "selected_pair_count": int(len(selected_pairs)),
        "exported_count": int(len(exported)),
        "failed_count": int(len(failed)),
        "exported": exported,
        "failed": failed,
    }
    index_path = output_dir / "export_index.json"
    index_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["index_path"] = str(index_path)
    return summary


def _resolve_trace_dir(reports_dir: Path, trace_dir: Optional[str]) -> Path:
    if trace_dir:
        explicit = Path(trace_dir)
        if explicit.exists():
            return explicit
        candidate = reports_dir / str(trace_dir)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Trace dir not found: {trace_dir}")

    preferred = ["stage5_trace_capture_default", "stage5_trace_capture_cost", "shield_trace"]
    for name in preferred:
        candidate = reports_dir / name
        if candidate.exists() and (candidate / "trace_summary.json").exists():
            return candidate

    discovered = sorted(
        [
            item
            for item in reports_dir.iterdir()
            if item.is_dir() and (item / "trace_summary.json").exists()
        ],
        key=lambda path: path.name,
    )
    if not discovered:
        raise FileNotFoundError(f"No trace directory with trace_summary.json found under {reports_dir}")
    return discovered[0]


def _resolve_pair_files(trace_dir: Path, anomaly_cases_path: Optional[Path]) -> List[Path]:
    if anomaly_cases_path is not None:
        with Path(anomaly_cases_path).open("r", encoding="utf-8") as f:
            payload = dict(json.load(f) or {})
        result: List[Path] = []
        for case in list(payload.get("cases", []) or []):
            pair_file = Path(str(case.get("pair_file", "")))
            if not pair_file.is_absolute():
                pair_file = trace_dir / pair_file
            if pair_file.exists():
                result.append(pair_file)
        return result

    trace_summary_path = trace_dir / "trace_summary.json"
    if trace_summary_path.exists():
        with trace_summary_path.open("r", encoding="utf-8") as f:
            trace_summary = dict(json.load(f) or {})
        pair_files = []
        for raw_path in list(trace_summary.get("pair_files", []) or []):
            path = Path(raw_path)
            if not path.is_absolute():
                path = trace_dir / path
            if path.exists():
                pair_files.append(path)
        if pair_files:
            return pair_files

    return sorted(trace_dir.glob("pair_*_seed_*.json"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paired stage5 trace files into GIF replays.")
    parser.add_argument("--run-id", required=True, help="SAFE_RL run id")
    parser.add_argument("--trace-dir", default="", help="Trace directory name under run reports, or absolute path")
    parser.add_argument("--run-root", default="safe_rl_output/runs", help="Run root directory")
    parser.add_argument("--anomaly-cases", default="", help="Optional anomaly_cases.json path")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K pair files to export")
    parser.add_argument("--output-root", default="qualitative_results/stage5_replays", help="GIF output root")
    parser.add_argument("--fps", type=int, default=6, help="GIF frames per second")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "dual", "triple", "baseline", "shielded", "distilled"),
        help="Replay mode passed through to replay renderer.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = export_paired_gifs(
        run_id=args.run_id,
        trace_dir=args.trace_dir or None,
        run_root=Path(args.run_root),
        anomaly_cases_path=Path(args.anomaly_cases) if args.anomaly_cases else None,
        top_k=int(args.top_k),
        output_root=Path(args.output_root),
        fps=int(args.fps),
        mode=args.mode,
    )
    print(f"[export_paired_gif] exported={summary['exported_count']} failed={summary['failed_count']}")
    print(f"[export_paired_gif] index={summary['index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
