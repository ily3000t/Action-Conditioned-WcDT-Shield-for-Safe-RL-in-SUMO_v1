import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - fallback when Pillow is unavailable
    Image = None
    ImageDraw = None
    ImageFont = None


TRACK_ORDER = ("baseline", "shielded", "distilled")
DEFAULT_FRAME_SIZE = (1280, 720)
DEFAULT_FPS = 6
_TWO_PI = 2.0 * math.pi


def normalize_heading_to_degrees(value: Any) -> float:
    heading = _safe_float(value, 0.0)
    if abs(heading) > _TWO_PI:
        return heading
    return float(math.degrees(heading))


def normalize_step(step: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(step or {})
    payload.setdefault("step_index", 0)
    payload.setdefault("raw_action", -1)
    payload.setdefault("final_action", -1)
    payload.setdefault("executed_action", payload.get("final_action", -1))
    payload.setdefault("replacement_happened", False)
    payload.setdefault("fallback_used", False)
    payload.setdefault("raw_risk", 0.0)
    payload.setdefault("final_risk", payload.get("raw_risk", 0.0))
    payload.setdefault("risk_reduction", _safe_float(payload.get("raw_risk", 0.0)) - _safe_float(payload.get("final_risk", 0.0)))
    payload.setdefault("constraint_reason", "")
    payload.setdefault("block_trigger", "")
    payload.setdefault("min_ttc", payload.get("ttc", 0.0))
    payload.setdefault("ttc", payload.get("min_ttc", 0.0))
    payload.setdefault("min_distance", 0.0)
    payload.setdefault("reward", 0.0)
    payload.setdefault("task_reward", payload.get("reward", 0.0))
    payload.setdefault("ego_speed", 0.0)
    payload.setdefault("collision", False)
    payload.setdefault("candidate_evaluations", [])
    payload.setdefault("history_scene", [])
    return payload


def build_aligned_steps(
    baseline_steps: Sequence[Dict[str, Any]],
    shielded_steps: Sequence[Dict[str, Any]],
    distilled_steps: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    max_steps = max(len(baseline_steps), len(shielded_steps), len(distilled_steps))
    aligned: List[Dict[str, Any]] = []
    for idx in range(max_steps):
        aligned.append(
            {
                "step_index": idx,
                "baseline": baseline_steps[idx] if idx < len(baseline_steps) else None,
                "shielded": shielded_steps[idx] if idx < len(shielded_steps) else None,
                "distilled": distilled_steps[idx] if idx < len(distilled_steps) else None,
            }
        )
    return aligned


def normalize_pair_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    baseline_steps = [normalize_step(item) for item in list(normalized.get("baseline_steps", []) or [])]
    shielded_steps = [normalize_step(item) for item in list(normalized.get("shielded_steps", []) or [])]
    distilled_steps_raw = list(normalized.get("distilled_steps", []) or [])
    distilled_steps = [normalize_step(item) for item in distilled_steps_raw]
    distilled_unavailable = bool(normalized.get("distilled_unavailable", len(distilled_steps) == 0))

    normalized["baseline_steps"] = baseline_steps
    normalized["shielded_steps"] = shielded_steps
    normalized["distilled_steps"] = distilled_steps
    normalized["distilled_unavailable"] = distilled_unavailable
    normalized.setdefault("distilled_episode_id", "")

    aligned_steps = list(normalized.get("aligned_steps", []) or [])
    if aligned_steps:
        patched_aligned = []
        for item in aligned_steps:
            row = dict(item or {})
            row.setdefault("step_index", len(patched_aligned))
            row.setdefault("baseline", None)
            row.setdefault("shielded", None)
            row.setdefault("distilled", None)
            patched_aligned.append(row)
        normalized["aligned_steps"] = patched_aligned
    else:
        normalized["aligned_steps"] = build_aligned_steps(
            baseline_steps=baseline_steps,
            shielded_steps=shielded_steps,
            distilled_steps=distilled_steps,
        )
    return normalized


def load_pair_payload(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return normalize_pair_payload(payload)


def render_pair_gif(
    pair_payload: Dict[str, Any],
    output_path: Path,
    mode: str = "auto",
    fps: int = DEFAULT_FPS,
) -> Path:
    normalized = normalize_pair_payload(pair_payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if Image is None or ImageDraw is None or ImageFont is None:
        _write_placeholder_gif(output_path)
        return output_path

    visible_tracks = _resolve_visible_tracks(mode=mode, payload=normalized)
    steps_by_track = {
        "baseline": list(normalized.get("baseline_steps", []) or []),
        "shielded": list(normalized.get("shielded_steps", []) or []),
        "distilled": list(normalized.get("distilled_steps", []) or []),
    }
    frame_count = max((len(steps_by_track[name]) for name in visible_tracks), default=1)
    frame_count = max(frame_count, 1)
    bounds = _scene_bounds(normalized, visible_tracks)

    frames: List[Image.Image] = []
    for idx in range(frame_count):
        frame = Image.new("RGB", DEFAULT_FRAME_SIZE, (249, 250, 252))
        draw = ImageDraw.Draw(frame)
        _draw_header(draw, normalized, idx, frame_count)
        _draw_track_panels(
            draw=draw,
            payload=normalized,
            visible_tracks=visible_tracks,
            steps_by_track=steps_by_track,
            frame_index=idx,
            bounds=bounds,
        )
        timeline_track = "shielded" if "shielded" in visible_tracks else visible_tracks[0]
        _draw_timeline_panel(
            draw=draw,
            steps=steps_by_track.get(timeline_track, []),
            frame_index=idx,
            panel_box=(24, 500, DEFAULT_FRAME_SIZE[0] - 24, DEFAULT_FRAME_SIZE[1] - 16),
        )
        frames.append(frame)

    duration_ms = max(1, int(1000 / max(1, int(fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


def _resolve_visible_tracks(mode: str, payload: Dict[str, Any]) -> List[str]:
    normalized_mode = str(mode or "auto").strip().lower()
    distilled_unavailable = bool(payload.get("distilled_unavailable", True))

    if normalized_mode in TRACK_ORDER:
        if normalized_mode == "distilled" and distilled_unavailable:
            return ["baseline", "shielded"]
        return [normalized_mode]
    if normalized_mode == "triple":
        return ["baseline", "shielded", "distilled"] if not distilled_unavailable else ["baseline", "shielded"]
    if normalized_mode == "dual":
        return ["baseline", "shielded"]
    return ["baseline", "shielded", "distilled"] if not distilled_unavailable else ["baseline", "shielded"]


def _draw_header(draw: "ImageDraw.ImageDraw", payload: Dict[str, Any], frame_index: int, frame_count: int):
    title_parts = [
        f"pair={int(payload.get('pair_index', -1))}",
        f"seed={int(payload.get('seed', -1))}",
        f"frame={frame_index + 1}/{frame_count}",
    ]
    if bool(payload.get("distilled_unavailable", False)):
        title_parts.append("distilled_unavailable=true")
    draw.text((24, 12), " | ".join(title_parts), fill=(30, 30, 35), font=ImageFont.load_default())


def _draw_track_panels(
    draw: "ImageDraw.ImageDraw",
    payload: Dict[str, Any],
    visible_tracks: Sequence[str],
    steps_by_track: Dict[str, Sequence[Dict[str, Any]]],
    frame_index: int,
    bounds: Tuple[float, float, float, float],
):
    panel_top = 44
    panel_bottom = 484
    panel_left = 24
    panel_right = DEFAULT_FRAME_SIZE[0] - 24
    count = max(1, len(visible_tracks))
    panel_width = (panel_right - panel_left) / count
    for idx, track_name in enumerate(visible_tracks):
        left = int(panel_left + idx * panel_width)
        right = int(panel_left + (idx + 1) * panel_width)
        draw.rectangle((left, panel_top, right - 8, panel_bottom), outline=(155, 165, 180), width=1, fill=(255, 255, 255))
        draw.text((left + 8, panel_top + 6), track_name, fill=(25, 25, 30), font=ImageFont.load_default())
        track_steps = list(steps_by_track.get(track_name, []) or [])
        step = track_steps[min(frame_index, len(track_steps) - 1)] if track_steps else normalize_step({})
        map_box = (left + 8, panel_top + 24, right - 16, panel_top + 276)
        stats_box = (left + 8, panel_top + 282, right - 16, panel_bottom - 8)
        _draw_scene_map(draw, step, bounds, map_box)
        _draw_step_stats(draw, step, stats_box)


def _draw_scene_map(
    draw: "ImageDraw.ImageDraw",
    step: Dict[str, Any],
    bounds: Tuple[float, float, float, float],
    panel_box: Tuple[int, int, int, int],
):
    x_min, x_max, y_min, y_max = bounds
    scene = _extract_latest_scene(step)
    lane_polylines = list(scene.get("lane_polylines", []) or [])
    if lane_polylines:
        for polyline in lane_polylines:
            points = []
            for point in list(polyline or []):
                if len(point) < 2:
                    continue
                px, py = _project(point[0], point[1], x_min, x_max, y_min, y_max, panel_box)
                points.append((px, py))
            if len(points) >= 2:
                draw.line(points, fill=(210, 212, 220), width=2)
    else:
        # fallback lane reference when lane geometry is unavailable
        left, top, right, bottom = panel_box
        lane_mid = int((top + bottom) / 2)
        draw.line((left, lane_mid, right, lane_mid), fill=(220, 224, 232), width=2)
        draw.line((left, lane_mid - 38, right, lane_mid - 38), fill=(233, 236, 241), width=1)
        draw.line((left, lane_mid + 38, right, lane_mid + 38), fill=(233, 236, 241), width=1)

    vehicles = list(scene.get("vehicles", []) or [])
    for vehicle in vehicles:
        _draw_vehicle(draw, vehicle, panel_box, bounds)


def _draw_vehicle(
    draw: "ImageDraw.ImageDraw",
    vehicle: Dict[str, Any],
    panel_box: Tuple[int, int, int, int],
    bounds: Tuple[float, float, float, float],
):
    vid = str(vehicle.get("vehicle_id", ""))
    color = (52, 110, 219)
    if "ego" in vid.lower():
        color = (235, 86, 86)
    elif "merge" in vid.lower():
        color = (239, 154, 64)
    elif "lead" in vid.lower():
        color = (72, 160, 92)

    x, y = _project(
        vehicle.get("x", 0.0),
        vehicle.get("y", 0.0),
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        panel_box,
    )
    heading_deg = normalize_heading_to_degrees(vehicle.get("heading", 0.0))
    length = max(10, int(_safe_float(vehicle.get("length", 4.8), 4.8) * 2.4))
    width = max(6, int(_safe_float(vehicle.get("width", 2.0), 2.0) * 2.0))

    half_l = int(length / 2)
    half_w = int(width / 2)
    draw.rectangle((x - half_l, y - half_w, x + half_l, y + half_w), fill=color, outline=(20, 20, 20), width=1)

    rad = math.radians(heading_deg)
    tip_x = int(x + math.cos(rad) * max(6, half_l))
    tip_y = int(y - math.sin(rad) * max(6, half_l))
    draw.line((x, y, tip_x, tip_y), fill=(15, 15, 15), width=2)
    draw.text((x + half_l + 2, y - half_w - 2), vid[:8], fill=(20, 20, 22), font=ImageFont.load_default())


def _draw_step_stats(draw: "ImageDraw.ImageDraw", step: Dict[str, Any], panel_box: Tuple[int, int, int, int]):
    left, top, right, _ = panel_box
    text_lines = [
        f"raw_action={int(step.get('raw_action', -1))} final_action={int(step.get('final_action', -1))}",
        f"shield_called={bool(step.get('replacement_happened', False) or step.get('block_trigger', ''))} replaced={bool(step.get('replacement_happened', False))}",
        f"raw_risk={_safe_float(step.get('raw_risk', 0.0)):.3f} final_risk={_safe_float(step.get('final_risk', 0.0)):.3f}",
        f"block_trigger={str(step.get('block_trigger', '') or 'none')}",
        f"constraint_reason={str(step.get('constraint_reason', '') or 'none')}",
        f"min_ttc={_safe_float(step.get('min_ttc', step.get('ttc', 0.0))):.3f} min_distance={_safe_float(step.get('min_distance', 0.0)):.3f}",
        f"speed={_safe_float(step.get('ego_speed', 0.0)):.3f}",
    ]
    y = top
    for line in text_lines:
        draw.text((left, y), line, fill=(25, 25, 30), font=ImageFont.load_default())
        y += 15
        if y > panel_box[3] - 12:
            break
    draw.line((left, panel_box[1] - 4, right, panel_box[1] - 4), fill=(235, 236, 239), width=1)


def _draw_timeline_panel(
    draw: "ImageDraw.ImageDraw",
    steps: Sequence[Dict[str, Any]],
    frame_index: int,
    panel_box: Tuple[int, int, int, int],
):
    left, top, right, bottom = panel_box
    draw.rectangle(panel_box, outline=(155, 165, 180), fill=(255, 255, 255), width=1)
    draw.text((left + 8, top + 6), "timeline: risk_raw / risk_final / block_trigger / replacement / min_ttc / min_distance", fill=(20, 20, 26), font=ImageFont.load_default())
    series_map = _build_timeline_series(steps)

    chart_top = top + 24
    chart_bottom = bottom - 22
    chart_left = left + 10
    chart_right = right - 10
    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(225, 228, 234), width=1)
    if not steps:
        return

    colors = {
        "risk_raw": (220, 53, 69),
        "risk_final": (63, 136, 197),
        "block_trigger": (130, 130, 130),
        "replacement_happened": (75, 160, 92),
        "min_ttc": (239, 154, 64),
        "min_distance": (130, 90, 180),
    }
    legend_x = chart_left + 8
    legend_y = chart_top + 4
    for key in ("risk_raw", "risk_final", "block_trigger", "replacement_happened", "min_ttc", "min_distance"):
        values = series_map[key]
        if not values:
            continue
        min_v = min(values)
        max_v = max(values)
        scale = max(1e-6, max_v - min_v)
        points = []
        for idx, value in enumerate(values):
            x = chart_left + int((chart_right - chart_left) * idx / max(1, len(values) - 1))
            y = chart_bottom - int((chart_bottom - chart_top) * (value - min_v) / scale)
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=colors[key], width=2)
        else:
            draw.point(points[0], fill=colors[key])
        draw.text((legend_x, legend_y), key, fill=colors[key], font=ImageFont.load_default())
        legend_y += 12

    cursor_x = chart_left + int((chart_right - chart_left) * min(frame_index, len(steps) - 1) / max(1, len(steps) - 1))
    draw.line((cursor_x, chart_top, cursor_x, chart_bottom), fill=(70, 70, 70), width=1)


def _build_timeline_series(steps: Sequence[Dict[str, Any]]) -> Dict[str, List[float]]:
    normalized = [normalize_step(item) for item in list(steps or [])]
    return {
        "risk_raw": [_safe_float(item.get("raw_risk", 0.0)) for item in normalized],
        "risk_final": [_safe_float(item.get("final_risk", 0.0)) for item in normalized],
        "block_trigger": [0.0 if str(item.get("block_trigger", "")).strip() in ("", "none") else 1.0 for item in normalized],
        "replacement_happened": [1.0 if bool(item.get("replacement_happened", False)) else 0.0 for item in normalized],
        "min_ttc": [_safe_float(item.get("min_ttc", item.get("ttc", 0.0))) for item in normalized],
        "min_distance": [_safe_float(item.get("min_distance", 0.0)) for item in normalized],
    }


def _extract_latest_scene(step: Dict[str, Any]) -> Dict[str, Any]:
    history_scene = list(step.get("history_scene", []) or [])
    if history_scene:
        return dict(history_scene[-1] or {})
    return {"vehicles": [], "lane_polylines": []}


def _scene_bounds(payload: Dict[str, Any], visible_tracks: Sequence[str]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for track in visible_tracks:
        for step in list(payload.get(f"{track}_steps", []) or []):
            scene = _extract_latest_scene(step)
            for vehicle in list(scene.get("vehicles", []) or []):
                xs.append(_safe_float(vehicle.get("x", 0.0)))
                ys.append(_safe_float(vehicle.get("y", 0.0)))
            for polyline in list(scene.get("lane_polylines", []) or []):
                for point in list(polyline or []):
                    if len(point) >= 2:
                        xs.append(_safe_float(point[0], 0.0))
                        ys.append(_safe_float(point[1], 0.0))
    if not xs or not ys:
        return (0.0, 120.0, -15.0, 15.0)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if abs(x_max - x_min) < 1.0:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1.0:
        y_max = y_min + 1.0
    margin_x = (x_max - x_min) * 0.08
    margin_y = (y_max - y_min) * 0.18
    return (x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y)


def _project(
    x: Any,
    y: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    panel_box: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    px = _safe_float(x, 0.0)
    py = _safe_float(y, 0.0)
    left, top, right, bottom = panel_box
    width = max(1.0, float(right - left))
    height = max(1.0, float(bottom - top))
    nx = (px - x_min) / max(1e-6, x_max - x_min)
    ny = (py - y_min) / max(1e-6, y_max - y_min)
    sx = left + int(nx * width)
    sy = bottom - int(ny * height)
    return sx, sy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _write_placeholder_gif(path: Path):
    path.write_bytes(
        b"GIF89a"
        b"\x01\x00\x01\x00"
        b"\x80\x00\x00"
        b"\x00\x00\x00"
        b"\xff\xff\xff"
        b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"
        b"\x02\x02\x44\x01\x00"
        b"\x3b"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one paired stage5 trace file into a GIF replay.")
    parser.add_argument("--pair-file", required=True, help="Path to pair_<idx>_seed_<n>.json")
    parser.add_argument("--output", required=True, help="Output GIF path")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="GIF frames per second")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "dual", "triple", "baseline", "shielded", "distilled"),
        help="Replay mode. auto: 3-track when distilled exists, otherwise 2-track.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = load_pair_payload(Path(args.pair_file))
    output_path = render_pair_gif(payload, output_path=Path(args.output), mode=args.mode, fps=args.fps)
    print(f"[replay_episode] wrote GIF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
