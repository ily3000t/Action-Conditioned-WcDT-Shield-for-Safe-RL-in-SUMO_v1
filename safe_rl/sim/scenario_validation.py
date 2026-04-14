import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_VEHICLE_LENGTH = 4.8
_DEFAULT_MIN_GAP = 2.5


def _resolve_cfg_path(cfg_path: Path) -> Path:
    return Path(cfg_path).resolve()


def _read_cfg_inputs(cfg_path: Path) -> Dict[str, List[Path]]:
    root = ET.parse(cfg_path).getroot()
    input_node = root.find("./input")
    outputs: Dict[str, List[Path]] = {
        "net_files": [],
        "route_files": [],
    }
    if input_node is None:
        return outputs

    def _collect(tag_name: str) -> List[Path]:
        node = input_node.find(tag_name)
        if node is None:
            return []
        raw_value = str(node.attrib.get("value", "")).strip()
        if not raw_value:
            return []
        items = []
        for part in raw_value.split(","):
            part = part.strip()
            if not part:
                continue
            items.append((cfg_path.parent / part).resolve())
        return items

    outputs["net_files"] = _collect("net-file")
    outputs["route_files"] = _collect("route-files")
    return outputs


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_scenario_assets(cfg_path: Path) -> List[Path]:
    cfg_path = _resolve_cfg_path(cfg_path)
    inputs = _read_cfg_inputs(cfg_path)
    assets: List[Path] = [cfg_path]
    assets.extend(inputs.get("net_files", []))
    assets.extend(inputs.get("route_files", []))

    for net_file in inputs.get("net_files", []):
        net_str = str(net_file)
        if not net_str.endswith(".net.xml"):
            continue
        prefix = net_str[:-8]
        for suffix in (".nod.xml", ".edg.xml", ".con.xml"):
            plain_path = Path(prefix + suffix).resolve()
            if plain_path.exists():
                assets.append(plain_path)

    dedup: List[Path] = []
    seen = set()
    for path in assets:
        resolved = Path(path).resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(resolved)
    return dedup


def build_scenario_fingerprint(cfg_path: Path, scenario_variant: str = "") -> Dict[str, Any]:
    cfg_path = _resolve_cfg_path(cfg_path)
    scenario_assets: List[Dict[str, Any]] = []
    for asset in resolve_scenario_assets(cfg_path):
        item: Dict[str, Any] = {
            "path": str(asset),
            "exists": bool(asset.exists()),
        }
        if asset.exists():
            item["sha256"] = _path_sha256(asset)
            item["size_bytes"] = int(asset.stat().st_size)
        scenario_assets.append(item)

    hasher = hashlib.sha256()
    for item in scenario_assets:
        hasher.update(str(item.get("path", "")).encode("utf-8"))
        hasher.update(b"|")
        hasher.update(str(item.get("sha256", "MISSING")).encode("utf-8"))
        hasher.update(b";")

    cfg_name = cfg_path.name
    scenario_name = cfg_name[:-8] if cfg_name.endswith(".sumocfg") else cfg_path.stem
    return {
        "scenario_name": str(scenario_name),
        "scenario_variant": str(scenario_variant or ""),
        "scenario_asset_hash": hasher.hexdigest(),
        "scenario_assets": scenario_assets,
    }


def _parse_nodes(nod_path: Path) -> Dict[str, Tuple[float, float]]:
    root = ET.parse(nod_path).getroot()
    nodes: Dict[str, Tuple[float, float]] = {}
    for node in root.findall("./node"):
        node_id = str(node.attrib.get("id", "")).strip()
        if not node_id:
            continue
        x = float(node.attrib.get("x", 0.0))
        y = float(node.attrib.get("y", 0.0))
        nodes[node_id] = (x, y)
    return nodes


def _parse_edges(edg_path: Path) -> Dict[str, Dict[str, Any]]:
    root = ET.parse(edg_path).getroot()
    edges: Dict[str, Dict[str, Any]] = {}
    for edge in root.findall("./edge"):
        edge_id = str(edge.attrib.get("id", "")).strip()
        if not edge_id:
            continue
        edges[edge_id] = {
            "from": str(edge.attrib.get("from", "")).strip(),
            "to": str(edge.attrib.get("to", "")).strip(),
            "num_lanes": int(edge.attrib.get("numLanes", 1)),
        }
    return edges


def _parse_routes(rou_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    root = ET.parse(rou_path).getroot()
    routes: Dict[str, List[str]] = {}
    for route in root.findall("./route"):
        route_id = str(route.attrib.get("id", "")).strip()
        if not route_id:
            continue
        edges = [part.strip() for part in str(route.attrib.get("edges", "")).split(" ") if part.strip()]
        routes[route_id] = edges

    vtypes: Dict[str, Dict[str, Any]] = {}
    for vtype in root.findall("./vType"):
        type_id = str(vtype.attrib.get("id", "")).strip()
        if not type_id:
            continue
        vtypes[type_id] = {
            "length": float(vtype.attrib.get("length", _DEFAULT_VEHICLE_LENGTH)),
            "minGap": float(vtype.attrib.get("minGap", _DEFAULT_MIN_GAP)),
        }

    vehicles: Dict[str, Dict[str, Any]] = {}
    for vehicle in root.findall("./vehicle"):
        vehicle_id = str(vehicle.attrib.get("id", "")).strip()
        if not vehicle_id:
            continue
        vehicles[vehicle_id] = {
            "id": vehicle_id,
            "type": str(vehicle.attrib.get("type", "")).strip(),
            "route": str(vehicle.attrib.get("route", "")).strip(),
            "depart": str(vehicle.attrib.get("depart", "")).strip(),
            "depart_lane": str(vehicle.attrib.get("departLane", "")).strip(),
            "depart_pos": str(vehicle.attrib.get("departPos", "")).strip(),
        }
    return routes, vtypes, vehicles


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _edge_lengths(nod_path: Path, edg_path: Path) -> Dict[str, float]:
    nodes = _parse_nodes(nod_path)
    edges = _parse_edges(edg_path)
    lengths: Dict[str, float] = {}
    for edge_id, edge in edges.items():
        start = nodes.get(edge["from"])
        end = nodes.get(edge["to"])
        if start is None or end is None:
            continue
        lengths[edge_id] = float(math.hypot(end[0] - start[0], end[1] - start[1]))
    return lengths


def validate_scenario_geometry(cfg_path: Path) -> Dict[str, Any]:
    cfg_path = _resolve_cfg_path(cfg_path)
    report: Dict[str, Any] = {
        "scenario_cfg": str(cfg_path),
        "passed": True,
        "errors": [],
        "warnings": [],
        "seed_positions": [],
    }
    if not cfg_path.exists():
        report["passed"] = False
        report["errors"].append(f"sumocfg not found: {cfg_path}")
        return report

    inputs = _read_cfg_inputs(cfg_path)
    if not inputs.get("route_files"):
        report["passed"] = False
        report["errors"].append("route-files missing in sumocfg")
        return report
    if not inputs.get("net_files"):
        report["passed"] = False
        report["errors"].append("net-file missing in sumocfg")
        return report

    net_file = inputs["net_files"][0]
    route_file = inputs["route_files"][0]
    if not route_file.exists():
        report["passed"] = False
        report["errors"].append(f"route file not found: {route_file}")
        return report
    if not net_file.exists():
        report["warnings"].append(f"net file not found: {net_file}")

    net_prefix = str(net_file)
    if net_prefix.endswith(".net.xml"):
        net_prefix = net_prefix[:-8]
    nod_path = Path(net_prefix + ".nod.xml")
    edg_path = Path(net_prefix + ".edg.xml")
    if not nod_path.exists() or not edg_path.exists():
        report["passed"] = False
        report["errors"].append(
            f"plain network files missing for validation: nod={nod_path.exists()}, edg={edg_path.exists()}"
        )
        return report

    edge_length_map = _edge_lengths(nod_path, edg_path)
    routes, vtypes, vehicles = _parse_routes(route_file)

    departures: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
    seed_lookup: Dict[str, Dict[str, Any]] = {}

    for vehicle_id, item in vehicles.items():
        depart_time = _safe_float(item.get("depart"), default=-1.0)
        if abs(depart_time) > 1e-6:
            continue

        route_id = str(item.get("route", ""))
        route_edges = routes.get(route_id, [])
        if not route_edges:
            report["errors"].append(f"seed '{vehicle_id}' route '{route_id}' not found or empty")
            continue
        first_edge = route_edges[0]
        edge_len = _safe_float(edge_length_map.get(first_edge), default=-1.0)
        if edge_len <= 0:
            report["errors"].append(f"seed '{vehicle_id}' first edge '{first_edge}' length unavailable")
            continue

        depart_pos = _safe_float(item.get("depart_pos"), default=-1.0)
        if depart_pos < 0:
            report["errors"].append(f"seed '{vehicle_id}' has invalid departPos '{item.get('depart_pos')}'")
            continue

        vehicle_type = str(item.get("type", ""))
        type_payload = vtypes.get(vehicle_type, {})
        veh_len = _safe_float(type_payload.get("length"), _DEFAULT_VEHICLE_LENGTH)
        min_gap = _safe_float(type_payload.get("minGap"), _DEFAULT_MIN_GAP)
        lane_index = _safe_int(item.get("depart_lane"))
        if lane_index is None:
            report["warnings"].append(
                f"seed '{vehicle_id}' departLane '{item.get('depart_lane')}' is non-numeric; overlap check skipped"
            )

        seed_payload = {
            "vehicle_id": vehicle_id,
            "type_id": vehicle_type,
            "route_id": route_id,
            "first_edge": first_edge,
            "depart_time": depart_time,
            "depart_lane": lane_index,
            "depart_pos": depart_pos,
            "edge_length": edge_len,
            "vehicle_length": veh_len,
            "min_gap": min_gap,
        }
        report["seed_positions"].append(seed_payload)
        seed_lookup[vehicle_id] = seed_payload

        if depart_pos > edge_len:
            report["errors"].append(
                f"seed '{vehicle_id}' departPos={depart_pos:.3f} exceeds first-edge length {edge_len:.3f} on {first_edge}"
            )

        if lane_index is not None:
            key = (first_edge, lane_index, depart_time)
            departures.setdefault(key, []).append(seed_payload)

    for (edge_id, lane, depart_time), items in departures.items():
        sorted_items = sorted(items, key=lambda x: float(x["depart_pos"]), reverse=True)
        for idx in range(1, len(sorted_items)):
            front = sorted_items[idx - 1]
            back = sorted_items[idx]
            actual_gap = float(front["depart_pos"]) - float(back["depart_pos"])
            required_gap = max(
                float(front["vehicle_length"]) + float(front["min_gap"]),
                float(back["vehicle_length"]) + float(back["min_gap"]),
            )
            if actual_gap < required_gap:
                report["errors"].append(
                    "seed overlap at depart=0: "
                    f"edge={edge_id}, lane={lane}, time={depart_time:.2f}, "
                    f"front={front['vehicle_id']}({front['depart_pos']:.3f}), "
                    f"back={back['vehicle_id']}({back['depart_pos']:.3f}), "
                    f"gap={actual_gap:.3f} < required={required_gap:.3f}"
                )

    def _check_explicit_pair(front_id: str, back_id: str):
        front = seed_lookup.get(front_id)
        back = seed_lookup.get(back_id)
        if front is None or back is None:
            report["warnings"].append(f"explicit seed pair check skipped: {front_id}/{back_id} not both present")
            return
        if (
            str(front.get("first_edge")) != str(back.get("first_edge"))
            or int(front.get("depart_lane", -1)) != int(back.get("depart_lane", -2))
            or abs(float(front.get("depart_time", 0.0)) - float(back.get("depart_time", 0.0))) > 1e-6
        ):
            report["warnings"].append(
                f"explicit seed pair check skipped: {front_id}/{back_id} not on same edge/lane/depart-time"
            )
            return
        actual_gap = abs(float(front["depart_pos"]) - float(back["depart_pos"]))
        required_gap = max(
            float(front["vehicle_length"]) + float(front["min_gap"]),
            float(back["vehicle_length"]) + float(back["min_gap"]),
        )
        if actual_gap < required_gap:
            report["errors"].append(
                f"explicit pair overlap: {front_id}/{back_id} gap={actual_gap:.3f} < required={required_gap:.3f}"
            )

    _check_explicit_pair("ramp_follow_seed", "merge_seed")
    _check_explicit_pair("main_back_seed", "ego")

    report["passed"] = len(report["errors"]) == 0
    return report
