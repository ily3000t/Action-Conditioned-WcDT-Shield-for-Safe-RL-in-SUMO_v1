import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

from safe_rl.config.config import SimConfig


def _candidate_path(path: str) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    return p if p.exists() else None


def prepare_sumo_python_path(config: SimConfig):
    if config.sumo_home:
        tools_path = Path(config.sumo_home) / "tools"
        if tools_path.is_dir():
            tools_path_str = str(tools_path)
            if tools_path_str not in sys.path:
                sys.path.insert(0, tools_path_str)
            os.environ.setdefault("SUMO_HOME", str(Path(config.sumo_home)))


def resolve_sumo_binary(config: SimConfig, use_gui: bool) -> str:
    explicit = config.sumo_gui_bin if use_gui else config.sumo_bin
    explicit_path = _candidate_path(explicit)
    if explicit_path is not None:
        return str(explicit_path)

    if config.sumo_home:
        bin_name = "sumo-gui.exe" if use_gui else "sumo.exe"
        home_bin = Path(config.sumo_home) / "bin" / bin_name
        if home_bin.is_file():
            return str(home_bin)
        # Fallback for non-Windows naming.
        alt_name = "sumo-gui" if use_gui else "sumo"
        home_bin_alt = Path(config.sumo_home) / "bin" / alt_name
        if home_bin_alt.is_file():
            return str(home_bin_alt)

    return "sumo-gui" if use_gui else "sumo"


def resolve_netconvert_binary(config: SimConfig) -> str:
    explicit_path = _candidate_path(config.netconvert_bin)
    if explicit_path is not None:
        return str(explicit_path)

    if config.sumo_home:
        candidates = [
            Path(config.sumo_home) / "bin" / "netconvert.exe",
            Path(config.sumo_home) / "bin" / "netconvert",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

    return "netconvert"


def parse_cfg_net_file(cfg_path: Path) -> Optional[Path]:
    if not cfg_path.is_file():
        return None
    try:
        root = ET.parse(cfg_path).getroot()
        net_file_node = root.find("./input/net-file")
        if net_file_node is None:
            return None
        value = net_file_node.attrib.get("value", "").strip()
        if not value:
            return None
        return (cfg_path.parent / value).resolve()
    except Exception:
        return None


def maybe_build_network_from_plain(cfg_path: Path, config: SimConfig) -> Tuple[bool, str]:
    """
    Returns (ok, message). If net file exists, ok=True immediately.
    If missing and plain files are present, tries netconvert.
    """
    net_path = parse_cfg_net_file(cfg_path)
    if net_path is None:
        return False, f"Unable to parse net-file from cfg: {cfg_path}"

    if net_path.is_file():
        return True, f"Network file already exists: {net_path}"

    net_str = str(net_path)
    if not net_str.endswith(".net.xml"):
        return False, f"Unsupported net-file naming (expected *.net.xml): {net_path}"

    prefix = net_str[:-8]
    nod_file = Path(prefix + ".nod.xml")
    edg_file = Path(prefix + ".edg.xml")
    con_file = Path(prefix + ".con.xml")

    if not (nod_file.is_file() and edg_file.is_file() and con_file.is_file()):
        return False, (
            "Missing plain network files for auto-build: "
            f"nod={nod_file.is_file()}, edg={edg_file.is_file()}, con={con_file.is_file()}"
        )

    if not config.auto_build_network:
        return False, "auto_build_network disabled and net file missing"

    netconvert = resolve_netconvert_binary(config)
    cmd = [
        netconvert,
        "--node-files", str(nod_file),
        "--edge-files", str(edg_file),
        "--connection-files", str(con_file),
        "--output-file", str(net_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = (
            f"netconvert failed (code={result.returncode}).\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout={result.stdout}\n"
            f"stderr={result.stderr}"
        )
        return False, message

    return True, f"Network generated successfully: {net_path}"
