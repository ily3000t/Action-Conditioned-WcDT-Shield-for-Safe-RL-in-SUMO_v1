#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from pathlib import Path


def _detect_netconvert() -> str:
    env_home = os.environ.get("SUMO_HOME", "")
    candidates = []
    if env_home:
        candidates.append(Path(env_home) / "bin" / "netconvert.exe")
        candidates.append(Path(env_home) / "bin" / "netconvert")

    # User-provided default path in this project context.
    candidates.append(Path(r"E:/Program Files/sumo-1.22.0/bin/netconvert.exe"))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "netconvert"


def main():
    root = Path(__file__).resolve().parent
    net_file = root / "highway_merge.net.xml"
    node_file = root / "highway_merge.nod.xml"
    edge_file = root / "highway_merge.edg.xml"
    con_file = root / "highway_merge.con.xml"

    netconvert = _detect_netconvert()
    cmd = [
        netconvert,
        "--node-files", str(node_file),
        "--edge-files", str(edge_file),
        "--connection-files", str(con_file),
        "--output-file", str(net_file),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"Generated: {net_file}")


if __name__ == "__main__":
    main()
