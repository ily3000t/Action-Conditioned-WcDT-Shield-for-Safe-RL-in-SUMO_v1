from pathlib import Path
import uuid

from safe_rl.sim.scenario_validation import validate_scenario_geometry


def _local_tmp_dir(tag: str) -> Path:
    path = Path("safe_rl_output/test_artifacts") / f"{tag}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_scenario(tmp_path: Path, vehicle_lines):
    nod = tmp_path / "toy.nod.xml"
    edg = tmp_path / "toy.edg.xml"
    con = tmp_path / "toy.con.xml"
    net = tmp_path / "toy.net.xml"
    rou = tmp_path / "toy.rou.xml"
    cfg = tmp_path / "toy.sumocfg"

    nod.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                "<nodes>",
                '  <node id="main_start" x="-100.0" y="0.0" type="priority"/>',
                '  <node id="merge" x="0.0" y="0.0" type="priority"/>',
                '  <node id="main_end" x="200.0" y="0.0" type="priority"/>',
                '  <node id="ramp_start" x="-100.0" y="40.0" type="priority"/>',
                "</nodes>",
            ]
        ),
        encoding="utf-8",
    )
    edg.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                "<edges>",
                '  <edge id="main_in" from="main_start" to="merge" numLanes="3" speed="33.33" priority="3"/>',
                '  <edge id="main_out" from="merge" to="main_end" numLanes="3" speed="33.33" priority="3"/>',
                '  <edge id="ramp_in" from="ramp_start" to="merge" numLanes="1" speed="22.22" priority="2"/>',
                "</edges>",
            ]
        ),
        encoding="utf-8",
    )
    con.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                "<connections>",
                '  <connection from="main_in" to="main_out" fromLane="0" toLane="0"/>',
                '  <connection from="main_in" to="main_out" fromLane="1" toLane="1"/>',
                '  <connection from="main_in" to="main_out" fromLane="2" toLane="2"/>',
                '  <connection from="ramp_in" to="main_out" fromLane="0" toLane="1"/>',
                "</connections>",
            ]
        ),
        encoding="utf-8",
    )
    net.write_text("<net></net>", encoding="utf-8")

    route_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '  <vType id="car_main" length="4.8" minGap="2.5"/>',
        '  <vType id="car_ramp" length="4.8" minGap="2.5"/>',
        '  <route id="route_main" edges="main_in main_out"/>',
        '  <route id="route_ramp" edges="ramp_in main_out"/>',
    ]
    route_lines.extend(vehicle_lines)
    route_lines.append("</routes>")
    rou.write_text("\n".join(route_lines), encoding="utf-8")

    cfg.write_text(
        "\n".join(
            [
                '<configuration>',
                "  <input>",
                f'    <net-file value="{net.name}"/>',
                f'    <route-files value="{rou.name}"/>',
                "  </input>",
                "</configuration>",
            ]
        ),
        encoding="utf-8",
    )
    return cfg


def test_validate_scenario_geometry_fails_when_seed_depart_pos_exceeds_edge_length():
    tmp_path = _local_tmp_dir("scenario_validation_overflow")
    cfg = _write_scenario(
        tmp_path,
        [
            '  <vehicle id="ego" type="car_main" route="route_main" depart="0.00" departLane="1" departPos="130"/>',
        ],
    )
    report = validate_scenario_geometry(cfg)
    assert report["passed"] is False
    assert any("exceeds first-edge length" in str(item) for item in report["errors"])


def test_validate_scenario_geometry_fails_when_same_lane_seeds_overlap():
    tmp_path = _local_tmp_dir("scenario_validation_overlap")
    cfg = _write_scenario(
        tmp_path,
        [
            '  <vehicle id="ego" type="car_main" route="route_main" depart="0.00" departLane="1" departPos="60"/>',
            '  <vehicle id="main_back_seed" type="car_main" route="route_main" depart="0.00" departLane="1" departPos="55"/>',
        ],
    )
    report = validate_scenario_geometry(cfg)
    assert report["passed"] is False
    assert any("seed overlap at depart=0" in str(item) for item in report["errors"])


def test_validate_scenario_geometry_passes_for_legal_seed_layout():
    tmp_path = _local_tmp_dir("scenario_validation_pass")
    cfg = _write_scenario(
        tmp_path,
        [
            '  <vehicle id="ego" type="car_main" route="route_main" depart="0.00" departLane="1" departPos="70"/>',
            '  <vehicle id="main_back_seed" type="car_main" route="route_main" depart="0.00" departLane="1" departPos="45"/>',
            '  <vehicle id="merge_seed" type="car_ramp" route="route_ramp" depart="0.00" departLane="0" departPos="50"/>',
            '  <vehicle id="ramp_follow_seed" type="car_ramp" route="route_ramp" depart="0.00" departLane="0" departPos="20"/>',
        ],
    )
    report = validate_scenario_geometry(cfg)
    assert report["passed"] is True
    assert report["errors"] == []
