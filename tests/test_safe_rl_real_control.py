from safe_rl.config.config import SimConfig
from safe_rl.sim.actions import encode_action
from safe_rl.sim.real_control import (
    RealSumoController,
    SKIP_INSUFFICIENT_LANE_COUNT,
    SKIP_MERGE_GUARD_BLOCKED,
    SKIP_NO_ROUTE_CONNECTION,
    SKIP_PLACEMENT_CONFLICT,
)


class _VehicleDomain:
    def __init__(self, state):
        self.state = state
        self.speed_modes = {}
        self.lane_change_modes = {}
        self.changed_lanes = []
        self.move_calls = []
        self.slow_down_calls = []

    def getIDList(self):
        return list(self.state.keys())

    def setSpeedMode(self, vehicle_id, mode):
        self.speed_modes[vehicle_id] = mode

    def setLaneChangeMode(self, vehicle_id, mode):
        self.lane_change_modes[vehicle_id] = mode

    def getPosition(self, vehicle_id):
        vehicle = self.state[vehicle_id]
        return vehicle["x"], vehicle.get("y", float(vehicle["lane_index"]) * 4.0)

    def getSpeed(self, vehicle_id):
        return self.state[vehicle_id]["speed"]

    def getLaneIndex(self, vehicle_id):
        return self.state[vehicle_id]["lane_index"]

    def getLaneID(self, vehicle_id):
        vehicle = self.state[vehicle_id]
        return f"{vehicle['road_id']}_{vehicle['lane_index']}"

    def getRoadID(self, vehicle_id):
        return self.state[vehicle_id]["road_id"]

    def getLanePosition(self, vehicle_id):
        return self.state[vehicle_id]["lane_pos"]

    def getLength(self, vehicle_id):
        return self.state[vehicle_id].get("length", 4.8)

    def getWidth(self, vehicle_id):
        return self.state[vehicle_id].get("width", 2.0)

    def getRoute(self, vehicle_id):
        return self.state[vehicle_id].get("route", [])

    def getRouteIndex(self, vehicle_id):
        return self.state[vehicle_id].get("route_index", 0)

    def setSpeed(self, vehicle_id, speed):
        self.state[vehicle_id]["speed"] = speed

    def slowDown(self, vehicle_id, speed, duration):
        self.state[vehicle_id]["speed"] = speed
        self.slow_down_calls.append((vehicle_id, speed, duration))

    def changeLane(self, vehicle_id, lane_index, _duration):
        self.state[vehicle_id]["lane_index"] = lane_index
        self.changed_lanes.append((vehicle_id, lane_index))

    def moveTo(self, vehicle_id, lane_id, lane_pos):
        road_id, lane_index = lane_id.rsplit("_", 1)
        self.state[vehicle_id]["road_id"] = road_id
        self.state[vehicle_id]["lane_index"] = int(lane_index)
        self.state[vehicle_id]["lane_pos"] = lane_pos
        self.state[vehicle_id]["x"] = lane_pos
        self.move_calls.append((vehicle_id, lane_id, lane_pos))


class _EdgeDomain:
    def __init__(self, lane_counts):
        self.lane_counts = lane_counts

    def getLaneNumber(self, road_id):
        return self.lane_counts[road_id]


class _LaneDomain:
    def __init__(self, lengths, links):
        self.lengths = lengths
        self.links = links

    def getLength(self, lane_id):
        return self.lengths.get(lane_id, 1000.0)

    def getLinks(self, lane_id):
        return self.links.get(lane_id, [])


class _SimulationDomain:
    def __init__(self):
        self.time = 0.0

    def getTime(self):
        return self.time


class _FakeAPI:
    def __init__(self, state, lane_counts=None, lane_lengths=None, lane_links=None):
        self.vehicle = _VehicleDomain(state)
        self.edge = _EdgeDomain(
            lane_counts
            or {"main_in": 3, "ramp_in": 1, "main_out": 3, ":merge_0": 3, ":merge_3": 1}
        )
        self.lane = _LaneDomain(lane_lengths or {}, lane_links or _default_lane_links())
        self.simulation = _SimulationDomain()

    def simulationStep(self):
        self.simulation.time += 0.1


def _default_lane_links():
    return {
        "main_in_0": [(":merge_0_0",)],
        "main_in_1": [(":merge_0_1",)],
        "main_in_2": [(":merge_0_2",)],
        "ramp_in_0": [(":merge_3_0",)],
        ":merge_0_0": [("main_out_0",)],
        ":merge_0_1": [("main_out_1",)],
        ":merge_0_2": [("main_out_2",)],
        ":merge_3_0": [("main_out_0",)],
    }


def _controller(state, lane_counts=None, lane_lengths=None, lane_links=None):
    config = SimConfig(step_length=0.1, ego_vehicle_id="ego")
    return RealSumoController(
        _FakeAPI(state, lane_counts=lane_counts, lane_lengths=lane_lengths, lane_links=lane_links),
        config,
    )


def test_apply_action_targets_explicit_ego():
    controller = _controller(
        {
            "ego": {"x": 100.0, "speed": 22.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 900.0},
            "lead": {"x": 112.0, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 912.0},
        }
    )

    meta = controller.apply_action(encode_action(1, -1))
    assert meta["applied"] is True
    assert meta["target_speed"] == 22.1
    assert meta["target_lane"] == 2
    assert controller.api.vehicle.changed_lanes == [("ego", 2)]


def test_apply_action_marks_lane_violation_when_out_of_bounds():
    controller = _controller(
        {
            "ego": {"x": 100.0, "speed": 20.0, "lane_index": 2, "road_id": "main_in", "lane_pos": 900.0},
        }
    )

    meta = controller.apply_action(encode_action(0, -1))
    assert meta["lane_violation"] is True
    assert meta["target_lane"] == 2
    assert controller.api.vehicle.changed_lanes == []


def test_apply_action_skips_lane_change_near_merge():
    controller = _controller(
        {
            "ego": {
                "x": 970.0,
                "speed": 20.0,
                "lane_index": 1,
                "road_id": "main_in",
                "lane_pos": 970.0,
                "route": ["main_in", "main_out"],
                "route_index": 0,
            },
        },
        lane_lengths={"main_in_1": 983.58},
    )

    meta = controller.apply_action(encode_action(0, -1))
    assert meta["lane_violation"] is False
    assert meta["target_lane"] == 1
    assert meta["lane_change_skipped_reason"] == SKIP_MERGE_GUARD_BLOCKED
    assert controller.api.vehicle.changed_lanes == []


def test_apply_action_skips_lane_change_when_route_disconnects():
    controller = _controller(
        {
            "ego": {
                "x": 100.0,
                "speed": 20.0,
                "lane_index": 0,
                "road_id": "main_in",
                "lane_pos": 100.0,
                "route": ["main_in", "main_out"],
                "route_index": 0,
            },
        },
        lane_links={
            "main_in_0": [(":merge_0_0",)],
            "main_in_1": [("ghost_out_1",)],
            "main_in_2": [(":merge_0_2",)],
            ":merge_0_0": [("main_out_0",)],
            ":merge_0_2": [("main_out_2",)],
        },
    )

    meta = controller.apply_action(encode_action(0, -1))
    assert meta["lane_violation"] is False
    assert meta["target_lane"] == 0
    assert meta["lane_change_skipped_reason"] == SKIP_NO_ROUTE_CONNECTION
    assert controller.api.vehicle.changed_lanes == []


def test_hard_brake_targets_same_lane_lead():
    controller = _controller(
        {
            "ego": {"x": 100.0, "speed": 22.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 900.0},
            "lead": {"x": 120.0, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 920.0},
            "left": {"x": 105.0, "speed": 20.0, "lane_index": 2, "road_id": "main_in", "lane_pos": 905.0},
        }
    )

    meta = controller.inject_risk_event("hard_brake")
    assert meta["applied"] is True
    assert meta["actual_event"] == "hard_brake"
    assert meta["target_vehicle_id"] == "lead"
    assert controller.api.vehicle.getSpeed("lead") < 18.0
    assert controller.api.vehicle.slow_down_calls


def test_unsafe_merge_prefers_ramp_vehicle_when_available():
    controller = _controller(
        {
            "ego": {
                "x": -40.0,
                "speed": 22.0,
                "lane_index": 1,
                "road_id": "main_in",
                "lane_pos": 960.0,
                "route": ["main_in", "main_out"],
                "route_index": 0,
            },
            "merge": {
                "x": -45.0,
                "speed": 18.0,
                "lane_index": 0,
                "road_id": "ramp_in",
                "lane_pos": 380.0,
                "route": ["ramp_in", "main_out"],
                "route_index": 0,
            },
            "left": {"x": -35.0, "speed": 21.0, "lane_index": 2, "road_id": "main_in", "lane_pos": 965.0},
        },
        lane_lengths={"ramp_in_0": 450.0},
    )

    meta = controller.inject_risk_event("unsafe_merge")
    assert meta["applied"] is True
    assert meta["actual_event"] == "unsafe_merge"
    assert meta["target_vehicle_id"] == "merge"
    assert controller.api.vehicle.move_calls


def test_move_to_respects_spacing_conflict():
    controller = _controller(
        {
            "ego": {"x": 100.0, "speed": 22.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 900.0},
            "lead": {"x": 150.0, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 950.0},
            "blocker": {"x": 112.5, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 912.5},
        }
    )

    moved = controller._move_vehicle_same_road(
        controller._snapshot("lead"),
        controller._snapshot("ego"),
        lane_index=1,
        gap=12.0,
    )

    assert moved is False
    assert controller.api.vehicle.move_calls == []


def test_cut_in_refuses_single_lane_ramp_like_target():
    controller = _controller(
        {
            "ego": {
                "x": 100.0,
                "speed": 22.0,
                "lane_index": 1,
                "road_id": "main_in",
                "lane_pos": 900.0,
                "route": ["main_in", "main_out"],
                "route_index": 0,
            },
            "merge": {
                "x": 103.0,
                "speed": 18.0,
                "lane_index": 0,
                "road_id": ":merge_3",
                "lane_pos": 10.0,
                "route": ["ramp_in", "main_out"],
                "route_index": 0,
            },
        },
        lane_counts={"main_in": 3, ":merge_3": 1, "main_out": 3, ":merge_0": 3},
        lane_links={
            "main_in_1": [(":merge_0_1",)],
            ":merge_3_0": [("main_out_0",)],
            ":merge_0_1": [("main_out_1",)],
        },
    )

    ego = controller._snapshot("ego")
    target = controller._snapshot("merge")
    meta = controller._event_cut_in(ego, [target], requested="cut_in")

    assert meta["applied"] is False
    assert meta["skipped_reason"] == SKIP_INSUFFICIENT_LANE_COUNT
    assert controller.api.vehicle.changed_lanes == []


def test_move_result_reports_placement_conflict_reason():
    controller = _controller(
        {
            "ego": {"x": 100.0, "speed": 22.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 900.0},
            "lead": {"x": 150.0, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 950.0},
            "blocker": {"x": 112.5, "speed": 18.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 912.5},
        }
    )

    result = controller._move_vehicle_same_road_result(
        controller._snapshot("lead"),
        controller._snapshot("ego"),
        lane_index=1,
        gap=12.0,
    )

    assert result["applied"] is False
    assert result["skipped_reason"] == SKIP_PLACEMENT_CONFLICT
