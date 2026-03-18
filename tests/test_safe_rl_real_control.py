from safe_rl.config.config import SimConfig
from safe_rl.sim.actions import encode_action
from safe_rl.sim.real_control import RealSumoController


class _VehicleDomain:
    def __init__(self, state):
        self.state = state
        self.speed_modes = {}
        self.lane_change_modes = {}
        self.changed_lanes = []
        self.move_calls = []

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

    def setSpeed(self, vehicle_id, speed):
        self.state[vehicle_id]["speed"] = speed

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
    def __init__(self, lengths):
        self.lengths = lengths

    def getLength(self, lane_id):
        return self.lengths.get(lane_id, 1000.0)


class _SimulationDomain:
    def __init__(self):
        self.time = 0.0

    def getTime(self):
        return self.time


class _FakeAPI:
    def __init__(self, state, lane_counts=None, lane_lengths=None):
        self.vehicle = _VehicleDomain(state)
        self.edge = _EdgeDomain(lane_counts or {"main_in": 3, "ramp_in": 1})
        self.lane = _LaneDomain(lane_lengths or {})
        self.simulation = _SimulationDomain()

    def simulationStep(self):
        self.simulation.time += 0.1


def _controller(state, lane_counts=None, lane_lengths=None):
    config = SimConfig(step_length=0.1, ego_vehicle_id="ego")
    return RealSumoController(_FakeAPI(state, lane_counts=lane_counts, lane_lengths=lane_lengths), config)


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


def test_unsafe_merge_prefers_ramp_vehicle_when_available():
    controller = _controller(
        {
            "ego": {"x": -40.0, "speed": 22.0, "lane_index": 1, "road_id": "main_in", "lane_pos": 960.0},
            "merge": {"x": -45.0, "speed": 18.0, "lane_index": 0, "road_id": "ramp_in", "lane_pos": 430.0},
            "left": {"x": -35.0, "speed": 21.0, "lane_index": 2, "road_id": "main_in", "lane_pos": 965.0},
        },
        lane_lengths={"ramp_in_0": 450.0},
    )

    meta = controller.inject_risk_event("unsafe_merge")
    assert meta["applied"] is True
    assert meta["actual_event"] == "unsafe_merge"
    assert meta["target_vehicle_id"] == "merge"
    assert controller.api.vehicle.move_calls
