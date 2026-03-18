import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from safe_rl.data.types import SceneState, TrafficLightState, VehicleState
from safe_rl.sim.actions import decode_action


RISK_EVENTS = ["hard_brake", "unsafe_merge", "close_follow", "merge_conflict", "cut_in"]


@dataclass
class _Vehicle:
    vehicle_id: str
    lane_id: int
    x: float
    speed: float
    length: float = 4.8
    width: float = 2.0


class MockTrafficCore:
    """
    Lightweight traffic dynamics fallback used when SUMO bindings are unavailable.
    It preserves the same backend interface so the full pipeline can still run.
    """

    def __init__(self, episode_steps: int = 300, step_length: float = 0.1, seed: int = 42):
        self.episode_steps = episode_steps
        self.step_length = step_length
        self._rng = random.Random(seed)
        self.step_index = 0
        self.ego_id = "ego"
        self.ego_lane_attempt_invalid = False
        self.ego: Optional[_Vehicle] = None
        self.others: Dict[str, _Vehicle] = {}

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.step_index = 0
        self.ego_lane_attempt_invalid = False
        self.ego = _Vehicle(vehicle_id=self.ego_id, lane_id=1, x=0.0, speed=22.0)
        self.others = {
            "lead": _Vehicle(vehicle_id="lead", lane_id=1, x=35.0, speed=20.0),
            "merge": _Vehicle(vehicle_id="merge", lane_id=2, x=20.0, speed=18.0),
            "left": _Vehicle(vehicle_id="left", lane_id=0, x=10.0, speed=21.0),
        }
        return self.get_scene(timestamp=0.0)

    def inject_risk_event(self, event_type: Optional[str] = None):
        if event_type is None:
            event_type = self._rng.choice(RISK_EVENTS)
        if event_type == "hard_brake":
            self.others["lead"].speed = max(0.0, self.others["lead"].speed - 10.0)
        elif event_type == "unsafe_merge":
            self.others["merge"].lane_id = self.ego.lane_id
            self.others["merge"].x = self.ego.x + 5.0
        elif event_type == "close_follow":
            self.others["lead"].lane_id = self.ego.lane_id
            self.others["lead"].x = self.ego.x + 8.0
        elif event_type == "merge_conflict":
            self.others["merge"].lane_id = self.ego.lane_id
            self.others["merge"].x = self.ego.x + 2.5
        elif event_type == "cut_in":
            self.others["left"].lane_id = self.ego.lane_id
            self.others["left"].x = self.ego.x + 3.0

    def step(self, action_id: int):
        action = decode_action(action_id)
        dt = self.step_length

        # Lateral move by one lane with hard boundaries.
        target_lane = self.ego.lane_id + action.lateral
        self.ego_lane_attempt_invalid = target_lane < 0 or target_lane > 2
        if not self.ego_lane_attempt_invalid:
            self.ego.lane_id = target_lane

        # Longitudinal speed control.
        accel = {-1: -2.5, 0: 0.0, 1: 1.5}[action.longitudinal]
        old_speed = self.ego.speed
        self.ego.speed = max(0.0, self.ego.speed + accel * dt)
        self.ego.x += self.ego.speed * dt

        # Other vehicles simple rollout.
        for other in self.others.values():
            if other.vehicle_id == "lead":
                noise = self._rng.uniform(-0.4, 0.2)
            else:
                noise = self._rng.uniform(-0.2, 0.2)
            other.speed = max(0.0, other.speed + noise)
            other.x += other.speed * dt

        collision = self._check_collision()
        task_reward = (self.ego.speed * dt * 0.1) - (0.05 if action.lateral != 0 else 0.0)
        if collision:
            task_reward -= 10.0
        if self.ego_lane_attempt_invalid:
            task_reward -= 1.0

        self.step_index += 1
        done = collision or self.step_index >= self.episode_steps
        info = {
            "collision": collision,
            "ego_speed": self.ego.speed,
            "ego_prev_speed": old_speed,
            "lane_violation": self.ego_lane_attempt_invalid,
        }
        return self.get_scene(timestamp=self.step_index * dt), task_reward, done, info

    def _lane_to_y(self, lane_id: int) -> float:
        return float(lane_id) * 4.0

    def _check_collision(self) -> bool:
        for other in self.others.values():
            if other.lane_id != self.ego.lane_id:
                continue
            if abs(other.x - self.ego.x) < (self.ego.length + other.length) * 0.45:
                return True
        return False

    def get_scene(self, timestamp: float) -> SceneState:
        ego_state = VehicleState(
            vehicle_id=self.ego.vehicle_id,
            x=self.ego.x,
            y=self._lane_to_y(self.ego.lane_id),
            vx=self.ego.speed,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            heading=0.0,
            lane_id=self.ego.lane_id,
            length=self.ego.length,
            width=self.ego.width,
        )
        vehicles: List[VehicleState] = [ego_state]
        for other in self.others.values():
            vehicles.append(
                VehicleState(
                    vehicle_id=other.vehicle_id,
                    x=other.x,
                    y=self._lane_to_y(other.lane_id),
                    vx=other.speed,
                    vy=0.0,
                    ax=0.0,
                    ay=0.0,
                    heading=0.0,
                    lane_id=other.lane_id,
                    length=other.length,
                    width=other.width,
                )
            )

        lane_polylines = []
        for lane_idx in range(3):
            y = self._lane_to_y(lane_idx)
            lane_polylines.append([[self.ego.x - 100.0, y], [self.ego.x + 200.0, y]])

        lights = [TrafficLightState(light_id="merge_light", x=self.ego.x + 120.0, y=8.0, state="GREEN")]
        return SceneState(timestamp=timestamp, ego_id=self.ego_id, vehicles=vehicles, traffic_lights=lights,
                         lane_polylines=lane_polylines)

    def min_distance_to_ego(self) -> float:
        min_distance = math.inf
        for other in self.others.values():
            if other.vehicle_id == self.ego.vehicle_id:
                continue
            distance = abs(other.x - self.ego.x)
            if distance < min_distance:
                min_distance = distance
        return float(min_distance if min_distance != math.inf else 1e6)
