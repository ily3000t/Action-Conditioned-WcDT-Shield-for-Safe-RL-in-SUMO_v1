import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from safe_rl.config.config import SimConfig
from safe_rl.data.risk import detect_collision
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.sim.actions import action_name, decode_action
from safe_rl.sim.mock_core import RISK_EVENTS


LONGITUDINAL_SPEED_DELTA = {-1: -1.5, 0: 0.0, 1: 1.0}
MOVE_PLACEMENT_CLEARANCE = 10.0
HARD_BRAKE_GAP = 16.0
CLOSE_FOLLOW_GAP = 13.0
CUT_IN_GAP = 12.0
MERGE_LANE_END_BUFFER = 10.0
MAX_CUT_IN_ALIGNMENT_DISTANCE = 35.0
RISK_EVENT_FALLBACKS = {
    "hard_brake": ["close_follow", "cut_in"],
    "close_follow": ["hard_brake", "cut_in", "unsafe_merge"],
    "unsafe_merge": ["cut_in", "merge_conflict", "close_follow"],
    "merge_conflict": ["unsafe_merge", "cut_in", "hard_brake"],
    "cut_in": ["unsafe_merge", "close_follow", "hard_brake"],
}


@dataclass
class RealVehicleSnapshot:
    vehicle_id: str
    x: float
    y: float
    speed: float
    lane_index: int
    lane_id: str
    road_id: str
    lane_pos: float
    length: float
    width: float


class RealSumoController:
    def __init__(self, api: Any, config: SimConfig, logger: Optional[logging.Logger] = None):
        self.api = api
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._rng = random.Random(config.random_seed)

    def has_ego(self) -> bool:
        return self.config.ego_vehicle_id in set(self._vehicle_ids())

    def warmup_until_ego(self, max_steps: int) -> bool:
        for _ in range(max_steps):
            if self.has_ego():
                self.prepare_vehicle(self.config.ego_vehicle_id)
                return True
            self.api.simulationStep()
        return self.has_ego()

    def prepare_vehicle(self, vehicle_id: str):
        if not self._vehicle_exists(vehicle_id):
            return
        self._safe_call(self.api.vehicle.setSpeedMode, vehicle_id, 0)
        self._safe_call(self.api.vehicle.setLaneChangeMode, vehicle_id, 0)

    def build_scene(self) -> SceneState:
        vehicles = [self._snapshot(vehicle_id) for vehicle_id in self._vehicle_ids()]
        vehicle_states = [
            VehicleState(
                vehicle_id=vehicle.vehicle_id,
                x=vehicle.x,
                y=vehicle.y,
                vx=vehicle.speed,
                vy=0.0,
                ax=0.0,
                ay=0.0,
                heading=0.0,
                lane_id=vehicle.lane_index,
                length=vehicle.length,
                width=vehicle.width,
            )
            for vehicle in vehicles
        ]
        return SceneState(
            timestamp=float(self._safe_call(self.api.simulation.getTime, default=0.0)),
            ego_id=self.config.ego_vehicle_id,
            vehicles=vehicle_states,
        )

    def apply_action(self, action_id: int) -> Dict[str, Any]:
        ego = self._ego_snapshot()
        if ego is None:
            return {
                "applied": False,
                "ego_missing": True,
                "lane_violation": False,
                "action_name": action_name(action_id),
            }

        self.prepare_vehicle(ego.vehicle_id)
        action = decode_action(action_id)
        target_speed = max(0.0, ego.speed + LONGITUDINAL_SPEED_DELTA[action.longitudinal] * self.config.step_length)
        self._safe_call(self.api.vehicle.setSpeed, ego.vehicle_id, target_speed)

        lane_count = self._lane_count_for_vehicle(ego)
        target_lane = ego.lane_index
        lane_violation = False
        if action.lateral != 0:
            target_lane = ego.lane_index - action.lateral
            if target_lane < 0 or target_lane >= lane_count:
                lane_violation = True
                target_lane = ego.lane_index
            elif target_lane != ego.lane_index:
                self._safe_call(
                    self.api.vehicle.changeLane,
                    ego.vehicle_id,
                    target_lane,
                    max(1.0, self.config.step_length * 2.0),
                )

        return {
            "applied": True,
            "ego_missing": False,
            "lane_violation": lane_violation,
            "target_speed": float(target_speed),
            "target_lane": int(target_lane),
            "action_name": action_name(action_id),
        }

    def inject_risk_event(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        requested = event_type or self._rng.choice(RISK_EVENTS)
        attempts = [requested] + [event for event in RISK_EVENT_FALLBACKS.get(requested, []) if event != requested]
        for candidate in attempts:
            meta = self._inject_specific_event(candidate, requested)
            if meta.get("applied", False):
                self.logger.info(
                    "Real SUMO risk event applied: requested=%s actual=%s target=%s move=%s",
                    requested,
                    meta.get("actual_event"),
                    meta.get("target_vehicle_id", ""),
                    meta.get("used_move", False),
                )
                return meta

        self.logger.warning("Real SUMO risk event failed: requested=%s, ego_present=%s", requested, self.has_ego())
        return {
            "applied": False,
            "requested_event": requested,
            "actual_event": "",
            "target_vehicle_id": "",
            "used_move": False,
            "reason": "no_feasible_target",
        }

    def summarize_step(self, scene: SceneState, action_meta: Dict[str, Any], risk_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ego_speed = 0.0
        if self.has_ego():
            ego_speed = float(self._safe_call(self.api.vehicle.getSpeed, self.config.ego_vehicle_id, default=0.0))
        return {
            "collision": bool(detect_collision(scene)),
            "ego_speed": ego_speed,
            "lane_violation": bool(action_meta.get("lane_violation", False)),
            "risk_event": risk_meta.get("actual_event", "") if risk_meta else "",
            "risk_target_vehicle": risk_meta.get("target_vehicle_id", "") if risk_meta else "",
            "risk_requested_event": risk_meta.get("requested_event", "") if risk_meta else "",
        }

    def _inject_specific_event(self, event_type: str, requested: str) -> Dict[str, Any]:
        ego = self._ego_snapshot()
        if ego is None:
            return {"applied": False, "requested_event": requested}

        vehicles = [vehicle for vehicle in self._all_snapshots() if vehicle.vehicle_id != ego.vehicle_id]
        handler = getattr(self, f"_event_{event_type}", None)
        if handler is None:
            return {"applied": False, "requested_event": requested}
        return handler(ego, vehicles, requested=requested)

    def _event_hard_brake(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        lead = self._nearest_same_lane_ahead(ego, vehicles)
        if lead is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(lead.vehicle_id)
        target_speed = max(0.0, min(lead.speed * 0.45, ego.speed * 0.7))
        self._safe_call(self.api.vehicle.setSpeed, lead.vehicle_id, target_speed)
        used_move = False
        if lead.x - ego.x > 30.0:
            used_move = self._move_vehicle_same_road(lead, ego, ego.lane_index, gap=HARD_BRAKE_GAP)
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "hard_brake",
            "target_vehicle_id": lead.vehicle_id,
            "used_move": used_move,
        }

    def _event_close_follow(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_same_lane_ahead(ego, vehicles)
        if target is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(target.vehicle_id)
        used_move = False
        if target.x - ego.x > 24.0:
            used_move = self._move_vehicle_same_road(target, ego, ego.lane_index, gap=CLOSE_FOLLOW_GAP)
        if not used_move:
            target_speed = max(0.0, min(target.speed, ego.speed * 0.8))
            self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, target_speed)
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "close_follow",
            "target_vehicle_id": target.vehicle_id,
            "used_move": used_move,
        }

    def _event_cut_in(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_adjacent_vehicle(ego, vehicles)
        if target is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(target.vehicle_id)
        used_move = False
        if target.road_id == ego.road_id and abs(target.x - ego.x) <= MAX_CUT_IN_ALIGNMENT_DISTANCE:
            used_move = self._move_vehicle_same_road(target, ego, ego.lane_index, gap=CUT_IN_GAP)
        if not used_move:
            self._safe_call(
                self.api.vehicle.changeLane,
                target.vehicle_id,
                ego.lane_index,
                max(1.0, self.config.step_length * 2.0),
            )
        target_speed = min(max(target.speed, ego.speed * 0.95), ego.speed + 1.0)
        self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(0.0, target_speed))
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "cut_in",
            "target_vehicle_id": target.vehicle_id,
            "used_move": used_move,
        }

    def _event_unsafe_merge(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(target.vehicle_id)
        used_move = self._move_vehicle_along_current_lane(target, gap_to_lane_end=MERGE_LANE_END_BUFFER)
        self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(target.speed, min(ego.speed, target.speed + 1.0)))
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "unsafe_merge",
            "target_vehicle_id": target.vehicle_id,
            "used_move": used_move,
        }

    def _event_merge_conflict(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        if ego.x < -120.0:
            return {"applied": False, "requested_event": requested}

        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(target.vehicle_id)
        used_move = self._move_vehicle_along_current_lane(target, gap_to_lane_end=MERGE_LANE_END_BUFFER)
        self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(target.speed, ego.speed))
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "merge_conflict",
            "target_vehicle_id": target.vehicle_id,
            "used_move": used_move,
        }

    def _vehicle_ids(self) -> List[str]:
        return list(self._safe_call(self.api.vehicle.getIDList, default=[]))

    def _vehicle_exists(self, vehicle_id: str) -> bool:
        return vehicle_id in set(self._vehicle_ids())

    def _all_snapshots(self) -> List[RealVehicleSnapshot]:
        snapshots = [self._snapshot(vehicle_id) for vehicle_id in self._vehicle_ids()]
        snapshots.sort(key=lambda vehicle: vehicle.x)
        return snapshots

    def _ego_snapshot(self) -> Optional[RealVehicleSnapshot]:
        if not self.has_ego():
            return None
        return self._snapshot(self.config.ego_vehicle_id)

    def _snapshot(self, vehicle_id: str) -> RealVehicleSnapshot:
        x, y = self._safe_call(self.api.vehicle.getPosition, vehicle_id, default=(0.0, 0.0))
        speed = float(self._safe_call(self.api.vehicle.getSpeed, vehicle_id, default=0.0))
        lane_index = int(self._safe_call(self.api.vehicle.getLaneIndex, vehicle_id, default=0))
        lane_id = str(self._safe_call(self.api.vehicle.getLaneID, vehicle_id, default=f"lane_{lane_index}"))
        road_id = str(self._safe_call(self.api.vehicle.getRoadID, vehicle_id, default=""))
        lane_pos = float(self._safe_call(self.api.vehicle.getLanePosition, vehicle_id, default=max(float(x), 0.0)))
        length = float(self._safe_call(self.api.vehicle.getLength, vehicle_id, default=4.8))
        width = float(self._safe_call(self.api.vehicle.getWidth, vehicle_id, default=2.0))
        return RealVehicleSnapshot(
            vehicle_id=vehicle_id,
            x=float(x),
            y=float(y),
            speed=speed,
            lane_index=lane_index,
            lane_id=lane_id,
            road_id=road_id,
            lane_pos=lane_pos,
            length=length,
            width=width,
        )

    def _lane_count_for_vehicle(self, vehicle: RealVehicleSnapshot) -> int:
        if vehicle.road_id:
            count = self._safe_call(self.api.edge.getLaneNumber, vehicle.road_id, default=None)
            if count is not None:
                return max(1, int(count))
        return max(1, vehicle.lane_index + 1)

    def _move_vehicle_same_road(self, target: RealVehicleSnapshot, ego: RealVehicleSnapshot, lane_index: int, gap: float) -> bool:
        if target.road_id != ego.road_id or not target.road_id:
            return False
        lane_id = f"{target.road_id}_{lane_index}"
        lane_length = float(
            self._safe_call(
                self.api.lane.getLength,
                lane_id,
                default=max(ego.lane_pos + gap + 1.0, 10.0),
            )
        )
        gap = max(gap, MOVE_PLACEMENT_CLEARANCE)
        target_pos = min(max(ego.lane_pos + gap, 1.0), max(1.0, lane_length - 1.0))
        if not self._placement_is_clear(target, target.road_id, lane_index, target_pos, clearance=gap):
            return False
        try:
            self.api.vehicle.moveTo(target.vehicle_id, lane_id, target_pos)
            return True
        except Exception:
            return False

    def _move_vehicle_along_current_lane(self, target: RealVehicleSnapshot, gap_to_lane_end: float) -> bool:
        if not target.lane_id or not target.road_id:
            return False
        gap_to_lane_end = max(gap_to_lane_end, MOVE_PLACEMENT_CLEARANCE)
        lane_length = float(
            self._safe_call(
                self.api.lane.getLength,
                target.lane_id,
                default=target.lane_pos + gap_to_lane_end + 1.0,
            )
        )
        target_pos = max(1.0, lane_length - gap_to_lane_end)
        if not self._placement_is_clear(target, target.road_id, target.lane_index, target_pos, clearance=MOVE_PLACEMENT_CLEARANCE):
            return False
        try:
            self.api.vehicle.moveTo(target.vehicle_id, target.lane_id, target_pos)
            return True
        except Exception:
            return False

    def _placement_is_clear(
        self,
        target: RealVehicleSnapshot,
        road_id: str,
        lane_index: int,
        target_pos: float,
        clearance: float,
    ) -> bool:
        for other in self._all_snapshots():
            if other.vehicle_id == target.vehicle_id:
                continue
            if other.road_id != road_id or other.lane_index != lane_index:
                continue
            min_gap = max(clearance, 0.5 * (other.length + target.length) + 3.0)
            if abs(other.lane_pos - target_pos) < min_gap:
                return False
        return True

    def _nearest_same_lane_ahead(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot]) -> Optional[RealVehicleSnapshot]:
        candidates = [vehicle for vehicle in vehicles if vehicle.lane_index == ego.lane_index and vehicle.x > ego.x]
        if not candidates:
            return None
        candidates.sort(key=lambda vehicle: vehicle.x - ego.x)
        return candidates[0]

    def _nearest_adjacent_vehicle(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot]) -> Optional[RealVehicleSnapshot]:
        candidates = [
            vehicle
            for vehicle in vehicles
            if abs(vehicle.lane_index - ego.lane_index) == 1 and not vehicle.road_id.startswith("ramp")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda vehicle: (abs(vehicle.x - ego.x), vehicle.lane_index))
        return candidates[0]

    def _nearest_ramp_vehicle(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot]) -> Optional[RealVehicleSnapshot]:
        candidates = [vehicle for vehicle in vehicles if vehicle.road_id.startswith("ramp")]
        if not candidates:
            return None
        candidates.sort(key=lambda vehicle: abs(vehicle.x - ego.x))
        return candidates[0]

    @staticmethod
    def _safe_call(func, *args, default=None):
        try:
            return func(*args)
        except Exception:
            return default
