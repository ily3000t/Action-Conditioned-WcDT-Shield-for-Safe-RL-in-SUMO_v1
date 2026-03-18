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
        ego_id = self.config.ego_vehicle_id
        return SceneState(
            timestamp=float(self._safe_call(self.api.simulation.getTime, default=0.0)),
            ego_id=ego_id,
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
                self._safe_call(self.api.vehicle.changeLane, ego.vehicle_id, target_lane, max(1.0, self.config.step_length * 2.0))

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
        target_speed = max(0.0, min(lead.speed * 0.2, ego.speed * 0.35))
        self._safe_call(self.api.vehicle.setSpeed, lead.vehicle_id, target_speed)
        used_move = self._move_vehicle_same_road(lead, ego, ego.lane_index, gap=7.0) if lead.x - ego.x > 12.0 else False
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "hard_brake",
            "target_vehicle_id": lead.vehicle_id,
            "used_move": used_move,
        }

    def _event_close_follow(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        candidates = [vehicle for vehicle in vehicles if vehicle.lane_index == ego.lane_index]
        if not candidates:
            return {"applied": False, "requested_event": requested}

        candidates.sort(key=lambda vehicle: (0 if vehicle.x >= ego.x else 1, abs(vehicle.x - ego.x)))
        target = candidates[0]
        self.prepare_vehicle(target.vehicle_id)
        used_move = self._move_vehicle_same_road(target, ego, ego.lane_index, gap=8.0)
        if not used_move:
            self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(0.0, min(target.speed, ego.speed * 0.55)))
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
        used_move = self._move_vehicle_same_road(target, ego, ego.lane_index, gap=5.0)
        if not used_move:
            self._safe_call(self.api.vehicle.changeLane, target.vehicle_id, ego.lane_index, max(1.0, self.config.step_length * 2.0))
        self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(target.speed, ego.speed))
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": "cut_in",
            "target_vehicle_id": target.vehicle_id,
            "used_move": used_move,
        }

    def _event_unsafe_merge(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is not None:
            self.prepare_vehicle(target.vehicle_id)
            used_move = self._move_vehicle_along_current_lane(target, gap_to_lane_end=6.0)
            self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(target.speed, ego.speed * 1.1))
            return {
                "applied": True,
                "requested_event": requested,
                "actual_event": "unsafe_merge",
                "target_vehicle_id": target.vehicle_id,
                "used_move": used_move,
            }
        return self._event_cut_in(ego, vehicles, requested="unsafe_merge")

    def _event_merge_conflict(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        if ego.x < -80.0:
            return {"applied": False, "requested_event": requested}

        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is None:
            return {"applied": False, "requested_event": requested}

        self.prepare_vehicle(target.vehicle_id)
        used_move = self._move_vehicle_along_current_lane(target, gap_to_lane_end=3.0)
        self._safe_call(self.api.vehicle.setSpeed, target.vehicle_id, max(target.speed, ego.speed * 1.2))
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
        lane_length = float(self._safe_call(self.api.lane.getLength, lane_id, default=max(ego.lane_pos + gap + 1.0, 10.0)))
        target_pos = min(max(ego.lane_pos + gap, 1.0), max(1.0, lane_length - 1.0))
        try:
            self.api.vehicle.moveTo(target.vehicle_id, lane_id, target_pos)
            return True
        except Exception:
            return False

    def _move_vehicle_along_current_lane(self, target: RealVehicleSnapshot, gap_to_lane_end: float) -> bool:
        if not target.lane_id:
            return False
        lane_length = float(self._safe_call(self.api.lane.getLength, target.lane_id, default=target.lane_pos + gap_to_lane_end + 1.0))
        target_pos = max(1.0, lane_length - max(gap_to_lane_end, 1.0))
        try:
            self.api.vehicle.moveTo(target.vehicle_id, target.lane_id, target_pos)
            return True
        except Exception:
            return False

    def _nearest_same_lane_ahead(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot]) -> Optional[RealVehicleSnapshot]:
        candidates = [vehicle for vehicle in vehicles if vehicle.lane_index == ego.lane_index and vehicle.x > ego.x]
        if not candidates:
            return None
        candidates.sort(key=lambda vehicle: vehicle.x - ego.x)
        return candidates[0]

    def _nearest_adjacent_vehicle(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot]) -> Optional[RealVehicleSnapshot]:
        candidates = [vehicle for vehicle in vehicles if abs(vehicle.lane_index - ego.lane_index) == 1 and not vehicle.road_id.startswith("ramp")]
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
