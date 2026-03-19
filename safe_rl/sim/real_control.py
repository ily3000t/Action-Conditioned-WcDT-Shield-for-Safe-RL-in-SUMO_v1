import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from safe_rl.config.config import SimConfig
from safe_rl.data.risk import detect_collision
from safe_rl.data.types import SceneState, VehicleState
from safe_rl.sim.actions import action_name, decode_action
from safe_rl.sim.mock_core import RISK_EVENTS


LONGITUDINAL_SPEED_DELTA = {-1: -1.5, 0: 0.0, 1: 1.0}
MOVE_PLACEMENT_CLEARANCE = 12.0
HARD_BRAKE_GAP = 22.0
CLOSE_FOLLOW_GAP = 18.0
CUT_IN_GAP = 16.0
MERGE_LANE_END_BUFFER = 34.0
MAX_CUT_IN_ALIGNMENT_DISTANCE = 28.0
JUNCTION_GUARD_DISTANCE = 30.0
HIGH_BRAKE_TARGET_SPEED = 6.0
SKIP_INVALID_LANE_INDEX = "invalid_lane_index"
SKIP_NO_ROUTE_CONNECTION = "no_route_connection"
SKIP_INSUFFICIENT_LANE_COUNT = "insufficient_lane_count"
SKIP_PLACEMENT_CONFLICT = "placement_conflict"
SKIP_MERGE_GUARD_BLOCKED = "merge_guard_blocked"
SKIP_NO_FEASIBLE_TARGET = "no_feasible_target"
RISK_EVENT_FALLBACKS = {
    "hard_brake": ["close_follow", "cut_in"],
    "close_follow": ["hard_brake", "cut_in", "unsafe_merge"],
    "unsafe_merge": ["merge_conflict", "close_follow", "hard_brake"],
    "merge_conflict": ["unsafe_merge", "close_follow", "hard_brake"],
    "cut_in": ["close_follow", "hard_brake", "unsafe_merge"],
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
                "skipped_reason": SKIP_NO_FEASIBLE_TARGET,
            }

        self.prepare_vehicle(ego.vehicle_id)
        action = decode_action(action_id)
        target_speed = max(0.0, ego.speed + LONGITUDINAL_SPEED_DELTA[action.longitudinal] * self.config.step_length)
        self._safe_call(self.api.vehicle.setSpeed, ego.vehicle_id, target_speed)

        target_lane = ego.lane_index
        lane_violation = False
        skipped_reason = ""
        if action.lateral != 0:
            desired_lane = ego.lane_index - action.lateral
            lane_target = self._resolve_lane_target(ego, desired_lane)
            lane_violation = bool(lane_target["lane_violation"])
            skipped_reason = str(lane_target["reason"] or "")
            if lane_target["allowed"] and lane_target["lane_index"] != ego.lane_index:
                self._safe_call(
                    self.api.vehicle.changeLane,
                    ego.vehicle_id,
                    lane_target["lane_index"],
                    max(1.0, self.config.step_length * 2.0),
                )
                target_lane = lane_target["lane_index"]
                skipped_reason = ""
            else:
                target_lane = ego.lane_index

        return {
            "applied": True,
            "ego_missing": False,
            "lane_violation": lane_violation,
            "target_speed": float(target_speed),
            "target_lane": int(target_lane),
            "action_name": action_name(action_id),
            "lane_change_skipped_reason": skipped_reason,
            "skipped_reason": skipped_reason,
        }

    def inject_risk_event(self, event_type: Optional[str] = None) -> Dict[str, Any]:
        requested = event_type or self._rng.choice(RISK_EVENTS)
        attempts = [requested] + [event for event in RISK_EVENT_FALLBACKS.get(requested, []) if event != requested]
        last_meta = {
            "applied": False,
            "requested_event": requested,
            "actual_event": "",
            "target_vehicle_id": "",
            "used_move": False,
            "reason": SKIP_NO_FEASIBLE_TARGET,
            "skipped_reason": SKIP_NO_FEASIBLE_TARGET,
        }
        for candidate in attempts:
            meta = self._inject_specific_event(candidate, requested)
            last_meta = meta
            if meta.get("applied", False):
                self.logger.info(
                    "Real SUMO risk event applied: requested=%s actual=%s target=%s move=%s",
                    requested,
                    meta.get("actual_event"),
                    meta.get("target_vehicle_id", ""),
                    meta.get("used_move", False),
                )
                return meta

        self.logger.warning(
            "Real SUMO risk event failed: requested=%s, ego_present=%s, skipped_reason=%s",
            requested,
            self.has_ego(),
            last_meta.get("skipped_reason", SKIP_NO_FEASIBLE_TARGET),
        )
        if not last_meta.get("skipped_reason"):
            last_meta["skipped_reason"] = SKIP_NO_FEASIBLE_TARGET
        if not last_meta.get("reason"):
            last_meta["reason"] = last_meta["skipped_reason"]
        return last_meta

    def summarize_step(self, scene: SceneState, action_meta: Dict[str, Any], risk_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ego_speed = 0.0
        if self.has_ego():
            ego_speed = float(self._safe_call(self.api.vehicle.getSpeed, self.config.ego_vehicle_id, default=0.0))
        info = {
            "collision": bool(detect_collision(scene)),
            "ego_speed": ego_speed,
            "lane_violation": bool(action_meta.get("lane_violation", False)),
            "risk_event": risk_meta.get("actual_event", "") if risk_meta else "",
            "risk_target_vehicle": risk_meta.get("target_vehicle_id", "") if risk_meta else "",
            "risk_requested_event": risk_meta.get("requested_event", "") if risk_meta else "",
        }
        skipped = str(action_meta.get("lane_change_skipped_reason", "") or action_meta.get("skipped_reason", "") or "")
        if skipped:
            info["lane_change_skipped_reason"] = skipped
        risk_skipped = str(risk_meta.get("skipped_reason", "") if risk_meta else "")
        if risk_skipped:
            info["risk_skipped_reason"] = risk_skipped
        return info

    def _inject_specific_event(self, event_type: str, requested: str) -> Dict[str, Any]:
        ego = self._ego_snapshot()
        if ego is None:
            return {
                "applied": False,
                "requested_event": requested,
                "actual_event": "",
                "target_vehicle_id": "",
                "used_move": False,
                "reason": SKIP_NO_FEASIBLE_TARGET,
                "skipped_reason": SKIP_NO_FEASIBLE_TARGET,
            }

        vehicles = [vehicle for vehicle in self._all_snapshots() if vehicle.vehicle_id != ego.vehicle_id]
        handler = getattr(self, f"_event_{event_type}", None)
        if handler is None:
            return {
                "applied": False,
                "requested_event": requested,
                "actual_event": "",
                "target_vehicle_id": "",
                "used_move": False,
                "reason": SKIP_NO_FEASIBLE_TARGET,
                "skipped_reason": SKIP_NO_FEASIBLE_TARGET,
            }
        return handler(ego, vehicles, requested=requested)

    def _event_hard_brake(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        lead = self._nearest_same_lane_ahead(ego, vehicles)
        if lead is None:
            return self._failed_event_meta(requested=requested, reason=SKIP_NO_FEASIBLE_TARGET)

        self.prepare_vehicle(lead.vehicle_id)
        target_speed = max(HIGH_BRAKE_TARGET_SPEED, min(lead.speed * 0.7, ego.speed * 0.85))
        self._set_speed_smooth(lead.vehicle_id, target_speed)
        move_result = {"applied": False, "skipped_reason": ""}
        if lead.x - ego.x > 34.0:
            move_result = self._move_vehicle_same_road_result(lead, ego, ego.lane_index, gap=HARD_BRAKE_GAP)
        return self._success_event_meta(
            requested=requested,
            actual_event="hard_brake",
            target_vehicle_id=lead.vehicle_id,
            used_move=bool(move_result["applied"]),
        )

    def _event_close_follow(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_same_lane_ahead(ego, vehicles)
        if target is None:
            return self._failed_event_meta(requested=requested, reason=SKIP_NO_FEASIBLE_TARGET)

        self.prepare_vehicle(target.vehicle_id)
        move_result = {"applied": False, "skipped_reason": ""}
        if target.x - ego.x > 30.0:
            move_result = self._move_vehicle_same_road_result(target, ego, ego.lane_index, gap=CLOSE_FOLLOW_GAP)
        target_speed = max(HIGH_BRAKE_TARGET_SPEED + 2.0, min(target.speed, ego.speed * 0.9))
        self._set_speed_smooth(target.vehicle_id, target_speed)
        return self._success_event_meta(
            requested=requested,
            actual_event="close_follow",
            target_vehicle_id=target.vehicle_id,
            used_move=bool(move_result["applied"]),
        )

    def _event_cut_in(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_adjacent_vehicle(ego, vehicles)
        if target is None:
            return self._failed_event_meta(requested=requested, reason=SKIP_NO_FEASIBLE_TARGET)

        self.prepare_vehicle(target.vehicle_id)
        move_result = {"applied": False, "skipped_reason": ""}
        lane_change_result = {"applied": False, "skipped_reason": ""}
        if target.road_id == ego.road_id and abs(target.x - ego.x) <= MAX_CUT_IN_ALIGNMENT_DISTANCE:
            move_result = self._move_vehicle_same_road_result(target, ego, ego.lane_index, gap=CUT_IN_GAP)
        if not move_result["applied"]:
            lane_change_result = self._request_lane_change_result(target, ego.lane_index)
        if not move_result["applied"] and not lane_change_result["applied"]:
            skipped_reason = lane_change_result["skipped_reason"] or move_result["skipped_reason"] or SKIP_NO_FEASIBLE_TARGET
            return self._failed_event_meta(
                requested=requested,
                target_vehicle_id=target.vehicle_id,
                reason=skipped_reason,
            )

        target_speed = min(max(target.speed, ego.speed * 0.95), ego.speed + 0.8)
        self._set_speed_smooth(target.vehicle_id, max(HIGH_BRAKE_TARGET_SPEED, target_speed))
        return self._success_event_meta(
            requested=requested,
            actual_event="cut_in",
            target_vehicle_id=target.vehicle_id,
            used_move=bool(move_result["applied"]),
        )

    def _event_unsafe_merge(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is None:
            return self._failed_event_meta(requested=requested, reason=SKIP_NO_FEASIBLE_TARGET)

        self.prepare_vehicle(target.vehicle_id)
        move_result = self._move_vehicle_along_current_lane_result(target, gap_to_lane_end=MERGE_LANE_END_BUFFER)
        self._set_speed_smooth(target.vehicle_id, max(target.speed, min(ego.speed * 0.95, target.speed + 0.8)))
        if not move_result["applied"]:
            return self._failed_event_meta(
                requested=requested,
                target_vehicle_id=target.vehicle_id,
                reason=move_result["skipped_reason"] or SKIP_NO_FEASIBLE_TARGET,
            )
        return self._success_event_meta(
            requested=requested,
            actual_event="unsafe_merge",
            target_vehicle_id=target.vehicle_id,
            used_move=True,
        )

    def _event_merge_conflict(self, ego: RealVehicleSnapshot, vehicles: List[RealVehicleSnapshot], requested: str) -> Dict[str, Any]:
        if ego.x < -120.0:
            return self._failed_event_meta(requested=requested, reason=SKIP_MERGE_GUARD_BLOCKED)

        target = self._nearest_ramp_vehicle(ego, vehicles)
        if target is None:
            return self._failed_event_meta(requested=requested, reason=SKIP_NO_FEASIBLE_TARGET)

        self.prepare_vehicle(target.vehicle_id)
        move_result = self._move_vehicle_along_current_lane_result(target, gap_to_lane_end=MERGE_LANE_END_BUFFER)
        self._set_speed_smooth(target.vehicle_id, max(target.speed, ego.speed * 0.92))
        if not move_result["applied"]:
            return self._failed_event_meta(
                requested=requested,
                target_vehicle_id=target.vehicle_id,
                reason=move_result["skipped_reason"] or SKIP_NO_FEASIBLE_TARGET,
            )
        return self._success_event_meta(
            requested=requested,
            actual_event="merge_conflict",
            target_vehicle_id=target.vehicle_id,
            used_move=True,
        )

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

    def _resolve_lane_target(self, vehicle: RealVehicleSnapshot, desired_lane: int) -> Dict[str, Any]:
        lane_count = self._lane_count_for_vehicle(vehicle)
        if desired_lane < 0:
            return {
                "allowed": False,
                "lane_index": vehicle.lane_index,
                "lane_violation": True,
                "reason": SKIP_INVALID_LANE_INDEX,
            }
        if desired_lane >= lane_count:
            reason = SKIP_INSUFFICIENT_LANE_COUNT if lane_count <= 1 else SKIP_INVALID_LANE_INDEX
            return {
                "allowed": False,
                "lane_index": vehicle.lane_index,
                "lane_violation": True,
                "reason": reason,
            }
        if desired_lane == vehicle.lane_index:
            return {
                "allowed": True,
                "lane_index": desired_lane,
                "lane_violation": False,
                "reason": "",
            }
        if self._near_lane_end(vehicle):
            return {
                "allowed": False,
                "lane_index": vehicle.lane_index,
                "lane_violation": False,
                "reason": SKIP_MERGE_GUARD_BLOCKED,
            }

        target_lane_id = self._lane_id(vehicle.road_id, desired_lane)
        if not target_lane_id:
            return {
                "allowed": False,
                "lane_index": vehicle.lane_index,
                "lane_violation": False,
                "reason": SKIP_INVALID_LANE_INDEX,
            }

        next_edge = self._next_route_edge(vehicle.vehicle_id)
        if next_edge and not self._lane_reaches_edge(target_lane_id, next_edge):
            return {
                "allowed": False,
                "lane_index": vehicle.lane_index,
                "lane_violation": False,
                "reason": SKIP_NO_ROUTE_CONNECTION,
            }

        return {
            "allowed": True,
            "lane_index": desired_lane,
            "lane_violation": False,
            "reason": "",
        }

    def _request_lane_change_result(self, vehicle: RealVehicleSnapshot, desired_lane: int) -> Dict[str, Any]:
        lane_target = self._resolve_lane_target(vehicle, desired_lane)
        if not lane_target["allowed"]:
            return {
                "applied": False,
                "target_lane": vehicle.lane_index,
                "lane_violation": bool(lane_target["lane_violation"]),
                "skipped_reason": str(lane_target["reason"] or SKIP_NO_FEASIBLE_TARGET),
            }
        self._safe_call(
            self.api.vehicle.changeLane,
            vehicle.vehicle_id,
            lane_target["lane_index"],
            max(1.0, self.config.step_length * 2.0),
        )
        return {
            "applied": True,
            "target_lane": lane_target["lane_index"],
            "lane_violation": False,
            "skipped_reason": "",
        }

    def _request_lane_change(self, vehicle: RealVehicleSnapshot, desired_lane: int) -> bool:
        return bool(self._request_lane_change_result(vehicle, desired_lane)["applied"])

    def _move_vehicle_same_road_result(
        self,
        target: RealVehicleSnapshot,
        ego: RealVehicleSnapshot,
        lane_index: int,
        gap: float,
    ) -> Dict[str, Any]:
        if target.road_id != ego.road_id or not target.road_id:
            return {"applied": False, "skipped_reason": SKIP_NO_ROUTE_CONNECTION}
        lane_target = self._resolve_lane_target(target, lane_index)
        if not lane_target["allowed"]:
            return {"applied": False, "skipped_reason": str(lane_target["reason"] or SKIP_NO_FEASIBLE_TARGET)}
        lane_id = self._lane_id(target.road_id, lane_target["lane_index"])
        if not lane_id:
            return {"applied": False, "skipped_reason": SKIP_INVALID_LANE_INDEX}
        lane_length = self._lane_length(lane_id, default=max(ego.lane_pos + gap + 1.0, 10.0))
        gap = max(gap, MOVE_PLACEMENT_CLEARANCE)
        target_pos = min(max(ego.lane_pos + gap, 1.0), max(1.0, lane_length - 1.0))
        if lane_length - target_pos < JUNCTION_GUARD_DISTANCE:
            return {"applied": False, "skipped_reason": SKIP_MERGE_GUARD_BLOCKED}
        if not self._placement_is_clear(target, target.road_id, lane_target["lane_index"], target_pos, clearance=gap):
            return {"applied": False, "skipped_reason": SKIP_PLACEMENT_CONFLICT}
        try:
            self.api.vehicle.moveTo(target.vehicle_id, lane_id, target_pos)
            return {"applied": True, "skipped_reason": ""}
        except Exception:
            return {"applied": False, "skipped_reason": SKIP_PLACEMENT_CONFLICT}

    def _move_vehicle_same_road(self, target: RealVehicleSnapshot, ego: RealVehicleSnapshot, lane_index: int, gap: float) -> bool:
        return bool(self._move_vehicle_same_road_result(target, ego, lane_index, gap)["applied"])

    def _move_vehicle_along_current_lane_result(self, target: RealVehicleSnapshot, gap_to_lane_end: float) -> Dict[str, Any]:
        if not target.lane_id or not target.road_id:
            return {"applied": False, "skipped_reason": SKIP_NO_ROUTE_CONNECTION}
        gap_to_lane_end = max(gap_to_lane_end, JUNCTION_GUARD_DISTANCE)
        lane_length = self._lane_length(target.lane_id, default=target.lane_pos + gap_to_lane_end + 1.0)
        target_pos = max(1.0, lane_length - gap_to_lane_end)
        next_edge = self._next_route_edge(target.vehicle_id)
        if next_edge and not self._lane_reaches_edge(target.lane_id, next_edge):
            return {"applied": False, "skipped_reason": SKIP_NO_ROUTE_CONNECTION}
        if lane_length - target_pos < JUNCTION_GUARD_DISTANCE:
            return {"applied": False, "skipped_reason": SKIP_MERGE_GUARD_BLOCKED}
        if not self._placement_is_clear(target, target.road_id, target.lane_index, target_pos, clearance=MOVE_PLACEMENT_CLEARANCE):
            return {"applied": False, "skipped_reason": SKIP_PLACEMENT_CONFLICT}
        try:
            self.api.vehicle.moveTo(target.vehicle_id, target.lane_id, target_pos)
            return {"applied": True, "skipped_reason": ""}
        except Exception:
            return {"applied": False, "skipped_reason": SKIP_PLACEMENT_CONFLICT}

    def _move_vehicle_along_current_lane(self, target: RealVehicleSnapshot, gap_to_lane_end: float) -> bool:
        return bool(self._move_vehicle_along_current_lane_result(target, gap_to_lane_end)["applied"])

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
        candidates = [vehicle for vehicle in vehicles if vehicle.road_id.startswith("ramp") or vehicle.road_id.startswith(":merge_3")]
        if not candidates:
            return None
        candidates.sort(key=lambda vehicle: abs(vehicle.x - ego.x))
        return candidates[0]

    def _set_speed_smooth(self, vehicle_id: str, target_speed: float) -> bool:
        target_speed = max(0.0, float(target_speed))
        slow_down = getattr(self.api.vehicle, "slowDown", None)
        duration = max(1.0, self.config.step_length * 12.0)
        if callable(slow_down):
            try:
                slow_down(vehicle_id, target_speed, duration)
                return True
            except Exception:
                pass
        try:
            self.api.vehicle.setSpeed(vehicle_id, target_speed)
            return True
        except Exception:
            return False

    def _lane_id(self, road_id: str, lane_index: int) -> str:
        if not road_id:
            return ""
        return f"{road_id}_{lane_index}"

    def _lane_length(self, lane_id: str, default: float) -> float:
        return float(self._safe_call(self.api.lane.getLength, lane_id, default=default))

    def _near_lane_end(self, vehicle: RealVehicleSnapshot) -> bool:
        if not vehicle.lane_id:
            return False
        lane_length = self._lane_length(vehicle.lane_id, default=vehicle.lane_pos + JUNCTION_GUARD_DISTANCE + 1.0)
        return (lane_length - vehicle.lane_pos) < JUNCTION_GUARD_DISTANCE

    def _next_route_edge(self, vehicle_id: str) -> Optional[str]:
        route = self._safe_call(getattr(self.api.vehicle, "getRoute", lambda *_args: None), vehicle_id, default=None)
        route_index = self._safe_call(getattr(self.api.vehicle, "getRouteIndex", lambda *_args: None), vehicle_id, default=None)
        if not route or route_index is None:
            return None
        try:
            next_index = int(route_index) + 1
        except Exception:
            return None
        if next_index < len(route):
            return str(route[next_index])
        return None

    def _lane_reaches_edge(self, lane_id: str, target_edge: str, depth: int = 2, visited: Optional[Set[str]] = None) -> bool:
        if not lane_id or not target_edge:
            return True
        if visited is None:
            visited = set()
        if lane_id in visited:
            return False
        visited.add(lane_id)

        edge_id = self._edge_from_lane_id(lane_id)
        if edge_id == target_edge:
            return True

        links = self._safe_call(getattr(self.api.lane, "getLinks", lambda *_args: []), lane_id, default=[])
        for link in links or []:
            next_lane_id = self._extract_link_lane_id(link)
            if not next_lane_id:
                continue
            next_edge_id = self._edge_from_lane_id(next_lane_id)
            if next_edge_id == target_edge:
                return True
            if depth > 0 and next_edge_id.startswith(":") and self._lane_reaches_edge(next_lane_id, target_edge, depth - 1, visited):
                return True
        return False

    def _extract_link_lane_id(self, link: Any) -> str:
        if isinstance(link, str):
            return link
        if isinstance(link, (list, tuple)) and link:
            head = link[0]
            if isinstance(head, str):
                return head
        return ""

    def _edge_from_lane_id(self, lane_id: str) -> str:
        if "_" not in lane_id:
            return lane_id
        return lane_id.rsplit("_", 1)[0]

    def _success_event_meta(self, requested: str, actual_event: str, target_vehicle_id: str, used_move: bool) -> Dict[str, Any]:
        return {
            "applied": True,
            "requested_event": requested,
            "actual_event": actual_event,
            "target_vehicle_id": target_vehicle_id,
            "used_move": bool(used_move),
            "reason": "",
            "skipped_reason": "",
        }

    def _failed_event_meta(self, requested: str, reason: str, target_vehicle_id: str = "") -> Dict[str, Any]:
        normalized_reason = str(reason or SKIP_NO_FEASIBLE_TARGET)
        return {
            "applied": False,
            "requested_event": requested,
            "actual_event": "",
            "target_vehicle_id": target_vehicle_id,
            "used_move": False,
            "reason": normalized_reason,
            "skipped_reason": normalized_reason,
        }

    @staticmethod
    def _safe_call(func, *args, default=None):
        try:
            return func(*args)
        except Exception:
            return default
