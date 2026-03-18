import logging
from pathlib import Path
from typing import Optional

from safe_rl.config.config import SimConfig
from safe_rl.sim.backend_interface import BackendStepResult, ISumoBackend
from safe_rl.sim.mock_core import MockTrafficCore
from safe_rl.sim.sumo_utils import (
    maybe_build_network_from_plain,
    prepare_sumo_python_path,
    resolve_sumo_binary,
)


_LOGGER = logging.getLogger(__name__)


class LibsumoBackend(ISumoBackend):
    def __init__(self, config: SimConfig):
        self.config = config
        self._libsumo = None
        self._started = False
        self._cfg_path: Optional[Path] = None
        self._sumo_binary: Optional[str] = None
        self._mock = MockTrafficCore(
            episode_steps=config.episode_steps,
            step_length=config.step_length,
            seed=config.random_seed,
        )
        self._use_mock = True

    def start(self):
        if self.config.force_mock:
            _LOGGER.warning("force_mock=true, using mock backend.")
            self._started = True
            self._use_mock = True
            return

        cfg_path = Path(self.config.sumo_cfg)
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        self._cfg_path = cfg_path

        prepare_sumo_python_path(self.config)

        try:
            import libsumo  # type: ignore

            self._libsumo = libsumo
            self._sumo_binary = resolve_sumo_binary(self.config, use_gui=False)

            if cfg_path.is_file():
                ok, message = maybe_build_network_from_plain(cfg_path, self.config)
                if not ok:
                    _LOGGER.warning("SUMO cfg check failed (%s), fallback to mock backend.", message)
                else:
                    _LOGGER.info(message)
                    self._libsumo.start([self._sumo_binary, "-c", str(cfg_path), "--seed", str(self.config.random_seed)])
                    self._use_mock = False
                    _LOGGER.info("libsumo backend started in real SUMO mode with cfg=%s", cfg_path)
            else:
                _LOGGER.warning("SUMO cfg not found (%s), fallback to mock backend.", cfg_path)
        except Exception as exc:
            _LOGGER.warning("libsumo unavailable (%s), fallback to mock backend.", exc)
        self._started = True

    def _load_with_seed(self, seed: Optional[int]):
        cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
        load_args = ["-c", str(cfg_path)]
        if seed is not None:
            load_args += ["--seed", str(int(seed))]
        self._libsumo.load(load_args)

    def _warmup_after_reset(self):
        max_steps = max(10, int(5.0 / max(self.config.step_length, 1e-3)))
        for _ in range(max_steps):
            vehicle_ids = self._libsumo.vehicle.getIDList()
            if vehicle_ids:
                return
            self._libsumo.simulationStep()
        _LOGGER.warning(
            "libsumo reset warmup finished with no active vehicle (steps=%d). Continue with placeholder scene.",
            max_steps,
        )

    def reset(self, seed: Optional[int] = None):
        if not self._started:
            self.start()
        if self._use_mock:
            return self._mock.reset(seed=seed)

        try:
            self._load_with_seed(seed)
        except Exception as exc:
            _LOGGER.warning("libsumo load failed (%s), restarting SUMO once.", exc)
            try:
                self._libsumo.close()
            except Exception:
                pass
            cfg_path = self._cfg_path or Path(self.config.sumo_cfg).resolve()
            self._libsumo.start([self._sumo_binary, "-c", str(cfg_path), "--seed", str(int(seed or self.config.random_seed))])
            self._load_with_seed(seed)

        self._warmup_after_reset()
        return self.get_state()

    def step(self, action_id: int) -> BackendStepResult:
        if self._use_mock:
            scene, task_reward, done, info = self._mock.step(action_id)
            return BackendStepResult(scene=scene, task_reward=task_reward, done=done, info=info)

        self._libsumo.simulationStep()
        scene = self.get_state()
        done = self._libsumo.simulation.getMinExpectedNumber() <= 0
        ego_speed = 0.0
        if scene.vehicles:
            try:
                ego_speed = float(self._libsumo.vehicle.getSpeed(scene.ego_id))
            except Exception:
                ego_speed = 0.0
        task_reward = float(ego_speed * self.config.step_length * 0.1)
        return BackendStepResult(scene=scene, task_reward=task_reward, done=done, info={"collision": False, "ego_speed": ego_speed})

    def inject_risk_event(self, event_type: Optional[str] = None):
        if self._use_mock:
            self._mock.inject_risk_event(event_type)
            return
        _LOGGER.info("libsumo risk event injection placeholder invoked: %s", event_type)

    def get_state(self):
        if self._use_mock:
            return self._mock.get_scene(timestamp=self._mock.step_index * self.config.step_length)

        from safe_rl.data.types import SceneState, VehicleState

        vehicle_ids = self._libsumo.vehicle.getIDList()
        vehicles = []
        for vid in vehicle_ids:
            x, y = self._libsumo.vehicle.getPosition(vid)
            speed = self._libsumo.vehicle.getSpeed(vid)
            lane_index = self._libsumo.vehicle.getLaneIndex(vid)
            vehicles.append(
                VehicleState(
                    vehicle_id=vid,
                    x=float(x),
                    y=float(y),
                    vx=float(speed),
                    vy=0.0,
                    ax=0.0,
                    ay=0.0,
                    heading=0.0,
                    lane_id=int(lane_index),
                )
            )
        ego_id = vehicle_ids[0] if vehicle_ids else "ego"
        return SceneState(timestamp=float(self._libsumo.simulation.getTime()), ego_id=ego_id, vehicles=vehicles)

    def close(self):
        if not self._started:
            return
        if not self._use_mock and self._libsumo is not None:
            try:
                self._libsumo.close()
            except Exception:
                pass
        self._started = False
