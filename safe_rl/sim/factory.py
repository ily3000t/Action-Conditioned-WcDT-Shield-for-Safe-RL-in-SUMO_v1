from safe_rl.config.config import SimConfig
from safe_rl.sim.backend_interface import ISumoBackend
from safe_rl.sim.libsumo_backend import LibsumoBackend
from safe_rl.sim.traci_backend import TraciBackend


def create_backend(config: SimConfig) -> ISumoBackend:
    backend_name = (config.backend or "traci").lower().strip()
    if backend_name == "libsumo":
        return LibsumoBackend(config)
    return TraciBackend(config)
