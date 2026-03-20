import re
import uuid
from pathlib import Path

from safe_rl.config.config import TensorboardConfig
from safe_rl.pipeline.tensorboard_logger import TensorboardManager


def _tb_root() -> Path:
    path = Path("safe_rl_output/test_artifacts/tensorboard") / uuid.uuid4().hex[:8]
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_tensorboard_manager_run_dir_and_writer():
    config = TensorboardConfig(
        enabled=True,
        root_dir=str(_tb_root()),
        run_name="quick check",
        flush_secs=1,
    )
    manager = TensorboardManager(config)

    if not manager.is_enabled():
        assert manager.get_writer("eval") is None
        return

    assert manager.run_dir is not None
    assert re.match(r"^\d{8}_\d{6}_quick_check$", manager.run_dir.name)

    writer = manager.get_writer("world_model")
    assert writer is not None
    writer.add_scalar("loss/step_total", 1.23, 0)
    manager.close()

    module_dir = manager.run_dir / "world_model"
    files = list(module_dir.glob("events.out.tfevents.*"))
    assert len(files) >= 1


def test_tensorboard_manager_disabled():
    config = TensorboardConfig(enabled=False, root_dir=str(_tb_root()), run_name="", flush_secs=1)
    manager = TensorboardManager(config)
    assert manager.is_enabled() is False
    assert manager.get_writer("eval") is None


def test_tensorboard_manager_stage_prefix_in_run_name():
    config = TensorboardConfig(
        enabled=True,
        root_dir=str(_tb_root()),
        run_name="quick",
        flush_secs=1,
    )
    manager = TensorboardManager(config, stage_prefix="stage3")

    if not manager.is_enabled():
        return

    assert manager.run_dir is not None
    assert re.match(r"^stage3_\d{8}_\d{6}_quick$", manager.run_dir.name)
    manager.close()
