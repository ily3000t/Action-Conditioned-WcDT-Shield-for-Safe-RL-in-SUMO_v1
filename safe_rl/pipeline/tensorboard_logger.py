import datetime as dt
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from safe_rl.config.config import SafeRLConfig, TensorboardConfig


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", (value or "").strip())
    return normalized.strip("_")


class TensorboardManager:
    def __init__(self, config: TensorboardConfig, stage_prefix: str = ""):
        self.config = config
        self.enabled = bool(config.enabled)
        self.available = False
        self.run_dir: Optional[Path] = None
        self._writer_cls = None
        self._writers: Dict[str, object] = {}

        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer_cls = SummaryWriter
            self.available = True
        except Exception:
            self.available = False

        if self.available:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            stage = _safe_name(stage_prefix)
            suffix = _safe_name(config.run_name)
            parts = []
            if stage:
                parts.append(stage)
            parts.append(ts)
            if suffix:
                parts.append(suffix)
            run_name = "_".join(parts)
            self.run_dir = Path(config.root_dir) / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def is_enabled(self) -> bool:
        return bool(self.enabled and self.available and self.run_dir is not None)

    def get_writer(self, module_name: str):
        if not self.is_enabled():
            return None

        key = _safe_name(module_name) or "default"
        if key in self._writers:
            return self._writers[key]

        module_dir = self.run_dir / key
        module_dir.mkdir(parents=True, exist_ok=True)
        writer = self._writer_cls(log_dir=str(module_dir), flush_secs=int(self.config.flush_secs))
        self._writers[key] = writer
        return writer

    def add_scalar(self, module_name: str, tag: str, value: float, step: int):
        writer = self.get_writer(module_name)
        if writer is None:
            return
        writer.add_scalar(tag, float(value), int(step))

    def log_run_metadata(self, config: SafeRLConfig, metadata: Dict[str, object]):
        writer = self.get_writer("eval")
        if writer is None:
            return
        writer.add_text("meta/config", json.dumps(asdict(config), ensure_ascii=False, indent=2), 0)
        if metadata:
            writer.add_text("meta/runtime", json.dumps(metadata, ensure_ascii=False, indent=2), 0)

    def close(self):
        for writer in self._writers.values():
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass
        self._writers.clear()