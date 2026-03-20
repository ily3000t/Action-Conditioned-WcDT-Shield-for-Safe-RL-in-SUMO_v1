from pathlib import Path
from typing import Any, Dict, List, Optional
import datetime as dt
import json


class IncrementalSessionEventLogger:
    def __init__(self, path: str, stage: str, run_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.path = Path(path)
        self.stage = str(stage)
        self.run_id = str(run_id)
        self.created_at = dt.datetime.now().isoformat(timespec="seconds")
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.events: List[Dict[str, Any]] = []
        self._flush()

    def set_metadata(self, **metadata: Any):
        self.metadata.update(metadata)
        self._flush()

    def append_event(self, event: Dict[str, Any]):
        payload = dict(event)
        payload.setdefault("event_index", len(self.events) + 1)
        payload.setdefault("logged_at", dt.datetime.now().isoformat(timespec="seconds"))
        self.events.append(payload)
        self._flush()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "metadata": self.metadata,
            "event_count": len(self.events),
            "events": list(self.events),
        }

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2, default=str)
