import pickle
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from safe_rl.data.types import InterventionRecord


class InterventionBuffer:
    def __init__(self, capacity: int = 100000, seed: int = 42):
        self.capacity = capacity
        self._records: List[InterventionRecord] = []
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._records)

    def push(self, record: InterventionRecord):
        if len(self._records) >= self.capacity:
            self._records.pop(0)
        self._records.append(record)

    def sample(self, batch_size: int) -> List[InterventionRecord]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if len(self._records) == 0:
            return []
        batch_size = min(batch_size, len(self._records))
        return self._rng.sample(self._records, batch_size)

    def all_records(self) -> List[InterventionRecord]:
        return list(self._records)

    def stats(self) -> Dict[str, float]:
        total = len(self._records)
        if total == 0:
            return {
                "size": 0,
                "mean_raw_risk": 0.0,
                "mean_final_risk": 0.0,
                "mean_risk_reduction": 0.0,
            }
        raw = [record.raw_risk for record in self._records]
        final = [record.final_risk for record in self._records]
        reduction = [r - f for r, f in zip(raw, final)]
        return {
            "size": float(total),
            "mean_raw_risk": float(sum(raw) / total),
            "mean_final_risk": float(sum(final) / total),
            "mean_risk_reduction": float(sum(reduction) / total),
        }

    def save(self, path: str):
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            pickle.dump(self._records, f)

    def save_json(self, path: str):
        import json

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump([asdict(record) for record in self._records], f, ensure_ascii=False)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._records = pickle.load(f)

    def extend(self, records: Sequence[InterventionRecord]):
        for record in records:
            self.push(record)
