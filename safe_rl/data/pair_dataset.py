import pickle
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from torch.utils.data import Dataset
except Exception:
    class Dataset:  # type: ignore
        pass

from safe_rl.data.types import RiskPairSample


class RiskPairDataset(Dataset):
    def __init__(self, samples: Sequence[RiskPairSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RiskPairSample:
        return self.samples[index]


def collate_risk_pairs(batch: Sequence[RiskPairSample]) -> List[RiskPairSample]:
    return list(batch)


def save_risk_pairs(path: str, samples: Sequence[RiskPairSample]):
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(list(samples), f)


def load_risk_pairs(path: str) -> List[RiskPairSample]:
    with open(path, "rb") as f:
        return list(pickle.load(f))


def summarize_pair_sources(samples: Sequence[RiskPairSample]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        source = str(sample.source or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))
