import pickle
import random
from pathlib import Path
from typing import List, Sequence, Tuple

from safe_rl.config.config import DatasetConfig, SimConfig
from safe_rl.data.risk import aggregate_future_risk
from safe_rl.data.types import ActionConditionedSample, EpisodeLog


class ActionConditionedDatasetBuilder:
    def __init__(self, sim_config: SimConfig, dataset_config: DatasetConfig):
        self.sim_config = sim_config
        self.dataset_config = dataset_config

    def build_samples(self, episodes: Sequence[EpisodeLog]) -> List[ActionConditionedSample]:
        samples: List[ActionConditionedSample] = []
        h = self.sim_config.history_steps
        f = self.sim_config.future_steps

        for episode in episodes:
            if len(episode.steps) < h + f + 1:
                continue
            for t in range(h, len(episode.steps) - f):
                history_scene = [step.scene for step in episode.steps[t - h:t]]
                candidate_action = episode.steps[t].raw_action
                future_scene = [step.scene for step in episode.steps[t + 1:t + 1 + f]]
                lane_violation = episode.steps[t].risk_labels.lane_violation
                risk_labels = aggregate_future_risk(
                    future_scene,
                    ttc_threshold=self.dataset_config.ttc_threshold,
                    lane_violation=lane_violation,
                )
                sample = ActionConditionedSample(
                    history_scene=history_scene,
                    candidate_action=candidate_action,
                    future_scene=future_scene,
                    risk_labels=risk_labels,
                    meta={
                        "episode_id": episode.episode_id,
                        "step_index": t,
                        "risky_mode": episode.risky_mode,
                    },
                )
                samples.append(sample)
        return samples

    def split_dataset(
        self,
        samples: Sequence[ActionConditionedSample],
        seed: int = 42,
    ) -> Tuple[List[ActionConditionedSample], List[ActionConditionedSample], List[ActionConditionedSample]]:
        samples = list(samples)
        random.Random(seed).shuffle(samples)
        total = len(samples)
        train_end = int(total * self.dataset_config.train_ratio)
        val_end = train_end + int(total * self.dataset_config.val_ratio)
        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]
        return train, val, test

    def save_splits(
        self,
        train: Sequence[ActionConditionedSample],
        val: Sequence[ActionConditionedSample],
        test: Sequence[ActionConditionedSample],
    ):
        output_dir = Path(self.dataset_config.dataset_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save(output_dir / "train.pkl", list(train))
        self._save(output_dir / "val.pkl", list(val))
        self._save(output_dir / "test.pkl", list(test))

    @staticmethod
    def _save(path: Path, value):
        with path.open("wb") as f:
            pickle.dump(value, f)

    @staticmethod
    def load(path: str) -> List[ActionConditionedSample]:
        with open(path, "rb") as f:
            return pickle.load(f)
