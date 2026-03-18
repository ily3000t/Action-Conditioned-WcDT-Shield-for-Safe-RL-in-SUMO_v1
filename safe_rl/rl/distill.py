from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn

from safe_rl.buffer import InterventionBuffer
from safe_rl.config.config import DistillConfig
from safe_rl.models.features import BASE_FEATURE_DIM, encode_history


class DistilledPolicyNet(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BASE_FEATURE_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 9),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DistilledPolicy:
    model: DistilledPolicyNet
    device: torch.device

    @torch.no_grad()
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        x = torch.from_numpy(observation.astype(np.float32)).to(self.device).unsqueeze(0)
        logits = self.model(x)
        return int(torch.argmax(logits, dim=-1).item())


class PolicyDistiller:
    def __init__(self, config: DistillConfig, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or "cpu")
        self.model = DistilledPolicyNet().to(self.device)

    def should_distill(self, buffer: InterventionBuffer) -> bool:
        return len(buffer) >= self.config.trigger_buffer_size

    def distill(self, buffer: InterventionBuffer, tb_writer=None) -> DistilledPolicy:
        if len(buffer) == 0:
            self.model.eval()
            return DistilledPolicy(model=self.model, device=self.device)

        records = buffer.all_records()
        x = np.stack([encode_history(record.history_scene) for record in records], axis=0).astype(np.float32)
        y = np.array([record.final_action for record in records], dtype=np.int64)

        x_t = torch.from_numpy(x).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        batch_size = max(1, self.config.batch_size)
        global_step = 0
        for epoch_idx in range(self.config.epochs):
            permutation = torch.randperm(x_t.shape[0], device=self.device)
            epoch_ce_sum = 0.0
            epoch_steps = 0

            for start in range(0, x_t.shape[0], batch_size):
                idx = permutation[start:start + batch_size]
                logits = self.model(x_t[idx])
                loss = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_ce = float(loss.item())
                epoch_ce_sum += step_ce
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar("loss/step_ce", step_ce, global_step)
                global_step += 1

            with torch.no_grad():
                all_logits = self.model(x_t)
                acc = float((torch.argmax(all_logits, dim=-1) == y_t).to(torch.float32).mean().item())

            epoch_ce = epoch_ce_sum / max(1, epoch_steps)
            if tb_writer is not None:
                tb_writer.add_scalar("loss/epoch_ce", epoch_ce, epoch_idx)
                tb_writer.add_scalar("metric/epoch_action_acc", acc, epoch_idx)

        self.model.eval()
        return DistilledPolicy(model=self.model, device=self.device)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
