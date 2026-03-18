import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from safe_rl.config.config import LightRiskConfig
from safe_rl.data.risk import risk_targets
from safe_rl.data.types import ActionConditionedSample, RiskPrediction
from safe_rl.models.features import ACTION_DIM, BASE_FEATURE_DIM, history_action_feature


class LightRiskMLP(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        input_dim = BASE_FEATURE_DIM + ACTION_DIM
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.risk_head = nn.Linear(hidden_dim, 4)
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor):
        feature = self.backbone(x)
        risk_logits = self.risk_head(feature)
        uncertainty = self.uncertainty_head(feature)
        return risk_logits, uncertainty


class _LightRiskDataset(Dataset):
    def __init__(self, samples: Sequence[ActionConditionedSample]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = history_action_feature(sample.history_scene, sample.candidate_action)
        y = np.array(risk_targets(sample.risk_labels), dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


@dataclass
class LightRiskPredictor:
    model: LightRiskMLP
    device: torch.device

    @torch.no_grad()
    def predict(self, history_scene, action_id: int) -> RiskPrediction:
        x = history_action_feature(history_scene, action_id)
        x_t = torch.from_numpy(x).to(self.device).unsqueeze(0)
        logits, uncertainty = self.model(x_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        unc = float(uncertainty.squeeze(0).item())
        return RiskPrediction(
            p_collision=float(probs[0]),
            p_ttc=float(probs[1]),
            p_lane_violation=float(probs[2]),
            p_overall=float(probs[3]),
            uncertainty=unc,
        )


class LightRiskTrainer:
    def __init__(self, config: LightRiskConfig, seed: int = 42, device: Optional[str] = None):
        self.config = config
        self.rng = random.Random(seed)
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LightRiskMLP(hidden_dim=config.hidden_dim).to(self.device)

    def fit(
        self,
        train_samples: Sequence[ActionConditionedSample],
        val_samples: Optional[Sequence[ActionConditionedSample]] = None,
        tb_writer=None,
    ) -> LightRiskPredictor:
        if len(train_samples) == 0:
            return LightRiskPredictor(model=self.model.eval(), device=self.device)

        print(f"[LightRisk] start training on {self.device}, samples={len(train_samples)}")
        train_loader = DataLoader(
            _LightRiskDataset(train_samples),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        bce = nn.BCEWithLogitsLoss()

        self.model.train()
        global_step = 0
        for epoch_idx in range(self.config.epochs):
            epoch_total_sum = 0.0
            epoch_risk_sum = 0.0
            epoch_uncertainty_sum = 0.0
            epoch_steps = 0

            for x, y in train_loader:
                x = x.to(self.device).to(torch.float32)
                y = y.to(self.device).to(torch.float32)
                logits, uncertainty = self.model(x)
                risk_loss = bce(logits, y)
                overall_error = torch.abs(torch.sigmoid(logits[:, 3]) - y[:, 3])
                uncertainty_loss = torch.mean((uncertainty.squeeze(-1) - overall_error.detach()) ** 2)
                loss = risk_loss + 0.1 * uncertainty_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_total = float(loss.item())
                step_risk = float(risk_loss.item())
                step_uncertainty = float(uncertainty_loss.item())
                epoch_total_sum += step_total
                epoch_risk_sum += step_risk
                epoch_uncertainty_sum += step_uncertainty
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar("loss/step_total", step_total, global_step)
                    tb_writer.add_scalar("loss/step_risk", step_risk, global_step)
                    tb_writer.add_scalar("loss/step_uncertainty", step_uncertainty, global_step)
                global_step += 1

            avg_total = epoch_total_sum / max(1, epoch_steps)
            avg_risk = epoch_risk_sum / max(1, epoch_steps)
            avg_uncertainty = epoch_uncertainty_sum / max(1, epoch_steps)
            print(f"[LightRisk] epoch {epoch_idx + 1}/{self.config.epochs}, loss={avg_total:.6f}")

            if tb_writer is not None:
                tb_writer.add_scalar("loss/epoch_total", avg_total, epoch_idx)
                tb_writer.add_scalar("loss/epoch_risk", avg_risk, epoch_idx)
                tb_writer.add_scalar("loss/epoch_uncertainty", avg_uncertainty, epoch_idx)

            if val_samples:
                val_total, val_risk, val_uncertainty = self._evaluate(val_samples, bce)
                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss_epoch_total", val_total, epoch_idx)
                    tb_writer.add_scalar("val/loss_epoch_risk", val_risk, epoch_idx)
                    tb_writer.add_scalar("val/loss_epoch_uncertainty", val_uncertainty, epoch_idx)

        self.model.eval()
        return LightRiskPredictor(model=self.model, device=self.device)

    @torch.no_grad()
    def _evaluate(self, samples: Sequence[ActionConditionedSample], bce: nn.Module):
        if len(samples) == 0:
            return 0.0, 0.0, 0.0

        loader = DataLoader(
            _LightRiskDataset(samples),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        was_training = self.model.training
        self.model.eval()
        total_sum = 0.0
        risk_sum = 0.0
        uncertainty_sum = 0.0
        steps = 0

        for x, y in loader:
            x = x.to(self.device).to(torch.float32)
            y = y.to(self.device).to(torch.float32)
            logits, uncertainty = self.model(x)
            risk_loss = bce(logits, y)
            overall_error = torch.abs(torch.sigmoid(logits[:, 3]) - y[:, 3])
            uncertainty_loss = torch.mean((uncertainty.squeeze(-1) - overall_error.detach()) ** 2)
            total_loss = risk_loss + 0.1 * uncertainty_loss

            total_sum += float(total_loss.item())
            risk_sum += float(risk_loss.item())
            uncertainty_sum += float(uncertainty_loss.item())
            steps += 1

        if was_training:
            self.model.train()

        denom = max(1, steps)
        return total_sum / denom, risk_sum / denom, uncertainty_sum / denom

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()


def create_untrained_light_predictor(config: LightRiskConfig, device: Optional[str] = None) -> LightRiskPredictor:
    trainer = LightRiskTrainer(config=config, device=device)
    trainer.model.eval()
    return LightRiskPredictor(model=trainer.model, device=trainer.device)
