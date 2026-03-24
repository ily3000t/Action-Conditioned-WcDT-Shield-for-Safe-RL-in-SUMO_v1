import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from safe_rl.config.config import LightRiskConfig
from safe_rl.data.pair_dataset import RiskPairDataset, collate_risk_pairs
from safe_rl.data.risk import risk_targets
from safe_rl.data.types import ActionConditionedSample, RiskPairSample, RiskPrediction
from safe_rl.models.features import ACTION_DIM, BASE_FEATURE_DIM, history_action_feature


PAIR_RANK_MARGIN = 0.02
SPREAD_TARGET_DELTA = 0.15
SPREAD_MIN_GAP = 0.02


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
        self.risk_type_head = nn.Linear(hidden_dim, 3)
        self.risk_score_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feature = self.backbone(x)
        risk_type_logits = self.risk_type_head(feature)
        risk_score = torch.sigmoid(self.risk_score_head(feature).squeeze(-1))
        uncertainty = self.uncertainty_head(feature).squeeze(-1)
        return {
            "risk_type_logits": risk_type_logits,
            "risk_score": risk_score,
            "uncertainty": uncertainty,
        }


class _LightRiskDataset(Dataset):
    def __init__(self, samples: Sequence[ActionConditionedSample]):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = history_action_feature(sample.history_scene, sample.candidate_action)
        collision, ttc_risk, lane_violation, overall_risk = risk_targets(sample.risk_labels)
        y_types = np.array([collision, ttc_risk, lane_violation], dtype=np.float32)
        y_score = np.array([overall_risk], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y_types), torch.from_numpy(y_score)


@dataclass
class LightRiskPredictor:
    model: LightRiskMLP
    device: torch.device

    @torch.no_grad()
    def predict(self, history_scene, action_id: int) -> RiskPrediction:
        x = history_action_feature(history_scene, action_id)
        x_t = torch.from_numpy(x).to(self.device).unsqueeze(0)
        output = self.model(x_t)
        type_probs = torch.sigmoid(output["risk_type_logits"]).squeeze(0).cpu().numpy()
        risk_score = float(output["risk_score"].squeeze(0).item())
        unc = float(output["uncertainty"].squeeze(0).item())
        return RiskPrediction(
            p_collision=float(type_probs[0]),
            p_ttc=float(type_probs[1]),
            p_lane_violation=float(type_probs[2]),
            p_overall=risk_score,
            uncertainty=unc,
            risk_score=risk_score,
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
        self.huber = nn.HuberLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.last_train_report: Dict[str, Any] = {}
        self.last_pair_metrics: Dict[str, float] = self._empty_pair_metrics()
        self.last_pair_ft_report: Dict[str, Any] = {}

    def fit(
        self,
        train_samples: Sequence[ActionConditionedSample],
        val_samples: Optional[Sequence[ActionConditionedSample]] = None,
        tb_writer=None,
    ) -> LightRiskPredictor:
        if len(train_samples) == 0:
            self.last_train_report = {
                "variant": "v2" if self.config.enable_v2 else "v1_compat",
                "epochs": 0,
                "train_samples": 0,
                "val_samples": len(val_samples or []),
                "epoch_metrics": [],
            }
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

        self.model.train()
        global_step = 0
        epoch_metrics: List[Dict[str, float]] = []
        for epoch_idx in range(self.config.epochs):
            epoch_total_sum = 0.0
            epoch_type_sum = 0.0
            epoch_score_sum = 0.0
            epoch_uncertainty_sum = 0.0
            epoch_steps = 0

            for x, y_types, y_score in train_loader:
                x = x.to(self.device).to(torch.float32)
                y_types = y_types.to(self.device).to(torch.float32)
                y_score = y_score.to(self.device).to(torch.float32).squeeze(-1)
                output = self.model(x)
                total_loss, type_loss, score_loss, uncertainty_loss = self._compute_pointwise_losses(output, y_types, y_score)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step_total = float(total_loss.item())
                step_type = float(type_loss.item())
                step_score = float(score_loss.item())
                step_uncertainty = float(uncertainty_loss.item())
                epoch_total_sum += step_total
                epoch_type_sum += step_type
                epoch_score_sum += step_score
                epoch_uncertainty_sum += step_uncertainty
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar("loss/step_total", step_total, global_step)
                    tb_writer.add_scalar("loss/step_type", step_type, global_step)
                    tb_writer.add_scalar("loss/step_score", step_score, global_step)
                    tb_writer.add_scalar("loss/step_uncertainty", step_uncertainty, global_step)
                global_step += 1

            avg_total = epoch_total_sum / max(1, epoch_steps)
            avg_type = epoch_type_sum / max(1, epoch_steps)
            avg_score = epoch_score_sum / max(1, epoch_steps)
            avg_uncertainty = epoch_uncertainty_sum / max(1, epoch_steps)
            epoch_metrics.append(
                {
                    "epoch": float(epoch_idx),
                    "loss_total": avg_total,
                    "loss_type": avg_type,
                    "loss_score": avg_score,
                    "loss_uncertainty": avg_uncertainty,
                }
            )
            print(f"[LightRisk] epoch {epoch_idx + 1}/{self.config.epochs}, loss={avg_total:.6f}")

            if tb_writer is not None:
                tb_writer.add_scalar("loss/epoch_total", avg_total, epoch_idx)
                tb_writer.add_scalar("loss/epoch_type", avg_type, epoch_idx)
                tb_writer.add_scalar("loss/epoch_score", avg_score, epoch_idx)
                tb_writer.add_scalar("loss/epoch_uncertainty", avg_uncertainty, epoch_idx)

            if val_samples:
                val_total, val_type, val_score, val_uncertainty = self._evaluate(val_samples)
                if tb_writer is not None:
                    tb_writer.add_scalar("val/loss_epoch_total", val_total, epoch_idx)
                    tb_writer.add_scalar("val/loss_epoch_type", val_type, epoch_idx)
                    tb_writer.add_scalar("val/loss_epoch_score", val_score, epoch_idx)
                    tb_writer.add_scalar("val/loss_epoch_uncertainty", val_uncertainty, epoch_idx)

        self.last_train_report = {
            "variant": "v2" if self.config.enable_v2 else "v1_compat",
            "epochs": int(self.config.epochs),
            "train_samples": int(len(train_samples)),
            "val_samples": int(len(val_samples or [])),
            "epoch_metrics": epoch_metrics,
        }
        self.model.eval()
        return LightRiskPredictor(model=self.model, device=self.device)

    def fine_tune_pairs(
        self,
        pair_samples: Sequence[RiskPairSample],
        replay_samples: Optional[Sequence[ActionConditionedSample]] = None,
        tb_writer=None,
    ) -> Dict[str, float]:
        before_pair_metrics = self.evaluate_pairs(pair_samples)
        before_pointwise_metrics = self._evaluate_pointwise_samples(replay_samples or [])

        if not self.config.pair_finetune or len(pair_samples) == 0:
            self.last_pair_metrics = before_pair_metrics
            self.last_pair_ft_report = {
                "enabled": bool(self.config.pair_finetune),
                "pair_count": int(len(pair_samples)),
                "replay_sample_count": int(len(replay_samples or [])),
                "before_pair_metrics": before_pair_metrics,
                "after_pair_metrics": before_pair_metrics,
                "before_pointwise_metrics": before_pointwise_metrics,
                "after_pointwise_metrics": before_pointwise_metrics,
                "epoch_metrics": [],
            }
            return self.last_pair_metrics

        loader = DataLoader(
            RiskPairDataset(pair_samples),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_risk_pairs,
        )
        replay_loader = self._build_replay_loader(replay_samples or [])
        replay_iter = iter(replay_loader) if replay_loader is not None else None

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.model.train()
        global_step = 0
        epoch_metrics: List[Dict[str, float]] = []
        for epoch_idx in range(max(1, int(self.config.pair_finetune_epochs))):
            epoch_total = 0.0
            epoch_pointwise = 0.0
            epoch_ranking = 0.0
            epoch_spread = 0.0
            epoch_steps = 0
            for batch in loader:
                ranking_loss, spread_loss = self._compute_pair_losses(batch)
                pointwise_total, _, _, _, replay_iter = self._compute_replay_losses(replay_iter, replay_loader)
                total_loss = (
                    float(self.config.pointwise_replay_weight) * pointwise_total
                    + float(self.config.ranking_loss_weight) * ranking_loss
                    + float(self.config.spread_loss_weight) * spread_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step_total = float(total_loss.item())
                step_pointwise = float(pointwise_total.item())
                step_ranking = float(ranking_loss.item())
                step_spread = float(spread_loss.item())
                epoch_total += step_total
                epoch_pointwise += step_pointwise
                epoch_ranking += step_ranking
                epoch_spread += step_spread
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar("loss/step_total", step_total, global_step)
                    tb_writer.add_scalar("loss/step_pointwise", step_pointwise, global_step)
                    tb_writer.add_scalar("loss/step_ranking", step_ranking, global_step)
                    tb_writer.add_scalar("loss/step_spread", step_spread, global_step)
                global_step += 1

            if epoch_steps > 0:
                epoch_summary = {
                    "epoch": float(epoch_idx),
                    "loss_total": epoch_total / epoch_steps,
                    "loss_pointwise": epoch_pointwise / epoch_steps,
                    "loss_ranking": epoch_ranking / epoch_steps,
                    "loss_spread": epoch_spread / epoch_steps,
                }
                epoch_metrics.append(epoch_summary)
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/epoch_total", epoch_summary["loss_total"], epoch_idx)
                    tb_writer.add_scalar("loss/epoch_pointwise", epoch_summary["loss_pointwise"], epoch_idx)
                    tb_writer.add_scalar("loss/epoch_ranking", epoch_summary["loss_ranking"], epoch_idx)
                    tb_writer.add_scalar("loss/epoch_spread", epoch_summary["loss_spread"], epoch_idx)

        self.model.eval()
        after_pair_metrics = self.evaluate_pairs(pair_samples)
        after_pointwise_metrics = self._evaluate_pointwise_samples(replay_samples or [])
        self.last_pair_metrics = after_pair_metrics
        self.last_pair_ft_report = {
            "enabled": True,
            "pair_count": int(len(pair_samples)),
            "replay_sample_count": int(len(replay_samples or [])),
            "before_pair_metrics": before_pair_metrics,
            "after_pair_metrics": after_pair_metrics,
            "before_pointwise_metrics": before_pointwise_metrics,
            "after_pointwise_metrics": after_pointwise_metrics,
            "epoch_metrics": epoch_metrics,
        }
        if tb_writer is not None:
            tb_writer.add_scalar("summary/before_pair_ranking_accuracy", float(before_pair_metrics.get("pair_ranking_accuracy", 0.0)), 0)
            tb_writer.add_scalar("summary/after_pair_ranking_accuracy", float(after_pair_metrics.get("pair_ranking_accuracy", 0.0)), 0)
            tb_writer.add_scalar("summary/before_same_state_score_gap", float(before_pair_metrics.get("same_state_score_gap", 0.0)), 0)
            tb_writer.add_scalar("summary/after_same_state_score_gap", float(after_pair_metrics.get("same_state_score_gap", 0.0)), 0)
            tb_writer.add_scalar("summary/before_score_spread", float(before_pair_metrics.get("score_spread", 0.0)), 0)
            tb_writer.add_scalar("summary/after_score_spread", float(after_pair_metrics.get("score_spread", 0.0)), 0)
        return self.last_pair_metrics

    @torch.no_grad()
    def evaluate_pairs(self, pair_samples: Sequence[RiskPairSample]) -> Dict[str, float]:
        if len(pair_samples) == 0:
            return self._empty_pair_metrics()

        correct = 0
        hard_correct = 0
        hard_total = 0
        score_gaps = []
        score_spreads = []
        brier_terms = []
        unique_scores = set()
        dataset = RiskPairDataset(pair_samples)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False, collate_fn=collate_risk_pairs)
        for batch in loader:
            x_a, x_b, preferred_a, target_a, target_b, _, _, _ = self._pair_batch_tensors(batch)
            score_a = self.model(x_a)["risk_score"]
            score_b = self.model(x_b)["risk_score"]
            pred_pref_a = score_a <= score_b
            correct += int(torch.sum(pred_pref_a == preferred_a).item())
            hard_negative = torch.tensor([bool(sample.meta.get("hard_negative", False)) for sample in batch], dtype=torch.bool, device=self.device)
            if hard_negative.any():
                hard_correct += int(torch.sum((pred_pref_a == preferred_a) & hard_negative).item())
                hard_total += int(torch.sum(hard_negative).item())
            gap = torch.abs(score_a - score_b).detach().cpu().numpy()
            score_gaps.extend(float(item) for item in gap)
            spread = torch.std(torch.stack([score_a, score_b], dim=-1), dim=-1).detach().cpu().numpy()
            score_spreads.extend(float(item) for item in spread)
            brier_terms.extend(float(item) for item in ((score_a - target_a) ** 2).detach().cpu().numpy())
            brier_terms.extend(float(item) for item in ((score_b - target_b) ** 2).detach().cpu().numpy())
            unique_scores.update(round(float(item), 6) for item in score_a.detach().cpu().numpy())
            unique_scores.update(round(float(item), 6) for item in score_b.detach().cpu().numpy())

        total = max(1, len(pair_samples))
        return {
            "pair_count": float(len(pair_samples)),
            "pair_ranking_accuracy": float(correct / total),
            "hard_negative_accuracy": float(hard_correct / max(1, hard_total)),
            "same_state_score_gap": float(np.mean(score_gaps)) if score_gaps else 0.0,
            "score_spread": float(np.mean(score_spreads)) if score_spreads else 0.0,
            "calibration_brier": float(np.mean(brier_terms)) if brier_terms else 0.0,
            "unique_score_count": float(len(unique_scores)),
        }

    def _compute_pointwise_losses(
        self,
        output: Dict[str, torch.Tensor],
        y_types: torch.Tensor,
        y_score: torch.Tensor,
    ):
        type_loss = self.bce(output["risk_type_logits"], y_types)
        score_loss = self.huber(output["risk_score"], y_score)
        overall_error = torch.abs(output["risk_score"] - y_score).detach()
        uncertainty_loss = torch.mean((output["uncertainty"] - overall_error) ** 2)
        total_loss = type_loss + score_loss + 0.1 * uncertainty_loss
        return total_loss, type_loss, score_loss, uncertainty_loss

    def _compute_pair_losses(self, batch: Sequence[RiskPairSample]):
        x_a, x_b, preferred_a, target_a, target_b, weight, _, trusted_for_spread = self._pair_batch_tensors(batch)
        out_a = self.model(x_a)
        out_b = self.model(x_b)
        score_a = out_a["risk_score"]
        score_b = out_b["risk_score"]

        safer = torch.where(preferred_a, score_a, score_b)
        riskier = torch.where(preferred_a, score_b, score_a)
        ranking_loss = torch.relu(PAIR_RANK_MARGIN - (riskier - safer))
        ranking_loss = torch.sum(ranking_loss * weight) / torch.clamp(torch.sum(weight), min=1.0)

        target_gap = torch.abs(target_a - target_b)
        spread_mask = ((target_gap > SPREAD_TARGET_DELTA) & trusted_for_spread).to(torch.float32)
        spread_gap = torch.abs(score_a - score_b)
        spread_loss = torch.relu(SPREAD_MIN_GAP - spread_gap) * spread_mask
        spread_denom = torch.clamp(torch.sum(spread_mask * weight), min=1.0)
        spread_loss = torch.sum(spread_loss * weight) / spread_denom
        return ranking_loss, spread_loss

    def _pair_batch_tensors(self, batch: Sequence[RiskPairSample]):
        x_a = []
        x_b = []
        preferred_a = []
        target_a = []
        target_b = []
        weight = []
        hard_negative = []
        trusted_for_spread = []
        for sample in batch:
            x_a.append(history_action_feature(sample.history_scene, sample.action_a))
            x_b.append(history_action_feature(sample.history_scene, sample.action_b))
            preferred_a.append(int(sample.preferred_action) == int(sample.action_a))
            target_a.append(float(sample.meta.get("target_risk_a", 0.0)))
            target_b.append(float(sample.meta.get("target_risk_b", 0.0)))
            weight.append(float(sample.weight))
            hard_negative.append(bool(sample.meta.get("hard_negative", False)))
            trusted_for_spread.append(bool(sample.meta.get("trusted_for_spread", False)))
        return (
            torch.tensor(np.array(x_a), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(x_b), dtype=torch.float32, device=self.device),
            torch.tensor(preferred_a, dtype=torch.bool, device=self.device),
            torch.tensor(target_a, dtype=torch.float32, device=self.device),
            torch.tensor(target_b, dtype=torch.float32, device=self.device),
            torch.tensor(weight, dtype=torch.float32, device=self.device),
            torch.tensor(hard_negative, dtype=torch.bool, device=self.device),
            torch.tensor(trusted_for_spread, dtype=torch.bool, device=self.device),
        )

    def _build_replay_loader(self, replay_samples: Sequence[ActionConditionedSample]):
        samples = list(replay_samples or [])
        if not samples:
            return None
        return DataLoader(
            _LightRiskDataset(samples),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _next_replay_batch(self, replay_iter, replay_loader):
        if replay_loader is None:
            return None, replay_iter
        try:
            batch = next(replay_iter)
        except StopIteration:
            replay_iter = iter(replay_loader)
            batch = next(replay_iter)
        return batch, replay_iter

    def _compute_replay_losses(self, replay_iter, replay_loader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        batch, replay_iter = self._next_replay_batch(replay_iter, replay_loader)
        if batch is None:
            return zero, zero, zero, zero, replay_iter
        x, y_types, y_score = batch
        x = x.to(self.device).to(torch.float32)
        y_types = y_types.to(self.device).to(torch.float32)
        y_score = y_score.to(self.device).to(torch.float32).squeeze(-1)
        output = self.model(x)
        total_loss, type_loss, score_loss, uncertainty_loss = self._compute_pointwise_losses(output, y_types, y_score)
        return total_loss, type_loss, score_loss, uncertainty_loss, replay_iter

    @torch.no_grad()
    def _evaluate(self, samples: Sequence[ActionConditionedSample]):
        if len(samples) == 0:
            return 0.0, 0.0, 0.0, 0.0

        loader = DataLoader(
            _LightRiskDataset(samples),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
        )

        was_training = self.model.training
        self.model.eval()
        total_sum = 0.0
        type_sum = 0.0
        score_sum = 0.0
        uncertainty_sum = 0.0
        steps = 0

        for x, y_types, y_score in loader:
            x = x.to(self.device).to(torch.float32)
            y_types = y_types.to(self.device).to(torch.float32)
            y_score = y_score.to(self.device).to(torch.float32).squeeze(-1)
            output = self.model(x)
            total_loss, type_loss, score_loss, uncertainty_loss = self._compute_pointwise_losses(output, y_types, y_score)
            total_sum += float(total_loss.item())
            type_sum += float(type_loss.item())
            score_sum += float(score_loss.item())
            uncertainty_sum += float(uncertainty_loss.item())
            steps += 1

        if was_training:
            self.model.train()

        denom = max(1, steps)
        return total_sum / denom, type_sum / denom, score_sum / denom, uncertainty_sum / denom

    def _evaluate_pointwise_samples(self, samples: Sequence[ActionConditionedSample]) -> Dict[str, float]:
        if not samples:
            return self._empty_pointwise_metrics()
        total, type_loss, score_loss, uncertainty_loss = self._evaluate(samples)
        return {
            "sample_count": float(len(samples)),
            "loss_total": float(total),
            "loss_type": float(type_loss),
            "loss_score": float(score_loss),
            "loss_uncertainty": float(uncertainty_loss),
        }

    def _empty_pointwise_metrics(self) -> Dict[str, float]:
        return {
            "sample_count": 0.0,
            "loss_total": 0.0,
            "loss_type": 0.0,
            "loss_score": 0.0,
            "loss_uncertainty": 0.0,
        }

    def _empty_pair_metrics(self) -> Dict[str, float]:
        return {
            "pair_count": 0.0,
            "pair_ranking_accuracy": 0.0,
            "hard_negative_accuracy": 0.0,
            "same_state_score_gap": 0.0,
            "score_spread": 0.0,
            "calibration_brier": 0.0,
            "unique_score_count": 0.0,
        }

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
