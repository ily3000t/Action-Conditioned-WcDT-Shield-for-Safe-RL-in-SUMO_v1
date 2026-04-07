from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

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
    AUX_RAW_LOSS_WEIGHT = 0.35
    COLLAPSE_TOP1_THRESHOLD = 0.95
    COLLAPSE_ENTROPY_THRESHOLD = 0.10

    def __init__(self, config: DistillConfig, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or "cpu")
        self.model = DistilledPolicyNet().to(self.device)
        self.last_training_report: Dict[str, Any] = {}

    def should_distill(
        self,
        buffer: InterventionBuffer,
        supervision_samples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> bool:
        sample_count = int(len(supervision_samples)) if supervision_samples else int(len(buffer))
        return sample_count >= self.config.trigger_buffer_size

    def _distribution_stats(self, values: np.ndarray) -> Dict[str, Any]:
        counts = np.bincount(values.astype(np.int64), minlength=9).astype(np.int64)
        total = int(counts.sum())
        if total <= 0:
            return {
                "counts": {str(idx): 0 for idx in range(9)},
                "top1_ratio": 0.0,
                "entropy": 0.0,
            }
        probs = counts.astype(np.float64) / float(total)
        entropy = float(-np.sum([p * np.log(p) for p in probs if p > 0.0]))
        return {
            "counts": {str(idx): int(counts[idx]) for idx in range(9)},
            "top1_ratio": float(np.max(probs)),
            "entropy": entropy,
        }

    def _load_supervision_dataset(
        self,
        buffer: InterventionBuffer,
        supervision_samples: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        features = []
        raw_actions = []
        final_actions = []
        intervened_mask = []
        skipped_invalid_features = 0
        skipped_invalid_actions = 0
        source = "intervention_buffer"

        if supervision_samples:
            source = "stage4_supervision_dataset"
            for item in supervision_samples:
                payload = dict(item or {})
                feature = np.array(payload.get("history_feature", []), dtype=np.float32).reshape(-1)
                if feature.size != BASE_FEATURE_DIM:
                    skipped_invalid_features += 1
                    continue
                raw_action = int(payload.get("raw_action", -1))
                final_action = int(payload.get("final_action", -1))
                if raw_action < 0 or raw_action > 8 or final_action < 0 or final_action > 8:
                    skipped_invalid_actions += 1
                    continue
                intervened = bool(payload.get("intervened", raw_action != final_action))
                features.append(feature)
                raw_actions.append(raw_action)
                final_actions.append(final_action)
                intervened_mask.append(intervened)
        else:
            records = buffer.all_records()
            for record in records:
                features.append(encode_history(record.history_scene).astype(np.float32))
                raw_actions.append(int(record.raw_action))
                final_actions.append(int(record.final_action))
                intervened_mask.append(True)

        if not features:
            return {
                "source": source,
                "x": np.zeros((0, BASE_FEATURE_DIM), dtype=np.float32),
                "raw_actions": np.zeros((0,), dtype=np.int64),
                "final_actions": np.zeros((0,), dtype=np.int64),
                "intervened_mask": np.zeros((0,), dtype=np.bool_),
                "skipped_invalid_history_feature_count": int(skipped_invalid_features),
                "skipped_invalid_action_count": int(skipped_invalid_actions),
            }

        return {
            "source": source,
            "x": np.stack(features, axis=0).astype(np.float32),
            "raw_actions": np.array(raw_actions, dtype=np.int64),
            "final_actions": np.array(final_actions, dtype=np.int64),
            "intervened_mask": np.array(intervened_mask, dtype=np.bool_),
            "skipped_invalid_history_feature_count": int(skipped_invalid_features),
            "skipped_invalid_action_count": int(skipped_invalid_actions),
        }

    def distill(
        self,
        buffer: InterventionBuffer,
        supervision_samples: Optional[Sequence[Dict[str, Any]]] = None,
        tb_writer=None,
    ) -> DistilledPolicy:
        dataset = self._load_supervision_dataset(buffer, supervision_samples)
        x = np.array(dataset["x"], dtype=np.float32)
        raw_actions = np.array(dataset["raw_actions"], dtype=np.int64)
        final_actions = np.array(dataset["final_actions"], dtype=np.int64)
        intervened_mask = np.array(dataset["intervened_mask"], dtype=np.bool_)
        source = str(dataset.get("source", "intervention_buffer"))

        if x.shape[0] == 0:
            self.last_training_report = {
                "source": source,
                "sample_count": 0,
                "skipped": True,
                "skip_reason": "no_valid_samples",
                "intervened_sample_count": 0,
                "non_intervened_sample_count": 0,
                "collapsed": False,
                "skipped_invalid_history_feature_count": int(dataset.get("skipped_invalid_history_feature_count", 0)),
                "skipped_invalid_action_count": int(dataset.get("skipped_invalid_action_count", 0)),
            }
            self.model.eval()
            return DistilledPolicy(model=self.model, device=self.device)

        main_targets = np.where(intervened_mask, final_actions, raw_actions).astype(np.int64)
        label_stats = self._distribution_stats(main_targets)

        x_t = torch.from_numpy(x).to(self.device)
        raw_t = torch.from_numpy(raw_actions).to(self.device)
        main_t = torch.from_numpy(main_targets).to(self.device)
        intervened_t = torch.from_numpy(intervened_mask.astype(np.bool_)).to(self.device)

        class_counts = np.bincount(main_targets, minlength=9).astype(np.float32)
        total = float(class_counts.sum())
        class_weights = np.ones((9,), dtype=np.float32)
        if total > 0:
            valid = class_counts > 0
            class_weights[valid] = total / class_counts[valid]
            class_weights[~valid] = 0.0
            valid_mean = float(np.mean(class_weights[valid])) if np.any(valid) else 1.0
            if valid_mean > 0:
                class_weights[valid] = class_weights[valid] / valid_mean
        class_weight_t = torch.from_numpy(class_weights).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion_main = nn.CrossEntropyLoss(weight=class_weight_t, reduction="none")
        criterion_aux = nn.CrossEntropyLoss(weight=class_weight_t, reduction="none")

        self.model.train()
        batch_size = max(1, int(self.config.batch_size))
        global_step = 0
        epoch_metrics = []
        for epoch_idx in range(int(self.config.epochs)):
            permutation = torch.randperm(x_t.shape[0], device=self.device)
            epoch_loss_sum = 0.0
            epoch_main_sum = 0.0
            epoch_aux_sum = 0.0
            epoch_steps = 0

            for start in range(0, x_t.shape[0], batch_size):
                idx = permutation[start:start + batch_size]
                logits = self.model(x_t[idx])
                main_loss = criterion_main(logits, main_t[idx]).mean()

                batch_intervened = intervened_t[idx]
                if torch.any(batch_intervened):
                    aux_logits = logits[batch_intervened]
                    aux_targets = raw_t[idx][batch_intervened]
                    aux_loss = criterion_aux(aux_logits, aux_targets).mean()
                else:
                    aux_loss = torch.zeros((), device=self.device, dtype=main_loss.dtype)

                total_loss = main_loss + float(self.AUX_RAW_LOSS_WEIGHT) * aux_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step_total = float(total_loss.item())
                step_main = float(main_loss.item())
                step_aux = float(aux_loss.item())
                epoch_loss_sum += step_total
                epoch_main_sum += step_main
                epoch_aux_sum += step_aux
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar("loss/step_total", step_total, global_step)
                    tb_writer.add_scalar("loss/step_main", step_main, global_step)
                    tb_writer.add_scalar("loss/step_aux_intervened_raw", step_aux, global_step)
                global_step += 1

            with torch.no_grad():
                all_logits = self.model(x_t)
                pred_actions = torch.argmax(all_logits, dim=-1)
                acc_main = float((pred_actions == main_t).to(torch.float32).mean().item())

            epoch_report = {
                "epoch": int(epoch_idx),
                "loss_total": float(epoch_loss_sum / max(1, epoch_steps)),
                "loss_main": float(epoch_main_sum / max(1, epoch_steps)),
                "loss_aux_intervened_raw": float(epoch_aux_sum / max(1, epoch_steps)),
                "action_acc_main": acc_main,
            }
            epoch_metrics.append(epoch_report)
            if tb_writer is not None:
                tb_writer.add_scalar("loss/epoch_total", epoch_report["loss_total"], epoch_idx)
                tb_writer.add_scalar("loss/epoch_main", epoch_report["loss_main"], epoch_idx)
                tb_writer.add_scalar("loss/epoch_aux_intervened_raw", epoch_report["loss_aux_intervened_raw"], epoch_idx)
                tb_writer.add_scalar("metric/epoch_action_acc_main", acc_main, epoch_idx)

        self.model.eval()
        with torch.no_grad():
            pred_logits = self.model(x_t)
            pred_actions_np = torch.argmax(pred_logits, dim=-1).detach().cpu().numpy().astype(np.int64)
        pred_stats = self._distribution_stats(pred_actions_np)
        collapsed = bool(
            float(pred_stats["top1_ratio"]) >= float(self.COLLAPSE_TOP1_THRESHOLD)
            or float(pred_stats["entropy"]) <= float(self.COLLAPSE_ENTROPY_THRESHOLD)
        )

        self.last_training_report = {
            "source": source,
            "sample_count": int(x.shape[0]),
            "intervened_sample_count": int(np.sum(intervened_mask)),
            "non_intervened_sample_count": int(np.sum(~intervened_mask)),
            "aux_raw_loss_weight": float(self.AUX_RAW_LOSS_WEIGHT),
            "label_action_counts": dict(label_stats["counts"]),
            "label_top1_ratio": float(label_stats["top1_ratio"]),
            "label_entropy": float(label_stats["entropy"]),
            "pred_action_counts": dict(pred_stats["counts"]),
            "pred_top1_ratio": float(pred_stats["top1_ratio"]),
            "pred_entropy": float(pred_stats["entropy"]),
            "collapsed": collapsed,
            "skipped": False,
            "epoch_metrics": epoch_metrics,
            "skipped_invalid_history_feature_count": int(dataset.get("skipped_invalid_history_feature_count", 0)),
            "skipped_invalid_action_count": int(dataset.get("skipped_invalid_action_count", 0)),
        }
        if tb_writer is not None:
            tb_writer.add_scalar("metric/pred_top1_ratio", float(pred_stats["top1_ratio"]), 0)
            tb_writer.add_scalar("metric/pred_entropy", float(pred_stats["entropy"]), 0)
            tb_writer.add_scalar("metric/collapsed_flag", float(collapsed), 0)
        return DistilledPolicy(model=self.model, device=self.device)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
