import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from net_works.scene_encoder import SceneEncoder
from net_works.traj_decoder import TrajDecoder
from safe_rl.config.config import WorldModelConfig
from safe_rl.data.pair_dataset import RiskPairDataset, collate_risk_pairs
from safe_rl.data.risk import risk_targets
from safe_rl.data.types import ActionConditionedSample, RiskLabels, RiskPairSample, RiskPrediction, SceneState, VehicleState, WorldPrediction
from safe_rl.models.action_encoder import ActionEncoder

PAIR_RANK_MARGIN = 0.02
SPREAD_TARGET_DELTA = 0.15
SPREAD_MIN_GAP = 0.02


def _post_process_output(generate_traj: torch.Tensor, predicted_his_traj: torch.Tensor) -> torch.Tensor:
    delt_t = 0.1
    batch_size = generate_traj.shape[0]
    num_obs = generate_traj.shape[1]
    vx = generate_traj[:, :, :, :, 0] / delt_t
    vy = generate_traj[:, :, :, :, 1] / delt_t
    start_x = predicted_his_traj[:, :, -1, 0].view(batch_size, num_obs, 1, 1)
    start_y = predicted_his_traj[:, :, -1, 1].view(batch_size, num_obs, 1, 1)
    start_heading = predicted_his_traj[:, :, -1, 2].view(batch_size, num_obs, 1, 1)
    x = torch.cumsum(generate_traj[:, :, :, :, 0], dim=-1) + start_x
    y = torch.cumsum(generate_traj[:, :, :, :, 1], dim=-1) + start_y
    heading = torch.cumsum(generate_traj[:, :, :, :, 2], dim=-1) + start_heading
    output = torch.stack((x, y, heading, vx, vy), dim=-1)
    return output


class SceneTensorizer:
    def __init__(
        self,
        history_steps: int,
        future_steps: int,
        max_other_num: int = 6,
        max_traffic: int = 8,
        max_lane_num: int = 32,
        max_point_num: int = 128,
    ):
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.max_other_num = max_other_num
        self.max_traffic = max_traffic
        self.max_lane_num = max_lane_num
        self.max_point_num = max_point_num

    @staticmethod
    def _vehicle_feature(vehicle: VehicleState) -> List[float]:
        return [vehicle.width, vehicle.length, 0.0, 1.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _light_state_to_value(state: str) -> float:
        mapping = {
            'RED': 1.0,
            'YELLOW': 2.0,
            'GREEN': 3.0,
        }
        return mapping.get((state or '').upper(), 0.0)

    @staticmethod
    def _find_vehicle(scene: SceneState, vehicle_id: str) -> Optional[VehicleState]:
        for vehicle in scene.vehicles:
            if vehicle.vehicle_id == vehicle_id:
                return vehicle
        return None

    def _pad_history(self, history_scene: List[SceneState]) -> List[SceneState]:
        if len(history_scene) >= self.history_steps:
            return history_scene[-self.history_steps:]
        if not history_scene:
            raise ValueError('history_scene must not be empty')
        missing = self.history_steps - len(history_scene)
        return [history_scene[0]] * missing + history_scene

    def _trajectory_for_vehicle(self, history_scene: List[SceneState], vehicle_id: str) -> torch.Tensor:
        traj = []
        fallback = None
        for scene in history_scene:
            vehicle = self._find_vehicle(scene, vehicle_id)
            if vehicle is None:
                if fallback is None:
                    traj.append([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    traj.append(fallback)
            else:
                fallback = [vehicle.x, vehicle.y, vehicle.heading, vehicle.vx, vehicle.vy]
                traj.append(fallback)
        return torch.tensor(traj, dtype=torch.float32)

    @staticmethod
    def _last_vehicle(scene: SceneState, ego_id: str) -> VehicleState:
        for vehicle in scene.vehicles:
            if vehicle.vehicle_id == ego_id:
                return vehicle
        if scene.vehicles:
            return scene.vehicles[0]
        return VehicleState(
            vehicle_id=ego_id if ego_id else 'ego',
            x=0.0,
            y=0.0,
            vx=0.0,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            heading=0.0,
            lane_id=0,
        )

    def _future_ego_target(self, future_scene: List[SceneState], ego_id: str) -> torch.Tensor:
        target = []
        last = None
        for scene in future_scene[: self.future_steps]:
            ego = self._last_vehicle(scene, ego_id)
            point = [ego.x, ego.y, ego.heading, ego.vx, ego.vy]
            target.append(point)
            last = point
        while len(target) < self.future_steps:
            target.append(last if last is not None else [0.0, 0.0, 0.0, 0.0, 0.0])
        return torch.tensor(target, dtype=torch.float32)

    def tensorize_batch(self, samples: Sequence[ActionConditionedSample]) -> Dict[str, torch.Tensor]:
        payload = self.tensorize_state_action_batch([(sample.history_scene, int(sample.candidate_action)) for sample in samples])
        target_future = []
        risk_type_target = []
        risk_score_target = []
        for sample in samples:
            history = self._pad_history(sample.history_scene)
            last_scene = history[-1]
            ego_id = last_scene.ego_id
            target_future.append(self._future_ego_target(sample.future_scene, ego_id))
            collision, ttc_risk, lane_violation, overall_risk = risk_targets(sample.risk_labels)
            risk_type_target.append(torch.tensor([collision, ttc_risk, lane_violation], dtype=torch.float32))
            risk_score_target.append(torch.tensor(overall_risk, dtype=torch.float32))
        payload['target_future'] = torch.stack(target_future, dim=0)
        payload['risk_type_target'] = torch.stack(risk_type_target, dim=0)
        payload['risk_score_target'] = torch.stack(risk_score_target, dim=0)
        return payload

    def tensorize_state_action_batch(self, history_action_items: Sequence[Tuple[List[SceneState], int]]) -> Dict[str, torch.Tensor]:
        other_his_traj_delt = []
        other_his_pos = []
        other_feature = []
        predicted_his_traj_delt = []
        predicted_his_pos = []
        predicted_his_traj = []
        predicted_feature = []
        lane_list = []
        traffic_light = []
        traffic_light_pos = []
        candidate_action = []

        for history_scene, action_id in history_action_items:
            history = self._pad_history(list(history_scene))
            last_scene = history[-1]
            ego_id = last_scene.ego_id
            ego_vehicle = self._last_vehicle(last_scene, ego_id)

            ego_his = self._trajectory_for_vehicle(history, ego_id)
            ego_delt = ego_his[1:] - ego_his[:-1]
            predicted_his_traj.append(ego_his.unsqueeze(0))
            predicted_his_traj_delt.append(ego_delt.unsqueeze(0))
            predicted_his_pos.append(torch.tensor([[ego_his[-1, 0], ego_his[-1, 1]]], dtype=torch.float32))
            predicted_feature.append(torch.tensor([self._vehicle_feature(ego_vehicle)], dtype=torch.float32))

            neighbors = [v for v in last_scene.vehicles if v.vehicle_id != ego_id]
            neighbors.sort(key=lambda v: abs(v.x - ego_vehicle.x) + abs(v.y - ego_vehicle.y))
            neighbors = neighbors[: self.max_other_num]

            other_traj_tensor = []
            other_pos_tensor = []
            other_feature_tensor = []
            for neighbor in neighbors:
                his = self._trajectory_for_vehicle(history, neighbor.vehicle_id)
                other_traj_tensor.append(his[1:] - his[:-1])
                other_pos_tensor.append(torch.tensor([his[-1, 0], his[-1, 1]], dtype=torch.float32))
                other_feature_tensor.append(torch.tensor(self._vehicle_feature(neighbor), dtype=torch.float32))

            while len(other_traj_tensor) < self.max_other_num:
                other_traj_tensor.append(torch.zeros((self.history_steps - 1, 5), dtype=torch.float32))
                other_pos_tensor.append(torch.zeros((2,), dtype=torch.float32))
                other_feature_tensor.append(torch.zeros((7,), dtype=torch.float32))

            other_his_traj_delt.append(torch.stack(other_traj_tensor, dim=0))
            other_his_pos.append(torch.stack(other_pos_tensor, dim=0))
            other_feature.append(torch.stack(other_feature_tensor, dim=0))

            lane_tensor = []
            for lane in last_scene.lane_polylines[: self.max_lane_num]:
                points = lane[: self.max_point_num]
                points_tensor = torch.tensor(points, dtype=torch.float32)
                if points_tensor.ndim == 1:
                    points_tensor = points_tensor.view(-1, 2)
                if points_tensor.shape[0] < self.max_point_num:
                    pad = torch.zeros((self.max_point_num - points_tensor.shape[0], 2), dtype=torch.float32)
                    points_tensor = torch.cat([points_tensor, pad], dim=0)
                lane_tensor.append(points_tensor)
            if not lane_tensor:
                lane_tensor.append(torch.zeros((self.max_point_num, 2), dtype=torch.float32))
            while len(lane_tensor) < self.max_lane_num:
                lane_tensor.append(torch.zeros((self.max_point_num, 2), dtype=torch.float32))
            lane_list.append(torch.stack(lane_tensor[: self.max_lane_num], dim=0))

            tl_state = []
            tl_pos = []
            for light in last_scene.traffic_lights[: self.max_traffic]:
                value = self._light_state_to_value(light.state)
                tl_state.append(torch.tensor([value] * self.history_steps, dtype=torch.float32))
                tl_pos.append(torch.tensor([light.x, light.y], dtype=torch.float32))
            while len(tl_state) < self.max_traffic:
                tl_state.append(torch.zeros((self.history_steps,), dtype=torch.float32))
                tl_pos.append(torch.zeros((2,), dtype=torch.float32))
            traffic_light.append(torch.stack(tl_state, dim=0))
            traffic_light_pos.append(torch.stack(tl_pos, dim=0))

            candidate_action.append(int(action_id))

        predicted_his_traj_tensor = torch.stack(predicted_his_traj, dim=0)
        predicted_his_traj_delt_tensor = torch.stack(predicted_his_traj_delt, dim=0)
        noise = torch.zeros_like(predicted_his_traj_delt_tensor)

        return {
            'noise': noise,
            'lane_list': torch.stack(lane_list, dim=0),
            'other_his_traj_delt': torch.stack(other_his_traj_delt, dim=0),
            'other_his_pos': torch.stack(other_his_pos, dim=0),
            'other_feature': torch.stack(other_feature, dim=0),
            'predicted_his_traj_delt': predicted_his_traj_delt_tensor,
            'predicted_his_pos': torch.stack(predicted_his_pos, dim=0),
            'predicted_his_traj': predicted_his_traj_tensor,
            'predicted_feature': torch.stack(predicted_feature, dim=0),
            'traffic_light': torch.stack(traffic_light, dim=0),
            'traffic_light_pos': torch.stack(traffic_light_pos, dim=0),
            'candidate_action': torch.tensor(candidate_action, dtype=torch.long),
        }

    def tensorize_inference(self, history_scene: List[SceneState], action_id: int) -> Dict[str, torch.Tensor]:
        return self.tensorize_state_action_batch([(history_scene, action_id)])


class ActionConditionedWorldModel(nn.Module):
    def __init__(self, config: WorldModelConfig, history_steps: int):
        super().__init__()
        self.config = config
        self.history_steps = history_steps
        self.scene_encoder = SceneEncoder(his_step=history_steps)
        self.action_encoder = ActionEncoder(num_actions=9, embedding_dim=32, output_dim=256)
        self.fusion = nn.Sequential(
            nn.Linear(512, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden_dim, 256),
        )
        self.traj_decoder = TrajDecoder(multimodal=config.multimodal, dim=256, future_step=config.future_steps)
        self.risk_type_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        self.risk_score_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        scene_feature = self.scene_encoder(
            batch['noise'],
            batch['lane_list'],
            batch['other_his_traj_delt'],
            batch['other_his_pos'],
            batch['other_feature'],
            batch['predicted_his_traj_delt'],
            batch['predicted_his_pos'],
            batch['predicted_feature'],
            batch['traffic_light'],
            batch['traffic_light_pos'],
        )
        action_feature = self.action_encoder(batch['candidate_action']).unsqueeze(1)
        fused_feature = self.fusion(torch.cat([scene_feature, action_feature], dim=-1))
        traj_delta, confidence = self.traj_decoder(fused_feature)
        traj = _post_process_output(traj_delta, batch['predicted_his_traj'])
        pooled = torch.mean(fused_feature, dim=1)
        risk_type_logits = self.risk_type_head(pooled)
        risk_score = torch.sigmoid(self.risk_score_head(pooled).squeeze(-1))
        uncertainty = self.uncertainty_head(pooled).squeeze(-1)
        return {
            'traj': traj,
            'confidence': confidence,
            'risk_type_logits': risk_type_logits,
            'risk_score': risk_score,
            'uncertainty': uncertainty,
        }


@dataclass
class WorldModelPredictor:
    model: ActionConditionedWorldModel
    tensorizer: SceneTensorizer
    device: torch.device

    @torch.no_grad()
    def predict(self, history_scene: List[SceneState], action_id: int) -> WorldPrediction:
        batch = self.tensorizer.tensorize_inference(history_scene, action_id)
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = self.model(batch)
        risk_type_prob = torch.sigmoid(output['risk_type_logits'][0]).cpu().numpy()
        base_risk_score = float(output['risk_score'][0].item())
        confidence = torch.softmax(output['confidence'][0, 0], dim=-1).cpu().numpy()
        uncertainty = float(output['uncertainty'][0].item())

        modality_risk: List[RiskPrediction] = []
        modality_scores = []
        for i in range(len(confidence)):
            scale = float(0.6 + 0.8 * confidence[i])
            risk_score = max(0.0, min(1.0, base_risk_score * scale))
            modality_scores.append(risk_score)
            modality_risk.append(
                RiskPrediction(
                    p_collision=float(risk_type_prob[0]),
                    p_ttc=float(risk_type_prob[1]),
                    p_lane_violation=float(risk_type_prob[2]),
                    p_overall=risk_score,
                    uncertainty=uncertainty,
                    risk_score=risk_score,
                )
            )

        aggregated = float(np.quantile(np.array(modality_scores, dtype=np.float32), 0.9))
        multimodal_future = output['traj'][0, 0].cpu().numpy()
        return WorldPrediction(
            multimodal_future=multimodal_future,
            modality_risk=modality_risk,
            aggregated_risk=aggregated,
            uncertainty=uncertainty,
        )


class WorldModelTrainer:
    def __init__(
        self,
        config: WorldModelConfig,
        history_steps: int,
        seed: int = 42,
        device: Optional[str] = None,
    ):
        self.config = config
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rng = random.Random(seed)
        self.model = ActionConditionedWorldModel(config=config, history_steps=history_steps).to(self.device)
        self.tensorizer = SceneTensorizer(history_steps=history_steps, future_steps=config.future_steps)
        self.huber = nn.HuberLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
        self.last_train_report: Dict[str, Any] = {}
        self.last_pair_metrics: Dict[str, float] = self._empty_pair_metrics()
        self.last_pair_ft_report: Dict[str, Any] = {}

    def fit(
        self,
        train_samples: Sequence[ActionConditionedSample],
        val_samples: Optional[Sequence[ActionConditionedSample]] = None,
        tb_writer=None,
    ) -> WorldModelPredictor:
        if len(train_samples) == 0:
            self.last_train_report = {
                'variant': 'v2' if self.config.enable_v2 else 'v1_compat',
                'epochs': 0,
                'train_samples': 0,
                'val_samples': len(val_samples or []),
                'epoch_metrics': [],
            }
            self.model.eval()
            return WorldModelPredictor(self.model, self.tensorizer, self.device)

        print(f'[WorldModel] start training on {self.device}, samples={len(train_samples)}')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        batch_size = max(1, self.config.batch_size)

        self.model.train()
        global_step = 0
        epoch_metrics: List[Dict[str, float]] = []
        for epoch_idx in range(self.config.epochs):
            shuffled = list(train_samples)
            self.rng.shuffle(shuffled)

            epoch_total_sum = 0.0
            epoch_traj_sum = 0.0
            epoch_type_sum = 0.0
            epoch_score_sum = 0.0
            epoch_uncertainty_sum = 0.0
            epoch_steps = 0

            for start in range(0, len(shuffled), batch_size):
                batch_samples = shuffled[start:start + batch_size]
                batch = self.tensorizer.tensorize_batch(batch_samples)
                batch = self._move_batch(batch)
                output = self.model(batch)

                total_loss, traj_loss, type_loss, score_loss, uncertainty_loss = self._compute_losses(batch, output)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step_total = float(total_loss.item())
                step_traj = float(traj_loss.item())
                step_type = float(type_loss.item())
                step_score = float(score_loss.item())
                step_uncertainty = float(uncertainty_loss.item())

                epoch_total_sum += step_total
                epoch_traj_sum += step_traj
                epoch_type_sum += step_type
                epoch_score_sum += step_score
                epoch_uncertainty_sum += step_uncertainty
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar('loss/step_total', step_total, global_step)
                    tb_writer.add_scalar('loss/step_traj', step_traj, global_step)
                    tb_writer.add_scalar('loss/step_type', step_type, global_step)
                    tb_writer.add_scalar('loss/step_score', step_score, global_step)
                    tb_writer.add_scalar('loss/step_uncertainty', step_uncertainty, global_step)
                global_step += 1

            avg_total = epoch_total_sum / max(1, epoch_steps)
            avg_traj = epoch_traj_sum / max(1, epoch_steps)
            avg_type = epoch_type_sum / max(1, epoch_steps)
            avg_score = epoch_score_sum / max(1, epoch_steps)
            avg_uncertainty = epoch_uncertainty_sum / max(1, epoch_steps)
            epoch_metrics.append(
                {
                    'epoch': float(epoch_idx),
                    'loss_total': avg_total,
                    'loss_traj': avg_traj,
                    'loss_type': avg_type,
                    'loss_score': avg_score,
                    'loss_uncertainty': avg_uncertainty,
                }
            )
            print(f'[WorldModel] epoch {epoch_idx + 1}/{self.config.epochs}, loss={avg_total:.6f}')

            if tb_writer is not None:
                tb_writer.add_scalar('loss/epoch_total', avg_total, epoch_idx)
                tb_writer.add_scalar('loss/epoch_traj', avg_traj, epoch_idx)
                tb_writer.add_scalar('loss/epoch_type', avg_type, epoch_idx)
                tb_writer.add_scalar('loss/epoch_score', avg_score, epoch_idx)
                tb_writer.add_scalar('loss/epoch_uncertainty', avg_uncertainty, epoch_idx)

            if val_samples:
                val_total, val_traj, val_type, val_score, val_uncertainty = self._evaluate(val_samples, batch_size)
                if tb_writer is not None:
                    tb_writer.add_scalar('val/loss_epoch_total', val_total, epoch_idx)
                    tb_writer.add_scalar('val/loss_epoch_traj', val_traj, epoch_idx)
                    tb_writer.add_scalar('val/loss_epoch_type', val_type, epoch_idx)
                    tb_writer.add_scalar('val/loss_epoch_score', val_score, epoch_idx)
                    tb_writer.add_scalar('val/loss_epoch_uncertainty', val_uncertainty, epoch_idx)

        self.last_train_report = {
            'variant': 'v2' if self.config.enable_v2 else 'v1_compat',
            'epochs': int(self.config.epochs),
            'train_samples': int(len(train_samples)),
            'val_samples': int(len(val_samples or [])),
            'epoch_metrics': epoch_metrics,
        }
        self.model.eval()
        return WorldModelPredictor(model=self.model, tensorizer=self.tensorizer, device=self.device)

    def fine_tune_pairs(
        self,
        pair_samples: Sequence[RiskPairSample],
        replay_samples: Optional[Sequence[ActionConditionedSample]] = None,
        tb_writer=None,
    ) -> Dict[str, float]:
        pair_count = int(len(pair_samples))
        replay_sample_count = int(len(replay_samples or []))
        trusted_pair_count = sum(1 for sample in pair_samples if bool(sample.meta.get("trusted_for_spread", False)))
        hard_negative_count = sum(1 for sample in pair_samples if bool(sample.meta.get("hard_negative", False)))
        before_pair_metrics = self.evaluate_pairs(pair_samples)
        before_pointwise_metrics = self._evaluate_risk_only_samples(replay_samples or [])

        if not self.config.pair_finetune or len(pair_samples) == 0:
            print(f"[WorldModel PairFT] skipped, enabled={bool(self.config.pair_finetune)}, pairs={pair_count}, trusted_pairs={trusted_pair_count}, hard_negatives={hard_negative_count}, replay_samples={replay_sample_count}")
            self.last_pair_metrics = before_pair_metrics
            self.last_pair_ft_report = {
                'enabled': bool(self.config.pair_finetune),
                'pair_count': int(len(pair_samples)),
                'replay_sample_count': int(len(replay_samples or [])),
                'before_pair_metrics': before_pair_metrics,
                'after_pair_metrics': before_pair_metrics,
                'before_pointwise_metrics': before_pointwise_metrics,
                'after_pointwise_metrics': before_pointwise_metrics,
                'epoch_metrics': [],
                'world_pair_ft_frozen_modules': [],
                'world_pair_ft_trainable_modules': [],
            }
            return self.last_pair_metrics

        print(f"[WorldModel PairFT] start on {self.device}, pairs={pair_count}, trusted_pairs={trusted_pair_count}, hard_negatives={hard_negative_count}, replay_samples={replay_sample_count}")
        loader = DataLoader(
            RiskPairDataset(pair_samples),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_risk_pairs,
        )
        replay_loader = self._build_replay_loader(replay_samples or [])
        replay_iter = iter(replay_loader) if replay_loader is not None else None
        grad_state, frozen_modules, trainable_modules = self._apply_pair_ft_freeze_policy()
        optimizer = torch.optim.Adam([parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=self.config.learning_rate)

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
                    tb_writer.add_scalar('loss/step_total', step_total, global_step)
                    tb_writer.add_scalar('loss/step_pointwise', step_pointwise, global_step)
                    tb_writer.add_scalar('loss/step_ranking', step_ranking, global_step)
                    tb_writer.add_scalar('loss/step_spread', step_spread, global_step)
                global_step += 1

            if epoch_steps > 0:
                epoch_summary = {
                    'epoch': float(epoch_idx),
                    'loss_total': epoch_total / epoch_steps,
                    'loss_pointwise': epoch_pointwise / epoch_steps,
                    'loss_ranking': epoch_ranking / epoch_steps,
                    'loss_spread': epoch_spread / epoch_steps,
                }
                epoch_metrics.append(epoch_summary)
                print(
                    f"[WorldModel PairFT] epoch {epoch_idx + 1}/{max(1, int(self.config.pair_finetune_epochs))}, "
                    f"total={epoch_summary['loss_total']:.6f}, pointwise={epoch_summary['loss_pointwise']:.6f}, "
                    f"ranking={epoch_summary['loss_ranking']:.6f}, spread={epoch_summary['loss_spread']:.6f}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar('loss/epoch_total', epoch_summary['loss_total'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_pointwise', epoch_summary['loss_pointwise'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_ranking', epoch_summary['loss_ranking'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_spread', epoch_summary['loss_spread'], epoch_idx)

        self._restore_grad_state(grad_state)
        self.model.eval()
        after_pair_metrics = self.evaluate_pairs(pair_samples)
        after_pointwise_metrics = self._evaluate_risk_only_samples(replay_samples or [])
        self.last_pair_metrics = after_pair_metrics
        self.last_pair_ft_report = {
            'enabled': True,
            'pair_count': int(len(pair_samples)),
            'replay_sample_count': int(len(replay_samples or [])),
            'before_pair_metrics': before_pair_metrics,
            'after_pair_metrics': after_pair_metrics,
            'before_pointwise_metrics': before_pointwise_metrics,
            'after_pointwise_metrics': after_pointwise_metrics,
            'epoch_metrics': epoch_metrics,
            'world_pair_ft_frozen_modules': frozen_modules,
            'world_pair_ft_trainable_modules': trainable_modules,
        }
        if tb_writer is not None:
            tb_writer.add_scalar('summary/before_pair_ranking_accuracy', float(before_pair_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/after_pair_ranking_accuracy', float(after_pair_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/before_same_state_score_gap', float(before_pair_metrics.get('same_state_score_gap', 0.0)), 0)
            tb_writer.add_scalar('summary/after_same_state_score_gap', float(after_pair_metrics.get('same_state_score_gap', 0.0)), 0)
            tb_writer.add_scalar('summary/before_score_spread', float(before_pair_metrics.get('score_spread', 0.0)), 0)
            tb_writer.add_scalar('summary/after_score_spread', float(after_pair_metrics.get('score_spread', 0.0)), 0)
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

        loader = DataLoader(
            RiskPairDataset(pair_samples),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_risk_pairs,
        )
        for batch in loader:
            batch_a, batch_b, preferred_a, target_a, target_b, _, hard_negative, _ = self._tensorize_pair_batch(batch)
            score_a = self.model(batch_a)['risk_score']
            score_b = self.model(batch_b)['risk_score']
            pred_pref_a = score_a <= score_b
            correct += int(torch.sum(pred_pref_a == preferred_a).item())
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
            'pair_count': float(len(pair_samples)),
            'pair_ranking_accuracy': float(correct / total),
            'hard_negative_accuracy': float(hard_correct / max(1, hard_total)),
            'same_state_score_gap': float(np.mean(score_gaps)) if score_gaps else 0.0,
            'score_spread': float(np.mean(score_spreads)) if score_spreads else 0.0,
            'calibration_brier': float(np.mean(brier_terms)) if brier_terms else 0.0,
            'unique_score_count': float(len(unique_scores)),
        }

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def _compute_losses(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]):
        pred_traj = output['traj'][:, 0]
        target_future = batch['target_future'].unsqueeze(1).expand_as(pred_traj)
        traj_loss = self.huber(pred_traj, target_future)
        traj_loss = torch.mean(traj_loss, dim=(-1, -2))
        traj_loss, _ = torch.min(traj_loss, dim=-1)
        traj_loss = torch.mean(traj_loss)

        type_loss, score_loss, uncertainty_loss = self._compute_risk_losses(batch, output)
        total_loss = traj_loss + type_loss + score_loss + self.config.uncertainty_weight * uncertainty_loss
        return total_loss, traj_loss, type_loss, score_loss, uncertainty_loss

    def _compute_risk_losses(self, batch: Dict[str, torch.Tensor], output: Dict[str, torch.Tensor]):
        risk_type_target = batch['risk_type_target']
        risk_score_target = batch['risk_score_target']
        type_loss = self.bce(output['risk_type_logits'], risk_type_target)
        score_loss = torch.mean(self.huber(output['risk_score'], risk_score_target))
        overall_error = torch.abs(output['risk_score'] - risk_score_target).detach()
        uncertainty_loss = torch.mean((output['uncertainty'] - overall_error) ** 2)
        return type_loss, score_loss, uncertainty_loss

    def _compute_pair_losses(self, batch: Sequence[RiskPairSample]):
        batch_a, batch_b, preferred_a, target_a, target_b, weight, _, trusted_for_spread = self._tensorize_pair_batch(batch)
        out_a = self.model(batch_a)
        out_b = self.model(batch_b)
        score_a = out_a['risk_score']
        score_b = out_b['risk_score']

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

    def _tensorize_pair_batch(self, batch: Sequence[RiskPairSample]):
        histories_a = []
        histories_b = []
        preferred_a = []
        target_a = []
        target_b = []
        weight = []
        hard_negative = []
        trusted_for_spread = []

        for sample in batch:
            histories_a.append((sample.history_scene, int(sample.action_a)))
            histories_b.append((sample.history_scene, int(sample.action_b)))
            preferred_a.append(int(sample.preferred_action) == int(sample.action_a))
            target_a.append(float(sample.meta.get('target_risk_a', 0.0)))
            target_b.append(float(sample.meta.get('target_risk_b', 0.0)))
            weight.append(float(sample.weight))
            hard_negative.append(bool(sample.meta.get('hard_negative', False)))
            trusted_for_spread.append(bool(sample.meta.get('trusted_for_spread', False)))

        batch_a = self._move_batch(self.tensorizer.tensorize_state_action_batch(histories_a))
        batch_b = self._move_batch(self.tensorizer.tensorize_state_action_batch(histories_b))
        return (
            batch_a,
            batch_b,
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
            samples,
            batch_size=max(1, self.config.batch_size),
            shuffle=True,
            drop_last=False,
            collate_fn=list,
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
        batch_samples, replay_iter = self._next_replay_batch(replay_iter, replay_loader)
        if batch_samples is None:
            return zero, zero, zero, zero, replay_iter
        batch = self._move_batch(self.tensorizer.tensorize_batch(batch_samples))
        output = self.model(batch)
        type_loss, score_loss, uncertainty_loss = self._compute_risk_losses(batch, output)
        total_loss = type_loss + score_loss + self.config.uncertainty_weight * uncertainty_loss
        return total_loss, type_loss, score_loss, uncertainty_loss, replay_iter

    def _apply_pair_ft_freeze_policy(self):
        grad_state = {name: parameter.requires_grad for name, parameter in self.model.named_parameters()}
        frozen_modules: List[str] = []
        trainable_modules: List[str] = []

        def _set_module_grad(module_name: str, enabled: bool):
            module = getattr(self.model, module_name)
            for parameter in module.parameters():
                parameter.requires_grad = enabled
            if enabled:
                trainable_modules.append(module_name)
            else:
                frozen_modules.append(module_name)

        backbone_mode = str(self.config.pair_ft_freeze_backbone or 'partial').strip().lower()
        if backbone_mode == 'all':
            _set_module_grad('scene_encoder', False)
            _set_module_grad('action_encoder', False)
            _set_module_grad('fusion', False)
        elif backbone_mode == 'partial':
            _set_module_grad('scene_encoder', False)
            _set_module_grad('action_encoder', False)
            _set_module_grad('fusion', True)
        else:
            _set_module_grad('scene_encoder', True)
            _set_module_grad('action_encoder', True)
            _set_module_grad('fusion', True)

        if bool(self.config.pair_ft_freeze_traj_decoder):
            _set_module_grad('traj_decoder', False)
        else:
            _set_module_grad('traj_decoder', True)

        _set_module_grad('risk_type_head', True)
        _set_module_grad('risk_score_head', True)
        _set_module_grad('uncertainty_head', True)
        return grad_state, sorted(set(frozen_modules)), sorted(set(trainable_modules))

    def _restore_grad_state(self, grad_state: Dict[str, bool]):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = bool(grad_state.get(name, True))

    @torch.no_grad()
    def _evaluate(self, samples: Sequence[ActionConditionedSample], batch_size: int):
        if len(samples) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        was_training = self.model.training
        self.model.eval()
        total_sum = 0.0
        traj_sum = 0.0
        type_sum = 0.0
        score_sum = 0.0
        uncertainty_sum = 0.0
        steps = 0

        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start:start + batch_size]
            batch = self.tensorizer.tensorize_batch(batch_samples)
            batch = self._move_batch(batch)
            output = self.model(batch)
            total_loss, traj_loss, type_loss, score_loss, uncertainty_loss = self._compute_losses(batch, output)

            total_sum += float(total_loss.item())
            traj_sum += float(traj_loss.item())
            type_sum += float(type_loss.item())
            score_sum += float(score_loss.item())
            uncertainty_sum += float(uncertainty_loss.item())
            steps += 1

        if was_training:
            self.model.train()

        denom = max(1, steps)
        return total_sum / denom, traj_sum / denom, type_sum / denom, score_sum / denom, uncertainty_sum / denom

    @torch.no_grad()
    def _evaluate_risk_only_samples(self, samples: Sequence[ActionConditionedSample]) -> Dict[str, float]:
        if not samples:
            return self._empty_pointwise_metrics()

        was_training = self.model.training
        self.model.eval()
        total_sum = 0.0
        type_sum = 0.0
        score_sum = 0.0
        uncertainty_sum = 0.0
        steps = 0

        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start:start + batch_size]
            batch = self.tensorizer.tensorize_batch(batch_samples)
            batch = self._move_batch(batch)
            output = self.model(batch)
            type_loss, score_loss, uncertainty_loss = self._compute_risk_losses(batch, output)
            total_loss = type_loss + score_loss + self.config.uncertainty_weight * uncertainty_loss
            total_sum += float(total_loss.item())
            type_sum += float(type_loss.item())
            score_sum += float(score_loss.item())
            uncertainty_sum += float(uncertainty_loss.item())
            steps += 1

        if was_training:
            self.model.train()

        denom = max(1, steps)
        return {
            'sample_count': float(len(samples)),
            'loss_total': total_sum / denom,
            'loss_type': type_sum / denom,
            'loss_score': score_sum / denom,
            'loss_uncertainty': uncertainty_sum / denom,
        }

    def _empty_pointwise_metrics(self) -> Dict[str, float]:
        return {
            'sample_count': 0.0,
            'loss_total': 0.0,
            'loss_type': 0.0,
            'loss_score': 0.0,
            'loss_uncertainty': 0.0,
        }

    def _empty_pair_metrics(self) -> Dict[str, float]:
        return {
            'pair_count': 0.0,
            'pair_ranking_accuracy': 0.0,
            'hard_negative_accuracy': 0.0,
            'same_state_score_gap': 0.0,
            'score_spread': 0.0,
            'calibration_brier': 0.0,
            'unique_score_count': 0.0,
        }

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()


def create_untrained_world_predictor(config: WorldModelConfig, history_steps: int, device: Optional[str] = None):
    trainer = WorldModelTrainer(config=config, history_steps=history_steps, device=device)
    trainer.model.eval()
    return WorldModelPredictor(model=trainer.model, tensorizer=trainer.tensorizer, device=trainer.device)
