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

PAIR_RANK_MARGIN = 0.08
SPREAD_TARGET_DELTA = 0.15
SPREAD_MIN_GAP = 0.08
STAGE4_MIX_EVERY_N_STEPS = 4


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
        risk_score_logit = self.risk_score_head(pooled).squeeze(-1)
        risk_score = torch.sigmoid(risk_score_logit)
        uncertainty = self.uncertainty_head(pooled).squeeze(-1)
        return {
            'traj': traj,
            'confidence': confidence,
            'risk_type_logits': risk_type_logits,
            'risk_score_logit': risk_score_logit,
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
            risk_score = max(0.0, min(1.0, base_risk_score))
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
        stage5_pair_samples: Optional[Sequence[RiskPairSample]] = None,
        stage1_probe_pair_samples: Optional[Sequence[RiskPairSample]] = None,
        stage4_pair_samples: Optional[Sequence[RiskPairSample]] = None,
    ) -> Dict[str, float]:
        replay_samples = list(replay_samples or [])
        pair_samples = list(pair_samples or [])
        stage5_pair_samples = list(stage5_pair_samples or [sample for sample in pair_samples if str(sample.source) == 'stage5_trace_first_replacement'])
        stage1_probe_pair_samples = list(stage1_probe_pair_samples or [sample for sample in pair_samples if str(sample.source) == 'stage1_probe_same_state'])
        stage4_pair_samples = list(
            stage4_pair_samples
            or [
                sample
                for sample in pair_samples
                if str(sample.source) in ('stage4_buffer', 'stage4_candidate_rank')
            ]
        )
        stage1_priority_mix_enabled = bool(
            getattr(self.config, 'pair_ft_stage1_priority_mix_enabled', False)
        )
        stage1_priority_mix_fraction = float(
            getattr(self.config, 'pair_ft_stage1_priority_mix_fraction', 0.35) or 0.35
        )
        stage1_priority_mix_fraction = float(
            min(1.0, max(0.0, stage1_priority_mix_fraction))
        )
        stage1_priority_trusted_only = bool(
            getattr(self.config, 'pair_ft_stage1_priority_trusted_only', True)
        )
        if stage1_priority_trusted_only:
            stage1_priority_pair_samples = [
                sample
                for sample in stage1_probe_pair_samples
                if bool((sample.meta or {}).get('trusted_for_spread', False))
            ]
        else:
            stage1_priority_pair_samples = list(stage1_probe_pair_samples)
        phaseb_stage1_anticollapse_weight_effective = float(
            getattr(self.config, 'pair_ft_stage1_phaseb_anticollapse_weight', 0.0) or 0.0
        )
        phaseb_stage1_anticollapse_score_range_floor = float(
            getattr(self.config, 'pair_ft_stage1_phaseb_score_range_floor', 0.02) or 0.02
        )
        phaseb_stage1_anticollapse_q_low = float(
            getattr(self.config, 'pair_ft_stage1_phaseb_score_range_quantile_low', 0.10) or 0.10
        )
        phaseb_stage1_anticollapse_q_high = float(
            getattr(self.config, 'pair_ft_stage1_phaseb_score_range_quantile_high', 0.90) or 0.90
        )
        phaseb_stage1_anticollapse_q_low = float(min(1.0, max(0.0, phaseb_stage1_anticollapse_q_low)))
        phaseb_stage1_anticollapse_q_high = float(min(1.0, max(0.0, phaseb_stage1_anticollapse_q_high)))
        if phaseb_stage1_anticollapse_q_high <= phaseb_stage1_anticollapse_q_low:
            phaseb_stage1_anticollapse_q_low = 0.10
            phaseb_stage1_anticollapse_q_high = 0.90
        phaseb_stage1_anticollapse_apply_on = str(
            getattr(self.config, 'pair_ft_stage1_phaseb_anticollapse_apply_on', 'priority_only')
            or 'priority_only'
        ).strip().lower()
        if phaseb_stage1_anticollapse_apply_on not in {'priority_only', 'all_stage1'}:
            phaseb_stage1_anticollapse_apply_on = 'priority_only'
        stage1_tail_apply_trusted_only = bool(
            getattr(self.config, 'pair_ft_stage1_tail_apply_trusted_only', True)
        )
        stage1_tail_epochs = max(
            0,
            int(getattr(self.config, 'pair_ft_stage1_tail_epochs', 0) or 0),
        )
        stage1_tail_pair_samples = self._filter_stage1_tail_pairs(
            stage1_probe_pair_samples,
            trusted_only=stage1_tail_apply_trusted_only,
        )
        stage1_tail_acceptance_enabled = bool(
            getattr(self.config, 'pair_ft_stage1_tail_acceptance_enabled', True)
        )
        stage1_tail_acceptance_acc_tolerance = float(
            getattr(self.config, 'pair_ft_stage1_tail_acceptance_acc_tolerance', 0.01) or 0.01
        )
        stage1_tail_acceptance_spread_tolerance = float(
            getattr(self.config, 'pair_ft_stage1_tail_acceptance_spread_tolerance', 0.001) or 0.001
        )
        stage1_tail_acceptance_gap_tolerance = float(
            getattr(self.config, 'pair_ft_stage1_tail_acceptance_gap_tolerance', 0.001) or 0.001
        )
        stage1_tail_sampling_mode = str(
            getattr(self.config, 'pair_ft_stage1_tail_sampling_mode', 'with_replacement')
            or 'with_replacement'
        ).strip().lower()
        if stage1_tail_sampling_mode not in {'with_replacement', 'without_replacement'}:
            stage1_tail_sampling_mode = 'with_replacement'
        tail_ranking_loss_weight_cfg = getattr(
            self.config,
            'pair_ft_stage1_tail_ranking_loss_weight',
            None,
        )
        if tail_ranking_loss_weight_cfg is not None:
            tail_ranking_loss_weight_cfg = float(tail_ranking_loss_weight_cfg)
        tail_resolution_loss_weight_cfg = getattr(
            self.config,
            'pair_ft_stage1_tail_resolution_loss_weight',
            None,
        )
        if tail_resolution_loss_weight_cfg is not None:
            tail_resolution_loss_weight_cfg = float(tail_resolution_loss_weight_cfg)
        tail_ranking_loss_weight_effective = float(
            tail_ranking_loss_weight_cfg
            if tail_ranking_loss_weight_cfg is not None
            else float(self.config.ranking_loss_weight)
        )
        tail_resolution_loss_weight_effective = float(
            tail_resolution_loss_weight_cfg
            if tail_resolution_loss_weight_cfg is not None
            else float(getattr(self.config, 'pair_ft_stage1_resolution_loss_weight', 0.0) or 0.0)
        )
        stage1_tail_anticollapse_weight_effective = float(
            getattr(self.config, 'pair_ft_stage1_tail_anticollapse_weight', 0.0) or 0.0
        )
        stage1_tail_score_range_floor_effective = float(
            getattr(self.config, 'pair_ft_stage1_tail_score_range_floor', 0.02) or 0.02
        )
        stage1_tail_score_range_quantile_low_effective = float(
            getattr(self.config, 'pair_ft_stage1_tail_score_range_quantile_low', 0.10) or 0.10
        )
        stage1_tail_score_range_quantile_high_effective = float(
            getattr(self.config, 'pair_ft_stage1_tail_score_range_quantile_high', 0.90) or 0.90
        )
        stage1_tail_score_range_quantile_low_effective = float(
            min(1.0, max(0.0, stage1_tail_score_range_quantile_low_effective))
        )
        stage1_tail_score_range_quantile_high_effective = float(
            min(1.0, max(0.0, stage1_tail_score_range_quantile_high_effective))
        )
        if stage1_tail_score_range_quantile_high_effective <= stage1_tail_score_range_quantile_low_effective:
            stage1_tail_score_range_quantile_low_effective = 0.10
            stage1_tail_score_range_quantile_high_effective = 0.90
        pair_count = int(len(pair_samples))
        replay_sample_count = int(len(replay_samples))
        eval_replay_samples = self._select_pair_ft_eval_samples(replay_samples)
        eval_replay_sample_count = int(len(eval_replay_samples))
        trusted_pair_count = sum(1 for sample in pair_samples if bool(sample.meta.get('trusted_for_spread', False)))
        hard_negative_count = sum(1 for sample in pair_samples if bool(sample.meta.get('hard_negative', False)))
        before_pair_metrics = self.evaluate_pairs(pair_samples)
        before_stage5_metrics = self.evaluate_pairs(stage5_pair_samples)
        before_stage1_probe_metrics = self.evaluate_pairs(stage1_probe_pair_samples)
        before_stage4_metrics = self.evaluate_pairs(stage4_pair_samples)
        stage4_high_gap_pair_samples = self._filter_high_gap_pairs(stage4_pair_samples)
        before_stage4_high_gap_metrics = self.evaluate_pairs(stage4_high_gap_pair_samples)
        stage4_aux_pair_samples = self._filter_stage4_aux_pairs(stage4_pair_samples)
        before_stage4_aux_metrics = self.evaluate_pairs(stage4_aux_pair_samples)
        before_stage4_aux_logit_gap_metrics = self._evaluate_pair_logit_gap_metrics(stage4_aux_pair_samples)
        before_pointwise_metrics = self._evaluate_risk_only_samples(eval_replay_samples)
        stage5_spread_eligible_count = self._spread_eligible_pair_count(stage5_pair_samples)
        stage1_probe_spread_eligible_count = self._spread_eligible_pair_count(stage1_probe_pair_samples)
        stage4_spread_eligible_count = self._spread_eligible_pair_count(stage4_pair_samples)
        stage5_pair_ids = [self._stage5_pair_identifier(sample, index) for index, sample in enumerate(stage5_pair_samples)]
        stage4_mix_every_n_steps = max(
            1,
            int(
                getattr(
                    self.config,
                    'pair_ft_stage4_mix_every_n_steps',
                    STAGE4_MIX_EVERY_N_STEPS,
                )
                or STAGE4_MIX_EVERY_N_STEPS
            ),
        )
        source_mix = {
            'phase_a_epochs': 0,
            'phase_b_epochs': 0,
            'stage5_steps': 0,
            'stage1_probe_steps': 0,
            'stage4_steps': 0,
            'stage1_tail_steps': 0,
            'stage5_pairs_seen': 0,
            'stage1_probe_pairs_seen': 0,
            'stage4_pairs_seen': 0,
            'stage1_tail_pairs_seen': 0,
            'stage5_pair_count': int(len(stage5_pair_samples)),
            'stage1_probe_pair_count': int(len(stage1_probe_pair_samples)),
            'stage4_pair_count': int(len(stage4_pair_samples)),
            'stage1_tail_pair_count': int(len(stage1_tail_pair_samples)),
            'phase_b_stage1_priority_enabled': bool(stage1_priority_mix_enabled),
            'phase_b_stage1_priority_fraction_configured': float(stage1_priority_mix_fraction),
            'phase_b_stage1_priority_trusted_only': bool(stage1_priority_trusted_only),
            'phase_b_stage1_priority_pair_count': int(len(stage1_priority_pair_samples)),
            'phase_b_stage1_priority_steps': 0,
            'phase_b_stage1_priority_pairs_seen': 0,
            'phase_b_stage1_anticollapse_weight_effective': float(
                phaseb_stage1_anticollapse_weight_effective
            ),
            'phase_b_stage1_anticollapse_apply_on_effective': str(
                phaseb_stage1_anticollapse_apply_on
            ),
            'phase_b_stage1_anticollapse_steps': 0,
            'phase_b_stage1_anticollapse_active_pair_count': 0,
            'phase_b_stage1_score_range_below_floor_fraction': 0.0,
            'stage1_tail_sampling_mode_effective': str(stage1_tail_sampling_mode),
            'stage1_tail_ranking_loss_weight_effective': float(tail_ranking_loss_weight_effective),
            'stage1_tail_resolution_loss_weight_effective': float(tail_resolution_loss_weight_effective),
            'stage1_tail_anticollapse_weight_effective': float(stage1_tail_anticollapse_weight_effective),
            'stage1_tail_score_range_floor_effective': float(stage1_tail_score_range_floor_effective),
            'stage1_tail_score_range_quantiles_effective': {
                'low': float(stage1_tail_score_range_quantile_low_effective),
                'high': float(stage1_tail_score_range_quantile_high_effective),
            },
            'stage4_mix_every_n_steps': int(stage4_mix_every_n_steps),
            'stage5_pair_cap': int(getattr(self.config, 'stage5_pair_max_seen_per_epoch', 0) or 0),
            'stage5_pair_seen_counts': {pair_id: 0 for pair_id in stage5_pair_ids},
            'stage5_cap_reached_pairs': 0,
        }

        if not self.config.pair_finetune or len(pair_samples) == 0:
            print(
                f"[WorldModel PairFT] skipped, enabled={bool(self.config.pair_finetune)}, pairs={pair_count}, "
                f"stage5_pairs={len(stage5_pair_samples)}, stage1_probe_pairs={len(stage1_probe_pair_samples)}, stage4_pairs={len(stage4_pair_samples)}, "
                f"trusted_pairs={trusted_pair_count}, hard_negatives={hard_negative_count}, replay_samples={replay_sample_count}, "
                f"eval_replay_samples={eval_replay_sample_count}"
            )
            self.last_pair_metrics = before_pair_metrics
            self.last_pair_ft_report = {
                'enabled': False,
                'pair_count': int(len(pair_samples)),
                'replay_sample_count': int(replay_sample_count),
                'eval_replay_sample_count': int(eval_replay_sample_count),
                'before_pair_metrics': before_pair_metrics,
                'after_pair_metrics': before_pair_metrics,
                'before_pointwise_metrics': before_pointwise_metrics,
                'after_pointwise_metrics': before_pointwise_metrics,
                'epoch_metrics': [],
                'resolution_space': 'score',
                'pair_ft_resolution_min_score_gap': float(getattr(self.config, 'pair_ft_resolution_min_score_gap', 0.03) or 0.03),
                'ignored_legacy_logit_margin': float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14),
                'pair_ft_resolution_min_logit_gap_compat': float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14),
                'stage1_resolution_space': 'score',
                'pair_ft_stage1_resolution_min_score_gap': float(
                    getattr(self.config, 'pair_ft_stage1_resolution_min_score_gap', 0.015) or 0.015
                ),
                'pair_ft_stage1_resolution_loss_weight': float(
                    getattr(self.config, 'pair_ft_stage1_resolution_loss_weight', 0.0) or 0.0
                ),
                'stage1_resolution_mode': str(
                    getattr(self.config, 'pair_ft_stage1_resolution_mode', 'fixed') or 'fixed'
                ).strip().lower(),
                'pair_ft_stage1_resolution_alpha': float(
                    getattr(self.config, 'pair_ft_stage1_resolution_alpha', 0.2) or 0.2
                ),
                'pair_ft_stage1_resolution_max_score_gap': float(
                    getattr(self.config, 'pair_ft_stage1_resolution_max_score_gap', 0.05) or 0.05
                ),
                'pair_ft_stage1_resolution_apply_trusted_only': bool(
                    getattr(self.config, 'pair_ft_stage1_resolution_apply_trusted_only', True)
                ),
                'stage1_tail_ranking_loss_weight_effective': float(tail_ranking_loss_weight_effective),
                'stage1_tail_resolution_loss_weight_effective': float(tail_resolution_loss_weight_effective),
                'stage1_tail_anticollapse_weight_effective': float(stage1_tail_anticollapse_weight_effective),
                'stage1_tail_score_range_floor_effective': float(stage1_tail_score_range_floor_effective),
                'stage1_tail_score_range_quantiles_effective': {
                    'low': float(stage1_tail_score_range_quantile_low_effective),
                    'high': float(stage1_tail_score_range_quantile_high_effective),
                },
                'stage1_tail_ranking_loss': 0.0,
                'stage1_tail_resolution_loss': 0.0,
                'stage1_tail_anticollapse_loss': 0.0,
                'stage1_tail_score_range_q10_q90': {'q10': 0.0, 'q90': 0.0, 'range': 0.0},
                'stage1_tail_floor_reject_reasons': {},
                'pair_ft_selection_accuracy_tie_epsilon_effective': float(
                    getattr(self.config, 'pair_ft_selection_accuracy_tie_epsilon', 1e-4) or 1e-4
                ),
                'stage1_tail_sampling_mode_effective': str(stage1_tail_sampling_mode),
                'world_pair_ft_frozen_modules': [],
                'world_pair_ft_trainable_modules': [],
                'world_pair_ft_source_mix': dict(source_mix),
                'stage5_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage5_metrics, before_stage5_metrics, 'pair_ranking_accuracy'),
                'stage1_probe_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage1_probe_metrics, before_stage1_probe_metrics, 'pair_ranking_accuracy'),
                'stage4_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage4_metrics, before_stage4_metrics, 'pair_ranking_accuracy'),
                'stage5_same_state_score_gap_before_after': self._metric_before_after(before_stage5_metrics, before_stage5_metrics, 'same_state_score_gap'),
                'stage1_probe_same_state_score_gap_before_after': self._metric_before_after(before_stage1_probe_metrics, before_stage1_probe_metrics, 'same_state_score_gap'),
                'stage4_same_state_score_gap_before_after': self._metric_before_after(before_stage4_metrics, before_stage4_metrics, 'same_state_score_gap'),
                'stage5_score_spread_before_after': self._metric_before_after(before_stage5_metrics, before_stage5_metrics, 'score_spread'),
                'stage1_probe_score_spread_before_after': self._metric_before_after(before_stage1_probe_metrics, before_stage1_probe_metrics, 'score_spread'),
                'stage4_score_spread_before_after': self._metric_before_after(before_stage4_metrics, before_stage4_metrics, 'score_spread'),
                'stage5_unique_score_count_before_after': self._metric_before_after(before_stage5_metrics, before_stage5_metrics, 'unique_score_count'),
                'stage1_probe_unique_score_count_before_after': self._metric_before_after(before_stage1_probe_metrics, before_stage1_probe_metrics, 'unique_score_count'),
                'stage4_unique_score_count_before_after': self._metric_before_after(before_stage4_metrics, before_stage4_metrics, 'unique_score_count'),
                'stage4_high_gap_pair_count': int(len(stage4_high_gap_pair_samples)),
                'stage4_high_gap_unique_score_count_before_after': self._metric_before_after(
                    before_stage4_high_gap_metrics,
                    before_stage4_high_gap_metrics,
                    'unique_score_count',
                ),
                'stage4_aux_pair_count': int(len(stage4_aux_pair_samples)),
                'stage4_aux_unique_score_count_before_after': self._metric_before_after(
                    before_stage4_aux_metrics,
                    before_stage4_aux_metrics,
                    'unique_score_count',
                ),
                'stage4_aux_logit_gap_before_after': {
                    'before': float(before_stage4_aux_logit_gap_metrics.get('mean_abs_logit_gap', 0.0)),
                    'after': float(before_stage4_aux_logit_gap_metrics.get('mean_abs_logit_gap', 0.0)),
                },
                'stage4_aux_score_spread_before_after': self._metric_before_after(
                    before_stage4_aux_metrics,
                    before_stage4_aux_metrics,
                    'score_spread',
                ),
                'stage4_aux_same_state_score_gap_before_after': self._metric_before_after(
                    before_stage4_aux_metrics,
                    before_stage4_aux_metrics,
                    'same_state_score_gap',
                ),
                'stage5_spread_eligible_pair_count': int(stage5_spread_eligible_count),
                'stage1_probe_spread_eligible_pair_count': int(stage1_probe_spread_eligible_count),
                'stage4_spread_eligible_pair_count': int(stage4_spread_eligible_count),
                'world_pair_ft_best_epoch': -1,
                'world_pair_ft_best_metrics': dict(before_stage5_metrics),
                'world_pair_ft_restored_best': False,
                'selection_path': 'legacy_tieaware',
                'selection_reason': 'pair_ft_skipped_or_no_pairs',
                'best_epoch_stage1_unique': float((before_stage1_probe_metrics or {}).get('unique_score_count', 0.0)),
                'best_epoch_eval_unique': float((before_stage5_metrics or {}).get('unique_score_count', 0.0)),
                'stage1_tail_enabled': bool(stage1_tail_epochs > 0),
                'stage1_tail_applied': False,
                'stage1_tail_epochs_configured': int(stage1_tail_epochs),
                'stage1_tail_epochs_executed': 0,
                'stage1_tail_pair_count': int(len(stage1_tail_pair_samples)),
                'phase_b_stage1_priority_enabled': bool(stage1_priority_mix_enabled),
                'phase_b_stage1_priority_fraction_configured': float(stage1_priority_mix_fraction),
                'phase_b_stage1_priority_trusted_only': bool(stage1_priority_trusted_only),
                'phase_b_stage1_priority_pair_count': int(len(stage1_priority_pair_samples)),
                'phase_b_stage1_priority_steps': 0,
                'phase_b_stage1_priority_pairs_seen': 0,
                'phase_b_stage1_anticollapse_weight_effective': float(
                    phaseb_stage1_anticollapse_weight_effective
                ),
                'phase_b_stage1_anticollapse_apply_on_effective': str(
                    phaseb_stage1_anticollapse_apply_on
                ),
                'phase_b_stage1_anticollapse_steps': 0,
                'phase_b_stage1_anticollapse_active_pair_count': 0,
                'phase_b_stage1_anticollapse_loss': 0.0,
                'phase_b_stage1_score_range_q10_q90': {'q10': 0.0, 'q90': 0.0, 'range': 0.0},
                'phase_b_stage1_score_range_below_floor_fraction': 0.0,
                'phase_b_stage1_score_range_p10': 0.0,
                'phase_b_stage1_score_range_p50': 0.0,
                'phase_b_stage1_score_range_p90': 0.0,
                'stage1_tail_internal_best_epoch': -1,
                'stage1_tail_internal_best_reason': 'tail_not_applied',
                'stage1_tail_internal_best_stage1_probe_unique': float(
                    (before_stage1_probe_metrics or {}).get('unique_score_count', 0.0)
                ),
                'stage1_tail_accepted': False,
                'stage1_tail_acceptance_reason': 'tail_not_applied',
                'stage1_tail_acceptance_thresholds': {
                    'enabled': bool(stage1_tail_acceptance_enabled),
                    'acc_tolerance': float(stage1_tail_acceptance_acc_tolerance),
                    'spread_tolerance': float(stage1_tail_acceptance_spread_tolerance),
                    'gap_tolerance': float(stage1_tail_acceptance_gap_tolerance),
                },
                'world_pair_ft_final_state_source': 'selected_best',
                'stage1_tail_stage1_probe_unique_before_after': self._metric_before_after(
                    before_stage1_probe_metrics,
                    before_stage1_probe_metrics,
                    'unique_score_count',
                ),
                'stage1_tail_stage1_probe_score_spread_before_after': self._metric_before_after(
                    before_stage1_probe_metrics,
                    before_stage1_probe_metrics,
                    'score_spread',
                ),
                'stage1_tail_stage1_probe_same_state_gap_before_after': self._metric_before_after(
                    before_stage1_probe_metrics,
                    before_stage1_probe_metrics,
                    'same_state_score_gap',
                ),
                'stage1_tail_stage1_probe_pair_ranking_accuracy_before_after': self._metric_before_after(
                    before_stage1_probe_metrics,
                    before_stage1_probe_metrics,
                    'pair_ranking_accuracy',
                ),
            }
            return self.last_pair_metrics

        print(
            f"[WorldModel PairFT] start on {self.device}, pairs={pair_count}, "
            f"stage5_pairs={len(stage5_pair_samples)}, stage1_probe_pairs={len(stage1_probe_pair_samples)}, stage4_pairs={len(stage4_pair_samples)}, "
            f"trusted_pairs={trusted_pair_count}, hard_negatives={hard_negative_count}, replay_samples={replay_sample_count}, "
            f"eval_replay_samples={eval_replay_sample_count}"
        )
        resolution_min_score_gap = float(getattr(self.config, 'pair_ft_resolution_min_score_gap', 0.03) or 0.03)
        resolution_legacy_min_logit_gap = float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14)
        stage1_resolution_min_score_gap = float(
            getattr(self.config, 'pair_ft_stage1_resolution_min_score_gap', 0.015) or 0.015
        )
        stage1_resolution_loss_weight = float(
            getattr(self.config, 'pair_ft_stage1_resolution_loss_weight', 0.0) or 0.0
        )
        tail_ranking_loss_weight_effective = float(
            tail_ranking_loss_weight_cfg
            if tail_ranking_loss_weight_cfg is not None
            else float(self.config.ranking_loss_weight)
        )
        tail_resolution_loss_weight_effective = float(
            tail_resolution_loss_weight_cfg
            if tail_resolution_loss_weight_cfg is not None
            else float(stage1_resolution_loss_weight)
        )
        stage1_resolution_mode = str(getattr(self.config, 'pair_ft_stage1_resolution_mode', 'fixed') or 'fixed').strip().lower()
        if stage1_resolution_mode not in {'fixed', 'adaptive'}:
            stage1_resolution_mode = 'fixed'
        stage1_resolution_alpha = float(getattr(self.config, 'pair_ft_stage1_resolution_alpha', 0.2) or 0.2)
        stage1_resolution_max_score_gap = float(
            getattr(self.config, 'pair_ft_stage1_resolution_max_score_gap', 0.05) or 0.05
        )
        stage1_resolution_apply_trusted_only = bool(
            getattr(self.config, 'pair_ft_stage1_resolution_apply_trusted_only', True)
        )
        selection_accuracy_tie_epsilon = float(
            getattr(self.config, 'pair_ft_selection_accuracy_tie_epsilon', 1e-4) or 1e-4
        )
        print(
            f"[WorldModel PairFT] resolution_space=score, min_score_gap={resolution_min_score_gap:.4f}, "
            f"resolution_weight={float(getattr(self.config, 'pair_ft_resolution_loss_weight', 0.0) or 0.0):.4f}"
        )
        print(
            f"[WorldModel PairFT] stage1_resolution_space=score, mode={stage1_resolution_mode}, "
            f"min_score_gap={stage1_resolution_min_score_gap:.4f}, "
            f"max_score_gap={stage1_resolution_max_score_gap:.4f}, alpha={stage1_resolution_alpha:.4f}, "
            f"resolution_weight={stage1_resolution_loss_weight:.4f}, "
            f"trusted_only={stage1_resolution_apply_trusted_only}"
        )
        print(
            "[WorldModel PairFT] compatibility note: pair_ft_resolution_min_logit_gap is loaded "
            "for backward compatibility but ignored because resolution_space=score."
        )
        print(
            f"[WorldModel PairFT] stage1_tail: enabled={bool(stage1_tail_epochs > 0)}, "
            f"epochs={int(stage1_tail_epochs)}, pair_count={int(len(stage1_tail_pair_samples))}, "
            f"trusted_only={bool(stage1_tail_apply_trusted_only)}, "
            f"sampling_mode={str(stage1_tail_sampling_mode)}, "
            f"ranking_weight={tail_ranking_loss_weight_effective:.4f}, "
            f"resolution_weight={tail_resolution_loss_weight_effective:.4f}, "
            f"anticollapse_weight={stage1_tail_anticollapse_weight_effective:.4f}, "
            f"range_floor={stage1_tail_score_range_floor_effective:.4f}, "
            f"range_q=({stage1_tail_score_range_quantile_low_effective:.2f},{stage1_tail_score_range_quantile_high_effective:.2f})"
        )
        print(
            f"[WorldModel PairFT] phase_b_stage1_priority_mix: enabled={bool(stage1_priority_mix_enabled)}, "
            f"fraction={stage1_priority_mix_fraction:.4f}, trusted_only={bool(stage1_priority_trusted_only)}, "
            f"pair_count={int(len(stage1_priority_pair_samples))}"
        )
        print(
            f"[WorldModel PairFT] phase_b_stage1_anticollapse: weight={phaseb_stage1_anticollapse_weight_effective:.4f}, "
            f"floor={phaseb_stage1_anticollapse_score_range_floor:.4f}, "
            f"q=({phaseb_stage1_anticollapse_q_low:.2f},{phaseb_stage1_anticollapse_q_high:.2f}), "
            f"apply_on={phaseb_stage1_anticollapse_apply_on}"
        )
        replay_loader = self._build_replay_loader(replay_samples)
        replay_iter = iter(replay_loader) if replay_loader is not None else None
        stage4_loader = self._build_pair_loader(stage4_pair_samples)
        stage4_iter = iter(stage4_loader) if stage4_loader is not None else None
        grad_state, frozen_modules, trainable_modules = self._apply_pair_ft_freeze_policy()
        optimizer = torch.optim.Adam([parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=self.config.learning_rate)

        self.model.train()
        global_step = 0
        epoch_metrics: List[Dict[str, float]] = []
        stage5_batch_size = max(1, min(int(self.config.batch_size), max(1, len(stage5_pair_samples)))) if stage5_pair_samples else 0
        stage1_probe_batch_size = max(1, min(int(self.config.batch_size), max(1, len(stage1_probe_pair_samples)))) if stage1_probe_pair_samples else 0
        stage1_priority_batch_size = (
            max(1, min(int(self.config.batch_size), max(1, len(stage1_priority_pair_samples))))
            if stage1_priority_pair_samples
            else 0
        )
        stage1_tail_batch_size = (
            max(1, min(int(self.config.batch_size), max(1, len(stage1_tail_pair_samples))))
            if stage1_tail_pair_samples
            else 0
        )
        total_epochs = max(1, int(self.config.pair_finetune_epochs))
        phase_a_epochs = min(total_epochs, 1 if (stage5_pair_samples or stage1_probe_pair_samples) else 0)
        phase_b_epochs = max(0, total_epochs - phase_a_epochs)
        source_mix['phase_a_epochs'] = int(phase_a_epochs)
        source_mix['phase_b_epochs'] = int(phase_b_epochs)
        best_eval_pairs = stage5_pair_samples or stage1_probe_pair_samples or pair_samples
        initial_metrics = dict(self.evaluate_pairs(best_eval_pairs))
        initial_stage1_probe_metrics = dict(self.evaluate_pairs(stage1_probe_pair_samples))
        legacy_best_metrics = dict(initial_metrics)
        legacy_best_stage1_probe_metrics = dict(initial_stage1_probe_metrics)
        legacy_best_epoch = -1
        legacy_best_reason = 'legacy_initial_metrics'
        legacy_best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        eligible_best_metrics: Optional[Dict[str, float]] = None
        eligible_best_stage1_probe_metrics: Optional[Dict[str, float]] = None
        eligible_best_epoch = -1
        eligible_best_reason = 'eligible_not_available'
        eligible_best_state: Optional[Dict[str, torch.Tensor]] = None
        if self._is_epoch_metrics_eligible_for_unique_guard(initial_metrics):
            eligible_best_metrics = dict(initial_metrics)
            eligible_best_stage1_probe_metrics = dict(initial_stage1_probe_metrics)
            eligible_best_reason = 'eligible_initial_metrics'
            eligible_best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
        patience = max(0, int(getattr(self.config, 'pair_ft_patience', 0) or 0))
        epochs_without_improvement = 0
        total_stage5_seen_counts = {pair_id: 0 for pair_id in stage5_pair_ids}
        stage5_cap_reached_ids = set()
        phaseb_stage1_anticollapse_steps = 0
        phaseb_stage1_anticollapse_active_pair_count = 0
        phaseb_stage1_anticollapse_loss_total = 0.0
        phaseb_stage1_score_range_q10_sum = 0.0
        phaseb_stage1_score_range_q90_sum = 0.0
        phaseb_stage1_score_range_sum = 0.0
        phaseb_stage1_score_range_values: List[float] = []
        phaseb_stage1_score_range_below_floor_steps = 0

        for epoch_idx in range(total_epochs):
            phase_name = 'phase_a_strong_only' if epoch_idx < phase_a_epochs else 'phase_b_source_mixed'
            epoch_total = 0.0
            epoch_pointwise = 0.0
            epoch_ranking = 0.0
            epoch_spread = 0.0
            epoch_resolution = 0.0
            epoch_stage1_resolution = 0.0
            epoch_stage4_resolution = 0.0
            epoch_steps = 0
            epoch_stage5_seen_counts = {pair_id: 0 for pair_id in stage5_pair_ids}
            epoch_stage4_aux_active_pairs = 0
            epoch_stage4_aux_logit_gap_sum = 0.0
            epoch_stage4_aux_score_gap_values: List[float] = []
            epoch_stage4_aux_below_score_margin_count = 0
            epoch_stage4_aux_below_logit_margin_count = 0
            epoch_stage4_pairs_seen = 0
            epoch_stage1_probe_active_pairs = 0
            epoch_stage1_probe_score_gap_values: List[float] = []
            epoch_stage1_probe_margin_values: List[float] = []
            epoch_stage1_probe_pred_gap_to_margin_ratios: List[float] = []
            epoch_stage1_probe_below_score_margin_count = 0
            epoch_stage1_probe_below_adaptive_margin_count = 0
            epoch_stage1_probe_pairs_seen = 0
            epoch_phaseb_stage1_anticollapse_steps = 0
            epoch_phaseb_stage1_anticollapse_active_pair_count = 0
            epoch_phaseb_stage1_anticollapse_loss_total = 0.0
            epoch_phaseb_stage1_score_range_q10_sum = 0.0
            epoch_phaseb_stage1_score_range_q90_sum = 0.0
            epoch_phaseb_stage1_score_range_sum = 0.0
            epoch_phaseb_stage1_score_range_values: List[float] = []
            epoch_phaseb_stage1_score_range_below_floor_steps = 0
            step_targets = [1]
            if stage5_pair_samples and stage5_batch_size > 0:
                step_targets.append(int(np.ceil(len(stage5_pair_samples) / float(stage5_batch_size))))
            if stage1_probe_pair_samples and stage1_probe_batch_size > 0:
                step_targets.append(int(np.ceil(len(stage1_probe_pair_samples) / float(stage1_probe_batch_size))))
            total_epoch_steps = max(step_targets)

            for step_idx in range(total_epoch_steps):
                ranking_terms: List[torch.Tensor] = []
                spread_terms: List[torch.Tensor] = []
                stage4_resolution_terms: List[torch.Tensor] = []
                stage1_resolution_terms: List[torch.Tensor] = []
                phaseb_stage1_anticollapse_terms: List[torch.Tensor] = []

                if stage5_pair_samples:
                    stage5_batch, selected_pair_ids = self._sample_stage5_batch_with_cap(
                        stage5_pair_samples=stage5_pair_samples,
                        pair_ids=stage5_pair_ids,
                        batch_size=stage5_batch_size,
                        seen_counts=epoch_stage5_seen_counts,
                        max_seen_per_epoch=int(getattr(self.config, 'stage5_pair_max_seen_per_epoch', 0) or 0),
                    )
                    if stage5_batch:
                        stage5_ranking_loss, stage5_spread_loss, _, _ = self._compute_pair_losses(
                            stage5_batch,
                            enable_resolution=False,
                        )
                        ranking_terms.append(stage5_ranking_loss)
                        spread_terms.append(stage5_spread_loss)
                        source_mix['stage5_steps'] += 1
                        source_mix['stage5_pairs_seen'] += len(stage5_batch)
                        for pair_id in selected_pair_ids:
                            epoch_stage5_seen_counts[pair_id] += 1
                            total_stage5_seen_counts[pair_id] += 1
                            if int(getattr(self.config, 'stage5_pair_max_seen_per_epoch', 0) or 0) > 0 and epoch_stage5_seen_counts[pair_id] >= int(getattr(self.config, 'stage5_pair_max_seen_per_epoch', 0) or 0):
                                stage5_cap_reached_ids.add(pair_id)

                if stage1_probe_pair_samples:
                    stage1_probe_batch = self._sample_pair_batch_with_replacement(stage1_probe_pair_samples, stage1_probe_batch_size)
                    if stage1_probe_batch:
                        enable_phaseb_stage1_anticollapse_on_standard = bool(
                            epoch_idx >= phase_a_epochs
                            and phaseb_stage1_anticollapse_weight_effective > 0.0
                            and phaseb_stage1_anticollapse_apply_on == 'all_stage1'
                        )
                        stage1_ranking_loss, stage1_spread_loss, stage1_resolution_loss, stage1_diag = self._compute_pair_losses(
                            stage1_probe_batch,
                            enable_resolution=False,
                            enable_stage1_resolution=True,
                            enable_stage1_tail_anticollapse=enable_phaseb_stage1_anticollapse_on_standard,
                            stage1_tail_score_range_floor=phaseb_stage1_anticollapse_score_range_floor,
                            stage1_tail_score_range_quantile_low=phaseb_stage1_anticollapse_q_low,
                            stage1_tail_score_range_quantile_high=phaseb_stage1_anticollapse_q_high,
                        )
                        ranking_terms.append(stage1_ranking_loss)
                        spread_terms.append(stage1_spread_loss)
                        stage1_resolution_terms.append(stage1_resolution_loss)
                        source_mix['stage1_probe_steps'] += 1
                        source_mix['stage1_probe_pairs_seen'] += len(stage1_probe_batch)
                        epoch_stage1_probe_pairs_seen += int(len(stage1_probe_batch))
                        stage1_active_count = int(stage1_diag.get('stage1_probe_active_pair_count', 0))
                        epoch_stage1_probe_active_pairs += stage1_active_count
                        epoch_stage1_probe_score_gap_values.extend(
                            float(value)
                            for value in list(stage1_diag.get('stage1_probe_score_gaps', []) or [])
                        )
                        epoch_stage1_probe_margin_values.extend(
                            float(value)
                            for value in list(stage1_diag.get('stage1_probe_margins', []) or [])
                        )
                        epoch_stage1_probe_pred_gap_to_margin_ratios.extend(
                            float(value)
                            for value in list(stage1_diag.get('stage1_probe_pred_gap_to_margin_ratios', []) or [])
                        )
                        epoch_stage1_probe_below_score_margin_count += int(
                            stage1_diag.get('stage1_probe_below_score_margin_count', 0)
                        )
                        epoch_stage1_probe_below_adaptive_margin_count += int(
                            stage1_diag.get('stage1_probe_below_adaptive_margin_count', 0)
                        )
                        if enable_phaseb_stage1_anticollapse_on_standard:
                            stage1_active_count = int(stage1_diag.get('stage1_probe_active_pair_count', 0))
                            if stage1_active_count > 0:
                                stage1_phaseb_anticollapse_loss_tensor = stage1_diag.get(
                                    'stage1_tail_anticollapse_loss_tensor',
                                    self._zero_loss(),
                                )
                                phaseb_stage1_anticollapse_terms.append(stage1_phaseb_anticollapse_loss_tensor)
                                epoch_phaseb_stage1_anticollapse_steps += 1
                                epoch_phaseb_stage1_anticollapse_active_pair_count += stage1_active_count
                                epoch_phaseb_stage1_anticollapse_loss_total += float(
                                    stage1_phaseb_anticollapse_loss_tensor.detach().item()
                                )
                                epoch_phaseb_stage1_score_range_q10_sum += float(
                                    stage1_diag.get('stage1_tail_score_range_q10', 0.0)
                                )
                                epoch_phaseb_stage1_score_range_q90_sum += float(
                                    stage1_diag.get('stage1_tail_score_range_q90', 0.0)
                                )
                                stage1_range_value = float(
                                    stage1_diag.get('stage1_tail_score_range', 0.0)
                                )
                                epoch_phaseb_stage1_score_range_sum += stage1_range_value
                                epoch_phaseb_stage1_score_range_values.append(stage1_range_value)
                                if stage1_range_value < float(phaseb_stage1_anticollapse_score_range_floor):
                                    epoch_phaseb_stage1_score_range_below_floor_steps += 1
                                phaseb_stage1_anticollapse_steps += 1
                                phaseb_stage1_anticollapse_active_pair_count += stage1_active_count
                                phaseb_stage1_anticollapse_loss_total += float(
                                    stage1_phaseb_anticollapse_loss_tensor.detach().item()
                                )
                                phaseb_stage1_score_range_q10_sum += float(
                                    stage1_diag.get('stage1_tail_score_range_q10', 0.0)
                                )
                                phaseb_stage1_score_range_q90_sum += float(
                                    stage1_diag.get('stage1_tail_score_range_q90', 0.0)
                                )
                                phaseb_stage1_score_range_sum += stage1_range_value
                                phaseb_stage1_score_range_values.append(stage1_range_value)
                                if stage1_range_value < float(phaseb_stage1_anticollapse_score_range_floor):
                                    phaseb_stage1_score_range_below_floor_steps += 1

                if (
                    epoch_idx >= phase_a_epochs
                    and stage1_priority_mix_enabled
                    and stage1_priority_pair_samples
                    and stage1_priority_batch_size > 0
                    and self.rng.random() < stage1_priority_mix_fraction
                ):
                    stage1_priority_batch = self._sample_pair_batch_with_replacement(
                        stage1_priority_pair_samples,
                        stage1_priority_batch_size,
                    )
                    if stage1_priority_batch:
                        enable_phaseb_stage1_anticollapse_on_priority = bool(
                            phaseb_stage1_anticollapse_weight_effective > 0.0
                            and phaseb_stage1_anticollapse_apply_on in {'priority_only', 'all_stage1'}
                        )
                        (
                            stage1_priority_ranking_loss,
                            stage1_priority_spread_loss,
                            stage1_priority_resolution_loss,
                            stage1_priority_diag,
                        ) = self._compute_pair_losses(
                            stage1_priority_batch,
                            enable_resolution=False,
                            enable_stage1_resolution=True,
                            enable_stage1_tail_anticollapse=enable_phaseb_stage1_anticollapse_on_priority,
                            stage1_tail_score_range_floor=phaseb_stage1_anticollapse_score_range_floor,
                            stage1_tail_score_range_quantile_low=phaseb_stage1_anticollapse_q_low,
                            stage1_tail_score_range_quantile_high=phaseb_stage1_anticollapse_q_high,
                        )
                        ranking_terms.append(stage1_priority_ranking_loss)
                        spread_terms.append(stage1_priority_spread_loss)
                        stage1_resolution_terms.append(stage1_priority_resolution_loss)
                        source_mix['phase_b_stage1_priority_steps'] += 1
                        source_mix['phase_b_stage1_priority_pairs_seen'] += len(stage1_priority_batch)
                        epoch_stage1_probe_pairs_seen += int(len(stage1_priority_batch))
                        stage1_priority_active_count = int(stage1_priority_diag.get('stage1_probe_active_pair_count', 0))
                        epoch_stage1_probe_active_pairs += stage1_priority_active_count
                        epoch_stage1_probe_score_gap_values.extend(
                            float(value)
                            for value in list(stage1_priority_diag.get('stage1_probe_score_gaps', []) or [])
                        )
                        epoch_stage1_probe_margin_values.extend(
                            float(value)
                            for value in list(stage1_priority_diag.get('stage1_probe_margins', []) or [])
                        )
                        epoch_stage1_probe_pred_gap_to_margin_ratios.extend(
                            float(value)
                            for value in list(stage1_priority_diag.get('stage1_probe_pred_gap_to_margin_ratios', []) or [])
                        )
                        epoch_stage1_probe_below_score_margin_count += int(
                            stage1_priority_diag.get('stage1_probe_below_score_margin_count', 0)
                        )
                        epoch_stage1_probe_below_adaptive_margin_count += int(
                            stage1_priority_diag.get('stage1_probe_below_adaptive_margin_count', 0)
                        )
                        if enable_phaseb_stage1_anticollapse_on_priority:
                            stage1_priority_active_count = int(
                                stage1_priority_diag.get('stage1_probe_active_pair_count', 0)
                            )
                            if stage1_priority_active_count > 0:
                                stage1_priority_anticollapse_loss_tensor = stage1_priority_diag.get(
                                    'stage1_tail_anticollapse_loss_tensor',
                                    self._zero_loss(),
                                )
                                phaseb_stage1_anticollapse_terms.append(stage1_priority_anticollapse_loss_tensor)
                                epoch_phaseb_stage1_anticollapse_steps += 1
                                epoch_phaseb_stage1_anticollapse_active_pair_count += stage1_priority_active_count
                                epoch_phaseb_stage1_anticollapse_loss_total += float(
                                    stage1_priority_anticollapse_loss_tensor.detach().item()
                                )
                                epoch_phaseb_stage1_score_range_q10_sum += float(
                                    stage1_priority_diag.get('stage1_tail_score_range_q10', 0.0)
                                )
                                epoch_phaseb_stage1_score_range_q90_sum += float(
                                    stage1_priority_diag.get('stage1_tail_score_range_q90', 0.0)
                                )
                                stage1_priority_range_value = float(
                                    stage1_priority_diag.get('stage1_tail_score_range', 0.0)
                                )
                                epoch_phaseb_stage1_score_range_sum += stage1_priority_range_value
                                epoch_phaseb_stage1_score_range_values.append(stage1_priority_range_value)
                                if stage1_priority_range_value < float(phaseb_stage1_anticollapse_score_range_floor):
                                    epoch_phaseb_stage1_score_range_below_floor_steps += 1
                                phaseb_stage1_anticollapse_steps += 1
                                phaseb_stage1_anticollapse_active_pair_count += stage1_priority_active_count
                                phaseb_stage1_anticollapse_loss_total += float(
                                    stage1_priority_anticollapse_loss_tensor.detach().item()
                                )
                                phaseb_stage1_score_range_q10_sum += float(
                                    stage1_priority_diag.get('stage1_tail_score_range_q10', 0.0)
                                )
                                phaseb_stage1_score_range_q90_sum += float(
                                    stage1_priority_diag.get('stage1_tail_score_range_q90', 0.0)
                                )
                                phaseb_stage1_score_range_sum += stage1_priority_range_value
                                phaseb_stage1_score_range_values.append(stage1_priority_range_value)
                                if stage1_priority_range_value < float(phaseb_stage1_anticollapse_score_range_floor):
                                    phaseb_stage1_score_range_below_floor_steps += 1

                if epoch_idx >= phase_a_epochs and stage4_loader is not None and (step_idx + 1) % int(stage4_mix_every_n_steps) == 0:
                    stage4_batch, stage4_iter = self._next_pair_batch(stage4_iter, stage4_loader)
                    if stage4_batch:
                        stage4_ranking_loss, _, stage4_resolution_loss, stage4_diag = self._compute_pair_losses(
                            stage4_batch,
                            enable_resolution=True,
                        )
                        ranking_terms.append(stage4_ranking_loss)
                        stage4_resolution_terms.append(stage4_resolution_loss)
                        source_mix['stage4_steps'] += 1
                        source_mix['stage4_pairs_seen'] += len(stage4_batch)
                        epoch_stage4_pairs_seen += int(len(stage4_batch))
                        stage4_aux_active_count = int(stage4_diag.get('stage4_aux_active_pair_count', 0))
                        epoch_stage4_aux_active_pairs += stage4_aux_active_count
                        epoch_stage4_aux_logit_gap_sum += float(stage4_diag.get('stage4_aux_logit_gap_mean', 0.0)) * float(stage4_aux_active_count)
                        epoch_stage4_aux_score_gap_values.extend(
                            float(value)
                            for value in list(stage4_diag.get('stage4_aux_score_gaps', []) or [])
                        )
                        epoch_stage4_aux_below_score_margin_count += int(stage4_diag.get('stage4_aux_below_score_margin_count', 0))
                        epoch_stage4_aux_below_logit_margin_count += int(stage4_diag.get('stage4_aux_below_margin_count', 0))

                ranking_loss = torch.mean(torch.stack(ranking_terms)) if ranking_terms else self._zero_loss()
                spread_loss = torch.mean(torch.stack(spread_terms)) if spread_terms else self._zero_loss()
                stage4_resolution_loss = (
                    torch.mean(torch.stack(stage4_resolution_terms)) if stage4_resolution_terms else self._zero_loss()
                )
                stage1_resolution_loss = (
                    torch.mean(torch.stack(stage1_resolution_terms)) if stage1_resolution_terms else self._zero_loss()
                )
                phaseb_stage1_anticollapse_loss = (
                    torch.mean(torch.stack(phaseb_stage1_anticollapse_terms))
                    if phaseb_stage1_anticollapse_terms
                    else self._zero_loss()
                )
                resolution_loss = stage4_resolution_loss + stage1_resolution_loss
                pointwise_total, _, _, _, replay_iter = self._compute_replay_losses(replay_iter, replay_loader)
                total_loss = (
                    float(self.config.pointwise_replay_weight) * pointwise_total
                    + float(self.config.ranking_loss_weight) * ranking_loss
                    + float(self.config.spread_loss_weight) * spread_loss
                    + float(getattr(self.config, 'pair_ft_resolution_loss_weight', 0.0) or 0.0) * stage4_resolution_loss
                    + float(getattr(self.config, 'pair_ft_stage1_resolution_loss_weight', 0.0) or 0.0) * stage1_resolution_loss
                    + float(phaseb_stage1_anticollapse_weight_effective) * phaseb_stage1_anticollapse_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                step_total = float(total_loss.item())
                step_pointwise = float(pointwise_total.item())
                step_ranking = float(ranking_loss.item())
                step_spread = float(spread_loss.item())
                step_resolution = float(resolution_loss.item())
                epoch_total += step_total
                epoch_pointwise += step_pointwise
                epoch_ranking += step_ranking
                epoch_spread += step_spread
                epoch_resolution += step_resolution
                epoch_stage4_resolution += float(stage4_resolution_loss.item())
                epoch_stage1_resolution += float(stage1_resolution_loss.item())
                epoch_steps += 1

                if tb_writer is not None:
                    tb_writer.add_scalar('loss/step_total', step_total, global_step)
                    tb_writer.add_scalar('loss/step_pointwise', step_pointwise, global_step)
                    tb_writer.add_scalar('loss/step_ranking', step_ranking, global_step)
                    tb_writer.add_scalar('loss/step_spread', step_spread, global_step)
                    tb_writer.add_scalar('loss/step_resolution', step_resolution, global_step)
                global_step += 1

            if epoch_steps > 0:
                epoch_summary = {
                    'epoch': float(epoch_idx),
                    'phase': phase_name,
                    'loss_total': epoch_total / epoch_steps,
                    'loss_pointwise': epoch_pointwise / epoch_steps,
                    'loss_ranking': epoch_ranking / epoch_steps,
                    'loss_spread': epoch_spread / epoch_steps,
                    'loss_resolution': epoch_resolution / epoch_steps,
                    'stage4_aux_resolution_loss': epoch_stage4_resolution / epoch_steps,
                    'stage4_aux_active_pair_count': float(epoch_stage4_aux_active_pairs),
                    'stage4_aux_active_pair_fraction': (
                        float(epoch_stage4_aux_active_pairs) / float(epoch_stage4_pairs_seen)
                        if epoch_stage4_pairs_seen > 0
                        else 0.0
                    ),
                    'stage4_aux_logit_gap_mean': (
                        float(epoch_stage4_aux_logit_gap_sum) / float(epoch_stage4_aux_active_pairs)
                        if epoch_stage4_aux_active_pairs > 0
                        else 0.0
                    ),
                    'stage4_aux_score_gap_mean': (
                        float(np.mean(epoch_stage4_aux_score_gap_values))
                        if epoch_stage4_aux_score_gap_values
                        else 0.0
                    ),
                    'stage4_aux_score_gap_p50': (
                        float(np.quantile(epoch_stage4_aux_score_gap_values, 0.50))
                        if epoch_stage4_aux_score_gap_values
                        else 0.0
                    ),
                    'stage4_aux_score_gap_p90': (
                        float(np.quantile(epoch_stage4_aux_score_gap_values, 0.90))
                        if epoch_stage4_aux_score_gap_values
                        else 0.0
                    ),
                    'stage4_aux_below_score_margin_count': float(epoch_stage4_aux_below_score_margin_count),
                    'stage4_aux_below_score_margin_fraction': (
                        float(epoch_stage4_aux_below_score_margin_count) / float(epoch_stage4_aux_active_pairs)
                        if epoch_stage4_aux_active_pairs > 0
                        else 0.0
                    ),
                    # Backward-compatible logit-space diagnostics.
                    'stage4_aux_below_margin_count': float(epoch_stage4_aux_below_logit_margin_count),
                    'stage4_aux_below_margin_fraction': (
                        float(epoch_stage4_aux_below_logit_margin_count) / float(epoch_stage4_aux_active_pairs)
                        if epoch_stage4_aux_active_pairs > 0
                        else 0.0
                    ),
                    'stage1_probe_resolution_loss': epoch_stage1_resolution / epoch_steps,
                    'stage1_probe_active_pair_count': float(epoch_stage1_probe_active_pairs),
                    'stage1_probe_active_pair_fraction': (
                        float(epoch_stage1_probe_active_pairs) / float(epoch_stage1_probe_pairs_seen)
                        if epoch_stage1_probe_pairs_seen > 0
                        else 0.0
                    ),
                    'stage1_probe_score_gap_mean': (
                        float(np.mean(epoch_stage1_probe_score_gap_values))
                        if epoch_stage1_probe_score_gap_values
                        else 0.0
                    ),
                    'stage1_probe_score_gap_p50': (
                        float(np.quantile(epoch_stage1_probe_score_gap_values, 0.50))
                        if epoch_stage1_probe_score_gap_values
                        else 0.0
                    ),
                    'stage1_probe_score_gap_p90': (
                        float(np.quantile(epoch_stage1_probe_score_gap_values, 0.90))
                        if epoch_stage1_probe_score_gap_values
                        else 0.0
                    ),
                    'stage1_probe_margin_mean': (
                        float(np.mean(epoch_stage1_probe_margin_values))
                        if epoch_stage1_probe_margin_values
                        else 0.0
                    ),
                    'stage1_probe_margin_p50': (
                        float(np.quantile(epoch_stage1_probe_margin_values, 0.50))
                        if epoch_stage1_probe_margin_values
                        else 0.0
                    ),
                    'stage1_probe_margin_p90': (
                        float(np.quantile(epoch_stage1_probe_margin_values, 0.90))
                        if epoch_stage1_probe_margin_values
                        else 0.0
                    ),
                    'stage1_probe_pred_gap_to_margin_ratio_mean': (
                        float(np.mean(epoch_stage1_probe_pred_gap_to_margin_ratios))
                        if epoch_stage1_probe_pred_gap_to_margin_ratios
                        else 0.0
                    ),
                    'stage1_probe_below_score_margin_count': float(epoch_stage1_probe_below_score_margin_count),
                    'stage1_probe_below_score_margin_fraction': (
                        float(epoch_stage1_probe_below_score_margin_count) / float(epoch_stage1_probe_active_pairs)
                        if epoch_stage1_probe_active_pairs > 0
                        else 0.0
                    ),
                    'stage1_probe_below_adaptive_margin_fraction': (
                        float(epoch_stage1_probe_below_adaptive_margin_count)
                        / float(epoch_stage1_probe_active_pairs)
                        if epoch_stage1_probe_active_pairs > 0
                        else 0.0
                    ),
                    'phase_b_stage1_anticollapse_loss': (
                        float(epoch_phaseb_stage1_anticollapse_loss_total)
                        / float(epoch_phaseb_stage1_anticollapse_steps)
                        if epoch_phaseb_stage1_anticollapse_steps > 0
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_q10': (
                        float(epoch_phaseb_stage1_score_range_q10_sum)
                        / float(epoch_phaseb_stage1_anticollapse_steps)
                        if epoch_phaseb_stage1_anticollapse_steps > 0
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_q90': (
                        float(epoch_phaseb_stage1_score_range_q90_sum)
                        / float(epoch_phaseb_stage1_anticollapse_steps)
                        if epoch_phaseb_stage1_anticollapse_steps > 0
                        else 0.0
                    ),
                    'phase_b_stage1_score_range': (
                        float(epoch_phaseb_stage1_score_range_sum)
                        / float(epoch_phaseb_stage1_anticollapse_steps)
                        if epoch_phaseb_stage1_anticollapse_steps > 0
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_below_floor_fraction': (
                        float(epoch_phaseb_stage1_score_range_below_floor_steps)
                        / float(epoch_phaseb_stage1_anticollapse_steps)
                        if epoch_phaseb_stage1_anticollapse_steps > 0
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_p10': (
                        float(np.quantile(epoch_phaseb_stage1_score_range_values, 0.10))
                        if epoch_phaseb_stage1_score_range_values
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_p50': (
                        float(np.quantile(epoch_phaseb_stage1_score_range_values, 0.50))
                        if epoch_phaseb_stage1_score_range_values
                        else 0.0
                    ),
                    'phase_b_stage1_score_range_p90': (
                        float(np.quantile(epoch_phaseb_stage1_score_range_values, 0.90))
                        if epoch_phaseb_stage1_score_range_values
                        else 0.0
                    ),
                    'phase_b_stage1_anticollapse_active_pair_count': float(
                        epoch_phaseb_stage1_anticollapse_active_pair_count
                    ),
                }
                eval_metrics = self.evaluate_pairs(best_eval_pairs)
                eval_stage1_probe_metrics = self.evaluate_pairs(stage1_probe_pair_samples)
                epoch_summary.update(
                    {
                        'eval_pair_ranking_accuracy': float(eval_metrics.get('pair_ranking_accuracy', 0.0)),
                        'eval_same_state_score_gap': float(eval_metrics.get('same_state_score_gap', 0.0)),
                        'eval_score_spread': float(eval_metrics.get('score_spread', 0.0)),
                        'eval_unique_score_count': float(eval_metrics.get('unique_score_count', 0.0)),
                        'stage1_probe_eval_pair_ranking_accuracy': float(
                            eval_stage1_probe_metrics.get('pair_ranking_accuracy', 0.0)
                        ),
                        'stage1_probe_eval_same_state_score_gap': float(
                            eval_stage1_probe_metrics.get('same_state_score_gap', 0.0)
                        ),
                        'stage1_probe_eval_score_spread': float(
                            eval_stage1_probe_metrics.get('score_spread', 0.0)
                        ),
                        'stage1_probe_eval_unique_score_count': float(
                            eval_stage1_probe_metrics.get('unique_score_count', 0.0)
                        ),
                    }
                )
                epoch_metrics.append(epoch_summary)
                print(
                    f"[WorldModel PairFT] epoch {epoch_idx + 1}/{total_epochs} ({phase_name}), "
                    f"total={epoch_summary['loss_total']:.6f}, pointwise={epoch_summary['loss_pointwise']:.6f}, "
                    f"ranking={epoch_summary['loss_ranking']:.6f}, spread={epoch_summary['loss_spread']:.6f}, "
                    f"resolution={epoch_summary['loss_resolution']:.6f}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar('loss/epoch_total', epoch_summary['loss_total'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_pointwise', epoch_summary['loss_pointwise'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_ranking', epoch_summary['loss_ranking'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_spread', epoch_summary['loss_spread'], epoch_idx)
                    tb_writer.add_scalar('loss/epoch_resolution', epoch_summary['loss_resolution'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_active_pair_count', epoch_summary['stage4_aux_active_pair_count'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_active_pair_fraction', epoch_summary['stage4_aux_active_pair_fraction'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_logit_gap_mean', epoch_summary['stage4_aux_logit_gap_mean'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_score_gap_mean', epoch_summary['stage4_aux_score_gap_mean'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_score_gap_p50', epoch_summary['stage4_aux_score_gap_p50'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_score_gap_p90', epoch_summary['stage4_aux_score_gap_p90'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_below_score_margin_count', epoch_summary['stage4_aux_below_score_margin_count'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_below_score_margin_fraction', epoch_summary['stage4_aux_below_score_margin_fraction'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_below_margin_count', epoch_summary['stage4_aux_below_margin_count'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage4_aux_below_margin_fraction', epoch_summary['stage4_aux_below_margin_fraction'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_active_pair_count', epoch_summary['stage1_probe_active_pair_count'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_active_pair_fraction', epoch_summary['stage1_probe_active_pair_fraction'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_resolution_loss', epoch_summary['stage1_probe_resolution_loss'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_score_gap_mean', epoch_summary['stage1_probe_score_gap_mean'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_score_gap_p50', epoch_summary['stage1_probe_score_gap_p50'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_score_gap_p90', epoch_summary['stage1_probe_score_gap_p90'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_margin_mean', epoch_summary['stage1_probe_margin_mean'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_margin_p50', epoch_summary['stage1_probe_margin_p50'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_margin_p90', epoch_summary['stage1_probe_margin_p90'], epoch_idx)
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_pred_gap_to_margin_ratio_mean',
                        epoch_summary['stage1_probe_pred_gap_to_margin_ratio_mean'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar('eval/epoch_stage1_probe_below_score_margin_count', epoch_summary['stage1_probe_below_score_margin_count'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_stage1_probe_below_score_margin_fraction', epoch_summary['stage1_probe_below_score_margin_fraction'], epoch_idx)
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_below_adaptive_margin_fraction',
                        epoch_summary['stage1_probe_below_adaptive_margin_fraction'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_anticollapse_loss',
                        epoch_summary['phase_b_stage1_anticollapse_loss'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_q10',
                        epoch_summary['phase_b_stage1_score_range_q10'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_q90',
                        epoch_summary['phase_b_stage1_score_range_q90'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range',
                        epoch_summary['phase_b_stage1_score_range'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_below_floor_fraction',
                        epoch_summary['phase_b_stage1_score_range_below_floor_fraction'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_p10',
                        epoch_summary['phase_b_stage1_score_range_p10'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_p50',
                        epoch_summary['phase_b_stage1_score_range_p50'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_score_range_p90',
                        epoch_summary['phase_b_stage1_score_range_p90'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_phase_b_stage1_anticollapse_active_pair_count',
                        epoch_summary['phase_b_stage1_anticollapse_active_pair_count'],
                        epoch_idx,
                    )
                if tb_writer is not None:
                    tb_writer.add_scalar('eval/epoch_pair_ranking_accuracy', epoch_summary['eval_pair_ranking_accuracy'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_same_state_score_gap', epoch_summary['eval_same_state_score_gap'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_score_spread', epoch_summary['eval_score_spread'], epoch_idx)
                    tb_writer.add_scalar('eval/epoch_unique_score_count', epoch_summary['eval_unique_score_count'], epoch_idx)
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_eval_pair_ranking_accuracy',
                        epoch_summary['stage1_probe_eval_pair_ranking_accuracy'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_eval_same_state_score_gap',
                        epoch_summary['stage1_probe_eval_same_state_score_gap'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_eval_score_spread',
                        epoch_summary['stage1_probe_eval_score_spread'],
                        epoch_idx,
                    )
                    tb_writer.add_scalar(
                        'eval/epoch_stage1_probe_eval_unique_score_count',
                        epoch_summary['stage1_probe_eval_unique_score_count'],
                        epoch_idx,
                    )

                improved_for_patience = False
                legacy_cmp, legacy_reason = self._compare_pair_ft_metrics_for_legacy_selection(
                    current_eval_metrics=eval_metrics,
                    best_eval_metrics=legacy_best_metrics,
                    current_stage1_probe_metrics=eval_stage1_probe_metrics,
                    best_stage1_probe_metrics=legacy_best_stage1_probe_metrics,
                )
                if legacy_cmp > 0:
                    legacy_best_metrics = dict(eval_metrics)
                    legacy_best_stage1_probe_metrics = dict(eval_stage1_probe_metrics)
                    legacy_best_epoch = int(epoch_idx)
                    legacy_best_reason = str(legacy_reason or 'legacy_tieaware_improved')
                    legacy_best_state = {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}
                    if eligible_best_metrics is None:
                        improved_for_patience = True

                if self._is_epoch_metrics_eligible_for_unique_guard(eval_metrics):
                    if (
                        eligible_best_metrics is None
                        or self._compare_pair_ft_metrics_for_selection(eval_metrics, eligible_best_metrics) > 0
                    ):
                        eligible_best_metrics = dict(eval_metrics)
                        eligible_best_stage1_probe_metrics = dict(eval_stage1_probe_metrics)
                        eligible_best_epoch = int(epoch_idx)
                        eligible_best_reason = 'eligible_tieaware_improved'
                        eligible_best_state = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in self.model.state_dict().items()
                        }
                        improved_for_patience = True

                if improved_for_patience:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if patience > 0 and epochs_without_improvement >= patience:
                        break

        if eligible_best_state is not None and eligible_best_metrics is not None:
            selected_best_state = eligible_best_state
            selected_best_metrics = eligible_best_metrics
            selected_best_stage1_probe_metrics = dict(
                eligible_best_stage1_probe_metrics or self._empty_pair_metrics()
            )
            selected_best_epoch = int(eligible_best_epoch)
            selection_path = 'eligible'
            selection_reason = str(eligible_best_reason or 'eligible_selected')
        else:
            selected_best_state = legacy_best_state
            selected_best_metrics = legacy_best_metrics
            selected_best_stage1_probe_metrics = dict(legacy_best_stage1_probe_metrics)
            selected_best_epoch = int(legacy_best_epoch)
            selection_path = 'legacy_tieaware'
            selection_reason = str(legacy_best_reason or 'legacy_tieaware_selected')

        self.model.load_state_dict(selected_best_state)
        stage1_tail_enabled = bool(stage1_tail_epochs > 0)
        stage1_tail_applied = False
        stage1_tail_epochs_executed = 0
        stage1_tail_internal_best_epoch = -1
        stage1_tail_internal_best_reason = 'tail_not_applied'
        stage1_tail_internal_best_stage1_probe_unique = float(
            (selected_best_stage1_probe_metrics or {}).get('unique_score_count', 0.0)
        )
        stage1_tail_accepted = False
        stage1_tail_acceptance_reason = 'tail_not_applied'
        stage1_tail_stage1_probe_metrics_before = self.evaluate_pairs(stage1_probe_pair_samples)
        stage1_tail_eval_metrics_before = self.evaluate_pairs(best_eval_pairs)
        stage1_tail_stage1_probe_metrics_after = dict(stage1_tail_stage1_probe_metrics_before)
        stage1_tail_eval_metrics_after = dict(stage1_tail_eval_metrics_before)
        stage1_tail_ranking_loss_total = 0.0
        stage1_tail_resolution_loss_total = 0.0
        stage1_tail_anticollapse_loss_total = 0.0
        stage1_tail_score_range_q10_sum = 0.0
        stage1_tail_score_range_q90_sum = 0.0
        stage1_tail_score_range_sum = 0.0
        stage1_tail_loss_steps = 0
        stage1_tail_floor_reject_reasons: Dict[str, int] = {}
        min_spread_floor = float(getattr(self.config, 'pair_ft_min_score_spread_floor', 0.0) or 0.0)
        min_gap_floor = float(getattr(self.config, 'pair_ft_min_same_state_gap_floor', 0.0) or 0.0)
        tail_internal_best_state: Optional[Dict[str, torch.Tensor]] = None
        tail_internal_best_stage1_probe_metrics: Optional[Dict[str, float]] = None
        tail_internal_best_eval_metrics: Optional[Dict[str, float]] = None
        if stage1_tail_enabled and stage1_tail_pair_samples and stage1_tail_batch_size > 0:
            tail_optimizer = torch.optim.Adam(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                lr=self.config.learning_rate,
            )
            self.model.train()
            tail_epoch_steps_target = max(
                1,
                int(np.ceil(len(stage1_tail_pair_samples) / float(stage1_tail_batch_size))),
            )
            for tail_epoch_idx in range(int(stage1_tail_epochs)):
                phase_name = 'phase_c_stage1_tail'
                epoch_total = 0.0
                epoch_ranking = 0.0
                epoch_resolution = 0.0
                epoch_anticollapse = 0.0
                epoch_stage1_resolution = 0.0
                epoch_steps = 0
                epoch_stage1_probe_active_pairs = 0
                epoch_stage1_probe_score_gap_values: List[float] = []
                epoch_stage1_probe_margin_values: List[float] = []
                epoch_stage1_probe_pred_gap_to_margin_ratios: List[float] = []
                epoch_stage1_probe_below_score_margin_count = 0
                epoch_stage1_probe_below_adaptive_margin_count = 0
                epoch_stage1_probe_pairs_seen = 0
                epoch_stage1_tail_score_range_q10_sum = 0.0
                epoch_stage1_tail_score_range_q90_sum = 0.0
                epoch_stage1_tail_score_range_sum = 0.0
                epoch_tail_batches_without_replacement: List[List[RiskPairSample]] = []
                if stage1_tail_sampling_mode == 'without_replacement':
                    epoch_tail_batches_without_replacement = self._build_pair_batches_without_replacement(
                        stage1_tail_pair_samples,
                        stage1_tail_batch_size,
                    )

                for _ in range(tail_epoch_steps_target):
                    if stage1_tail_sampling_mode == 'without_replacement':
                        if not epoch_tail_batches_without_replacement:
                            continue
                        stage1_tail_batch = epoch_tail_batches_without_replacement.pop(0)
                    else:
                        stage1_tail_batch = self._sample_pair_batch_with_replacement(
                            stage1_tail_pair_samples,
                            stage1_tail_batch_size,
                        )
                    if not stage1_tail_batch:
                        continue
                    stage1_ranking_loss, _, stage1_resolution_loss, stage1_diag = self._compute_pair_losses(
                        stage1_tail_batch,
                        enable_resolution=False,
                        enable_stage1_resolution=True,
                        enable_stage1_tail_anticollapse=True,
                        stage1_tail_score_range_floor=stage1_tail_score_range_floor_effective,
                        stage1_tail_score_range_quantile_low=stage1_tail_score_range_quantile_low_effective,
                        stage1_tail_score_range_quantile_high=stage1_tail_score_range_quantile_high_effective,
                    )
                    stage1_tail_anticollapse_loss = stage1_diag.get('stage1_tail_anticollapse_loss_tensor', self._zero_loss())
                    total_loss = (
                        float(tail_ranking_loss_weight_effective) * stage1_ranking_loss
                        + float(tail_resolution_loss_weight_effective) * stage1_resolution_loss
                        + float(stage1_tail_anticollapse_weight_effective) * stage1_tail_anticollapse_loss
                    )
                    tail_optimizer.zero_grad()
                    total_loss.backward()
                    tail_optimizer.step()

                    source_mix['stage1_tail_steps'] += 1
                    source_mix['stage1_tail_pairs_seen'] += len(stage1_tail_batch)
                    epoch_stage1_probe_pairs_seen += int(len(stage1_tail_batch))

                    stage1_active_count = int(stage1_diag.get('stage1_probe_active_pair_count', 0))
                    epoch_stage1_probe_active_pairs += stage1_active_count
                    epoch_stage1_probe_score_gap_values.extend(
                        float(value)
                        for value in list(stage1_diag.get('stage1_probe_score_gaps', []) or [])
                    )
                    epoch_stage1_probe_margin_values.extend(
                        float(value)
                        for value in list(stage1_diag.get('stage1_probe_margins', []) or [])
                    )
                    epoch_stage1_probe_pred_gap_to_margin_ratios.extend(
                        float(value)
                        for value in list(stage1_diag.get('stage1_probe_pred_gap_to_margin_ratios', []) or [])
                    )
                    epoch_stage1_probe_below_score_margin_count += int(
                        stage1_diag.get('stage1_probe_below_score_margin_count', 0)
                    )
                    epoch_stage1_probe_below_adaptive_margin_count += int(
                        stage1_diag.get('stage1_probe_below_adaptive_margin_count', 0)
                    )
                    epoch_stage1_tail_score_range_q10_sum += float(stage1_diag.get('stage1_tail_score_range_q10', 0.0))
                    epoch_stage1_tail_score_range_q90_sum += float(stage1_diag.get('stage1_tail_score_range_q90', 0.0))
                    epoch_stage1_tail_score_range_sum += float(stage1_diag.get('stage1_tail_score_range', 0.0))
                    stage1_tail_score_range_q10_sum += float(stage1_diag.get('stage1_tail_score_range_q10', 0.0))
                    stage1_tail_score_range_q90_sum += float(stage1_diag.get('stage1_tail_score_range_q90', 0.0))
                    stage1_tail_score_range_sum += float(stage1_diag.get('stage1_tail_score_range', 0.0))

                    step_total = float(total_loss.item())
                    step_ranking = float(stage1_ranking_loss.item())
                    step_resolution = float(stage1_resolution_loss.item())
                    step_anticollapse = float(stage1_tail_anticollapse_loss.detach().item())
                    stage1_tail_ranking_loss_total += step_ranking
                    stage1_tail_resolution_loss_total += step_resolution
                    stage1_tail_anticollapse_loss_total += step_anticollapse
                    stage1_tail_loss_steps += 1
                    epoch_total += step_total
                    epoch_ranking += step_ranking
                    epoch_resolution += step_resolution
                    epoch_anticollapse += step_anticollapse
                    epoch_stage1_resolution += step_resolution
                    epoch_steps += 1

                    if tb_writer is not None:
                        tb_writer.add_scalar('loss/step_total', step_total, global_step)
                        tb_writer.add_scalar('loss/step_pointwise', 0.0, global_step)
                        tb_writer.add_scalar('loss/step_ranking', step_ranking, global_step)
                        tb_writer.add_scalar('loss/step_spread', 0.0, global_step)
                        tb_writer.add_scalar('loss/step_resolution', step_resolution, global_step)
                        tb_writer.add_scalar('loss/step_tail_anticollapse', step_anticollapse, global_step)
                    global_step += 1

                if epoch_steps > 0:
                    stage1_tail_applied = True
                    stage1_tail_epochs_executed += 1
                    epoch_summary = {
                        'epoch': float(total_epochs + tail_epoch_idx),
                        'phase': phase_name,
                        'loss_total': epoch_total / epoch_steps,
                        'loss_pointwise': 0.0,
                        'loss_ranking': epoch_ranking / epoch_steps,
                        'loss_spread': 0.0,
                        'loss_resolution': epoch_resolution / epoch_steps,
                        'stage1_tail_anticollapse_loss': epoch_anticollapse / epoch_steps,
                        'stage1_tail_score_range_q10': epoch_stage1_tail_score_range_q10_sum / epoch_steps,
                        'stage1_tail_score_range_q90': epoch_stage1_tail_score_range_q90_sum / epoch_steps,
                        'stage1_tail_score_range': epoch_stage1_tail_score_range_sum / epoch_steps,
                        'stage4_aux_resolution_loss': 0.0,
                        'stage4_aux_active_pair_count': 0.0,
                        'stage4_aux_active_pair_fraction': 0.0,
                        'stage4_aux_logit_gap_mean': 0.0,
                        'stage4_aux_score_gap_mean': 0.0,
                        'stage4_aux_score_gap_p50': 0.0,
                        'stage4_aux_score_gap_p90': 0.0,
                        'stage4_aux_below_score_margin_count': 0.0,
                        'stage4_aux_below_score_margin_fraction': 0.0,
                        'stage4_aux_below_margin_count': 0.0,
                        'stage4_aux_below_margin_fraction': 0.0,
                        'stage1_probe_resolution_loss': epoch_stage1_resolution / epoch_steps,
                        'stage1_probe_active_pair_count': float(epoch_stage1_probe_active_pairs),
                        'stage1_probe_active_pair_fraction': (
                            float(epoch_stage1_probe_active_pairs) / float(epoch_stage1_probe_pairs_seen)
                            if epoch_stage1_probe_pairs_seen > 0
                            else 0.0
                        ),
                        'stage1_probe_score_gap_mean': (
                            float(np.mean(epoch_stage1_probe_score_gap_values))
                            if epoch_stage1_probe_score_gap_values
                            else 0.0
                        ),
                        'stage1_probe_score_gap_p50': (
                            float(np.quantile(epoch_stage1_probe_score_gap_values, 0.50))
                            if epoch_stage1_probe_score_gap_values
                            else 0.0
                        ),
                        'stage1_probe_score_gap_p90': (
                            float(np.quantile(epoch_stage1_probe_score_gap_values, 0.90))
                            if epoch_stage1_probe_score_gap_values
                            else 0.0
                        ),
                        'stage1_probe_margin_mean': (
                            float(np.mean(epoch_stage1_probe_margin_values))
                            if epoch_stage1_probe_margin_values
                            else 0.0
                        ),
                        'stage1_probe_margin_p50': (
                            float(np.quantile(epoch_stage1_probe_margin_values, 0.50))
                            if epoch_stage1_probe_margin_values
                            else 0.0
                        ),
                        'stage1_probe_margin_p90': (
                            float(np.quantile(epoch_stage1_probe_margin_values, 0.90))
                            if epoch_stage1_probe_margin_values
                            else 0.0
                        ),
                        'stage1_probe_pred_gap_to_margin_ratio_mean': (
                            float(np.mean(epoch_stage1_probe_pred_gap_to_margin_ratios))
                            if epoch_stage1_probe_pred_gap_to_margin_ratios
                            else 0.0
                        ),
                        'stage1_probe_below_score_margin_count': float(epoch_stage1_probe_below_score_margin_count),
                        'stage1_probe_below_score_margin_fraction': (
                            float(epoch_stage1_probe_below_score_margin_count) / float(epoch_stage1_probe_active_pairs)
                            if epoch_stage1_probe_active_pairs > 0
                            else 0.0
                        ),
                        'stage1_probe_below_adaptive_margin_fraction': (
                            float(epoch_stage1_probe_below_adaptive_margin_count)
                            / float(epoch_stage1_probe_active_pairs)
                            if epoch_stage1_probe_active_pairs > 0
                            else 0.0
                        ),
                        'phase_b_stage1_anticollapse_loss': 0.0,
                        'phase_b_stage1_score_range_q10': 0.0,
                        'phase_b_stage1_score_range_q90': 0.0,
                        'phase_b_stage1_score_range': 0.0,
                        'phase_b_stage1_score_range_below_floor_fraction': 0.0,
                        'phase_b_stage1_score_range_p10': 0.0,
                        'phase_b_stage1_score_range_p50': 0.0,
                        'phase_b_stage1_score_range_p90': 0.0,
                        'phase_b_stage1_anticollapse_active_pair_count': 0.0,
                        'stage1_tail_sampling_mode_effective': str(stage1_tail_sampling_mode),
                    }
                    eval_metrics = self.evaluate_pairs(best_eval_pairs)
                    eval_stage1_probe_metrics = self.evaluate_pairs(stage1_probe_pair_samples)
                    epoch_summary.update(
                        {
                            'eval_pair_ranking_accuracy': float(eval_metrics.get('pair_ranking_accuracy', 0.0)),
                            'eval_same_state_score_gap': float(eval_metrics.get('same_state_score_gap', 0.0)),
                            'eval_score_spread': float(eval_metrics.get('score_spread', 0.0)),
                            'eval_unique_score_count': float(eval_metrics.get('unique_score_count', 0.0)),
                            'stage1_probe_eval_pair_ranking_accuracy': float(
                                eval_stage1_probe_metrics.get('pair_ranking_accuracy', 0.0)
                            ),
                            'stage1_probe_eval_same_state_score_gap': float(
                                eval_stage1_probe_metrics.get('same_state_score_gap', 0.0)
                            ),
                            'stage1_probe_eval_score_spread': float(
                                eval_stage1_probe_metrics.get('score_spread', 0.0)
                            ),
                        'stage1_probe_eval_unique_score_count': float(
                                eval_stage1_probe_metrics.get('unique_score_count', 0.0)
                            ),
                        }
                    )
                    floor_passed, floor_reason = self._stage1_tail_passes_floor(
                        candidate_stage1_probe_metrics=eval_stage1_probe_metrics,
                        pre_tail_stage1_probe_metrics=stage1_tail_stage1_probe_metrics_before,
                        min_spread_floor=min_spread_floor,
                        min_gap_floor=min_gap_floor,
                        acc_tolerance=stage1_tail_acceptance_acc_tolerance,
                        spread_tolerance=stage1_tail_acceptance_spread_tolerance,
                        gap_tolerance=stage1_tail_acceptance_gap_tolerance,
                    )
                    epoch_summary['stage1_tail_floor_passed'] = bool(floor_passed)
                    epoch_summary['stage1_tail_floor_reason'] = str(floor_reason)
                    epoch_summary['stage1_tail_is_internal_best'] = False
                    epoch_summary['stage1_tail_internal_best_reason'] = ''
                    if floor_passed:
                        should_update_best = False
                        best_update_reason = 'tail_first_floor_pass'
                        if tail_internal_best_stage1_probe_metrics is None:
                            should_update_best = True
                        else:
                            best_cmp, best_update_reason = self._compare_stage1_tail_metrics(
                                current_stage1_probe_metrics=eval_stage1_probe_metrics,
                                best_stage1_probe_metrics=tail_internal_best_stage1_probe_metrics,
                            )
                            should_update_best = best_cmp > 0
                        if should_update_best:
                            tail_internal_best_state = {
                                name: tensor.detach().cpu().clone()
                                for name, tensor in self.model.state_dict().items()
                            }
                            tail_internal_best_stage1_probe_metrics = dict(eval_stage1_probe_metrics)
                            tail_internal_best_eval_metrics = dict(eval_metrics)
                            stage1_tail_internal_best_epoch = int(total_epochs + tail_epoch_idx)
                            stage1_tail_internal_best_reason = str(best_update_reason)
                            stage1_tail_internal_best_stage1_probe_unique = float(
                                eval_stage1_probe_metrics.get('unique_score_count', 0.0)
                            )
                            epoch_summary['stage1_tail_is_internal_best'] = True
                            epoch_summary['stage1_tail_internal_best_reason'] = str(best_update_reason)
                    else:
                        stage1_tail_floor_reject_reasons[str(floor_reason)] = (
                            int(stage1_tail_floor_reject_reasons.get(str(floor_reason), 0)) + 1
                        )
                    if (not floor_passed) and tail_internal_best_state is None:
                        stage1_tail_internal_best_reason = str(floor_reason)
                    epoch_metrics.append(epoch_summary)

            self.model.eval()
        if stage1_tail_applied and tail_internal_best_state is not None:
            self.model.load_state_dict(tail_internal_best_state)
            stage1_tail_stage1_probe_metrics_after = dict(
                tail_internal_best_stage1_probe_metrics or self.evaluate_pairs(stage1_probe_pair_samples)
            )
            stage1_tail_eval_metrics_after = dict(
                tail_internal_best_eval_metrics or self.evaluate_pairs(best_eval_pairs)
            )
            stage1_tail_accepted, stage1_tail_acceptance_reason = self._evaluate_stage1_tail_acceptance(
                pre_tail_stage1_probe_metrics=stage1_tail_stage1_probe_metrics_before,
                pre_tail_eval_metrics=stage1_tail_eval_metrics_before,
                candidate_stage1_probe_metrics=stage1_tail_stage1_probe_metrics_after,
                candidate_eval_metrics=stage1_tail_eval_metrics_after,
                acceptance_enabled=stage1_tail_acceptance_enabled,
                acc_tolerance=stage1_tail_acceptance_acc_tolerance,
                spread_tolerance=stage1_tail_acceptance_spread_tolerance,
                gap_tolerance=stage1_tail_acceptance_gap_tolerance,
                min_spread_floor=min_spread_floor,
                min_gap_floor=min_gap_floor,
            )
            if not stage1_tail_accepted:
                self.model.load_state_dict(selected_best_state)
                stage1_tail_stage1_probe_metrics_after = dict(stage1_tail_stage1_probe_metrics_before)
                stage1_tail_eval_metrics_after = dict(stage1_tail_eval_metrics_before)
        elif stage1_tail_applied:
            self.model.load_state_dict(selected_best_state)
            stage1_tail_internal_best_reason = 'tail_no_internal_best'
            stage1_tail_acceptance_reason = 'tail_no_internal_best'
            stage1_tail_stage1_probe_metrics_after = dict(stage1_tail_stage1_probe_metrics_before)
            stage1_tail_eval_metrics_after = dict(stage1_tail_eval_metrics_before)
            stage1_tail_accepted = False

        self._restore_grad_state(grad_state)
        self.model.eval()
        after_pair_metrics = self.evaluate_pairs(pair_samples)
        after_stage5_metrics = self.evaluate_pairs(stage5_pair_samples)
        after_stage1_probe_metrics = self.evaluate_pairs(stage1_probe_pair_samples)
        after_stage4_metrics = self.evaluate_pairs(stage4_pair_samples)
        after_stage4_high_gap_metrics = self.evaluate_pairs(stage4_high_gap_pair_samples)
        after_stage4_aux_metrics = self.evaluate_pairs(stage4_aux_pair_samples)
        after_stage4_aux_logit_gap_metrics = self._evaluate_pair_logit_gap_metrics(stage4_aux_pair_samples)
        after_pointwise_metrics = self._evaluate_risk_only_samples(eval_replay_samples)
        source_mix['stage5_pair_seen_counts'] = dict(total_stage5_seen_counts)
        source_mix['stage5_cap_reached_pairs'] = int(len(stage5_cap_reached_ids))
        source_mix['phase_b_stage1_anticollapse_steps'] = int(phaseb_stage1_anticollapse_steps)
        source_mix['phase_b_stage1_anticollapse_active_pair_count'] = int(
            phaseb_stage1_anticollapse_active_pair_count
        )
        source_mix['phase_b_stage1_score_range_below_floor_fraction'] = (
            float(phaseb_stage1_score_range_below_floor_steps) / float(phaseb_stage1_anticollapse_steps)
            if phaseb_stage1_anticollapse_steps > 0
            else 0.0
        )
        self.last_pair_metrics = after_pair_metrics
        self.last_pair_ft_report = {
            'enabled': True,
            'pair_count': int(len(pair_samples)),
            'replay_sample_count': int(replay_sample_count),
            'eval_replay_sample_count': int(eval_replay_sample_count),
            'before_pair_metrics': before_pair_metrics,
            'after_pair_metrics': after_pair_metrics,
            'before_pointwise_metrics': before_pointwise_metrics,
            'after_pointwise_metrics': after_pointwise_metrics,
            'epoch_metrics': epoch_metrics,
            'resolution_space': 'score',
            'pair_ft_resolution_min_score_gap': float(getattr(self.config, 'pair_ft_resolution_min_score_gap', 0.03) or 0.03),
            'ignored_legacy_logit_margin': float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14),
            'pair_ft_resolution_min_logit_gap_compat': float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14),
            'stage1_resolution_space': 'score',
            'stage1_resolution_mode': str(stage1_resolution_mode),
            'pair_ft_stage1_resolution_min_score_gap': float(stage1_resolution_min_score_gap),
            'pair_ft_stage1_resolution_loss_weight': float(stage1_resolution_loss_weight),
            'pair_ft_stage1_resolution_alpha': float(stage1_resolution_alpha),
            'pair_ft_stage1_resolution_max_score_gap': float(stage1_resolution_max_score_gap),
            'pair_ft_stage1_resolution_apply_trusted_only': bool(stage1_resolution_apply_trusted_only),
            'stage1_tail_ranking_loss_weight_effective': float(tail_ranking_loss_weight_effective),
            'stage1_tail_resolution_loss_weight_effective': float(tail_resolution_loss_weight_effective),
            'stage1_tail_anticollapse_weight_effective': float(stage1_tail_anticollapse_weight_effective),
            'stage1_tail_score_range_floor_effective': float(stage1_tail_score_range_floor_effective),
            'stage1_tail_score_range_quantiles_effective': {
                'low': float(stage1_tail_score_range_quantile_low_effective),
                'high': float(stage1_tail_score_range_quantile_high_effective),
            },
            'stage1_tail_ranking_loss': (
                float(stage1_tail_ranking_loss_total) / float(stage1_tail_loss_steps)
                if stage1_tail_loss_steps > 0
                else 0.0
            ),
            'stage1_tail_resolution_loss': (
                float(stage1_tail_resolution_loss_total) / float(stage1_tail_loss_steps)
                if stage1_tail_loss_steps > 0
                else 0.0
            ),
            'stage1_tail_anticollapse_loss': (
                float(stage1_tail_anticollapse_loss_total) / float(stage1_tail_loss_steps)
                if stage1_tail_loss_steps > 0
                else 0.0
            ),
            'stage1_tail_score_range_q10_q90': {
                'q10': (
                    float(stage1_tail_score_range_q10_sum) / float(stage1_tail_loss_steps)
                    if stage1_tail_loss_steps > 0
                    else 0.0
                ),
                'q90': (
                    float(stage1_tail_score_range_q90_sum) / float(stage1_tail_loss_steps)
                    if stage1_tail_loss_steps > 0
                    else 0.0
                ),
                'range': (
                    float(stage1_tail_score_range_sum) / float(stage1_tail_loss_steps)
                    if stage1_tail_loss_steps > 0
                    else 0.0
                ),
            },
            'stage1_tail_floor_reject_reasons': dict(stage1_tail_floor_reject_reasons),
            'pair_ft_selection_accuracy_tie_epsilon_effective': float(selection_accuracy_tie_epsilon),
            'stage1_tail_sampling_mode_effective': str(stage1_tail_sampling_mode),
            'phase_b_stage1_anticollapse_weight_effective': float(
                phaseb_stage1_anticollapse_weight_effective
            ),
            'phase_b_stage1_anticollapse_apply_on_effective': str(
                phaseb_stage1_anticollapse_apply_on
            ),
            'phase_b_stage1_anticollapse_steps': int(phaseb_stage1_anticollapse_steps),
            'phase_b_stage1_anticollapse_active_pair_count': int(
                phaseb_stage1_anticollapse_active_pair_count
            ),
            'phase_b_stage1_anticollapse_loss': (
                float(phaseb_stage1_anticollapse_loss_total) / float(phaseb_stage1_anticollapse_steps)
                if phaseb_stage1_anticollapse_steps > 0
                else 0.0
            ),
            'phase_b_stage1_score_range_q10_q90': {
                'q10': (
                    float(phaseb_stage1_score_range_q10_sum) / float(phaseb_stage1_anticollapse_steps)
                    if phaseb_stage1_anticollapse_steps > 0
                    else 0.0
                ),
                'q90': (
                    float(phaseb_stage1_score_range_q90_sum) / float(phaseb_stage1_anticollapse_steps)
                    if phaseb_stage1_anticollapse_steps > 0
                    else 0.0
                ),
                'range': (
                    float(phaseb_stage1_score_range_sum) / float(phaseb_stage1_anticollapse_steps)
                    if phaseb_stage1_anticollapse_steps > 0
                    else 0.0
                ),
            },
            'phase_b_stage1_score_range_below_floor_fraction': (
                float(phaseb_stage1_score_range_below_floor_steps) / float(phaseb_stage1_anticollapse_steps)
                if phaseb_stage1_anticollapse_steps > 0
                else 0.0
            ),
            'phase_b_stage1_score_range_p10': (
                float(np.quantile(phaseb_stage1_score_range_values, 0.10))
                if phaseb_stage1_score_range_values
                else 0.0
            ),
            'phase_b_stage1_score_range_p50': (
                float(np.quantile(phaseb_stage1_score_range_values, 0.50))
                if phaseb_stage1_score_range_values
                else 0.0
            ),
            'phase_b_stage1_score_range_p90': (
                float(np.quantile(phaseb_stage1_score_range_values, 0.90))
                if phaseb_stage1_score_range_values
                else 0.0
            ),
            'world_pair_ft_frozen_modules': frozen_modules,
            'world_pair_ft_trainable_modules': trainable_modules,
            'world_pair_ft_source_mix': dict(source_mix),
            'stage5_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage5_metrics, after_stage5_metrics, 'pair_ranking_accuracy'),
            'stage1_probe_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage1_probe_metrics, after_stage1_probe_metrics, 'pair_ranking_accuracy'),
            'stage4_pair_ranking_accuracy_before_after': self._metric_before_after(before_stage4_metrics, after_stage4_metrics, 'pair_ranking_accuracy'),
            'stage5_same_state_score_gap_before_after': self._metric_before_after(before_stage5_metrics, after_stage5_metrics, 'same_state_score_gap'),
            'stage1_probe_same_state_score_gap_before_after': self._metric_before_after(before_stage1_probe_metrics, after_stage1_probe_metrics, 'same_state_score_gap'),
            'stage4_same_state_score_gap_before_after': self._metric_before_after(before_stage4_metrics, after_stage4_metrics, 'same_state_score_gap'),
            'stage5_score_spread_before_after': self._metric_before_after(before_stage5_metrics, after_stage5_metrics, 'score_spread'),
            'stage1_probe_score_spread_before_after': self._metric_before_after(before_stage1_probe_metrics, after_stage1_probe_metrics, 'score_spread'),
            'stage4_score_spread_before_after': self._metric_before_after(before_stage4_metrics, after_stage4_metrics, 'score_spread'),
            'stage5_unique_score_count_before_after': self._metric_before_after(before_stage5_metrics, after_stage5_metrics, 'unique_score_count'),
            'stage1_probe_unique_score_count_before_after': self._metric_before_after(before_stage1_probe_metrics, after_stage1_probe_metrics, 'unique_score_count'),
            'stage4_unique_score_count_before_after': self._metric_before_after(before_stage4_metrics, after_stage4_metrics, 'unique_score_count'),
            'stage4_high_gap_pair_count': int(len(stage4_high_gap_pair_samples)),
            'stage4_high_gap_unique_score_count_before_after': self._metric_before_after(
                before_stage4_high_gap_metrics,
                after_stage4_high_gap_metrics,
                'unique_score_count',
            ),
            'stage4_aux_pair_count': int(len(stage4_aux_pair_samples)),
            'stage4_aux_unique_score_count_before_after': self._metric_before_after(
                before_stage4_aux_metrics,
                after_stage4_aux_metrics,
                'unique_score_count',
            ),
            'stage4_aux_logit_gap_before_after': {
                'before': float(before_stage4_aux_logit_gap_metrics.get('mean_abs_logit_gap', 0.0)),
                'after': float(after_stage4_aux_logit_gap_metrics.get('mean_abs_logit_gap', 0.0)),
            },
            'stage4_aux_score_spread_before_after': self._metric_before_after(
                before_stage4_aux_metrics,
                after_stage4_aux_metrics,
                'score_spread',
            ),
            'stage4_aux_same_state_score_gap_before_after': self._metric_before_after(
                before_stage4_aux_metrics,
                after_stage4_aux_metrics,
                'same_state_score_gap',
            ),
            'stage5_spread_eligible_pair_count': int(stage5_spread_eligible_count),
            'stage1_probe_spread_eligible_pair_count': int(stage1_probe_spread_eligible_count),
            'stage4_spread_eligible_pair_count': int(stage4_spread_eligible_count),
            'world_pair_ft_best_epoch': int(selected_best_epoch),
            'world_pair_ft_best_metrics': dict(selected_best_metrics),
            'world_pair_ft_restored_best': True,
            'selection_path': str(selection_path),
            'selection_reason': str(selection_reason),
            'best_epoch_stage1_unique': float(
                (selected_best_stage1_probe_metrics or {}).get('unique_score_count', 0.0)
            ),
            'best_epoch_eval_unique': float((selected_best_metrics or {}).get('unique_score_count', 0.0)),
            'stage1_tail_enabled': bool(stage1_tail_enabled),
            'stage1_tail_applied': bool(stage1_tail_applied),
            'stage1_tail_epochs_configured': int(stage1_tail_epochs),
            'stage1_tail_epochs_executed': int(stage1_tail_epochs_executed),
            'stage1_tail_pair_count': int(len(stage1_tail_pair_samples)),
            'phase_b_stage1_priority_enabled': bool(stage1_priority_mix_enabled),
            'phase_b_stage1_priority_fraction_configured': float(stage1_priority_mix_fraction),
            'phase_b_stage1_priority_trusted_only': bool(stage1_priority_trusted_only),
            'phase_b_stage1_priority_pair_count': int(len(stage1_priority_pair_samples)),
            'phase_b_stage1_priority_steps': int(source_mix.get('phase_b_stage1_priority_steps', 0)),
            'phase_b_stage1_priority_pairs_seen': int(source_mix.get('phase_b_stage1_priority_pairs_seen', 0)),
            'stage1_tail_internal_best_epoch': int(stage1_tail_internal_best_epoch),
            'stage1_tail_internal_best_reason': str(stage1_tail_internal_best_reason),
            'stage1_tail_internal_best_stage1_probe_unique': float(stage1_tail_internal_best_stage1_probe_unique),
            'stage1_tail_accepted': bool(stage1_tail_accepted),
            'stage1_tail_acceptance_reason': str(stage1_tail_acceptance_reason),
            'stage1_tail_acceptance_thresholds': {
                'enabled': bool(stage1_tail_acceptance_enabled),
                'acc_tolerance': float(stage1_tail_acceptance_acc_tolerance),
                'spread_tolerance': float(stage1_tail_acceptance_spread_tolerance),
                'gap_tolerance': float(stage1_tail_acceptance_gap_tolerance),
            },
            'world_pair_ft_final_state_source': (
                'selected_best_plus_stage1_tail' if stage1_tail_accepted else 'selected_best'
            ),
            'stage1_tail_stage1_probe_unique_before_after': self._metric_before_after(
                stage1_tail_stage1_probe_metrics_before,
                stage1_tail_stage1_probe_metrics_after,
                'unique_score_count',
            ),
            'stage1_tail_stage1_probe_score_spread_before_after': self._metric_before_after(
                stage1_tail_stage1_probe_metrics_before,
                stage1_tail_stage1_probe_metrics_after,
                'score_spread',
            ),
            'stage1_tail_stage1_probe_same_state_gap_before_after': self._metric_before_after(
                stage1_tail_stage1_probe_metrics_before,
                stage1_tail_stage1_probe_metrics_after,
                'same_state_score_gap',
            ),
            'stage1_tail_stage1_probe_pair_ranking_accuracy_before_after': self._metric_before_after(
                stage1_tail_stage1_probe_metrics_before,
                stage1_tail_stage1_probe_metrics_after,
                'pair_ranking_accuracy',
            ),
        }
        if tb_writer is not None:
            tb_writer.add_scalar('summary/before_pair_ranking_accuracy', float(before_pair_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/after_pair_ranking_accuracy', float(after_pair_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/before_same_state_score_gap', float(before_pair_metrics.get('same_state_score_gap', 0.0)), 0)
            tb_writer.add_scalar('summary/after_same_state_score_gap', float(after_pair_metrics.get('same_state_score_gap', 0.0)), 0)
            tb_writer.add_scalar('summary/before_score_spread', float(before_pair_metrics.get('score_spread', 0.0)), 0)
            tb_writer.add_scalar('summary/after_score_spread', float(after_pair_metrics.get('score_spread', 0.0)), 0)
            tb_writer.add_scalar('summary/stage5_before_pair_ranking_accuracy', float(before_stage5_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/stage5_after_pair_ranking_accuracy', float(after_stage5_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/stage1_probe_before_pair_ranking_accuracy', float(before_stage1_probe_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/stage1_probe_after_pair_ranking_accuracy', float(after_stage1_probe_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/stage4_before_pair_ranking_accuracy', float(before_stage4_metrics.get('pair_ranking_accuracy', 0.0)), 0)
            tb_writer.add_scalar('summary/stage4_after_pair_ranking_accuracy', float(after_stage4_metrics.get('pair_ranking_accuracy', 0.0)), 0)
        return self.last_pair_metrics

    def _metric_before_after(self, before_metrics: Dict[str, float], after_metrics: Dict[str, float], key: str) -> Dict[str, float]:
        return {
            'before': float((before_metrics or {}).get(key, 0.0)),
            'after': float((after_metrics or {}).get(key, 0.0)),
        }

    def _filter_high_gap_pairs(self, pair_samples: Sequence[RiskPairSample]) -> List[RiskPairSample]:
        selected: List[RiskPairSample] = []
        for sample in pair_samples:
            target_a = float(sample.meta.get('target_risk_a', 0.0))
            target_b = float(sample.meta.get('target_risk_b', 0.0))
            if abs(target_a - target_b) >= float(SPREAD_TARGET_DELTA):
                selected.append(sample)
        return selected

    def _filter_stage4_aux_pairs(self, pair_samples: Sequence[RiskPairSample]) -> List[RiskPairSample]:
        threshold = float(getattr(self.config, 'stage4_aux_target_gap_threshold', 0.068) or 0.0)
        selected: List[RiskPairSample] = []
        for sample in pair_samples:
            if str(sample.source) != 'stage4_candidate_rank':
                continue
            meta = dict(sample.meta or {})
            aux_gap = float(meta.get('stage4_aux_gap', meta.get('target_gap', 0.0)) or 0.0)
            if bool(meta.get('stage4_aux_candidate', False)):
                selected.append(sample)
                continue
            if aux_gap >= threshold:
                selected.append(sample)
        return selected

    def _filter_stage1_tail_pairs(
        self,
        pair_samples: Sequence[RiskPairSample],
        trusted_only: bool = True,
    ) -> List[RiskPairSample]:
        selected: List[RiskPairSample] = []
        for sample in pair_samples:
            if str(sample.source) != 'stage1_probe_same_state':
                continue
            if trusted_only and not bool((sample.meta or {}).get('trusted_for_spread', False)):
                continue
            selected.append(sample)
        return selected

    def _is_epoch_metrics_eligible_for_unique_guard(self, metrics: Dict[str, float]) -> bool:
        min_unique_floor = float(getattr(self.config, 'pair_ft_min_unique_score_floor', 12) or 12)
        min_spread_floor = float(getattr(self.config, 'pair_ft_min_score_spread_floor', 0.0) or 0.0)
        min_gap_floor = float(getattr(self.config, 'pair_ft_min_same_state_gap_floor', 0.0) or 0.0)
        unique_count = float((metrics or {}).get('unique_score_count', 0.0))
        score_spread = float((metrics or {}).get('score_spread', 0.0))
        same_state_gap = float((metrics or {}).get('same_state_score_gap', 0.0))
        return (
            unique_count >= min_unique_floor
            and score_spread >= min_spread_floor
            and same_state_gap >= min_gap_floor
        )

    def _compare_pair_ft_metrics_for_selection(self, current_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> int:
        tie_epsilon = float(getattr(self.config, 'pair_ft_selection_accuracy_tie_epsilon', 1e-4) or 1e-4)
        current_acc = float((current_metrics or {}).get('pair_ranking_accuracy', 0.0))
        best_acc = float((best_metrics or {}).get('pair_ranking_accuracy', 0.0))
        if current_acc > best_acc + tie_epsilon:
            return 1
        if current_acc < best_acc - tie_epsilon:
            return -1

        current_unique = float((current_metrics or {}).get('unique_score_count', 0.0))
        best_unique = float((best_metrics or {}).get('unique_score_count', 0.0))
        if current_unique > best_unique + 1e-9:
            return 1
        if current_unique < best_unique - 1e-9:
            return -1

        current_gap = float((current_metrics or {}).get('same_state_score_gap', 0.0))
        best_gap = float((best_metrics or {}).get('same_state_score_gap', 0.0))
        if current_gap > best_gap + 1e-9:
            return 1
        if current_gap < best_gap - 1e-9:
            return -1

        current_spread = float((current_metrics or {}).get('score_spread', 0.0))
        best_spread = float((best_metrics or {}).get('score_spread', 0.0))
        if current_spread > best_spread + 1e-9:
            return 1
        if current_spread < best_spread - 1e-9:
            return -1
        return 0

    def _compare_pair_ft_metrics_for_legacy_selection(
        self,
        current_eval_metrics: Dict[str, float],
        best_eval_metrics: Dict[str, float],
        current_stage1_probe_metrics: Dict[str, float],
        best_stage1_probe_metrics: Dict[str, float],
    ) -> Tuple[int, str]:
        min_spread_floor = float(getattr(self.config, 'pair_ft_min_score_spread_floor', 0.0) or 0.0)
        min_gap_floor = float(getattr(self.config, 'pair_ft_min_same_state_gap_floor', 0.0) or 0.0)
        current_spread = float((current_eval_metrics or {}).get('score_spread', 0.0))
        best_spread = float((best_eval_metrics or {}).get('score_spread', 0.0))
        current_gap = float((current_eval_metrics or {}).get('same_state_score_gap', 0.0))
        best_gap = float((best_eval_metrics or {}).get('same_state_score_gap', 0.0))
        current_collapsed = (current_spread < min_spread_floor) or (current_gap < min_gap_floor)
        best_collapsed = (best_spread < min_spread_floor) or (best_gap < min_gap_floor)
        if current_collapsed and not best_collapsed:
            return -1, 'legacy_rejected_collapse_floor'
        if not current_collapsed and best_collapsed:
            return 1, 'legacy_noncollapsed_over_collapsed'

        tie_epsilon = float(getattr(self.config, 'pair_ft_selection_accuracy_tie_epsilon', 1e-4) or 1e-4)
        current_acc = float((current_eval_metrics or {}).get('pair_ranking_accuracy', 0.0))
        best_acc = float((best_eval_metrics or {}).get('pair_ranking_accuracy', 0.0))
        if current_acc > best_acc + tie_epsilon:
            return 1, 'legacy_accuracy_above_tie_epsilon'
        if current_acc < best_acc - tie_epsilon:
            return -1, 'legacy_accuracy_below_tie_epsilon'

        stage1_current_available = self._metrics_have_pairs(current_stage1_probe_metrics)
        stage1_best_available = self._metrics_have_pairs(best_stage1_probe_metrics)
        if stage1_current_available and stage1_best_available:
            tie_cmp, tie_reason = self._compare_unique_gap_spread_metrics(
                current_metrics=current_stage1_probe_metrics,
                best_metrics=best_stage1_probe_metrics,
            )
            if tie_cmp != 0:
                return tie_cmp, f'legacy_stage1_probe_{tie_reason}'

        tie_cmp, tie_reason = self._compare_unique_gap_spread_metrics(
            current_metrics=current_eval_metrics,
            best_metrics=best_eval_metrics,
        )
        if tie_cmp != 0:
            return tie_cmp, f'legacy_eval_{tie_reason}'
        return 0, 'legacy_tie_no_change'

    def _metrics_have_pairs(self, metrics: Optional[Dict[str, float]]) -> bool:
        return float((metrics or {}).get('pair_count', 0.0)) > 0.0

    def _compare_unique_gap_spread_metrics(
        self,
        current_metrics: Dict[str, float],
        best_metrics: Dict[str, float],
    ) -> Tuple[int, str]:
        current_unique = float((current_metrics or {}).get('unique_score_count', 0.0))
        best_unique = float((best_metrics or {}).get('unique_score_count', 0.0))
        if current_unique > best_unique + 1e-9:
            return 1, 'unique_higher'
        if current_unique < best_unique - 1e-9:
            return -1, 'unique_lower'

        current_gap = float((current_metrics or {}).get('same_state_score_gap', 0.0))
        best_gap = float((best_metrics or {}).get('same_state_score_gap', 0.0))
        if current_gap > best_gap + 1e-9:
            return 1, 'same_state_gap_higher'
        if current_gap < best_gap - 1e-9:
            return -1, 'same_state_gap_lower'

        current_spread = float((current_metrics or {}).get('score_spread', 0.0))
        best_spread = float((best_metrics or {}).get('score_spread', 0.0))
        if current_spread > best_spread + 1e-9:
            return 1, 'score_spread_higher'
        if current_spread < best_spread - 1e-9:
            return -1, 'score_spread_lower'
        return 0, 'metrics_equal'

    def _compare_stage1_tail_metrics(
        self,
        current_stage1_probe_metrics: Dict[str, float],
        best_stage1_probe_metrics: Dict[str, float],
    ) -> Tuple[int, str]:
        tie_cmp, tie_reason = self._compare_unique_gap_spread_metrics(
            current_metrics=current_stage1_probe_metrics,
            best_metrics=best_stage1_probe_metrics,
        )
        if tie_cmp != 0:
            return tie_cmp, f'stage1_{tie_reason}'

        current_acc = float((current_stage1_probe_metrics or {}).get('pair_ranking_accuracy', 0.0))
        best_acc = float((best_stage1_probe_metrics or {}).get('pair_ranking_accuracy', 0.0))
        if current_acc > best_acc + 1e-9:
            return 1, 'stage1_pair_ranking_accuracy_higher'
        if current_acc < best_acc - 1e-9:
            return -1, 'stage1_pair_ranking_accuracy_lower'
        return 0, 'stage1_metrics_equal'

    def _stage1_tail_passes_floor(
        self,
        candidate_stage1_probe_metrics: Dict[str, float],
        pre_tail_stage1_probe_metrics: Dict[str, float],
        min_spread_floor: float,
        min_gap_floor: float,
        acc_tolerance: float,
        spread_tolerance: float,
        gap_tolerance: float,
    ) -> Tuple[bool, str]:
        pre_acc = float((pre_tail_stage1_probe_metrics or {}).get('pair_ranking_accuracy', 0.0))
        pre_gap = float((pre_tail_stage1_probe_metrics or {}).get('same_state_score_gap', 0.0))
        pre_spread = float((pre_tail_stage1_probe_metrics or {}).get('score_spread', 0.0))
        candidate_acc = float((candidate_stage1_probe_metrics or {}).get('pair_ranking_accuracy', 0.0))
        candidate_gap = float((candidate_stage1_probe_metrics or {}).get('same_state_score_gap', 0.0))
        candidate_spread = float((candidate_stage1_probe_metrics or {}).get('score_spread', 0.0))

        min_acc_allowed = pre_acc - float(acc_tolerance)
        min_gap_allowed = max(float(min_gap_floor), pre_gap - float(gap_tolerance))
        min_spread_allowed = max(float(min_spread_floor), pre_spread - float(spread_tolerance))

        if candidate_acc < min_acc_allowed:
            return False, 'floor_stage1_acc_below_pre_tail_tolerance'
        if candidate_gap < min_gap_allowed:
            return False, 'floor_stage1_gap_below_pre_tail_tolerance'
        if candidate_spread < min_spread_allowed:
            return False, 'floor_stage1_spread_below_pre_tail_tolerance'
        return True, 'floor_pass'

    def _evaluate_stage1_tail_acceptance(
        self,
        pre_tail_stage1_probe_metrics: Dict[str, float],
        pre_tail_eval_metrics: Dict[str, float],
        candidate_stage1_probe_metrics: Dict[str, float],
        candidate_eval_metrics: Dict[str, float],
        acceptance_enabled: bool,
        acc_tolerance: float,
        spread_tolerance: float,
        gap_tolerance: float,
        min_spread_floor: float,
        min_gap_floor: float,
    ) -> Tuple[bool, str]:
        if not bool(acceptance_enabled):
            return True, 'acceptance_disabled'

        pre_stage1_unique = float((pre_tail_stage1_probe_metrics or {}).get('unique_score_count', 0.0))
        pre_eval_unique = float((pre_tail_eval_metrics or {}).get('unique_score_count', 0.0))
        candidate_stage1_unique = float((candidate_stage1_probe_metrics or {}).get('unique_score_count', 0.0))
        candidate_eval_unique = float((candidate_eval_metrics or {}).get('unique_score_count', 0.0))
        has_unique_gain = (
            candidate_stage1_unique > pre_stage1_unique + 1e-9
            or candidate_eval_unique > pre_eval_unique + 1e-9
        )
        if not has_unique_gain:
            return False, 'rejected_no_unique_gain'

        floor_passed, floor_reason = self._stage1_tail_passes_floor(
            candidate_stage1_probe_metrics=candidate_stage1_probe_metrics,
            pre_tail_stage1_probe_metrics=pre_tail_stage1_probe_metrics,
            min_spread_floor=min_spread_floor,
            min_gap_floor=min_gap_floor,
            acc_tolerance=acc_tolerance,
            spread_tolerance=spread_tolerance,
            gap_tolerance=gap_tolerance,
        )
        if not floor_passed:
            return False, f'rejected_{floor_reason}'
        return True, 'accepted_unique_gain_and_non_degradation'

    def _spread_eligible_pair_count(self, pair_samples: Sequence[RiskPairSample]) -> int:
        count = 0
        for sample in pair_samples:
            target_a = float(sample.meta.get('target_risk_a', 0.0))
            target_b = float(sample.meta.get('target_risk_b', 0.0))
            trusted = bool(sample.meta.get('trusted_for_spread', False))
            if trusted and abs(target_a - target_b) >= float(SPREAD_TARGET_DELTA):
                count += 1
        return count

    def _stage5_pair_identifier(self, sample: RiskPairSample, fallback_index: int) -> str:
        meta = dict(sample.meta or {})
        source_path = str(meta.get('source_path', ''))
        if source_path:
            return source_path
        pair_index = meta.get('pair_index', fallback_index)
        seed = meta.get('seed', 'na')
        return f"stage5_pair::{seed}::{pair_index}"

    def _sample_stage5_batch_with_cap(
        self,
        stage5_pair_samples: Sequence[RiskPairSample],
        pair_ids: Sequence[str],
        batch_size: int,
        seen_counts: Dict[str, int],
        max_seen_per_epoch: int,
    ) -> tuple[List[RiskPairSample], List[str]]:
        indexed = list(zip(pair_ids, stage5_pair_samples))
        if max_seen_per_epoch > 0:
            indexed = [(pair_id, sample) for pair_id, sample in indexed if int(seen_counts.get(pair_id, 0)) < max_seen_per_epoch]
        if not indexed:
            return [], []
        target_batch = max(1, min(int(batch_size), len(indexed)))
        if len(indexed) <= target_batch:
            selected = indexed
        else:
            selected = self.rng.sample(indexed, target_batch)
        batch_ids = [pair_id for pair_id, _ in selected]
        batch_samples = [sample for _, sample in selected]
        return batch_samples, batch_ids

    def _is_better_pair_ft_metrics(self, current_metrics: Dict[str, float], best_metrics: Dict[str, float]) -> bool:
        min_spread_floor = float(getattr(self.config, 'pair_ft_min_score_spread_floor', 0.0) or 0.0)
        min_gap_floor = float(getattr(self.config, 'pair_ft_min_same_state_gap_floor', 0.0) or 0.0)
        current_spread = float((current_metrics or {}).get('score_spread', 0.0))
        best_spread = float((best_metrics or {}).get('score_spread', 0.0))
        current_gap = float((current_metrics or {}).get('same_state_score_gap', 0.0))
        best_gap = float((best_metrics or {}).get('same_state_score_gap', 0.0))
        current_collapsed = (current_spread < min_spread_floor) or (current_gap < min_gap_floor)
        best_collapsed = (best_spread < min_spread_floor) or (best_gap < min_gap_floor)
        if current_collapsed and not best_collapsed:
            return False
        if not current_collapsed and best_collapsed:
            return True

        current_acc = float((current_metrics or {}).get('pair_ranking_accuracy', 0.0))
        best_acc = float((best_metrics or {}).get('pair_ranking_accuracy', 0.0))
        if current_acc > best_acc + 1e-9:
            return True
        if current_acc < best_acc - 1e-9:
            return False
        return current_gap > best_gap + 1e-9

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
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            for batch in loader:
                batch_a, batch_b, preferred_a, target_a, target_b, _, hard_negative, _, _, _ = self._tensorize_pair_batch(batch)
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
        finally:
            if was_training:
                self.model.train()

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

    @torch.no_grad()
    def _evaluate_pair_logit_gap_metrics(self, pair_samples: Sequence[RiskPairSample]) -> Dict[str, float]:
        if len(pair_samples) == 0:
            return {
                'pair_count': 0.0,
                'mean_abs_logit_gap': 0.0,
            }

        abs_logit_gaps: List[float] = []
        loader = DataLoader(
            RiskPairDataset(pair_samples),
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_risk_pairs,
        )
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            for batch in loader:
                batch_a, batch_b, _, _, _, _, _, _, _, _ = self._tensorize_pair_batch(batch)
                out_a = self.model(batch_a)
                out_b = self.model(batch_b)
                score_a = out_a['risk_score']
                score_b = out_b['risk_score']
                score_a_logit = out_a.get('risk_score_logit')
                score_b_logit = out_b.get('risk_score_logit')
                if score_a_logit is None:
                    score_a_logit = torch.logit(torch.clamp(score_a, min=1e-6, max=1.0 - 1e-6))
                if score_b_logit is None:
                    score_b_logit = torch.logit(torch.clamp(score_b, min=1e-6, max=1.0 - 1e-6))
                abs_logit_gap = torch.abs(score_a_logit - score_b_logit).detach().cpu().numpy()
                abs_logit_gaps.extend(float(item) for item in abs_logit_gap)
        finally:
            if was_training:
                self.model.train()

        return {
            'pair_count': float(len(pair_samples)),
            'mean_abs_logit_gap': float(np.mean(abs_logit_gaps)) if abs_logit_gaps else 0.0,
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

    def _compute_pair_losses(
        self,
        batch: Sequence[RiskPairSample],
        enable_resolution: bool = False,
        enable_stage1_resolution: bool = False,
        enable_stage1_tail_anticollapse: bool = False,
        stage1_tail_score_range_floor: float = 0.02,
        stage1_tail_score_range_quantile_low: float = 0.10,
        stage1_tail_score_range_quantile_high: float = 0.90,
    ):
        (
            batch_a,
            batch_b,
            preferred_a,
            target_a,
            target_b,
            weight,
            _,
            trusted_for_spread,
            stage4_aux_mask,
            stage1_probe_mask,
        ) = self._tensorize_pair_batch(batch)
        out_a = self.model(batch_a)
        out_b = self.model(batch_b)
        score_a = out_a['risk_score']
        score_b = out_b['risk_score']
        score_a_logit = out_a.get('risk_score_logit')
        score_b_logit = out_b.get('risk_score_logit')
        if score_a_logit is None:
            score_a_logit = torch.logit(torch.clamp(score_a, min=1e-6, max=1.0 - 1e-6))
        if score_b_logit is None:
            score_b_logit = torch.logit(torch.clamp(score_b, min=1e-6, max=1.0 - 1e-6))

        safer = torch.where(preferred_a, score_a_logit, score_b_logit)
        riskier = torch.where(preferred_a, score_b_logit, score_a_logit)
        target_gap = torch.abs(target_a - target_b)
        target_gap_meta_values: List[float] = []
        for sample in batch:
            meta = dict(sample.meta or {})
            fallback_gap = abs(float(meta.get('target_risk_a', 0.0)) - float(meta.get('target_risk_b', 0.0)))
            target_gap_meta_values.append(float(meta.get('target_gap', fallback_gap) or fallback_gap))
        target_gap_meta = torch.tensor(target_gap_meta_values, dtype=torch.float32, device=self.device)
        tie_gap_epsilon = max(0.0, float(getattr(self.config, 'pair_ft_tie_gap_epsilon', 0.0) or 0.0))
        tie_mask = (target_gap >= tie_gap_epsilon).to(torch.float32)
        gap_scale = torch.clamp((target_gap - tie_gap_epsilon) / max(1e-6, 1.0 - tie_gap_epsilon), min=0.0, max=1.0)
        ranking_weight = weight * tie_mask * (0.25 + 0.75 * gap_scale)
        ranking_loss = torch.relu(PAIR_RANK_MARGIN - (riskier - safer))
        ranking_denom = torch.clamp(torch.sum(ranking_weight), min=1.0)
        ranking_loss = torch.sum(ranking_loss * ranking_weight) / ranking_denom

        spread_mask = ((target_gap > SPREAD_TARGET_DELTA) & trusted_for_spread).to(torch.float32)
        spread_gap = torch.abs(score_a_logit - score_b_logit)
        spread_loss = torch.relu(SPREAD_MIN_GAP - spread_gap) * spread_mask
        spread_denom = torch.clamp(torch.sum(spread_mask * weight), min=1.0)
        spread_loss = torch.sum(spread_loss * weight) / spread_denom
        stage4_resolution_mask = stage4_aux_mask.to(torch.float32) if enable_resolution else torch.zeros_like(weight)
        stage1_resolution_apply_trusted_only = bool(
            getattr(self.config, 'pair_ft_stage1_resolution_apply_trusted_only', True)
        )
        stage1_probe_active_mask = stage1_probe_mask
        if stage1_resolution_apply_trusted_only:
            stage1_probe_active_mask = stage1_probe_active_mask & trusted_for_spread
        stage1_resolution_mask = (
            stage1_probe_active_mask.to(torch.float32)
            if enable_stage1_resolution
            else torch.zeros_like(weight)
        )
        resolution_score_gap = torch.abs(score_a - score_b)
        resolution_logit_gap = torch.abs(score_a_logit - score_b_logit)
        min_score_gap = float(getattr(self.config, 'pair_ft_resolution_min_score_gap', 0.03) or 0.03)
        stage1_min_score_gap = float(getattr(self.config, 'pair_ft_stage1_resolution_min_score_gap', 0.015) or 0.015)
        stage1_resolution_mode = str(getattr(self.config, 'pair_ft_stage1_resolution_mode', 'fixed') or 'fixed').strip().lower()
        if stage1_resolution_mode not in {'fixed', 'adaptive'}:
            stage1_resolution_mode = 'fixed'
        stage1_resolution_alpha = float(getattr(self.config, 'pair_ft_stage1_resolution_alpha', 0.2) or 0.2)
        stage1_resolution_max_score_gap = float(
            getattr(self.config, 'pair_ft_stage1_resolution_max_score_gap', 0.05) or 0.05
        )
        min_logit_gap = float(getattr(self.config, 'pair_ft_resolution_min_logit_gap', 0.14) or 0.14)
        stage4_resolution_terms = torch.relu(
            torch.tensor(min_score_gap, dtype=resolution_score_gap.dtype, device=resolution_score_gap.device)
            - resolution_score_gap
        )
        if stage1_resolution_mode == 'adaptive':
            stage1_margins = torch.clamp(
                stage1_resolution_alpha * target_gap_meta,
                min=stage1_min_score_gap,
                max=stage1_resolution_max_score_gap,
            )
        else:
            stage1_margins = torch.full_like(resolution_score_gap, fill_value=stage1_min_score_gap)
        stage1_resolution_terms = torch.relu(stage1_margins - resolution_score_gap)
        stage4_active_resolution_mask = stage4_resolution_mask > 0.0
        stage4_active_resolution_count = int(torch.sum(stage4_resolution_mask).item())
        stage1_active_resolution_mask = stage1_resolution_mask > 0.0
        stage1_active_resolution_count = int(torch.sum(stage1_resolution_mask).item())
        if enable_resolution and stage4_active_resolution_count > 0:
            stage4_resolution_loss = torch.sum(stage4_resolution_terms * stage4_resolution_mask) / torch.clamp(
                torch.sum(stage4_resolution_mask), min=1.0
            )
            active_resolution_logit_gap = resolution_logit_gap[stage4_active_resolution_mask]
            active_resolution_score_gap = resolution_score_gap[stage4_active_resolution_mask]
            stage4_aux_logit_gap_mean = (
                float(torch.mean(active_resolution_logit_gap).item())
                if active_resolution_logit_gap.numel() > 0
                else 0.0
            )
            stage4_aux_score_gap_values = (
                active_resolution_score_gap.detach().cpu().numpy().astype(float).tolist()
                if active_resolution_score_gap.numel() > 0
                else []
            )
            stage4_aux_below_margin_count = int(torch.sum((active_resolution_logit_gap < min_logit_gap).to(torch.int32)).item())
            stage4_aux_below_margin_fraction = (
                float(stage4_aux_below_margin_count) / float(stage4_active_resolution_count)
                if stage4_active_resolution_count > 0
                else 0.0
            )
            stage4_aux_below_score_margin_count = int(torch.sum((active_resolution_score_gap < min_score_gap).to(torch.int32)).item())
            stage4_aux_below_score_margin_fraction = (
                float(stage4_aux_below_score_margin_count) / float(stage4_active_resolution_count)
                if stage4_active_resolution_count > 0
                else 0.0
            )
        else:
            stage4_resolution_loss = self._zero_loss()
            stage4_aux_logit_gap_mean = 0.0
            stage4_aux_score_gap_values = []
            stage4_aux_below_margin_count = 0
            stage4_aux_below_margin_fraction = 0.0
            stage4_aux_below_score_margin_count = 0
            stage4_aux_below_score_margin_fraction = 0.0
        if enable_stage1_resolution and stage1_active_resolution_count > 0:
            stage1_resolution_loss = torch.sum(stage1_resolution_terms * stage1_resolution_mask) / torch.clamp(
                torch.sum(stage1_resolution_mask), min=1.0
            )
            stage1_active_score_gap = resolution_score_gap[stage1_active_resolution_mask]
            stage1_active_margins = stage1_margins[stage1_active_resolution_mask]
            stage1_probe_score_gap_values = (
                stage1_active_score_gap.detach().cpu().numpy().astype(float).tolist()
                if stage1_active_score_gap.numel() > 0
                else []
            )
            stage1_probe_margin_values = (
                stage1_active_margins.detach().cpu().numpy().astype(float).tolist()
                if stage1_active_margins.numel() > 0
                else []
            )
            stage1_probe_pred_gap_to_margin_ratios = (
                (stage1_active_score_gap / torch.clamp(stage1_active_margins, min=1e-6))
                .detach()
                .cpu()
                .numpy()
                .astype(float)
                .tolist()
                if stage1_active_score_gap.numel() > 0
                else []
            )
            stage1_probe_below_score_margin_count = int(
                torch.sum((stage1_active_score_gap < stage1_min_score_gap).to(torch.int32)).item()
            )
            stage1_probe_below_score_margin_fraction = (
                float(stage1_probe_below_score_margin_count) / float(stage1_active_resolution_count)
                if stage1_active_resolution_count > 0
                else 0.0
            )
            stage1_probe_below_adaptive_margin_count = int(
                torch.sum((stage1_active_score_gap < stage1_active_margins).to(torch.int32)).item()
            )
            stage1_probe_below_adaptive_margin_fraction = (
                float(stage1_probe_below_adaptive_margin_count) / float(stage1_active_resolution_count)
                if stage1_active_resolution_count > 0
                else 0.0
            )
        else:
            stage1_resolution_loss = self._zero_loss()
            stage1_probe_score_gap_values = []
            stage1_probe_margin_values = []
            stage1_probe_pred_gap_to_margin_ratios = []
            stage1_probe_below_score_margin_count = 0
            stage1_probe_below_score_margin_fraction = 0.0
            stage1_probe_below_adaptive_margin_count = 0
            stage1_probe_below_adaptive_margin_fraction = 0.0
        stage1_tail_q10 = 0.0
        stage1_tail_q90 = 0.0
        stage1_tail_score_range = 0.0
        if enable_stage1_tail_anticollapse and stage1_active_resolution_count > 0:
            active_score_a = score_a[stage1_active_resolution_mask]
            active_score_b = score_b[stage1_active_resolution_mask]
            active_scores = torch.cat([active_score_a, active_score_b], dim=0)
            if active_scores.numel() > 0:
                q_low = torch.quantile(active_scores, stage1_tail_score_range_quantile_low)
                q_high = torch.quantile(active_scores, stage1_tail_score_range_quantile_high)
                score_range = q_high - q_low
                range_floor_tensor = torch.tensor(
                    stage1_tail_score_range_floor,
                    dtype=score_range.dtype,
                    device=score_range.device,
                )
                stage1_tail_anticollapse_loss = torch.relu(range_floor_tensor - score_range)
                stage1_tail_q10 = float(q_low.detach().item())
                stage1_tail_q90 = float(q_high.detach().item())
                stage1_tail_score_range = float(score_range.detach().item())
            else:
                stage1_tail_anticollapse_loss = self._zero_loss()
        else:
            stage1_tail_anticollapse_loss = self._zero_loss()
        resolution_loss = stage4_resolution_loss + stage1_resolution_loss
        diagnostics = {
            'stage4_aux_active_pair_count': int(stage4_active_resolution_count),
            'stage4_aux_batch_size': int(weight.shape[0]),
            'stage4_aux_logit_gap_mean': float(stage4_aux_logit_gap_mean),
            'stage4_aux_score_gaps': stage4_aux_score_gap_values,
            'stage4_aux_score_gap_mean': float(np.mean(stage4_aux_score_gap_values)) if stage4_aux_score_gap_values else 0.0,
            'stage4_aux_score_gap_p50': float(np.quantile(stage4_aux_score_gap_values, 0.50)) if stage4_aux_score_gap_values else 0.0,
            'stage4_aux_score_gap_p90': float(np.quantile(stage4_aux_score_gap_values, 0.90)) if stage4_aux_score_gap_values else 0.0,
            'stage4_aux_below_score_margin_count': int(stage4_aux_below_score_margin_count),
            'stage4_aux_below_score_margin_fraction': float(stage4_aux_below_score_margin_fraction),
            'stage4_aux_below_margin_count': int(stage4_aux_below_margin_count),
            'stage4_aux_below_margin_fraction': float(stage4_aux_below_margin_fraction),
            'stage4_aux_resolution_loss': float(stage4_resolution_loss.detach().item()),
            'resolution_space': 'score',
            'pair_ft_resolution_min_score_gap': float(min_score_gap),
            'ignored_legacy_logit_margin': float(min_logit_gap),
            'pair_ft_resolution_min_logit_gap_compat': float(min_logit_gap),
            'stage1_probe_active_pair_count': int(stage1_active_resolution_count),
            'stage1_probe_score_gaps': stage1_probe_score_gap_values,
            'stage1_probe_score_gap_mean': float(np.mean(stage1_probe_score_gap_values)) if stage1_probe_score_gap_values else 0.0,
            'stage1_probe_score_gap_p50': float(np.quantile(stage1_probe_score_gap_values, 0.50)) if stage1_probe_score_gap_values else 0.0,
            'stage1_probe_score_gap_p90': float(np.quantile(stage1_probe_score_gap_values, 0.90)) if stage1_probe_score_gap_values else 0.0,
            'stage1_probe_margins': stage1_probe_margin_values,
            'stage1_probe_margin_mean': float(np.mean(stage1_probe_margin_values)) if stage1_probe_margin_values else 0.0,
            'stage1_probe_margin_p50': float(np.quantile(stage1_probe_margin_values, 0.50)) if stage1_probe_margin_values else 0.0,
            'stage1_probe_margin_p90': float(np.quantile(stage1_probe_margin_values, 0.90)) if stage1_probe_margin_values else 0.0,
            'stage1_probe_pred_gap_to_margin_ratios': stage1_probe_pred_gap_to_margin_ratios,
            'stage1_probe_pred_gap_to_margin_ratio_mean': (
                float(np.mean(stage1_probe_pred_gap_to_margin_ratios))
                if stage1_probe_pred_gap_to_margin_ratios
                else 0.0
            ),
            'stage1_probe_below_score_margin_count': int(stage1_probe_below_score_margin_count),
            'stage1_probe_below_score_margin_fraction': float(stage1_probe_below_score_margin_fraction),
            'stage1_probe_below_adaptive_margin_count': int(stage1_probe_below_adaptive_margin_count),
            'stage1_probe_below_adaptive_margin_fraction': float(stage1_probe_below_adaptive_margin_fraction),
            'stage1_probe_resolution_loss': float(stage1_resolution_loss.detach().item()),
            'stage1_resolution_space': 'score',
            'stage1_resolution_mode': str(stage1_resolution_mode),
            'pair_ft_stage1_resolution_alpha': float(stage1_resolution_alpha),
            'pair_ft_stage1_resolution_max_score_gap': float(stage1_resolution_max_score_gap),
            'pair_ft_stage1_resolution_apply_trusted_only': bool(stage1_resolution_apply_trusted_only),
            'pair_ft_stage1_resolution_min_score_gap': float(stage1_min_score_gap),
            'stage1_tail_anticollapse_loss': float(stage1_tail_anticollapse_loss.detach().item()),
            'stage1_tail_score_range_q10': float(stage1_tail_q10),
            'stage1_tail_score_range_q90': float(stage1_tail_q90),
            'stage1_tail_score_range': float(stage1_tail_score_range),
            'stage1_tail_score_range_floor': float(stage1_tail_score_range_floor),
            'stage1_tail_score_range_quantile_low': float(stage1_tail_score_range_quantile_low),
            'stage1_tail_score_range_quantile_high': float(stage1_tail_score_range_quantile_high),
            'stage1_tail_anticollapse_loss_tensor': stage1_tail_anticollapse_loss,
        }
        return ranking_loss, spread_loss, resolution_loss, diagnostics

    def _tensorize_pair_batch(self, batch: Sequence[RiskPairSample]):
        histories_a = []
        histories_b = []
        preferred_a = []
        target_a = []
        target_b = []
        weight = []
        hard_negative = []
        trusted_for_spread = []
        stage4_aux_mask = []
        stage1_probe_mask = []

        for sample in batch:
            histories_a.append((sample.history_scene, int(sample.action_a)))
            histories_b.append((sample.history_scene, int(sample.action_b)))
            preferred_a.append(int(sample.preferred_action) == int(sample.action_a))
            target_a.append(float(sample.meta.get('target_risk_a', 0.0)))
            target_b.append(float(sample.meta.get('target_risk_b', 0.0)))
            weight.append(float(sample.weight))
            hard_negative.append(bool(sample.meta.get('hard_negative', False)))
            trusted_for_spread.append(bool(sample.meta.get('trusted_for_spread', False)))
            stage4_aux_mask.append(
                bool(str(sample.source) == 'stage4_candidate_rank')
                and bool(sample.meta.get('stage4_aux_candidate', False))
            )
            stage1_probe_mask.append(bool(str(sample.source) == 'stage1_probe_same_state'))

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
            torch.tensor(stage4_aux_mask, dtype=torch.bool, device=self.device),
            torch.tensor(stage1_probe_mask, dtype=torch.bool, device=self.device),
        )

    def _select_pair_ft_eval_samples(self, replay_samples: Sequence[ActionConditionedSample]) -> List[ActionConditionedSample]:
        replay_samples = list(replay_samples)
        max_samples = int(getattr(self.config, "pair_ft_eval_max_samples", 0) or 0)
        if max_samples <= 0 or len(replay_samples) <= max_samples:
            return replay_samples
        if max_samples == 1:
            return [replay_samples[0]]
        step = (len(replay_samples) - 1) / float(max_samples - 1)
        return [replay_samples[int(round(index * step))] for index in range(max_samples)]

    def _build_pair_loader(self, pair_samples: Sequence[RiskPairSample]):
        samples = list(pair_samples or [])
        if not samples:
            return None
        return DataLoader(
            RiskPairDataset(samples),
            batch_size=max(1, min(int(self.config.batch_size), len(samples))),
            shuffle=True,
            drop_last=False,
            collate_fn=collate_risk_pairs,
        )

    def _next_pair_batch(self, pair_iter, pair_loader):
        if pair_loader is None:
            return None, pair_iter
        try:
            batch = next(pair_iter)
        except StopIteration:
            pair_iter = iter(pair_loader)
            batch = next(pair_iter)
        return list(batch), pair_iter

    def _sample_pair_batch_with_replacement(self, pair_samples: Sequence[RiskPairSample], batch_size: int) -> List[RiskPairSample]:
        samples = list(pair_samples or [])
        if not samples:
            return []
        return [self.rng.choice(samples) for _ in range(max(1, batch_size))]

    def _build_pair_batches_without_replacement(
        self,
        pair_samples: Sequence[RiskPairSample],
        batch_size: int,
    ) -> List[List[RiskPairSample]]:
        samples = list(pair_samples or [])
        if not samples:
            return []
        self.rng.shuffle(samples)
        effective_batch = max(1, int(batch_size))
        return [
            samples[start:start + effective_batch]
            for start in range(0, len(samples), effective_batch)
        ]

    def _pair_ft_epoch_steps(self, stage5_pair_samples: Sequence[RiskPairSample], stage4_loader, stage5_batch_size: int) -> int:
        if stage4_loader is not None:
            return max(1, len(stage4_loader))
        if stage5_pair_samples:
            return max(1, int(np.ceil(len(stage5_pair_samples) / float(max(1, stage5_batch_size)))))
        return 1

    def _zero_loss(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

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
