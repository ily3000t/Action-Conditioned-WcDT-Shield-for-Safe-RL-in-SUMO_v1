# WcDT SAFE_RL

## 主入口（一句话）
普通用户只看并只用 `safe_rl/config/default_safe_rl.yaml`。

## 基础运行

```bash
# 全链路
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml

# 分阶段复用同一 run
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id <run_id>
```

## 配置分层（已收敛）

- 主入口层（唯一正式入口）
  - `safe_rl/config/default_safe_rl.yaml`

- 进阶流程层
  - `safe_rl/config/advanced/stage2_v2_world_pair_focus.yaml`
  - `safe_rl/config/advanced/stage5_pair_bootstrap.yaml`
  - `safe_rl/config/advanced/stage45_recovery.yaml`

- 可视化与评估层
  - `safe_rl/config/visualization/stage5_eval_hardening.yaml`
  - `safe_rl/config/visualization/stage45_cost_desensitize.yaml`
  - `safe_rl/config/visualization/stage5_trace_capture_default.yaml`
  - `safe_rl/config/visualization/stage5_trace_capture_cost.yaml`

- 实验层
  - `safe_rl/config/experiments/shield_sweep.yaml`
  - `safe_rl/config/experiments/shield_trace_holdout_c1.yaml`

- 调试层
  - `safe_rl/config/debug/quick_check.yaml`
  - `safe_rl/config/debug/smoke_test.yaml`
  - `safe_rl/config/debug/debug_real_sumo.yaml`
  - `safe_rl/config/debug/debug_stage3_sumo.yaml`
  - `safe_rl/config/debug/shield_sanity.yaml`

- 兼容/待删除层（不推荐新实验继续使用）
  - `safe_rl/config/deprecated/*`
  - 例如：`shield_trace_c_strong.yaml`、`stage45_balance_candidate.yaml`、`safe_rl_balanced_profile.yaml`

> 说明：配置加载器已支持“按文件名回溯查找”，旧路径命令在大多数情况下仍可兼容，但请逐步迁移到新目录。

## 常用辅助命令

```bash
# 闭环复验（stage4 -> stage2 -> stage5）
python run_safe_rl_v2_pipeline.py --run-id <run_id>

# v2 批处理（stage1 -> stage5 bootstrap -> stage2 focus -> stage5 holdout）
run_safe_rl_v2_pipeline.cmd <run_id>
```

## 第二阶段修改方案（Resolution Loss，仅 Stage4-aux 子集）
当前状态是：aux 窗口已打通（pair 数足够），但 `stage4_aux_unique_after` 仍低于 12。

建议按下面顺序做第二阶段：

1. 仅在 world pair-ft 中增加“Stage4-aux 子集轻量分辨率约束”
- 只对 `source=stage4_candidate_rank` 且 `stage4_aux_candidate=true` 的样本生效。
- 不改主 pointwise/ranking/spread 的权重结构。
- 新增弱权重 `resolution_loss_weight`（建议 0.02 起步）。

2. 分辨率损失设计（轻量、可控）
- 目标：拉开同状态下的风险分数档位，避免 unique 塌缩。
- 推荐形式：对 batch 内风险分数做最小间隔/方差增强约束（只在 Stage4-aux 子集）。

3. 评估与门禁不变
- Stage5 hard-gate 语义不放宽。
- 继续以 `model_quality_aux_stage4` 的 reason code 判读是否脱离 `critical`。

4. 验收指标
- `stage4_aux_pair_count >= 128`（应保持）
- `stage4_aux_unique_after >= 12`
- 若满足上述条件，再看 `critical -> degraded` 是否触发。

## 可视化链路（V1）

```bash
# trace capture
python safe_rl_main.py --config safe_rl/config/visualization/stage5_trace_capture_default.yaml --stage stage5 --run-id <run_id>

# anomaly 选择
python -m safe_rl.visualization.select_anomaly_cases --run-id <run_id> --trace-dir stage5_trace_capture_default --top-k 20

# GIF 导出
python -m safe_rl.visualization.export_paired_gif --run-id <run_id> --trace-dir stage5_trace_capture_default --top-k 10

# GUI 深挖
python -m safe_rl.visualization.replay_in_sumo_gui --run-id <run_id> --seed <seed> --mode shielded --trace-dir stage5_trace_capture_default --base-config safe_rl/config/visualization/stage5_trace_capture_default.yaml
```
