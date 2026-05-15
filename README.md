# WcDT SAFE_RL

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

## 配置分层

- 主入口层
  - `safe_rl/config/default_safe_rl.yaml`
- 进阶流程层
  - `safe_rl/config/advanced/`
- 可视化与评估层
  - `safe_rl/config/visualization/`
- 实验层
  - `safe_rl/config/experiments/`
- 调试层
  - `safe_rl/config/debug/`

## 兼容性说明

- 已移除 `safe_rl/config/deprecated/`。
- 配置加载器改为显式路径，不再按文件名递归兜底。
- 旧路径命令会直接报错，请迁移到上述分层目录。

## 常用辅助命令

```bash
# 闭环复验（默认 stage4 -> stage2，Stage5 按 Stage2 质量状态条件触发）
python run_safe_rl_v2_pipeline.py --run-id <run_id>

# v2 批处理
run_safe_rl_v2_pipeline.cmd <run_id>
```

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

## Stage1 Data Audit / Stage1 SUMO Replay

```bash
# 1) Stage1 数据分布审计（pair_all / pair_trusted / candidate）
python -m safe_rl.visualization.stage1_data_audit --run-id <run_id>

# 2) Stage1 样本选择（用于人工 case review）
python -m safe_rl.visualization.select_stage1_probe_samples --run-id <run_id> --top-k 40 --device cpu

# 3) Stage1 原轨迹回放（raw_action_prefix + risk_event_schedule）
python -m safe_rl.visualization.stage1_sumo_replay --run-id 202605_stage1r_compact_v3 --mode raw_replay --episode-id ep_00037 --until-step 120 --config safe_rl/config/default_safe_rl.yaml

# 4) Stage1 A/B 分支对比回放（同 seed + 同 prefix 双重重放）
python -m safe_rl.visualization.stage1_sumo_replay --run-id <run_id> --mode compare_ab --episode-id ep_00037 --step-index 52 --action-a 0 --action-b 8 --horizon 20 --config safe_rl/config/default_safe_rl.yaml
```

说明：
- Stage5 可视化链路（trace/anomaly/gif/gui）用于 Stage5 pair 检查；
- Stage1 Data Audit / Stage1 SUMO Replay 用于定位 Stage1 监督分布与 pair 结构问题。

## Stage1-R Phased Runbook

```bash
# R0: audit only (no data semantic change)
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r0_audit.yaml --stage stage1 --run-id 20260414_200057

# R1: calibrated risk mapping only (keep sampling unchanged)
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r1_calibrated_risk.yaml --stage stage1 --run-id 20260414_200057

# R2: stratified sampling only + Stage2 distribution gate (critical-only block)
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r2_stratified_sampling.yaml --stage stage1 --run-id 20260414_200057
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r2_stratified_sampling.yaml --stage stage2 --run-id 20260414_200057
```

R3 compact merge rule:
- Trigger only when `stage1_scene_sanity_report.json` is critical by trigger thresholds.
- Must use a new run id (do not mix with `20260414_200057`).

```bash
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r3_compact_merge.yaml --stage stage1 --run-id 202605_stage1r_compact_v1
python safe_rl_main.py --config safe_rl/config/advanced/stage1_r3_compact_merge.yaml --stage stage2 --run-id 202605_stage1r_compact_v1
```

Compare-AB output fields include:
- `both_saturated`
- `saturation_reason_a`
- `saturation_reason_b`
- `action_sensitive`
- `diagnosis`
