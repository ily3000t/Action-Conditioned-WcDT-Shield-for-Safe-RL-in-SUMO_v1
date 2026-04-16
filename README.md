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
