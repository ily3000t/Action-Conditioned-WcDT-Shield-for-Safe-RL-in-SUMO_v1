# SAFE_RL 模块速查说明

这份文档是 `safe_rl/` 目录内部的速查版说明。

如果你想看完整运行手册、配置模式、分阶段命令、TensorBoard、输出工件和排障建议，请优先查看：

- [README.md](/E:/wcdt_all/WcDT/README.md)

## 1. 这个目录在整个项目里的作用

`safe_rl/` 实现的是当前基于 SUMO 的安全强化学习一期闭环：

- Stage 1：SUMO 数据采集与数据集构建
- Stage 2：轻量风险模型与动作条件化世界模型训练
- Stage 3：接入 Safety Shield 的 PPO 在线训练
- Stage 4：采集 intervention buffer
- Stage 5：蒸馏与 same-policy shield off/on 配对评估

主入口：

- [safe_rl_main.py](/E:/wcdt_all/WcDT/safe_rl_main.py)

## 2. 目录结构

- [config](/E:/wcdt_all/WcDT/safe_rl/config)：配置 dataclass 和所有 YAML 模式
- [sim](/E:/wcdt_all/WcDT/safe_rl/sim)：SUMO 后端、动作落地、风险注入、mock/real 控制
- [data](/E:/wcdt_all/WcDT/safe_rl/data)：采集器、样本构建、风险标签、warning 统计
- [models](/E:/wcdt_all/WcDT/safe_rl/models)：轻量风险模型、世界模型、特征编码
- [shield](/E:/wcdt_all/WcDT/safe_rl/shield)：候选动作生成、风险聚合、安全替换
- [rl](/E:/wcdt_all/WcDT/safe_rl/rl)：环境、PPO、distill
- [buffer](/E:/wcdt_all/WcDT/safe_rl/buffer)：intervention buffer
- [eval](/E:/wcdt_all/WcDT/safe_rl/eval)：评估与 paired episode 聚合
- [pipeline](/E:/wcdt_all/WcDT/safe_rl/pipeline)：5 阶段调度、工件管理、报告、TensorBoard

## 3. 最常用配置模式

- [quick_check.yaml](/E:/wcdt_all/WcDT/safe_rl/config/quick_check.yaml)：最快确认链路通不通
- [smoke_test.yaml](/E:/wcdt_all/WcDT/safe_rl/config/smoke_test.yaml)：更完整但仍偏调试
- [default_safe_rl.yaml](/E:/wcdt_all/WcDT/safe_rl/config/default_safe_rl.yaml)：默认正式训练配置
- [debug_real_sumo.yaml](/E:/wcdt_all/WcDT/safe_rl/config/debug_real_sumo.yaml)：真实 SUMO Stage 1 排障
- [debug_stage3_sumo.yaml](/E:/wcdt_all/WcDT/safe_rl/config/debug_stage3_sumo.yaml)：真实 SUMO Stage 3 会话排障
- [shield_sanity.yaml](/E:/wcdt_all/WcDT/safe_rl/config/shield_sanity.yaml)：激进阈值验证 shield 决策链

## 4. 最常用运行命令

### 4.1 一键跑完整链路

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 4.2 分阶段重跑

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

### 4.3 真实 SUMO 排障

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id exp_001
```

### 4.4 Shield sanity check

```bash
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id exp_001
```

### 4.5 Shield sweep

```bash
python safe_rl_main.py --config safe_rl/config/shield_sweep.yaml --stage stage4 --run-id 20260320_210439
python safe_rl_main.py --config safe_rl/config/shield_sweep.yaml --stage stage5 --run-id 20260320_210439
```

## 5. 当前评估口径

Stage 5 当前正式口径已经固定为：

- `system_baseline = 同一 PPO policy + shield 关闭`
- `system_shielded = 同一 PPO policy + shield 开启`
- baseline 和 shielded 使用相同 `seed`
- baseline 和 shielded 使用相同 `risky_mode=True`
- baseline 和 shielded 使用相同 `scenario_source = sim.sumo_cfg`

现在报告里要区分两类结论：

- `performance_passed`：系统整体性能是否达标
- `attribution_passed`：shield 是否已经被独立验证

## 6. 当前重要报告文件

所有运行产物都在：

```text
safe_rl_output/runs/<run_id>/
```

重点关注：

- `reports/pipeline_report.json`
- `reports/stage4_buffer_report.json`
- `reports/stage5_paired_episode_results.json`
- `reports/stage3_runtime_config.json`
- `reports/stage3_session_events.json`
- `reports/warning_summary.json`

## 7. TensorBoard 当前重点

当前最值得看的不是静态配置，而是这几类曲线：

- `ppo/stability/*`
- `ppo/risk/*`
- `buffer/*`
- `eval/*risk*`
- `eval/summary/shield_contribution_validated`

启动方式：

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

## 8. 一句话建议

如果你后面只是想知道“先看哪个文件”：

- 训练/评估总体结论：看 `pipeline_report.json`
- shield 是否真的替换动作：看 `stage5_paired_episode_results.json`
- Stage 3 为什么崩：看 `stage3_session_events.json`
- Stage 1 为什么 warning 多：看 `warning_summary.json`
- buffer 为什么是 0：看 `stage4_buffer_report.json`

## Shield Trace C1/C2/C_strong

这是只重跑 `stage5` 的轻量调参入口，用来比较 C 基线、C1、C2 三组 trace 结果。

运行命令：

```bash
python safe_rl_main.py --config safe_rl/config/shield_trace_c1.yaml --stage stage5 --run-id 20260320_210439
python safe_rl_main.py --config safe_rl/config/shield_trace_c2.yaml --stage stage5 --run-id 20260320_210439
python safe_rl_main.py --config safe_rl/config/shield_trace_c_strong.yaml --stage stage5 --run-id 20260320_210439
```

结果总表：

- `safe_rl_output/runs/20260320_210439/reports/shield_trace_tuning_summary.json`

总表重点字段：

- `effective_shield_config`
- `blocked_by_margin_count`
- `raw_passthrough_count`
- `merge_lateral_guard_block_count`
- `candidate_selected_count`
- `mean_intervention_count`
- `mean_risk_reduction`
- `mean_reward_gap_to_baseline_policy`

## ????

???????????????? 4 ???????

```bash
python run_safe_rl_v2_pipeline.py
```

?????

```bash
python run_safe_rl_v2_pipeline.py --run-id 20260320_210439
python run_safe_rl_v2_pipeline.py --python "E:\\Programs\\EnvAnaconda3\\envs\\pytorch\\python.exe"
python run_safe_rl_v2_pipeline.py --dry-run
```
