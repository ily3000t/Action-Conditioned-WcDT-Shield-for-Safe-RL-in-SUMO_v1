# WcDT SAFE_RL 运行手册

本文件只包含两部分：
- 运行方式
- 配置文件用途说明

本仓库当前包含两条主线：
- 原始 `WcDT / Waymo` 训练与结果展示代码
- 新增的 `safe_rl` 一期闭环：`SUMO 数据采集 -> 风险建模 -> Safety Shield -> PPO -> 干预样本池 -> 蒸馏 -> 配对评估`

## 1. 运行入口
### 主入口：`safe_rl_main.py`
```bash
python safe_rl_main.py --config <配置文件>
python safe_rl_main.py --config <配置文件> --stage <阶段名> --run-id <实验ID>
```
支持的阶段名：

- `all`
- `stage1`
- `stage2`
- `stage3`
- `stage4`
- `stage5`

规则：

- `--stage all` 时，如果不传 `--run-id`，系统会自动生成一个新的 `run_id`
- `--stage != all` 时，必须显式传入 `--run-id`
- 分阶段运行时，后续阶段必须复用同一个 `run_id`


###  批处理入口：`run_safe_rl_v2_pipeline.py`
###  兼容入口：`main.py`（通过任务系统间接调用 SAFE_RL）

## 2. 运行前准备

1. 激活 Python 环境（示例）

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__)"
```

2. 首次运行前构建 `highway_merge` 场景

```bash
python scenarios/highway_merge/build_network.py
```

3. 如果 SUMO 安装路径与默认不同，请在配置中修改：
- `sim.sumo_home`
- `sim.sumo_bin`
- `sim.sumo_gui_bin`
- `sim.netconvert_bin`

## 3. 通用命令模板

全链路运行（自动创建新 `run_id`）：

```bash
python safe_rl_main.py --config <config_path>
```

单阶段运行（必须指定已有 `run_id`）：

```bash
python safe_rl_main.py --config <config_path> --stage <stage1|stage2|stage3|stage4|stage5> --run-id <run_id>
```

支持阶段名：
- `all`
- `stage1`
- `stage2`
- `stage3`
- `stage4`
- `stage5`

## 4. 配置文件清单

### 4.1 基础与调试配置

| 配置文件 | 用途 | 典型命令 |
|---|---|---|
| `safe_rl/config/default_safe_rl.yaml` | 默认正式配置（全链路基线） | `python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml` |
| `safe_rl/config/quick_check.yaml` | 最小规模快速连通检查 | `python safe_rl_main.py --config safe_rl/config/quick_check.yaml` |
| `safe_rl/config/smoke_test.yaml` | 小规模功能烟测 | `python safe_rl_main.py --config safe_rl/config/smoke_test.yaml` |
| `safe_rl/config/debug_real_sumo.yaml` | 真实 TraCI 下 Stage1 采集排障 | `python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001` |
| `safe_rl/config/debug_stage3_sumo.yaml` | 真实 TraCI 下 Stage3 会话排障 | `python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id <run_id>` |

### 4.2 Stage2 训练专项配置

| 配置文件 | 用途 | 典型命令 |
|---|---|---|
| `safe_rl/config/stage2_world_base_only.yaml` | 仅做 world base 训练（不做 pair finetune） | `python safe_rl_main.py --config safe_rl/config/stage2_world_base_only.yaml --stage stage2 --run-id <run_id>` |
| `safe_rl/config/stage2_v2_world_pair_focus.yaml` | world pair-finetune 重点版本（Stage2 v2） | `python safe_rl_main.py --config safe_rl/config/stage2_v2_world_pair_focus.yaml --stage stage2 --run-id <run_id>` |

### 4.3 Stage4/Stage5 Shield 调参与平衡配置

| 配置文件 | 用途 | 典型命令 |
|---|---|---|
| `safe_rl/config/shield_sanity.yaml` | 验证 shield 是否触发与替换 | `python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id <run_id>`<br>`python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/stage45_recovery.yaml` | Stage4/5 应急回退参数（仅故障恢复时使用） | `python safe_rl_main.py --config safe_rl/config/stage45_recovery.yaml --stage stage4 --run-id <run_id>`<br>`python safe_rl_main.py --config safe_rl/config/stage45_recovery.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/safe_rl_balanced_profile.yaml` | Stage4/5 平衡正式 profile（推荐） | `python safe_rl_main.py --config safe_rl/config/safe_rl_balanced_profile.yaml --stage stage4 --run-id <run_id>`<br>`python safe_rl_main.py --config safe_rl/config/safe_rl_balanced_profile.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/stage45_balance_candidate.yaml` | 已降级为兼容别名（等价于 balanced profile，不建议新实验使用） | `python safe_rl_main.py --config safe_rl/config/stage45_balance_candidate.yaml --stage stage4 --run-id <run_id>`<br>`python safe_rl_main.py --config safe_rl/config/stage45_balance_candidate.yaml --stage stage5 --run-id <run_id>` |

### 4.4 Shield Sweep / Trace / Bootstrap 配置

| 配置文件 | 用途 | 典型命令 |
|---|---|---|
| `safe_rl/config/shield_sweep.yaml` | 多组阈值 sweep（Stage4+5） | `python safe_rl_main.py --config safe_rl/config/shield_sweep.yaml --stage stage4 --run-id <run_id>`<br>`python safe_rl_main.py --config safe_rl/config/shield_sweep.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/stage5_pair_bootstrap.yaml` | Stage5 生成 bootstrap pair 数据 | `python safe_rl_main.py --config safe_rl/config/stage5_pair_bootstrap.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_c.yaml` | trace 基准 C | `python safe_rl_main.py --config safe_rl/config/shield_trace_c.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_c1.yaml` | trace C1 | `python safe_rl_main.py --config safe_rl/config/shield_trace_c1.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_c2.yaml` | trace C2 | `python safe_rl_main.py --config safe_rl/config/shield_trace_c2.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_c_strong.yaml` | trace C_strong | `python safe_rl_main.py --config safe_rl/config/shield_trace_c_strong.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_d1.yaml` | trace D1 | `python safe_rl_main.py --config safe_rl/config/shield_trace_d1.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_d2.yaml` | trace D2 | `python safe_rl_main.py --config safe_rl/config/shield_trace_d2.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_d3.yaml` | trace D3 | `python safe_rl_main.py --config safe_rl/config/shield_trace_d3.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_e1.yaml` | trace E1 | `python safe_rl_main.py --config safe_rl/config/shield_trace_e1.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_e2.yaml` | trace E2 | `python safe_rl_main.py --config safe_rl/config/shield_trace_e2.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_e3.yaml` | trace E3 | `python safe_rl_main.py --config safe_rl/config/shield_trace_e3.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_f1.yaml` | trace F1 | `python safe_rl_main.py --config safe_rl/config/shield_trace_f1.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_f2.yaml` | trace F2 | `python safe_rl_main.py --config safe_rl/config/shield_trace_f2.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_f3.yaml` | trace F3 | `python safe_rl_main.py --config safe_rl/config/shield_trace_f3.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_g1.yaml` | trace G1 | `python safe_rl_main.py --config safe_rl/config/shield_trace_g1.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_g2.yaml` | trace G2 | `python safe_rl_main.py --config safe_rl/config/shield_trace_g2.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_g3.yaml` | trace G3 | `python safe_rl_main.py --config safe_rl/config/shield_trace_g3.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_g4.yaml` | trace G4 | `python safe_rl_main.py --config safe_rl/config/shield_trace_g4.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_g5.yaml` | trace G5 | `python safe_rl_main.py --config safe_rl/config/shield_trace_g5.yaml --stage stage5 --run-id <run_id>` |
| `safe_rl/config/shield_trace_holdout_c1.yaml` | trace holdout 校验 | `python safe_rl_main.py --config safe_rl/config/shield_trace_holdout_c1.yaml --stage stage5 --run-id <run_id>` |

### 4.5 配置分级（保留/降级，不删除）

- 核心保留：`default_safe_rl.yaml`、`safe_rl_balanced_profile.yaml`、`stage2_v2_world_pair_focus.yaml`、`stage5_pair_bootstrap.yaml`、`shield_trace_holdout_c1.yaml`
- 实验矩阵保留：`shield_trace_c*.yaml / d*.yaml / e*.yaml / f*.yaml / g*.yaml`、`shield_sweep.yaml`
- 调试保留：`quick_check.yaml`、`smoke_test.yaml`、`debug_real_sumo.yaml`、`debug_stage3_sumo.yaml`
- 降级保留：`stage45_balance_candidate.yaml`（兼容别名），`stage45_recovery.yaml`（仅应急回退）

## 5. 常见运行方式

### 5.1 全链路一次跑完（默认正式配置）

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 5.2 分阶段运行（同一个 run_id）

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

### 5.3 在已有 run 上只复跑平衡后的 Stage4 + Stage5

```bash
python safe_rl_main.py --config safe_rl/config/safe_rl_balanced_profile.yaml --stage stage4 --run-id <existing_run_id>
python safe_rl_main.py --config safe_rl/config/safe_rl_balanced_profile.yaml --stage stage5 --run-id <existing_run_id>
```

### 5.4 使用内置批处理脚本执行 v2 流程

```bash
python run_safe_rl_v2_pipeline.py
```

指定 run_id：

```bash
python run_safe_rl_v2_pipeline.py --run-id 20260331_172102
```

只打印命令不执行：

```bash
python run_safe_rl_v2_pipeline.py --dry-run
```

## 6. 兼容入口（可选）

如需沿用项目任务系统，在 `config.yaml` 中配置任务为 `SAFE_RL_PIPELINE` 后可运行：

```bash
python main.py
```

## 7. SAFE_RL 代码结构总览

`safe_rl/` 是当前闭环的核心目录：

- [safe_rl/config](/E:/wcdt_all/WcDT/safe_rl/config)：配置定义与 YAML
- [safe_rl/sim](/E:/wcdt_all/WcDT/safe_rl/sim)：SUMO 后端、动作映射、真实/模拟控制
- [safe_rl/data](/E:/wcdt_all/WcDT/safe_rl/data)：数据采集、风险标签、样本构建、warning 统计
- [safe_rl/models](/E:/wcdt_all/WcDT/safe_rl/models)：轻量风险模型、动作条件化世界模型、特征编码
- [safe_rl/shield](/E:/wcdt_all/WcDT/safe_rl/shield)：候选动作生成、风险聚合、安全屏蔽决策
- [safe_rl/rl](/E:/wcdt_all/WcDT/safe_rl/rl)：环境包装、PPO、蒸馏
- [safe_rl/buffer](/E:/wcdt_all/WcDT/safe_rl/buffer)：干预样本池
- [safe_rl/eval](/E:/wcdt_all/WcDT/safe_rl/eval)：离线/在线评估与配对统计
- [safe_rl/pipeline](/E:/wcdt_all/WcDT/safe_rl/pipeline)：5 个阶段的总调度、工件管理、报告写出、TensorBoard

### 7.1 各子模块作用

#### `safe_rl/config`

- [config.py](/E:/wcdt_all/WcDT/safe_rl/config/config.py)：定义整个 Safe RL 的 dataclass 配置结构
- `*.yaml`：不同运行模式对应的配置文件

#### `safe_rl/sim`

- [actions.py](/E:/wcdt_all/WcDT/safe_rl/sim/actions.py)：定义 9 个离散动作
- [backend_interface.py](/E:/wcdt_all/WcDT/safe_rl/sim/backend_interface.py)：统一后端接口 `ISumoBackend`
- [traci_backend.py](/E:/wcdt_all/WcDT/safe_rl/sim/traci_backend.py)：真实 `TraCI` + mock 路径
- [libsumo_backend.py](/E:/wcdt_all/WcDT/safe_rl/sim/libsumo_backend.py)：真实 `libsumo` + mock 路径
- [real_control.py](/E:/wcdt_all/WcDT/safe_rl/sim/real_control.py)：真实 SUMO 动作落地、风险注入、route/lane 守卫
- [mock_core.py](/E:/wcdt_all/WcDT/safe_rl/sim/mock_core.py)：无真实 SUMO 时的简化仿真
- [factory.py](/E:/wcdt_all/WcDT/safe_rl/sim/factory.py)：按配置创建后端

#### `safe_rl/data`

- [collector.py](/E:/wcdt_all/WcDT/safe_rl/data/collector.py)：Stage 1 采集 episode、保存原始日志、统计失败与 warning
- [dataset_builder.py](/E:/wcdt_all/WcDT/safe_rl/data/dataset_builder.py)：把原始 episode 切成 `ActionConditionedSample`
- [risk.py](/E:/wcdt_all/WcDT/safe_rl/data/risk.py)：碰撞、TTC、车道违规等风险标签计算
- [warning_summary.py](/E:/wcdt_all/WcDT/safe_rl/data/warning_summary.py)：解析 SUMO 日志，统计结构性 warning
- [types.py](/E:/wcdt_all/WcDT/safe_rl/data/types.py)：采样、预测、干预、评估等统一数据结构

#### `safe_rl/models`

- [light_risk_model.py](/E:/wcdt_all/WcDT/safe_rl/models/light_risk_model.py)：在线粗筛的轻量风险模型
- [world_model.py](/E:/wcdt_all/WcDT/safe_rl/models/world_model.py)：动作条件化世界模型
- [action_encoder.py](/E:/wcdt_all/WcDT/safe_rl/models/action_encoder.py)：候选动作编码
- [features.py](/E:/wcdt_all/WcDT/safe_rl/models/features.py)：历史场景特征编码

#### `safe_rl/shield`

- [candidate_generator.py](/E:/wcdt_all/WcDT/safe_rl/shield/candidate_generator.py)：围绕原动作生成候选动作
- [risk_aggregator.py](/E:/wcdt_all/WcDT/safe_rl/shield/risk_aggregator.py)：多模态风险聚合
- [safety_shield.py](/E:/wcdt_all/WcDT/safe_rl/shield/safety_shield.py)：完整安全屏蔽器，实现“判断风险 -> 选替代动作 -> 记录诊断”

#### `safe_rl/rl`

- [env.py](/E:/wcdt_all/WcDT/safe_rl/rl/env.py)：`SafeDrivingEnv`，把 shield 接到策略输出和环境执行之间
- [ppo.py](/E:/wcdt_all/WcDT/safe_rl/rl/ppo.py)：PPO 训练器，支持 `SB3` 和 fallback rollout
- [distill.py](/E:/wcdt_all/WcDT/safe_rl/rl/distill.py)：用 intervention buffer 做纯行为蒸馏

#### `safe_rl/buffer`

- [intervention_buffer.py](/E:/wcdt_all/WcDT/safe_rl/buffer/intervention_buffer.py)：记录被 shield 干预的样本，并给出统计信息

#### `safe_rl/eval`

- [evaluator.py](/E:/wcdt_all/WcDT/safe_rl/eval/evaluator.py)：Stage 5 正式评估
- [metrics.py](/E:/wcdt_all/WcDT/safe_rl/eval/metrics.py)：按 episode 聚合 reward、risk、replacement、fallback 等指标

#### `safe_rl/pipeline`

- [pipeline.py](/E:/wcdt_all/WcDT/safe_rl/pipeline/pipeline.py)：整个 5 阶段闭环主调度器
- [tensorboard_logger.py](/E:/wcdt_all/WcDT/safe_rl/pipeline/tensorboard_logger.py)：TensorBoard writer 管理
- [telemetry.py](/E:/wcdt_all/WcDT/safe_rl/pipeline/telemetry.py)：Stage 3 稳定性与 Stage 4 buffer 过程曲线
- [session_event_logger.py](/E:/wcdt_all/WcDT/safe_rl/pipeline/session_event_logger.py)：Stage 3 会话事件增量记录

## 8. TensorBoard 
当前 TensorBoard 不再只是看 loss，还重点看稳定性、风险和 buffer 过程。

日志路径：

```text
safe_rl_output/runs/<run_id>/tensorboard/
```

启动命令：

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

浏览器地址：

```text
http://localhost:6006
```
