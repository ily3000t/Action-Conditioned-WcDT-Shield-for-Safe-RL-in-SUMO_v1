# WcDT + SAFE_RL 使用与维护手册

本仓库当前包含两条主线：

- 原始 `WcDT / Waymo` 训练与结果展示代码
- 新增的 `safe_rl` 一期闭环：`SUMO 数据采集 -> 风险建模 -> Safety Shield -> PPO -> 干预样本池 -> 蒸馏 -> 配对评估`

这份文档以 `safe_rl` 为重点，目标是把下面三件事写清楚：

1. 现有代码分别负责什么
2. 现在应该怎么运行
3. 每个配置模式和阶段模式分别代表什么

## 1. 仓库里最常用的入口

### 1.1 SAFE_RL 直接入口

主入口文件：

- [safe_rl_main.py](/E:/wcdt_all/WcDT/safe_rl_main.py)

最推荐的运行方式就是直接调用它：

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

### 1.2 通过旧主入口 `main.py` 间接调用

仓库原本的统一入口仍然保留：

- [main.py](/E:/wcdt_all/WcDT/main.py)
- [tasks/safe_rl_pipeline_task.py](/E:/wcdt_all/WcDT/tasks/safe_rl_pipeline_task.py)

如果你的 `config.yaml` 里把任务设成 `SAFE_RL_PIPELINE`，也可以通过：

```bash
python main.py
```

来间接触发 Safe RL。

但从日常调试、分阶段重跑、真实 SUMO 排障的角度看，推荐优先使用 [safe_rl_main.py](/E:/wcdt_all/WcDT/safe_rl_main.py)。

## 2. SAFE_RL 代码结构总览

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

### 2.1 各子模块作用

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

## 3. 运行前准备

### 3.1 Python 环境

建议使用当前项目的 `pytorch` 环境：

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__)"
```

### 3.2 SUMO 环境

当前项目默认按下面路径配置：

```text
E:\Program Files\sumo-1.22.0
```

典型可执行文件：

```text
E:\Program Files\sumo-1.22.0\bin\sumo.exe
E:\Program Files\sumo-1.22.0\bin\sumo-gui.exe
E:\Program Files\sumo-1.22.0\bin\netconvert.exe
```

如果你的安装路径不同，需要修改配置文件中的：

- `sim.sumo_home`
- `sim.sumo_bin`
- `sim.sumo_gui_bin`
- `sim.netconvert_bin`

### 3.3 生成 `highway_merge` 路网

首次运行前建议先生成路网：

```bash
python scenarios/highway_merge/build_network.py
```

相关文件：

- [build_network.py](/E:/wcdt_all/WcDT/scenarios/highway_merge/build_network.py)
- [highway_merge.sumocfg](/E:/wcdt_all/WcDT/scenarios/highway_merge/highway_merge.sumocfg)
- [highway_merge.rou.xml](/E:/wcdt_all/WcDT/scenarios/highway_merge/highway_merge.rou.xml)

如果你没有改节点、边、连接定义，通常不需要每次重建。

## 4. 配置模式说明

当前最常用的配置文件有 6 份。

### 4.1 `quick_check.yaml`

文件：

- [quick_check.yaml](/E:/wcdt_all/WcDT/safe_rl/config/quick_check.yaml)

用途：

- 最快确认整条链路能不能从 `stage1` 跑到 `stage5`
- 改代码后做快速回归
- 看 TensorBoard 和报告文件是否正常生成

特点：

- 真实 `traci`
- episode 很少
- epoch 很少
- PPO 使用 fallback 路径
- 主要看“通不通”，不看最终性能结论

运行命令：

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

### 4.2 `smoke_test.yaml`

文件：

- [smoke_test.yaml](/E:/wcdt_all/WcDT/safe_rl/config/smoke_test.yaml)

用途：

- 比 `quick_check` 更完整的调试链路
- 适合做小规模功能验证

特点：

- 默认 `sumo_cfg` 为空，必要时可能走 mock
- 规模仍然偏小
- 适合逻辑回归，不适合做正式结论

运行命令：

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

### 4.3 `default_safe_rl.yaml`

文件：

- [default_safe_rl.yaml](/E:/wcdt_all/WcDT/safe_rl/config/default_safe_rl.yaml)

用途：

- 默认正式实验配置
- 用于完整训练和正式评估

特点：

- `backend=traci`
- `normal_episodes=200`
- `risky_episodes=200`
- `world_model.epochs=20`
- `ppo.total_timesteps=50000`
- `eval.eval_episodes=30`
- `collision_action=teleport`

运行命令：

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 4.4 `debug_real_sumo.yaml`

文件：

- [debug_real_sumo.yaml](/E:/wcdt_all/WcDT/safe_rl/config/debug_real_sumo.yaml)

用途：

- 专门用于真实 `TraCI` 的 Stage 1/采集排障
- 看风险注入是否真实生效
- 看连接断开、warning 分桶、采集失败隔离是否正常

特点：

- `collision_action=warn`
- episode 数适中
- 训练参数较轻
- 重点是排障，不是做正式结果

典型命令：

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

### 4.5 `debug_stage3_sumo.yaml`

文件：

- [debug_stage3_sumo.yaml](/E:/wcdt_all/WcDT/safe_rl/config/debug_stage3_sumo.yaml)

用途：

- 专门用于真实 `TraCI` 的 Stage 3 会话排障
- 重点看 `reset/load/restart/fatal step` 诊断链条
- 检查 `stage3_runtime_config.json` 和 `stage3_session_events.json`

特点：

- `collision_action=warn`
- `ppo.total_timesteps=1024`
- `n_steps=128`
- `eval.eval_episodes=2`
- 比默认正式配置轻很多，适合专门调 Stage 3

典型命令：

```bash
python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id <已有stage1_stage2产物的run_id>
```

### 4.6 `shield_sanity.yaml`

文件：

- [shield_sanity.yaml](/E:/wcdt_all/WcDT/safe_rl/config/shield_sanity.yaml)

用途：

- 专门验证 shield 是否真的被触发
- 做激进阈值的小实验
- 用于检查 `intervention_rate`、`replacement_count`、`buffer.size` 是否大于 0

特点：

- `risk_threshold=0.05`
- `uncertainty_threshold=1.0`
- `coarse_top_k=7`
- `eval.eval_episodes=10`

典型命令：

```bash
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id <run_id>
```

## 5. 阶段模式说明

### `stage1`：采集 SUMO 数据并构建数据集

输入：

- SUMO 场景配置
- 采集相关参数

输出：

- 原始 episode 日志
- `train.pkl / val.pkl / test.pkl`
- `collector_failures.json`
- `warning_summary.json`

适合重跑的场景：

- 改了 SUMO 场景
- 改了风险注入
- 改了标签逻辑
- 改了数据切片逻辑

### `stage2`：训练轻量风险模型和世界模型

输入：

- Stage 1 的数据集

输出：

- `light_risk.pt`
- `world_model.pt`
- `stage2_training_report.json`

适合重跑的场景：

- 改了模型结构
- 改了 loss
- 改了特征编码

### `stage3`：加载模型，构建 shield，训练在线策略

输入：

- Stage 2 模型工件

输出：

- `policy_meta.json`
- `ppo_sb3.zip`（仅 `use_sb3=true` 时）
- `stage3_runtime_config.json`
- `stage3_session_events.json`

适合重跑的场景：

- 改了 PPO
- 改了 `SafeDrivingEnv`
- 改了 shield 在线接入逻辑
- 需要单独调试真实 SUMO 会话稳定性

### `stage4`：加载模型和策略，采集干预样本池

输入：

- Stage 2 模型工件
- Stage 3 策略工件

输出：

- `intervention_buffer.pkl`
- `stage4_buffer_report.json`

当前口径：

- `stage4` 使用当前已训练 PPO policy
- 开启 shield
- 在 risky rollout 中采集干预样本
- 不是 heuristic baseline，也不是另一套 policy

### `stage5`：蒸馏并评估，输出最终报告

输入：

- 测试集
- 模型工件
- 策略工件
- intervention buffer

输出：

- `pipeline_report.json`
- `stage5_paired_episode_results.json`

当前官方评估口径：

- `system_baseline = 同一 PPO policy + shield 关闭`
- `system_shielded = 同一 PPO policy + shield 开启`
- `baseline` 和 `shielded` 使用相同 `seed`
- `baseline` 和 `shielded` 使用相同 `risky_mode=True`
- `baseline` 和 `shielded` 使用相同 `scenario_source = sim.sumo_cfg`

## 6. 最常用运行命令

### 6.1 一键快速检查整条链路

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

### 6.2 一键 smoke 测试

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

### 6.3 一键默认正式训练

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 6.4 分阶段模板

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

### 6.5 真实 SUMO Stage 1 排障

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

### 6.6 真实 SUMO Stage 3 排障

```bash
python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id exp_001
```

说明：

- 这里的 `run_id` 应该已经有 `stage1` 和 `stage2` 产物
- Stage 3 会生成 episode 级 SUMO log、runtime config 和 session event 报告

### 6.7 Shield sanity check

```bash
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id exp_001
```

说明：

- 这组命令用于验证 shield 是否真的触发、是否真的替换动作
- 会覆盖当前 `run_id` 下的 stage4/stage5 相关工件
- 如果你想保留正式报告，建议新建一个单独的 `run_id`

## 7. 输出目录结构与文件含义

每次运行都会在下面创建独立工作区：

```text
safe_rl_output/runs/<run_id>/
```

当前常见产物如下：

```text
safe_rl_output/runs/<run_id>/
├─ raw/                                 # stage1 原始 episode 日志
├─ datasets/
│  ├─ train.pkl
│  ├─ val.pkl
│  └─ test.pkl
├─ models/
│  ├─ light_risk.pt
│  └─ world_model.pt
├─ policies/
│  ├─ policy_meta.json
│  └─ ppo_sb3.zip                       # 仅 use_sb3=true 时存在
├─ buffers/
│  └─ intervention_buffer.pkl
├─ reports/
│  ├─ collector_failures.json
│  ├─ warning_summary.json
│  ├─ stage2_training_report.json
│  ├─ stage3_runtime_config.json
│  ├─ stage3_session_events.json
│  ├─ stage4_buffer_report.json
│  ├─ stage5_paired_episode_results.json
│  └─ pipeline_report.json
├─ sumo_logs/
│  ├─ traci_runtime.log                 # session 级日志
│  └─ episodes/                         # episode 级日志
├─ tensorboard/
└─ manifest.json
```

### 7.1 这些文件分别适合看什么

- `collector_failures.json`：Stage 1 单 episode 失败情况
- `warning_summary.json`：SUMO warning 分桶统计，区分 normal / risky
- `stage2_training_report.json`：Stage 2 训练设备与摘要
- `stage3_runtime_config.json`：Stage 3 真实启动参数，确认到底用的是 `teleport` 还是 `warn`
- `stage3_session_events.json`：Stage 3 reset/load/restart/fatal step 诊断日志
- `stage4_buffer_report.json`：buffer 来源、口径、大小和风险统计
- `stage5_paired_episode_results.json`：同一 PPO policy、shield off/on 的逐 episode 配对结果
- `pipeline_report.json`：最终汇总结论和验收结果
- `manifest.json`：阶段完成状态与各工件路径索引

## 8. 最终报告里重点看哪些字段

最终汇总文件：

- [pipeline_report.json](/E:/wcdt_all/WcDT/safe_rl_output)

当前需要重点看这些字段：

- `comparison_mode`
- `paired_eval`
- `system_baseline`
- `system_shielded`
- `intervention_buffer`
- `performance_passed`
- `attribution_passed`
- `shield_contribution_validated`
- `sanity_check_passed`
- `conclusion_text`

### 8.1 当前结论口径

现在项目已经明确把两类结论拆开：

- `performance_passed`：系统整体性能是否达标
- `attribution_passed`：是否已经独立证明 shield 确实发挥了作用

兼容字段：

- `acceptance_passed`：等价于 `performance_passed`
- `shield_contribution_validated`：等价于 `attribution_passed`

这意味着：

- 即使 PPO 训练后系统整体更强，也不代表 shield 一定已经被独立验证
- 只有在 `intervention_rate > 0`、`mean_risk_reduction > 0`、`replacement_count > 0` 时，才会判定 `attribution_passed=true`

## 9. TensorBoard 怎么看

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

### 9.1 当前重点模块

- `light_risk`
- `world_model`
- `ppo`
- `buffer`
- `distill`
- `eval`

### 9.2 现在最值得看的曲线

#### `ppo`

- `stability/*`：reset/load/restart/fatal step 稳定性
- `risk/step_raw`
- `risk/step_final`
- `risk/step_reduction`
- `risk/episode_mean_reduction`

#### `buffer`

- `buffer/size`
- `buffer/push_rate`
- `buffer/running_mean_raw_risk`
- `buffer/running_mean_final_risk`
- `buffer/running_mean_risk_reduction`
- `buffer/episode_pushes`

#### `eval`

- `baseline/episode_mean_raw_risk`
- `shielded/summary_mean_risk_reduction`
- `distilled/summary_mean_final_risk`
- `summary/collision_reduction`
- `summary/efficiency_drop`
- `summary/shield_contribution_validated`

## 10. 测试命令

### 10.1 跑全部测试

```bash
pytest tests -q
```

### 10.2 只跑 SAFE_RL 关键测试

```bash
pytest tests/test_safe_rl_actions.py tests/test_safe_rl_buffer.py tests/test_safe_rl_collector.py tests/test_safe_rl_config.py tests/test_safe_rl_env.py tests/test_safe_rl_libsumo_backend.py tests/test_safe_rl_real_control.py tests/test_safe_rl_risk.py tests/test_safe_rl_shield.py tests/test_safe_rl_stages.py tests/test_safe_rl_telemetry.py tests/test_safe_rl_tensorboard.py tests/test_safe_rl_traci_backend.py -q
```

### 10.3 做语法回归

```bash
python -m compileall safe_rl tests
```

## 11. 常见排障入口

### 11.1 看到 `SUMO cfg not found ... fallback to mock backend`

说明：

- 当前找不到真实 `.sumocfg`
- 程序会回退到 mock backend

优先检查：

- `sim.sumo_cfg` 是否正确
- `scenarios/highway_merge/highway_merge.sumocfg` 是否存在
- 是否忘了先执行 `python scenarios/highway_merge/build_network.py`

### 11.2 看到 `Connection closed by SUMO` / `Socket reset by peer`

优先看：

- `reports/stage3_session_events.json`
- `reports/stage3_runtime_config.json`
- `sumo_logs/traci_runtime.log`
- `sumo_logs/episodes/*.log`

### 11.3 真实 TraCI 的 warning 很多

优先看：

- `reports/warning_summary.json`

它已经按 `normal / risky / overall` 分桶统计：

- `illegal_lane_index`
- `no_connection_next_edge`
- `emergency_stop_no_connection`
- `junction_collision`
- `lanechange_collision`
- `emergency_braking_high`
- `other`

### 11.4 想确认 shield 到底有没有起作用

优先看：

- `reports/stage4_buffer_report.json`
- `reports/stage5_paired_episode_results.json`
- `reports/pipeline_report.json`
- TensorBoard 里的 `buffer/*` 和 `eval/*risk*`

如果下面这些量仍然全是 0：

- `intervention_rate`
- `replacement_count`
- `buffer.size`
- `mean_risk_reduction`

那当前能说明的是 PPO policy 变强了，但还不能说明 shield 已经被独立验证。

## 12. 关于 `TraCI` 和 `libsumo`

### 12.1 `TraCI`

当前真实 `TraCI` 路线已经支持：

- 显式 `ego` 控制
- 真实动作落地
- 真实风险注入
- `reset/load/restart` 容错
- episode 级和 session 级 SUMO 日志
- Stage 1 单 episode 失败隔离
- Stage 3 会话事件记录

### 12.2 `libsumo`

代码路径已经补齐到与 `TraCI` 同等级：

- 真实动作落地
- 风险注入
- `_last_scene` 回退
- fatal step 容错
- reset/restart 诊断
- session 级诊断信息

但它是否能在你的机器上真实运行，还取决于本机 Python 环境是否能成功：

```python
import libsumo
```

如果本机没有可用的 `libsumo` Python binding，那么真实 `libsumo` 仍然不能直接跑。

## 13. 推荐使用顺序

如果你只是想确认代码有没有跑通：

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

如果你刚改了真实 SUMO 采集或风险注入：

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

如果你刚改了 Stage 3 会话管理、环境重置或真实后端稳定性：

```bash
python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id exp_001
```

如果你刚改了 shield、buffer 或 attribution 评估口径：

```bash
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id exp_001
```

如果你准备做完整正式实验：

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

## 14. 一页速查命令

### 14.1 生成路网

```bash
python scenarios/highway_merge/build_network.py
```

### 14.2 快速检查链路

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

### 14.3 更完整的 smoke 检查

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

### 14.4 默认正式训练

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 14.5 Stage 1 真实 SUMO 排障

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

### 14.6 Stage 3 会话排障

```bash
python safe_rl_main.py --config safe_rl/config/debug_stage3_sumo.yaml --stage stage3 --run-id exp_001
```

### 14.7 Shield sanity

```bash
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/shield_sanity.yaml --stage stage5 --run-id exp_001
```

### 14.8 查看 TensorBoard

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

### 14.9 跑测试

```bash
pytest tests -q
```

## Shield Trace C1/C2/C_strong

用于固定 `run_id=20260320_210439`，只重跑 `stage5` 做轻量 trace 调参，不重训 `stage1~4`。

运行命令：

```bash
python safe_rl_main.py --config safe_rl/config/shield_trace_c1.yaml --stage stage5 --run-id 20260320_210439
python safe_rl_main.py --config safe_rl/config/shield_trace_c2.yaml --stage stage5 --run-id 20260320_210439
python safe_rl_main.py --config safe_rl/config/shield_trace_c_strong.yaml --stage stage5 --run-id 20260320_210439
```

配置区别：

- `shield_trace_c1.yaml`
  - `replacement_min_risk_margin = 0.08`
  - `raw_passthrough_risk_threshold = 0.24`
- `shield_trace_c2.yaml`
  - `replacement_min_risk_margin = 0.10`
  - `raw_passthrough_risk_threshold = 0.25`
- `shield_trace_c_strong.yaml`
  - `replacement_min_risk_margin = 0.15`
  - `raw_passthrough_risk_threshold = 0.30`

输出目录：

- `safe_rl_output/runs/20260320_210439/reports/shield_trace_c1/`
- `safe_rl_output/runs/20260320_210439/reports/shield_trace_c2/`
- `safe_rl_output/runs/20260320_210439/reports/shield_trace_c_strong/`
- `safe_rl_output/runs/20260320_210439/reports/shield_trace_tuning_summary.json`

`shield_trace_tuning_summary.json` 会汇总：

- `C_baseline`
- `C1`
- `C2`
- `C_strong`

重点比较字段：

- `effective_shield_config`
- `blocked_by_margin_count`
- `raw_passthrough_count`
- `merge_lateral_guard_block_count`
- `candidate_selected_count`
- `regression_pair_count`
- `mean_intervention_count`
- `mean_risk_reduction`
- `mean_reward_gap_to_baseline_policy`
