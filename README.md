# WcDT + SAFE_RL 运行说明

本仓库当前包含两条主线：

- 原始 `WcDT / Waymo` 相关代码
- 新增的 `safe_rl` 一期闭环：SUMO 数据采集 -> 动作条件化风险模型 / 世界模型 -> 安全屏蔽 -> PPO -> 干预蒸馏 -> 评估

本文档重点说明 `safe_rl` 这条链路的运行方式、分阶段调试方式、测试命令和排障入口。

## 1. 当前 SAFE_RL 的入口

主入口文件：

- [safe_rl_main.py](/E:/wcdt_all/WcDT/safe_rl_main.py)

当前支持两种运行方式：

- 一键整条链路运行：`--stage all`
- 分阶段运行：`--stage stage1|stage2|stage3|stage4|stage5`

命令行参数：

```bash
python safe_rl_main.py --config <配置文件>
python safe_rl_main.py --config <配置文件> --stage <阶段名> --run-id <实验ID>
```

说明：

- `--stage all` 时，如果不传 `--run-id`，系统会自动生成一个新的 `run_id`
- `--stage != all` 时，必须显式传 `--run-id`
- 分阶段执行时，后续阶段必须复用同一个 `run_id`

## 2. 环境准备

### 2.1 Python 环境

建议在你的 `pytorch` 环境中运行：

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__)"
```

### 2.2 SUMO 环境

当前工程默认按下面的 Windows 安装路径配置：

```text
E:\Program Files\sumo-1.22.0
```

典型可执行文件：

```text
E:\Program Files\sumo-1.22.0\bin\sumo.exe
E:\Program Files\sumo-1.22.0\bin\sumo-gui.exe
E:\Program Files\sumo-1.22.0\bin\netconvert.exe
```

对应配置默认写在这些文件里：

- [default_safe_rl.yaml](/E:/wcdt_all/WcDT/safe_rl/config/default_safe_rl.yaml)
- [quick_check.yaml](/E:/wcdt_all/WcDT/safe_rl/config/quick_check.yaml)
- [smoke_test.yaml](/E:/wcdt_all/WcDT/safe_rl/config/smoke_test.yaml)
- [debug_real_sumo.yaml](/E:/wcdt_all/WcDT/safe_rl/config/debug_real_sumo.yaml)

如果你的 SUMO 安装路径不同，请先修改这些 YAML 里的：

- `sim.sumo_home`
- `sim.sumo_bin`
- `sim.sumo_gui_bin`
- `sim.netconvert_bin`

### 2.3 生成 highway_merge 路网

首次运行前，建议先生成场景网络：

```bash
python scenarios/highway_merge/build_network.py
```

对应脚本：

- [build_network.py](/E:/wcdt_all/WcDT/scenarios/highway_merge/build_network.py)

如果 `.net.xml` 已经存在，且你没有改动节点/边/连接文件，这一步通常不需要重复执行。

## 3. SAFE_RL 配置文件说明

当前常用配置如下：

### 3.1 快速检查链路是否打通

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

适用场景：

- 最快验证全链路能否从 `stage1` 跑到 `stage5`
- 改了代码后做快速 smoke 检查
- 看 TensorBoard 是否正常落日志

特点：

- episode 数量少
- epoch 少
- timesteps 少
- 主要用于“链路通不通”，不是看最终指标

### 3.2 更完整但仍偏调试的检查

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

适用场景：

- 比 `quick_check` 更完整
- 仍然属于调试配置
- 用来验证训练、屏蔽、蒸馏和评估结果是否大致正常

### 3.3 默认正式配置

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

适用场景：

- 跑一期主实验
- 生成相对完整的模型、buffer、评估报告
- 用于正式比较 shield 前后效果

### 3.4 真实 SUMO 排障配置

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

适用场景：

- 专门检查真实 `TraCI` 后端
- 观察真实风险注入是否稳定
- 排查 `Connection closed by SUMO`、碰撞密度过高、日志异常等问题

当前这份配置的特点：

- `backend=traci`
- `normal_episodes=10`
- `risky_episodes=10`
- `episode_steps=150`
- `risk_event_prob` 较低
- 训练参数较轻，方便快速排障

## 4. 一键整链路运行命令

### 4.1 最快全链路检查

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

### 4.2 调试型全链路检查

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

### 4.3 默认正式全链路运行

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

这三条命令都会自动生成新的 `run_id`，并在：

```text
safe_rl_output/runs/<run_id>/
```

下保存完整中间产物。

## 5. 分阶段运行命令

如果你只想重跑某一步，而不是从头到尾全跑，使用 `--stage + --run-id`。

### 5.1 各阶段含义

- `stage1`：采集 SUMO 数据并构建数据集
- `stage2`：训练轻量风险模型和世界模型
- `stage3`：加载模型，构建 shield，训练在线策略
- `stage4`：加载模型和策略，采集干预样本池
- `stage5`：蒸馏并评估，输出最终报告

### 5.2 示例：默认配置按阶段运行

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

### 5.3 示例：快速配置按阶段跑

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage1 --run-id quick_debug_001
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage2 --run-id quick_debug_001
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage3 --run-id quick_debug_001
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage4 --run-id quick_debug_001
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage5 --run-id quick_debug_001
```

### 5.4 分阶段运行的重要规则

- 必须复用同一个 `run_id`
- 后一阶段依赖前一阶段产物
- 如果依赖文件缺失，程序会直接报错，不会自动补跑前置阶段

例如：

- `stage2` 依赖 `datasets/train.pkl|val.pkl|test.pkl`
- `stage3` 依赖 `models/light_risk.pt` 和 `models/world_model.pt`
- `stage4` 依赖模型和策略工件
- `stage5` 依赖测试集、模型、策略、干预池

## 6. 真实 SUMO / TraCI 调试命令

### 6.1 只调 Stage 1 采集

这是最常用的真实 SUMO 调试入口：

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

### 6.2 用 quick_check 配置只跑真实采集

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml --stage stage1 --run-id traci_stage1_check_001
```

### 6.3 真实 SUMO 问题排查重点

如果你看到类似：

- `Connection closed by SUMO`
- `FatalTraCIError`
- 高频碰撞 warning

优先查看当前 `run_id` 下的两个文件：

- `safe_rl_output/runs/<run_id>/reports/collector_failures.json`
- `safe_rl_output/runs/<run_id>/sumo_logs/traci_runtime.log`

说明：

- `collector_failures.json` 记录了哪些 episode 失败、失败原因、异常文本
- `traci_runtime.log` 是真实 SUMO 运行日志，适合定位 SUMO 自己为什么断开连接

### 6.4 当前真实 TraCI 稳定化行为

目前真实 `TraCI` 后端已经做了这些保护：

- `simulationStep()` 断连后按“单 episode 失败”处理
- 不再让 Stage 1 因一个 episode 崩掉而整体中断
- 会在当前 `run_id` 下写 `sumo_logs/traci_runtime.log`
- Stage 1 结束后会写失败报告 `collector_failures.json`
- 风险注入已做降强度处理，减少一步造出车辆重叠导致的崩溃

## 7. TensorBoard 查看命令

当前 SAFE_RL 五个模块已经接入 TensorBoard：

- `light_risk`
- `world_model`
- `ppo`
- `distill`
- `eval`

日志按 `run_id` 写入：

```text
safe_rl_output/runs/<run_id>/tensorboard/
```

启动 TensorBoard：

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

如果你只想看某一个实验，可以把 `--logdir` 指到某个具体 `run_id`。

## 8. 运行输出目录说明

每次运行都会在下面生成独立工作区：

```text
safe_rl_output/runs/<run_id>/
```

常见目录和文件如下：

```text
safe_rl_output/runs/<run_id>/
├─ raw/                        # stage1 原始 episode 日志
├─ datasets/
│  ├─ train.pkl
│  ├─ val.pkl
│  └─ test.pkl
├─ models/
│  ├─ light_risk.pt
│  └─ world_model.pt
├─ policies/
│  ├─ policy_meta.json
│  └─ ppo_sb3.zip             # 仅 use_sb3=true 时存在
├─ buffers/
│  └─ intervention_buffer.pkl
├─ reports/
│  ├─ pipeline_report.json
│  └─ collector_failures.json
├─ sumo_logs/
│  └─ traci_runtime.log
├─ tensorboard/
└─ manifest.json
```

其中：

- `manifest.json`：记录当前 run 的阶段完成状态、产物路径、时间戳
- `pipeline_report.json`：最终评估结果
- `collector_failures.json`：Stage 1 失败 episode 报告

## 9. 测试命令

当前 SAFE_RL 相关测试文件包括：

- `tests/test_safe_rl_actions.py`
- `tests/test_safe_rl_buffer.py`
- `tests/test_safe_rl_collector.py`
- `tests/test_safe_rl_config.py`
- `tests/test_safe_rl_real_control.py`
- `tests/test_safe_rl_risk.py`
- `tests/test_safe_rl_shield.py`
- `tests/test_safe_rl_stages.py`
- `tests/test_safe_rl_tensorboard.py`
- `tests/test_safe_rl_traci_backend.py`

### 9.1 跑全部 SAFE_RL 测试

```bash
pytest tests/test_safe_rl_actions.py tests/test_safe_rl_buffer.py tests/test_safe_rl_collector.py tests/test_safe_rl_config.py tests/test_safe_rl_real_control.py tests/test_safe_rl_risk.py tests/test_safe_rl_shield.py tests/test_safe_rl_stages.py tests/test_safe_rl_tensorboard.py tests/test_safe_rl_traci_backend.py -q
```

### 9.2 直接跑 tests 目录

```bash
pytest tests -q
```

### 9.3 真实后端和阶段逻辑重点回归

```bash
pytest tests/test_safe_rl_config.py tests/test_safe_rl_real_control.py tests/test_safe_rl_traci_backend.py tests/test_safe_rl_collector.py tests/test_safe_rl_stages.py -q
```

### 9.4 TensorBoard 相关测试

```bash
pytest tests/test_safe_rl_tensorboard.py -q
```

## 10. 推荐使用流程

如果你只是想确认代码现在能不能跑：

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

如果你刚改完真实 SUMO 相关代码，优先这样：

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

如果你改的是模型训练逻辑，推荐这样分步调试：

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_model_debug
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_model_debug
```

如果你改的是 shield、PPO 或蒸馏逻辑，推荐复用已有 run：

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_model_debug
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_model_debug
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_model_debug
```

## 11. 当前已知说明

### 11.1 关于 TraCI

当前真实 `TraCI` 路线已经支持：

- 显式 `ego` 控车
- 真实动作落地
- 真实风险注入
- 断连容错
- SUMO runtime log
- Stage 1 单 episode 失败隔离

### 11.2 关于 Libsumo

代码里已经补上了 `libsumo` 的真实动作和风险注入路径，但是否能在本机真实启用，还取决于你本地 Python 环境里是否能成功：

```python
import libsumo
```

如果本机没有可导入的 `libsumo` Python binding，那么真实 `libsumo` 路径仍然不可直接运行。

### 11.3 关于默认正式配置耗时

`default_safe_rl.yaml` 目前属于正式实验配置，参数相对重：

- `normal_episodes=200`
- `risky_episodes=200`
- `world_model.epochs=20`
- `ppo.total_timesteps=50000`
- `eval.eval_episodes=30`

所以它明显比 `quick_check` 和 `smoke_test` 更耗时，建议先确认链路没问题再跑。

## 12. 最常用命令速查

### 12.1 生成路网

```bash
python scenarios/highway_merge/build_network.py
```

### 12.2 快速全链路

```bash
python safe_rl_main.py --config safe_rl/config/quick_check.yaml
```

### 12.3 调试型全链路

```bash
python safe_rl_main.py --config safe_rl/config/smoke_test.yaml
```

### 12.4 默认正式配置

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 12.5 真实 SUMO Stage 1 调试

```bash
python safe_rl_main.py --config safe_rl/config/debug_real_sumo.yaml --stage stage1 --run-id debug_real_001
```

### 12.6 分阶段运行模板

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

### 12.7 查看 TensorBoard

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

### 12.8 运行测试

```bash
pytest tests -q
```
