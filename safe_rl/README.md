# Safe RL 第一阶段使用说明

本文档用于说明 **Safe RL 第一阶段流水线** 的运行方式、整体执行流程，以及项目中各个核心文件/模块的职责。

这份文档适合作为项目内的中文说明页，方便后续开发、调试、联调和交接。

先在 SUMO 或 mock 后端里随机动作采正常/风险 episode，切成“历史场景 + 当前动作 -> 未来场景 + 风险标签”的样本；再先训一个轻量风险分类器，再训一个动作条件化世界模型；然后用这两个模型组成 Safety Shield，把在线动作先粗筛再精筛；再在带 shield 的环境里跑 PPO 或 fallback policy；随后收集所有被改写动作做 intervention buffer；样本够了再训练一个蒸馏策略；最后分别评估世界模型和系统闭环。

## 1. 项目概览

Safe RL 第一阶段流水线的目标可以概括为：

1. 从仿真环境中收集轨迹与风险数据；
2. 构建动作条件数据集；
3. 训练轻量级风险模型与动作条件世界模型；
4. 在安全屏蔽（Safety Shield）约束下训练在线策略；
5. 收集干预数据，并在条件满足时进行策略蒸馏与评估。

整个流程是一个“**数据收集 → 模型训练 → 安全控制 → 干预回流 → 再评估**”的闭环。

---

## 2. 运行时状态说明

如果运行日志中出现如下提示：

```text
SUMO cfg not found (...), fallback to mock backend.
```

其含义是：

- 当前流水线**仍然会继续正常运行**；
- 数据收集、模型训练、安全屏蔽、评估流程都可以继续执行；
- 但当前使用的不是完整 SUMO 交通仿真，而是内置的 **mock 动力学后端**。

### 这意味着什么

适合的场景：

- 集成检查
- 功能调试
- 快速冒烟测试
- 验证训练/推理链路是否打通

不适合的场景：

- 最终真实性结论
- 严格 benchmark 对比
- 对外展示的正式实验结果

### 如何切换到真实 SUMO

需要完成以下两步：

1. 在 `safe_rl/config/default_safe_rl.yaml` 中，将 `sim.sumo_cfg` 设置为本地有效的 `.sumocfg` 文件路径；
2. 安装并使用 `traci` 或 `libsumo` 作为仿真后端。

---

## 3. 快速开始

### 方式一：独立运行 Safe RL 流水线

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

### 方式二：通过旧版任务系统运行

先在 `config.yaml` 中加入任务：

```yaml
task_config:
  task_list:
    - "SAFE_RL_PIPELINE"
```

然后执行：

```bash
python main.py
```

---

## 4. 流水线阶段说明

整个第一阶段流水线共分为 5 个步骤：

### 阶段 1：收集仿真日志并构建数据集

- 从后端环境中采集正常场景与危险场景的 episode；
- 保存原始日志；
- 构建动作条件样本；
- 划分训练集/验证集/测试集。

### 阶段 2：训练预测模型

训练两类核心模型：

- **轻量级风险模型（Light Risk Model）**：判断当前状态/动作是否具有风险；
- **动作条件世界模型（Action-Conditioned World Model）**：预测动作执行后的未来结果，同时输出风险和不确定性相关信息。

### 阶段 3：带安全屏蔽的在线策略训练

- 策略先给出候选动作；
- Safety Shield 对动作进行安全检查；
- 若发现风险过高，则替换为更安全的动作；
- 将最终动作送入环境执行。

### 阶段 4：收集干预记录

- 记录哪些时刻策略动作被屏蔽器替换；
- 保存干预前后动作及其上下文；
- 写入 intervention buffer，供后续蒸馏训练使用。

### 阶段 5：策略蒸馏与评估

- 当干预样本达到阈值后，触发策略蒸馏；
- 使用干预缓冲区中的数据训练蒸馏策略；
- 对世界模型、在线策略和整体闭环系统进行评估。

---

## 5. 目录与文件职责说明

下面按模块说明主要文件的作用，方便你快速定位代码。

### 5.1 根目录入口与系统集成

| 文件 | 作用 |
| --- | --- |
| `safe_rl_main.py` | Safe RL 流水线的独立命令行入口。 |
| `tasks/safe_rl_pipeline_task.py` | 任务封装器，使 `main.py` 可以通过任务列表运行 Safe RL。 |
| `main.py` | 任务工厂入口，现已支持 `SAFE_RL_PIPELINE`。 |
| `common/data.py` | 新增 `TaskType.SAFE_RL_PIPELINE` 枚举。 |
| `common/data_config.py` | 新增 `task_config.safe_rl_config` 配置路径字段。 |
| `tasks/__init__.py` | 导出 `SafeRLPipelineTask`。 |
| `config.yaml` | 旧版任务系统的配置入口，可挂接 Safe RL 配置文件。 |

---

### 5.2 `safe_rl/` 包

| 文件 | 作用 |
| --- | --- |
| `safe_rl/__init__.py` | 包入口，导出配置加载器等核心接口。 |
| `safe_rl/README.md` | 英文版使用说明与架构说明文档。 |

---

### 5.3 `safe_rl/config`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/config/__init__.py` | 重新导出配置类与配置加载器。 |
| `safe_rl/config/config.py` | 定义 Safe RL 各配置段对应的 dataclass，并负责 YAML 加载。 |
| `safe_rl/config/default_safe_rl.yaml` | 第一阶段完整运行的默认配置。 |
| `safe_rl/config/smoke_test.yaml` | 用于快速验证流程是否通的轻量配置。 |

---

### 5.4 `safe_rl/sim`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/sim/__init__.py` | 导出仿真动作与后端工厂。 |
| `safe_rl/sim/actions.py` | 定义 9 离散动作的编码/解码、动作距离与兜底动作。 |
| `safe_rl/sim/backend_interface.py` | 定义 `ISumoBackend` 接口和 step 返回结果结构。 |
| `safe_rl/sim/factory.py` | 按配置创建后端实例（`traci` 或 `libsumo`）。 |
| `safe_rl/sim/traci_backend.py` | TraCI 后端实现；若 SUMO 不可用可自动回退到 mock core。 |
| `safe_rl/sim/libsumo_backend.py` | Libsumo 后端实现；同样支持自动回退。 |
| `safe_rl/sim/mock_core.py` | 内置 mock 交通动力学核心，并支持风险事件注入。 |

---

### 5.5 `safe_rl/data`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/data/__init__.py` | 导出数据类型与风险计算工具。 |
| `safe_rl/data/types.py` | 核心数据类定义：场景、风险标签、预测结果、屏蔽决策、干预记录等。 |
| `safe_rl/data/risk.py` | 计算 TTC、最小距离、碰撞标记与风险标签。 |
| `safe_rl/data/collector.py` | 从仿真后端采集正常/危险 episode，并保存原始日志。 |
| `safe_rl/data/dataset_builder.py` | 构建并划分 `ActionConditionedSample` 数据集。 |

---

### 5.6 `safe_rl/models`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/models/__init__.py` | 模型模块统一导出；对依赖 torch 的部分使用安全可选导入。 |
| `safe_rl/models/features.py` | 历史信息与动作特征提取工具。 |
| `safe_rl/models/action_encoder.py` | 将离散动作嵌入潜在特征空间。 |
| `safe_rl/models/light_risk_model.py` | 轻量级风险分类器及其训练器、预测器。 |
| `safe_rl/models/world_model.py` | 动作条件世界模型，以及风险头、不确定性头、训练器与预测器。 |

---

### 5.7 `safe_rl/shield`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/shield/__init__.py` | 导出 Shield 模块接口。 |
| `safe_rl/shield/candidate_generator.py` | 生成局部候选动作集合（第一阶段为固定大小方案）。 |
| `safe_rl/shield/risk_aggregator.py` | 对候选动作的风险进行带不确定性加权的尾部聚合。 |
| `safe_rl/shield/safety_shield.py` | 屏蔽器主逻辑：粗筛、细评分、最小修改替换、兜底动作选择。 |

---

### 5.8 `safe_rl/rl`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/rl/__init__.py` | 导出 RL 模块。 |
| `safe_rl/rl/env.py` | `SafeDrivingEnv` 封装，支持带 Shield 的动作执行，并返回干预信息。 |
| `safe_rl/rl/ppo.py` | PPO 适配器；若可用则接入 SB3，同时保留启发式兜底实现。 |
| `safe_rl/rl/distill.py` | 基于 intervention buffer 的策略蒸馏网络与训练器。 |

---

### 5.9 `safe_rl/buffer`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/buffer/__init__.py` | 导出 buffer 模块。 |
| `safe_rl/buffer/intervention_buffer.py` | 负责干预记录的存储、采样、持久化与统计汇总。 |

---

### 5.10 `safe_rl/eval`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/eval/__init__.py` | 导出评估模块。 |
| `safe_rl/eval/metrics.py` | 负责 episode 级和系统级指标聚合，以及验收检查。 |
| `safe_rl/eval/evaluator.py` | 提供世界模型评估、策略评估与端到端评估辅助函数。 |

---

### 5.11 `safe_rl/pipeline`

| 文件 | 作用 |
| --- | --- |
| `safe_rl/pipeline/__init__.py` | 导出流水线模块。 |
| `safe_rl/pipeline/pipeline.py` | 负责调度完整的 5 阶段闭环流程。 |

---

### 5.12 测试文件

| 文件 | 作用 |
| --- | --- |
| `tests/test_safe_rl_actions.py` | 测试动作空间与候选动作生成逻辑。 |
| `tests/test_safe_rl_shield.py` | 测试 Shield 的替换行为与直通行为。 |
| `tests/test_safe_rl_buffer.py` | 测试 intervention buffer 的写入、采样、保存与加载。 |
| `tests/test_safe_rl_config.py` | 测试配置加载的基本正确性。 |

---

## 6. 默认输出目录

默认情况下，所有输出会写入 `safe_rl_output/` 目录下，主要包括：

```text
safe_rl_output/
├── raw_logs/
├── datasets/
├── models/
├── buffers/
└── reports/
    └── pipeline_report.json
```

各目录含义如下：

- `raw_logs/`：原始仿真日志
- `datasets/`：构建好的数据集
- `models/`：训练得到的模型文件
- `buffers/`：干预缓冲区相关数据
- `reports/pipeline_report.json`：流水线运行后的汇总报告

---

## 7. 性能与运行建议

### GPU 与 CPU 使用说明

如果训练日志中出现 `on cuda`，说明模型训练部分正在使用 GPU。

即便如此，CPU 占用仍可能较高，原因通常包括：

- 数据张量化
- 日志写入
- 环境 step 执行
- 数据集构建与预处理

这不是异常情况，别被任务管理器里那几根柱子吓一跳。

### 快速调试建议

如果当前目标只是验证流程能否跑通，建议优先使用：

```text
safe_rl/config/smoke_test.yaml
```

这样可以显著缩短迭代时间，更适合：

- 初次部署验证
- 改代码后的快速回归
- CI / 本地冒烟测试

---

## 8. 建议的使用顺序

如果你是第一次接手这个项目，推荐按下面的顺序理解和运行：

1. 先看 `safe_rl/config/default_safe_rl.yaml`，搞清楚配置项；
2. 再看 `safe_rl/pipeline/pipeline.py`，理解整体阶段流；
3. 然后看 `safe_rl/data/`、`safe_rl/models/`、`safe_rl/shield/` 三大核心模块；
4. 最后再进入 `safe_rl/rl/` 和 `safe_rl/eval/` 看训练与评估闭环。

这样不容易一上来就被代码拍在地上摩擦。

---

## 9. 一句话总结

Safe RL 第一阶段的核心，不是“单纯训练一个策略”，而是建立一条完整的安全闭环链路：

> **从数据出发，训练预测模型；再用预测模型约束策略；再把干预经验回流到策略蒸馏中，最终形成可评估、可迭代的安全强化学习系统。**





