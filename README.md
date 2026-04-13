# WcDT SAFE_RL 运行手册

本文档只包含两部分：
- 运行方式
- 配置文件入口说明

## 1. 主入口

```bash
python safe_rl_main.py --config <config_path>
python safe_rl_main.py --config <config_path> --stage <all|stage1|stage2|stage3|stage4|stage5> --run-id <run_id>
```

规则：
- `--stage all` 可不传 `--run-id`，系统会自动生成新的 `run_id`
- `--stage != all` 必须传入已存在的 `run_id`
- 分阶段运行时，后续阶段必须复用同一个 `run_id`

## 2. 运行前准备

```bash
conda activate pytorch
python -c "import torch; print(torch.__version__)"
python scenarios/highway_merge/build_network.py
```

如 SUMO 安装路径与默认不一致，请在配置中检查并修改：
- `sim.sumo_home`
- `sim.sumo_bin`
- `sim.sumo_gui_bin`
- `sim.netconvert_bin`

## 3. 常用运行命令

1. 全链路一次跑完（默认正式入口）

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
```

2. 分阶段运行（同一个 `run_id`）

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage1 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage2 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage3 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id exp_001
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id exp_001
```

3. 现有 run 上仅复跑 Stage4 + Stage5

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage4 --run-id <existing_run_id>
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml --stage stage5 --run-id <existing_run_id>
```

4. Stage2 world pair-finetune 专项（高级入口）

```bash
python safe_rl_main.py --config safe_rl/config/stage2_v2_world_pair_focus.yaml --stage stage2 --run-id <run_id>
```

5. Stage5 pair bootstrap（补数入口）

```bash
python safe_rl_main.py --config safe_rl/config/stage5_pair_bootstrap.yaml --stage stage5 --run-id <run_id>
```

6. Stage4/5 应急回退（仅故障恢复）

```bash
python safe_rl_main.py --config safe_rl/config/stage45_recovery.yaml --stage stage4 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/stage45_recovery.yaml --stage stage5 --run-id <run_id>
```

## 4. 配置入口分层（主入口 vs 实验入口）

### 4.1 主入口（普通用户只看这里）

| 配置文件 | 说明 |
|---|---|
| `safe_rl/config/default_safe_rl.yaml` | 唯一正式默认入口（全链路基线，默认 `shield.profile=balanced`） |

说明：`default_safe_rl.yaml` 已默认启用 balanced，不需要再切换 `safe_rl_balanced_profile.yaml`。

### 4.2 高级入口（按需使用）

| 配置文件 | 说明 |
|---|---|
| `safe_rl/config/stage2_v2_world_pair_focus.yaml` | Stage2 world pair-finetune 专项入口 |
| `safe_rl/config/stage5_pair_bootstrap.yaml` | Stage5 pair 数据补齐入口 |
| `safe_rl/config/stage45_recovery.yaml` | Stage4/5 应急回退入口 |

### 4.3 调试入口（开发者附录）

- `safe_rl/config/quick_check.yaml`
- `safe_rl/config/smoke_test.yaml`
- `safe_rl/config/debug_real_sumo.yaml`
- `safe_rl/config/debug_stage3_sumo.yaml`
- `safe_rl/config/shield_sanity.yaml`

### 4.4 实验矩阵入口（论文/扫参/trace）

- `safe_rl/config/shield_sweep.yaml`
- `safe_rl/config/shield_trace_c*.yaml / d*.yaml / e*.yaml / f*.yaml / g*.yaml`
- `safe_rl/config/shield_trace_holdout_c1.yaml`

### 4.5 兼容别名与降级入口（不作为主入口）

- `safe_rl/config/safe_rl_balanced_profile.yaml`：兼容别名（与 default 的 balanced 行为一致）
- `safe_rl/config/stage45_balance_candidate.yaml`：deprecated alias，仅兼容旧命令
- `safe_rl/config/stage2_world_base_only.yaml`：ablation 入口，不作为主流程推荐

## 5. 其他入口

1. 批处理入口（v2 流程）

```bash
python run_safe_rl_v2_pipeline.py
python run_safe_rl_v2_pipeline.py --run-id 20260331_172102
python run_safe_rl_v2_pipeline.py --dry-run
```

2. 兼容任务入口

```bash
python main.py
```

## 6. TensorBoard

```bash
tensorboard --logdir safe_rl_output/runs --port 6006
```

日志目录：

```text
safe_rl_output/runs/<run_id>/tensorboard/
```

## 7. Stage4 Zero-Intervention Diagnostics

- `warning_summary.json` is updated in a stage-merge style: existing Stage1 collector summary is preserved, and Stage2/Stage4 health is appended under both top-level keys and `by_stage`.
- If Stage4 still has zero replacements and `stage4_buffer_report.json -> shield_activation_diagnostics -> raw_risk_stats.p99` stays far below `thresholds.raw_threshold_used` for a long window, likely causes are:
  - risk-model score resolution/calibration is too weak
  - Stage4 policy visits a more conservative state distribution than Stage1/2 training data
  - this is not necessarily a shield replacement-logic bug

## 8. New Evaluation/Desensitization Profiles

- `default_safe_rl.yaml` remains the main entry and keeps `shield.profile=balanced`.
- Use independent profiles to avoid mixed-variable attribution:
  - `stage5_eval_hardening.yaml`: evaluation only (`eval.eval_episodes=90`, expanded seeds).
  - `stage45_cost_desensitize.yaml`: Stage4/5 behavior desensitization only (`blocked_distance_margin_slope`, distill lr/epochs).

Run commands:

```bash
# Evaluation hardening only
python safe_rl_main.py --config safe_rl/config/stage5_eval_hardening.yaml --stage stage5 --run-id <run_id>

# Cost desensitization only
python safe_rl_main.py --config safe_rl/config/stage45_cost_desensitize.yaml --stage stage4 --run-id <run_id>
python safe_rl_main.py --config safe_rl/config/stage45_cost_desensitize.yaml --stage stage5 --run-id <run_id>
```

Notes:
- `stage5_eval_hardening.yaml` uses seed-group holdout (same scenario, different seed groups).
- Seed-group holdout is **not** scenario distribution holdout.
