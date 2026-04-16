# SAFE_RL 子模块说明

本目录主要包含 SAFE_RL 的训练与评估实现。配置文件已完成分层收敛。

## 配置入口

- 正式主入口：`safe_rl/config/default_safe_rl.yaml`
- 进阶流程：`safe_rl/config/advanced/`
- 可视化/评估：`safe_rl/config/visualization/`
- 调试：`safe_rl/config/debug/`
- 实验：`safe_rl/config/experiments/`
- 兼容/待删除：`safe_rl/config/deprecated/`

## 常用命令

```bash
python safe_rl_main.py --config safe_rl/config/default_safe_rl.yaml
python run_safe_rl_v2_pipeline.py --run-id <run_id>
```

详细运行说明请看仓库根目录 README。
