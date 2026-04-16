# SAFE_RL Config Layout

- `default_safe_rl.yaml`: 唯一正式主入口。
- `advanced/`: 进阶流程配置（Stage2/Stage5 bootstrap、recovery）。
- `visualization/`: 评估增强、行为减敏、trace capture。
- `experiments/`: sweep/holdout 实验配置。
- `debug/`: 快速排障与小规模调试。

说明：
- 已移除 `deprecated/`。
- 加载器仅支持显式路径；不存在路径会直接报错。
