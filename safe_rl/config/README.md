# SAFE_RL Config Layout

- `default_safe_rl.yaml`: 唯一正式主入口。
- `advanced/`: 进阶流程配置（stage2/5 bootstrap、recovery）。
- `visualization/`: 评估增强、行为减敏、trace capture。
- `debug/`: 快速排障和小规模调试。
- `experiments/`: sweep/holdout 实验配置。
- `deprecated/`: 兼容保留与待删除配置，不建议新实验使用。

说明：加载器支持按文件名回溯查找，旧路径在多数情况下可兼容；建议逐步改用新目录路径。
