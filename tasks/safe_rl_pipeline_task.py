#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: safe_rl_pipeline_task.py
"""

from common import LoadConfigResultDate, TaskType
from safe_rl.pipeline import run_safe_rl_pipeline
from tasks.base_task import BaseTask


class SafeRLPipelineTask(BaseTask):
    def __init__(self):
        super(SafeRLPipelineTask, self).__init__()
        self.task_type = TaskType.SAFE_RL_PIPELINE

    def execute(self, result_info: LoadConfigResultDate):
        config_path = result_info.task_config.safe_rl_config.strip() if result_info.task_config.safe_rl_config else None
        result = run_safe_rl_pipeline(config_path=config_path)
        result_info.task_logger.logger.info(f"safe_rl pipeline result: {result}")
