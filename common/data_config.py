#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data_config.py
@Author: YangChen
@Date: 2023/12/20
"""
from typing import List

from common.data import TaskType, BaseConfig


class TaskConfig(BaseConfig):
    __task_list: List[TaskType] = None
    output_dir: str = ""
    log_dir: str = ""
    model_dir: str = ""
    result_dir: str = ""
    pre_train_model: str = ""
    waymo_train_dir: str = ""
    waymo_val_dir: str = ""
    waymo_test_dir: str = ""
    image_dir: str = ""
    data_output: str = ""
    data_preprocess_dir: str = ""
    train_dir: str = ""
    val_dir: str = ""
    test_dir: str = ""
    safe_rl_config: str = ""

    @property
    def task_list(self) -> List[TaskType]:
        return self.__task_list

    @task_list.setter
    def task_list(self, task_list: List[str]):
        self.__task_list = [TaskType(task_name) for task_name in task_list]

    def check_config(self):
        if len(self.task_list) < 0:
            raise Warning("task_list is None")
