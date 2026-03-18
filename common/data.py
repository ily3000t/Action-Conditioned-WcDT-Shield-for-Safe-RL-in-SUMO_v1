#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Project: WcDT
@Name: data.py
@Author: YangChen
@Date: 2023/12/20
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any


def object_dict_print(obj: Any) -> str:
    """
    Print attributes in a basic object.
    """
    result_str = ""
    for key, value in obj.__dict__.items():
        if isinstance(value, list) and len(value) > 0:
            list_str = f"{key}:\n\t["
            for row in value:
                list_str += f"{str(row)}, "
            list_str = list_str[:-2]
            list_str += "]\n"
            result_str += list_str
        else:
            result_str += f"{key}: {value} \n"
    return result_str


class TaskType(Enum):
    LOAD_CONFIG = "LOAD_CONFIG"
    DATA_PREPROCESS = "DATA_PREPROCESS"
    DATA_SPLIT = "DATA_SPLIT"
    DATA_COUNT = "DATA_COUNT"
    TRAIN_MODEL = "TRAIN_MODEL"
    SHOW_RESULTS = "SHOW_RESULTS"
    EVAL_MODEL = "EVAL_MODEL"
    GENE_SUBMISSION = "GENE_SUBMISSION"
    SAFE_RL_PIPELINE = "SAFE_RL_PIPELINE"
    UNKNOWN = "UNKNOWN"

    def __str__(self):
        return self.value


@dataclass
class BaseConfig(object):

    def __str__(self) -> str:
        return object_dict_print(self)


class TaskLogger(object):
    """
    Output logger.
    Args:
        log_path: log file path.
    """

    def __init__(self, log_path: str):
        super(TaskLogger, self).__init__()
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.INFO)

        # Avoid duplicated handlers when running multiple tasks in one process.
        if self.logger.handlers:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)

        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_path, encoding="UTF-8", mode='w')

        format_str = "%(asctime)s -%(name)s -%(levelname)-8s -%(filename)s(line: %(lineno)s):  %(message)s"
        formatter = logging.Formatter(fmt=format_str, datefmt="%Y/%m/%d %X")
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

    def get_logger(self) -> logging.Logger:
        return self.logger


if __name__ == "__main__":
    pass
