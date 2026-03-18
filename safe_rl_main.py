#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from typing import Optional

from safe_rl.pipeline import run_safe_rl_pipeline


VALID_STAGES = ("all", "stage1", "stage2", "stage3", "stage4", "stage5")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Safe RL pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to safe_rl yaml config")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=VALID_STAGES,
        help="Execute one stage or the full chain",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run workspace id. Required for single-stage execution.",
    )
    return parser


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.stage != "all" and not args.run_id:
        parser.error("--run-id is required when --stage is not all")

    return args


def main(argv: Optional[list] = None):
    args = parse_args(argv)

    result = run_safe_rl_pipeline(
        config_path=args.config,
        stage=args.stage,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()