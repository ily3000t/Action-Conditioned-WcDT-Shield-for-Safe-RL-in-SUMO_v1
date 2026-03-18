"""
Safe RL extension package for SUMO-based safety-aware reinforcement learning.

This package is intentionally parallel to the legacy Waymo/WcDT pipeline.
"""

from safe_rl.config.config import SafeRLConfig, load_safe_rl_config

__all__ = ["SafeRLConfig", "load_safe_rl_config"]
