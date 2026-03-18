from dataclasses import dataclass

import numpy as np

from safe_rl.config.config import PPOConfig
from safe_rl.sim.actions import encode_action


class PolicyAdapter:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        raise NotImplementedError


class HeuristicPolicy(PolicyAdapter):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        # Observation layout comes from safe_rl.models.features.scene_statistics
        ego_speed = float(observation[2])
        min_distance = float(observation[6])
        min_ttc = float(observation[7])
        if min_ttc < 2.0 or min_distance < 8.0:
            return encode_action(-1, 0)
        if ego_speed < 18.0 and min_ttc > 6.0:
            return encode_action(1, 0)
        return encode_action(0, 0)


@dataclass
class SB3PolicyAdapter(PolicyAdapter):
    model: any

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        if isinstance(action, np.ndarray):
            return int(action.item())
        return int(action)


class SafePPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config

    def train(self, env, tb_writer=None) -> PolicyAdapter:
        print(
            "[PPO] start training: "
            f"use_sb3={self.config.use_sb3}, total_timesteps={self.config.total_timesteps}, "
            f"n_steps={self.config.n_steps}, batch_size={self.config.batch_size}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("config/use_sb3", float(bool(self.config.use_sb3)), 0)
            tb_writer.add_scalar("config/total_timesteps", float(self.config.total_timesteps), 0)

        if self.config.use_sb3:
            try:
                from stable_baselines3 import PPO  # type: ignore

                tb_log_root = None
                if tb_writer is not None and hasattr(tb_writer, "log_dir"):
                    tb_log_root = str(tb_writer.log_dir)

                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=self.config.learning_rate,
                    gamma=self.config.gamma,
                    n_steps=self.config.n_steps,
                    batch_size=self.config.batch_size,
                    n_epochs=self.config.n_epochs,
                    verbose=1,
                    tensorboard_log=tb_log_root,
                )
                model.learn(total_timesteps=self.config.total_timesteps, tb_log_name="sb3")
                print("[PPO] sb3 training finished")
                return SB3PolicyAdapter(model=model)
            except Exception as exc:
                print(f"[PPO] sb3 unavailable ({exc}), fallback to heuristic rollout")

        policy = HeuristicPolicy()
        self._run_fallback_rollout(env, policy, tb_writer=tb_writer)
        print("[PPO] fallback rollout finished")
        return policy

    def _run_fallback_rollout(self, env, policy: PolicyAdapter, tb_writer=None):
        total_steps = max(1, self.config.total_timesteps)
        reset_output = env.reset()
        if isinstance(reset_output, tuple):
            obs = reset_output[0]
        else:
            obs = reset_output

        episode_idx = 0
        episode_reward = 0.0
        episode_steps = 0
        episode_intervened = 0

        log_interval = max(500, total_steps // 20)
        for step_idx in range(total_steps):
            action = policy.predict(obs, deterministic=False)
            step_output = env.step(action)
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            else:
                obs, reward, done, info = step_output

            reward = float(reward)
            intervened = float(bool(info.get("intervened", False)))
            risk_raw = float(info.get("risk_raw", 0.0))
            risk_final = float(info.get("risk_final", 0.0))

            episode_reward += reward
            episode_steps += 1
            episode_intervened += int(intervened > 0)

            if tb_writer is not None:
                tb_writer.add_scalar("rollout/step_reward", reward, step_idx)
                tb_writer.add_scalar("rollout/step_intervened", intervened, step_idx)
                tb_writer.add_scalar("rollout/step_risk_raw", risk_raw, step_idx)
                tb_writer.add_scalar("rollout/step_risk_final", risk_final, step_idx)

            if done:
                if tb_writer is not None:
                    tb_writer.add_scalar("rollout/episode_reward", episode_reward, episode_idx)
                    tb_writer.add_scalar("rollout/episode_steps", float(episode_steps), episode_idx)
                    tb_writer.add_scalar(
                        "rollout/episode_intervention_rate",
                        float(episode_intervened) / max(1.0, float(episode_steps)),
                        episode_idx,
                    )
                episode_idx += 1
                episode_reward = 0.0
                episode_steps = 0
                episode_intervened = 0

                reset_output = env.reset()
                obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output

            if (step_idx + 1) % log_interval == 0:
                print(f"[PPO] fallback rollout progress: {step_idx + 1}/{total_steps}")
