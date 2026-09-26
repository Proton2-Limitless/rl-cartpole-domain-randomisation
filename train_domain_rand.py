"""
Trains PPO on CartPole-v1 with domain randomisation: cart mass, pole mass,
and pole length are resampled every episode via CartPoleDRWrapper.

Run in Colab:
    !pip install stable-baselines3 gymnasium
    !python train_domain_rand.py

Must be in the same folder as cartpole_dr_wrapper.py.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from cartpole_dr_wrapper import CartPoleDRWrapper

TOTAL_TIMESTEPS = 200_000
MODEL_PATH = "models/ppo_domain_rand"
LOG_PATH = "logs/domain_rand"

os.makedirs("models", exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


def make_env():
    return Monitor(CartPoleDRWrapper(gym.make("CartPole-v1")))


def main():
    env = make_vec_env(make_env, n_envs=4)
    eval_env = make_vec_env(make_env, n_envs=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{LOG_PATH}/best_model",
        log_path=LOG_PATH,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_PATH)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    model.save(MODEL_PATH)
    print(f"\nSaved domain-randomised model to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
