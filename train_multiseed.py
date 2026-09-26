"""
Trains fixed and domain-randomised PPO agents on CartPole-v1 across
multiple random seeds, to check whether the length-sweep crossover found
in extreme_eval.py (the DR agent failing earlier than the fixed agent,
around pole length ~1.1-1.3) holds up across seeds or was a one-off
artifact of that single training run.

Trains SEEDS x {fixed, domain_rand} = 6 models total, saved as
models/ppo_fixed_seed{N}.zip and models/ppo_domain_rand_seed{N}.zip.

Run in Colab (expect roughly 6x the wall-clock time of the original
single training run -- budget 20-30 minutes total on CPU):
    !python train_multiseed.py

Must be in the same folder as cartpole_dr_wrapper.py.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from cartpole_dr_wrapper import CartPoleDRWrapper

SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 200_000

os.makedirs("models", exist_ok=True)


def make_fixed_env():
    return Monitor(gym.make("CartPole-v1"))


def make_dr_env(seed):
    return Monitor(CartPoleDRWrapper(gym.make("CartPole-v1"), seed=seed))


def train_one(condition, seed):
    print(f"\n=== Training {condition} agent, seed={seed} ===")
    if condition == "fixed":
        env = make_vec_env(make_fixed_env, n_envs=4, seed=seed)
    else:
        env = make_vec_env(lambda: make_dr_env(seed), n_envs=4, seed=seed)

    model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    path = f"models/ppo_{condition}_seed{seed}"
    model.save(path)
    print(f"Saved {path}.zip")


def main():
    for seed in SEEDS:
        train_one("fixed", seed)
        train_one("domain_rand", seed)
    print("\nAll 6 models trained and saved.")


if __name__ == "__main__":
    main()