"""
Extreme domain-randomisation stress test for CartPole-v1.

Project 1's original held-out evaluation (10 combos within/near the DR
training range) showed both agents saturating at 500/500 reward -- a
ceiling effect, not real evidence either way about domain randomisation.

This script instead sweeps ONE physical parameter at a time, far outside
BOTH agents' training ranges, to find where each agent's performance
actually starts to degrade. Produces three line plots (reward vs
parameter value) -- one per parameter -- each showing the fixed and
domain-randomised agent side by side, with the DR training range shaded
so you can see whether the DR agent's advantage (if any) shows up
specifically outside that shaded band.

Run after evaluate.py (needs models/ppo_fixed.zip and
models/ppo_domain_rand.zip already saved):
    !python extreme_eval.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

N_EPISODES_PER_POINT = 5  # fewer than evaluate.py's 10 -- sweeps have far more points to cover

# Defaults for the two parameters NOT being swept in each sweep
DEFAULT_MASSCART = 1.0
DEFAULT_MASSPOLE = 0.1
DEFAULT_LENGTH = 0.5

# Sweep ranges -- deliberately far wider than the DR training ranges
# (masscart 0.5-1.5, masspole 0.05-0.2, length 0.25-0.75) so both agents
# are pushed well past anything either one saw during training.
LENGTH_SWEEP = np.round(np.arange(0.1, 3.01, 0.2), 2)
MASSCART_SWEEP = np.round(np.arange(0.1, 5.01, 0.3), 2)
MASSPOLE_SWEEP = np.round(np.arange(0.01, 1.01, 0.07), 2)


def set_params(env, masscart, masspole, length):
    u = env.unwrapped
    u.masscart = masscart
    u.masspole = masspole
    u.length = length
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length


def evaluate_agent(model, masscart, masspole, length, n_episodes=N_EPISODES_PER_POINT):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        set_params(env, masscart, masspole, length)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def run_sweep(fixed_model, dr_model, param_name, values):
    rows = []
    for v in values:
        if param_name == "length":
            masscart, masspole, length = DEFAULT_MASSCART, DEFAULT_MASSPOLE, v
        elif param_name == "masscart":
            masscart, masspole, length = v, DEFAULT_MASSPOLE, DEFAULT_LENGTH
        elif param_name == "masspole":
            masscart, masspole, length = DEFAULT_MASSCART, v, DEFAULT_LENGTH
        else:
            raise ValueError(param_name)

        fixed_mean, fixed_std = evaluate_agent(fixed_model, masscart, masspole, length)
        dr_mean, dr_std = evaluate_agent(dr_model, masscart, masspole, length)
        rows.append(
            {
                "value": v,
                "fixed_mean": fixed_mean,
                "fixed_std": fixed_std,
                "dr_mean": dr_mean,
                "dr_std": dr_std,
            }
        )
        print(f"{param_name}={v:6.2f}   fixed={fixed_mean:6.1f}   dr={dr_mean:6.1f}")
    return pd.DataFrame(rows)


def plot_sweep(df, title, xlabel, default_value, dr_range, ax):
    ax.plot(df["value"], df["fixed_mean"], marker="o", label="Fixed-parameter agent")
    ax.fill_between(
        df["value"],
        df["fixed_mean"] - df["fixed_std"],
        df["fixed_mean"] + df["fixed_std"],
        alpha=0.15,
    )
    ax.plot(df["value"], df["dr_mean"], marker="s", label="Domain-randomised agent")
    ax.fill_between(
        df["value"],
        df["dr_mean"] - df["dr_std"],
        df["dr_mean"] + df["dr_std"],
        alpha=0.15,
    )
    ax.axvline(default_value, color="gray", linestyle=":", linewidth=1)
    ax.axvspan(dr_range[0], dr_range[1], color="orange", alpha=0.08)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean episode reward")
    ax.set_title(title)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=8)


def main():
    fixed_model = PPO.load("models/ppo_fixed")
    dr_model = PPO.load("models/ppo_domain_rand")

    os.makedirs("results", exist_ok=True)

    print("\n--- Sweeping pole length (default=0.5, DR trained on 0.25-0.75) ---")
    length_df = run_sweep(fixed_model, dr_model, "length", LENGTH_SWEEP)
    length_df.to_csv("results/sweep_length.csv", index=False)

    print("\n--- Sweeping cart mass (default=1.0, DR trained on 0.5-1.5) ---")
    masscart_df = run_sweep(fixed_model, dr_model, "masscart", MASSCART_SWEEP)
    masscart_df.to_csv("results/sweep_masscart.csv", index=False)

    print("\n--- Sweeping pole mass (default=0.1, DR trained on 0.05-0.2) ---")
    masspole_df = run_sweep(fixed_model, dr_model, "masspole", MASSPOLE_SWEEP)
    masspole_df.to_csv("results/sweep_masspole.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_sweep(length_df, "Sweeping pole length", "Pole length", DEFAULT_LENGTH, (0.25, 0.75), axes[0])
    plot_sweep(masscart_df, "Sweeping cart mass", "Cart mass", DEFAULT_MASSCART, (0.5, 1.5), axes[1])
    plot_sweep(masspole_df, "Sweeping pole mass", "Pole mass", DEFAULT_MASSPOLE, (0.05, 0.2), axes[2])
    plt.tight_layout()
    plt.savefig("results/extreme_sweep_comparison.png", dpi=150)

    print("\nSaved results/sweep_length.csv, sweep_masscart.csv, sweep_masspole.csv")
    print("Saved results/extreme_sweep_comparison.png")


if __name__ == "__main__":
    main()