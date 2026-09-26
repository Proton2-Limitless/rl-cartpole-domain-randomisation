"""
Evaluates the fixed and domain-randomised agents from train_multiseed.py
(3 seeds each) on the same pole-length sweep as extreme_eval.py, to check
whether the crossover found in the single-seed run -- DR agent failing
around length ~1.1, fixed agent holding on to ~1.3 -- is a consistent
effect or an artifact of that one training run.

For each length value, every seed's mean reward is recorded separately,
then summarised as an across-seed mean and std (distinct from the
within-seed episode-to-episode noise). The plot shows each individual
seed as a faint line plus a bold line for the across-seed average, so
you can see at a glance whether the 3 seeds roughly agree or scatter.

Run after train_multiseed.py:
    !python multiseed_length_sweep.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

SEEDS = [0, 1, 2]
N_EPISODES_PER_POINT = 5
DEFAULT_MASSCART = 1.0
DEFAULT_MASSPOLE = 0.1
LENGTH_SWEEP = np.round(np.arange(0.1, 3.01, 0.2), 2)


def set_params(env, masscart, masspole, length):
    u = env.unwrapped
    u.masscart = masscart
    u.masspole = masspole
    u.length = length
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length


def evaluate_agent(model, length, n_episodes=N_EPISODES_PER_POINT):
    env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        set_params(env, DEFAULT_MASSCART, DEFAULT_MASSPOLE, length)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return float(np.mean(rewards))


def main():
    rows = []
    for seed in SEEDS:
        fixed_model = PPO.load(f"models/ppo_fixed_seed{seed}")
        dr_model = PPO.load(f"models/ppo_domain_rand_seed{seed}")
        for length in LENGTH_SWEEP:
            fixed_mean = evaluate_agent(fixed_model, length)
            dr_mean = evaluate_agent(dr_model, length)
            rows.append({"seed": seed, "length": length, "condition": "fixed", "mean_reward": fixed_mean})
            rows.append({"seed": seed, "length": length, "condition": "domain_rand", "mean_reward": dr_mean})
            print(f"seed={seed}  length={length:5.2f}  fixed={fixed_mean:6.1f}  dr={dr_mean:6.1f}")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/multiseed_length_sweep_raw.csv", index=False)

    summary = df.groupby(["condition", "length"])["mean_reward"].agg(["mean", "std"]).reset_index()
    summary.to_csv("results/multiseed_length_sweep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"fixed": "tab:blue", "domain_rand": "tab:orange"}
    labels = {"fixed": "Fixed-parameter agent", "domain_rand": "Domain-randomised agent"}

    for condition in ["fixed", "domain_rand"]:
        cond_df = df[df["condition"] == condition]
        for seed in SEEDS:
            seed_df = cond_df[cond_df["seed"] == seed].sort_values("length")
            ax.plot(
                seed_df["length"], seed_df["mean_reward"],
                color=colors[condition], alpha=0.25, linewidth=1,
            )
        cond_summary = summary[summary["condition"] == condition].sort_values("length")
        ax.plot(
            cond_summary["length"], cond_summary["mean"],
            color=colors[condition], linewidth=2.5, marker="o", label=labels[condition],
        )
        ax.fill_between(
            cond_summary["length"],
            cond_summary["mean"] - cond_summary["std"],
            cond_summary["mean"] + cond_summary["std"],
            color=colors[condition], alpha=0.15,
        )

    ax.axvspan(0.25, 0.75, color="gray", alpha=0.08, label="DR training range")
    ax.set_xlabel("Pole length")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Fixed vs Domain-Randomised PPO across 3 seeds each: pole-length sweep")
    ax.set_ylim(0, 520)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/multiseed_length_sweep.png", dpi=150)

    print("\nSaved results/multiseed_length_sweep_raw.csv")
    print("Saved results/multiseed_length_sweep_summary.csv")
    print("Saved results/multiseed_length_sweep.png")

    print("\n--- Crossover region (length 0.9-1.5) ---")
    print(summary[(summary["length"] >= 0.9) & (summary["length"] <= 1.5)].to_string(index=False))


if __name__ == "__main__":
    main()