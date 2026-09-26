"""
Loads both trained agents and evaluates each on 10 held-out (masscart,
masspole, length) combinations. Produces results/eval_results.csv and
results/comparison_bar_chart.png.

Run after both train_fixed.py and train_domain_rand.py have completed:
    !python evaluate.py

Must be in the same folder as cartpole_dr_wrapper.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO

N_EPISODES_PER_COMBO = 10

# 10 held-out (masscart, masspole, length) combinations. Most sit inside
# the domain-randomised agent's training ranges (masscart 0.5-1.5,
# masspole 0.05-0.2, length 0.25-0.75); combo #9 is deliberately outside
# all three ranges, to check extrapolation rather than just interpolation.
HELD_OUT_COMBOS = [
    (0.60, 0.06, 0.30),
    (0.80, 0.08, 0.40),
    (1.00, 0.10, 0.50),  # matches CartPole-v1 defaults
    (1.20, 0.12, 0.60),
    (1.40, 0.14, 0.70),
    (0.55, 0.18, 0.35),
    (1.45, 0.07, 0.65),
    (0.50, 0.20, 0.25),
    (1.50, 0.05, 0.75),
    (0.30, 0.25, 0.90),  # outside the DR training range
]


def set_params(env, masscart, masspole, length):
    u = env.unwrapped
    u.masscart = masscart
    u.masspole = masspole
    u.length = length
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length


def evaluate_agent(model, combo, n_episodes=N_EPISODES_PER_COMBO):
    masscart, masspole, length = combo
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


def main():
    fixed_model = PPO.load("models/ppo_fixed")
    dr_model = PPO.load("models/ppo_domain_rand")

    rows = []
    for i, combo in enumerate(HELD_OUT_COMBOS):
        fixed_mean, fixed_std = evaluate_agent(fixed_model, combo)
        dr_mean, dr_std = evaluate_agent(dr_model, combo)
        rows.append(
            {
                "combo_id": i,
                "masscart": combo[0],
                "masspole": combo[1],
                "length": combo[2],
                "fixed_mean_reward": fixed_mean,
                "fixed_std_reward": fixed_std,
                "dr_mean_reward": dr_mean,
                "dr_std_reward": dr_std,
            }
        )
        print(f"Combo {i}: fixed={fixed_mean:6.1f}   dr={dr_mean:6.1f}")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)

    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        df["fixed_mean_reward"],
        width,
        yerr=df["fixed_std_reward"],
        label="Fixed-parameter agent",
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        df["dr_mean_reward"],
        width,
        yerr=df["dr_std_reward"],
        label="Domain-randomised agent",
        capsize=4,
    )
    ax.set_xlabel("Held-out parameter combination")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Fixed vs Domain-Randomised PPO on CartPole: Held-Out Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i}" for i in df["combo_id"]])
    ax.legend()
    ax.set_ylim(0, 520)
    plt.tight_layout()
    plt.savefig("results/comparison_bar_chart.png", dpi=150)

    print("\nSaved results/eval_results.csv and results/comparison_bar_chart.png")
    print(
        f"\nOverall mean — fixed: {df['fixed_mean_reward'].mean():.1f}, "
        f"domain-randomised: {df['dr_mean_reward'].mean():.1f}"
    )


if __name__ == "__main__":
    main()
