# Research Report — Domain Randomisation on CartPole

**Author:** [your name]
**Date:** [date]
**Repo:** [GitHub link] · **Model:** [Hugging Face link]

## Abstract
*(3–5 sentences, write this last)* We investigate whether training a PPO agent with domain randomisation over cart mass, pole mass, and pole length improves generalisation to unseen physical parameter combinations, compared to an agent trained on a single fixed configuration. [One-sentence result.]

## 1. Motivation
Sim-to-real transfer in robotics fails when a policy overfits to the exact physical constants of its training simulator. Domain randomisation — training across a distribution of simulated physical parameters rather than a single point estimate — is a standard mitigation. This experiment is a minimal, fast-to-run test of that idea using CartPole as a stand-in for a real robotic system, before scaling to continuous-control and manipulation tasks.

## 2. Method

### 2.1 Environment
`CartPole-v1` (Gymnasium). Default physical constants: cart mass = 1.0, pole mass = 0.1, pole length = 0.5.

### 2.2 Domain randomisation wrapper
[Describe your actual sampled ranges here once implemented.] At the start of every episode, cart mass, pole mass, and pole length are resampled uniformly from [range] before the episode begins.

### 2.3 Agents
Both agents use PPO (Stable-Baselines3), [hyperparameters — fill in: learning rate, n_steps, batch size, total timesteps]. The **fixed agent** trains only on the default constants. The **domain-randomised agent** trains with the wrapper active on every episode.

### 2.4 Evaluation protocol
Both trained agents are evaluated on 10 held-out parameter combinations not used during the fixed agent's training, [N] episodes per combination, reporting mean episode reward.

## 3. Results

[Insert `comparison_bar_chart.png` here]

| Combination | Cart mass | Pole mass | Pole length | Fixed agent (mean reward) | DR agent (mean reward) |
|---|---|---|---|---|---|
| 1 | | | | | |
| ... | | | | | |

**Overall:** fixed agent mean = ___, DR agent mean = ___.

## 4. Discussion

[Interpret the numbers honestly. If DR wins: by how much, and does the margin grow for combinations further from the default? If DR does *not* clearly win: is this plausible (CartPole is a very easy, low-dimensional task — it may not be sensitive enough to physics variation to show a strong DR benefit; note this as a limitation and a reason Project 2/3 move to harder tasks), and what would you change next time (wider randomisation range, more training timesteps, measuring robustness rather than raw reward)?]

## 5. Limitations
- CartPole is a low-dimensional, largely solved task — a ceiling effect (both agents near-maximal reward) is possible and would understate any real DR benefit
- Single random seed per agent [or note if you used multiple seeds — strongly recommended if time allows]
- Held-out combinations were hand-picked, not randomly sampled — some selection bias is possible

## 6. Conclusion and next steps
[One paragraph.] This result directly motivates Project 2, which moves to a continuous-action-space task (Pendulum, via SAC) and Project 3 (FetchReach manipulation), where the effect of domain randomisation is expected to be more pronounced.

## References
1. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," 2017.
2. Schulman et al., "Proximal Policy Optimization Algorithms," 2017.
3. [Add Sutton & Barto, and any advisor-relevant paper, e.g. Dr Samma's, once identified.]
