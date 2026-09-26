# Research Report — Domain Randomisation on CartPole

**Author:** Habib Yusuf
**Date:** Sep 26
**Repo:** [\[GitHub link\]](https://github.com/Proton2-Limitless/rl-cartpole-domain-randomisation) · 
**Model:** [Hugging Face link]

## Abstract

We investigate whether training a PPO agent with domain randomisation over cart mass, pole mass, and pole length improves generalisation to unseen physical parameters, compared to an agent trained on a single fixed configuration. An initial evaluation on 10 held-out parameter combinations found both agents saturated at maximum reward — a ceiling effect indicating the test combinations were not extreme enough to be discriminating. A systematic one-parameter-at-a-time sweep, repeated across 3 independently trained seeds per condition, isolated pole length as the dominant factor in task difficulty and found that the domain-randomised agent maintains ceiling performance up to 47% further beyond its training range than the fixed agent does beyond its single training point, a pattern consistent across all 3 seeds.

## 1. Motivation

Sim-to-real transfer in robotics fails when a policy overfits to the exact physical constants of its training simulator. Domain randomisation — training across a distribution of simulated physical parameters rather than a single point estimate — is a standard mitigation. This experiment is a minimal, fast-to-run test of that idea using CartPole as a stand-in for a real robotic system, before scaling to continuous-control and manipulation tasks (Projects 2 and 3).

## 2. Method

### 2.1 Environment
`CartPole-v1` (Gymnasium). Default physical constants: cart mass = 1.0, pole mass = 0.1, pole length = 0.5.

### 2.2 Domain randomisation wrapper
A custom `gymnasium.Wrapper` resamples cart mass (uniform, 0.5–1.5), pole mass (uniform, 0.05–0.2), and pole length (uniform, 0.25–0.75) at the start of every episode, before the environment's `reset()` generates the initial state.

### 2.3 Agents
Both agents use PPO (Stable-Baselines3 `MlpPolicy`, default hyperparameters), trained for 200,000 timesteps. The **fixed agent** trains only on the default constants. The **domain-randomised agent** trains with the wrapper active on every episode.

### 2.4 Evaluation protocol
Three stages, each building on what the previous stage revealed:

1. **Held-out combination evaluation:** both trained agents evaluated on 10 hand-picked (masscart, masspole, length) combinations, 10 episodes each.
2. **Single-parameter extreme sweep:** each of the three physical parameters swept independently across a much wider range than either agent's training distribution, holding the other two at their defaults, 5 episodes per point, single seed per condition.
3. **Multi-seed confirmation:** the length sweep (the only parameter found to matter in stage 2) repeated across 3 independently trained seeds per condition, 5 episodes per point per seed, to separate a genuine effect from single-run training variance.

## 3. Results

### 3.1 Stage 1 — held-out combinations: ceiling effect

All 10 combinations produced 500.0 mean reward for both agents. No combination in this hand-picked set pushed pole length far enough outside the trained range to be discriminating.

### 3.2 Stage 2 — single-parameter sweep: length isolated as the key variable

Cart mass (swept 0.1–4.9) and pole mass (swept 0.01–0.99) produced flat 500.0 reward for both agents across the entire range — neither parameter meaningfully affects task difficulty at these scales. Pole length, swept 0.1–2.9, was the only parameter where performance varied, with a single-seed crossover suggesting (misleadingly, see 3.3) that the fixed agent generalised further than the domain-randomised agent.

### 3.3 Stage 3 — multi-seed length sweep: the confirmed result

| Pole length | Fixed agent (mean ± std, 3 seeds) | Domain-randomised agent (mean ± std, 3 seeds) |
|---|---|---|
| 0.10–0.70 | 500.0 ± 0.0 | 500.0 ± 0.0 |
| 0.90 | 340.5 ± 276.3 | 500.0 ± 0.0 |
| 1.10 | 35.9 ± 20.6 | 500.0 ± 0.0 |
| 1.30 | 17.4 ± 3.6 | 342.6 ± 272.6 |
| 1.50 | 16.1 ± 0.2 | 186.6 ± 271.7 |
| 1.70–2.90 | ~18–23 (floor) | ~18–33 (floor) |

At length 1.1, all 3 fixed-agent seeds have already collapsed (individual values: 14.0, 39.0, 54.8) while all 3 domain-randomised seeds remain at exactly 500.0. This reverses the impression given by the single-seed sweep in Stage 2, and does so consistently across every seed, not just on average.

[Insert `results/multiseed_length_sweep.png` here]

## 4. Discussion

The headline result is that the domain-randomised agent's usable operating range extends further past its own training boundary (0.75) than the fixed agent's does past its single training point (0.5) — the DR agent stays at ceiling until length 1.1 (a 47% extrapolation past 0.75), while the fixed agent starts failing by 0.9 (only an 80% deviation from 0.5, but from a single point rather than a range) and has clearly failed by 1.1. Neither agent was ever trained on lengths above 0.75, so this is a genuine extrapolation result, not an artifact of the DR agent simply having "seen" the test lengths during training.

A plausible mechanism: the domain-randomised agent, having had to solve the task across a distribution of pole lengths, likely learned a control policy with a wider margin of stability (e.g. more conservative, higher-gain corrections) that happens to transfer to lengths beyond its training distribution too. The fixed agent, optimised against a single dynamics configuration, has no such pressure and can specialise as tightly as PPO's optimisation allows — which turns out to generalise poorly.

The initial single-seed sweep (Stage 2) showed the *opposite* pattern, which underlines why the multi-seed confirmation mattered: a single training run's idiosyncrasies (parameter initialisation, exploration noise, the specific minibatches sampled) can produce a misleading result in either direction. Only after repeating with 3 seeds did a consistent, directionally-stable effect emerge.

## 5. Limitations

- 3 seeds per condition is a meaningful improvement over 1, but still a small sample — a claim about the exact crossover point (length ≈ 0.9–1.1) should be read as approximate, not precise
- Cart mass and pole mass showed no effect at the scales tested; it remains possible that even more extreme values, or joint variation of multiple parameters simultaneously (rather than one at a time), would reveal effects not visible in this design
- All evaluation combinations beyond length ≈ 1.7 collapse to floor performance for both agents — the task becomes fundamentally different physics at that point (the pole falls too fast to correct in one timestep), so this analysis cannot speak to arbitrarily extreme conditions, only the specific regime around each agent's training boundary

## 6. Conclusion and next steps

Despite an initial null result (Stage 1's ceiling effect) and a misleading single-seed result (Stage 2), a properly designed multi-seed sweep confirms that domain randomisation over pole length produces measurably better extrapolation beyond the training distribution than training on a single fixed configuration, specifically in the region just outside that distribution. This motivates Project 2, which moves to a continuous-action-space task (Pendulum, via SAC), where the same multi-seed, parameter-sweep evaluation methodology developed here will be reused directly.

## References
1. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," 2017.
2. Schulman et al., "Proximal Policy Optimization Algorithms," 2017.
3. [Add Sutton & Barto, and any advisor-relevant paper, e.g. Dr Samma's, once identified.]