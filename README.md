# Project 1 — Domain Randomisation on CartPole

Does training a PPO agent with randomised physical parameters (cart mass, pole length, pole mass) produce a policy that generalises better to unseen physics than a policy trained on fixed parameters?

This is a small-scale, fast-to-run prototype of the core PhD thesis question: **does domain randomisation improve sim-to-real / cross-domain robustness?**

## Result summary
*(Fill in after running — keep this at the top so anyone opening the repo sees the headline immediately)*

- Fixed-parameter agent, evaluated on 10 held-out parameter combinations: **mean reward = ___**
- Domain-randomised agent, evaluated on the same 10 combinations: **mean reward = ___**
- See `results/comparison_bar_chart.png`

## What's in this repo

```
train_fixed.py            # trains PPO on default CartPole-v1
train_domain_rand.py      # trains PPO with a custom randomising wrapper
cartpole_dr_wrapper.py    # the Gymnasium wrapper that randomises mass/length each reset
evaluate.py                # runs both trained agents on 10 held-out parameter sets
results/                   # saved plots, csv of eval scores, training curves
models/                    # saved .zip policy files (SB3 format)
RESEARCH_REPORT.md         # short written report of method + findings
HF_MODEL_CARD.md           # card to paste into the Hugging Face repo README
```

## Reproduce it

```bash
pip install stable-baselines3 gymnasium torch
python train_fixed.py            # ~10 min on Colab free GPU
python train_domain_rand.py      # ~15 min
python evaluate.py               # produces results/comparison_bar_chart.png
```

## Method in one paragraph

Both agents are PPO (Stable-Baselines3 default hyperparameters) trained on `CartPole-v1`. The domain-randomised agent uses a wrapper that resamples cart mass, pole mass, and pole length from a uniform range at the start of every episode by writing directly into the underlying `env.unwrapped` physics attributes before `reset()` returns. The fixed agent trains on the environment's default constants throughout. Both agents are evaluated, after training, on 10 held-out parameter combinations neither agent was trained on directly (the randomised agent trains on a continuous range, so "held-out" here means specific point combinations not deliberately sampled during training).

## Status

- [ ] Fixed agent trained to ~500/500 reward
- [ ] Domain-randomisation wrapper implemented and unit-tested
- [ ] Randomised agent trained
- [ ] Both agents evaluated on 10 held-out configs
- [ ] Bar chart produced
- [ ] Pushed to GitHub with this README
- [ ] Model pushed to Hugging Face
- [ ] Feeds into Project 6 (arXiv note)

See `PHASE_IMPLEMENTATION.md` for the detailed step-by-step plan.
