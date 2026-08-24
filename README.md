# Flappy Bird with PPO — ETH Zurich CAS (FDD Week 4)

> *"Remember Flappy Bird? Same idea, but this time you teach the bird to fly — with PPO — on a clean little discrete-grid version of the game."*
> — Mattia De Martino & Ankita Ghosh, ETH Zurich, August 2026

## Project Overview

This project applies **Proximal Policy Optimization (PPO)** — a state-of-the-art reinforcement learning algorithm — to teach an agent ("Faby") to navigate a discrete-grid version of Flappy Bird. It is the Week 4 capstone of the ETH Zurich *From Data to Solutions* (FDD) CAS program.

The key insight: training Faby as a **single-degree-of-freedom vertical hopper** is a stepping stone toward training complex real-world systems like robot dogs (Spot) navigating uneven terrain.

---

## The Environment (Flappy MDP)

The game is modelled as a **Markov Decision Process**:

- **State**: `(y, v, d₁, h₁, ...)` — bird height, vertical velocity, distance and gap-centre of upcoming pipes
- **Actions** (discrete impulses):
  - `0` — No flap: gravity only (`u = 0`, cost λ = 0.0)
  - `1` — Weak flap: deterministic kick (`u = +2`, cost λ = 0.5)
  - `2` — Strong flap: stochastic kick (`u = +3 + ε`, ε ~ Uniform[-1,+1], cost λ = 0.7)
- **Reward per survived step**: `r = 1 − λ`
- **Terminal condition**: collision with a pipe or boundary
- **Dynamics**: `y ← clip(y + v)`, `v ← clip(v + u + noise − g)`

### Two Environments

| Instance | Grid | State Space | Purpose |
|----------|------|-------------|---------|
| **Small** | Coarse | Enumerable | Validate against true mathematical optimum (Value Iteration) |
| **Large** | Fine | ~10¹⁷ states | Real submission — function approximation (PPO) required |

---

## RL Concepts Used

| Concept | Role in this project |
|---------|---------------------|
| Policy Gradient | Foundation of the learning objective |
| Actor-Critic | Separate policy network (actor) and value network (critic) |
| GAE (λ) | Smooth advantage estimates across time to reduce variance |
| PPO Clipped Surrogate | Prevents destructively large policy updates |
| Domain Randomization | (Bonus) Trains robust policies via randomized physics |

---

## Project Tasks

### Implementation (PPO core)
- **Task 1** — `build_observation`: Select and scale state features for the neural network
- **Task 2** — `compute_gae`: Implement Generalized Advantage Estimation
- **Task 3** — `ppo_loss`: Implement the clipped PPO surrogate objective

### Calibration (computed, not guessed)
- **C1** — Discount factor `γ`: derived from mean episode length / delta
- **C2** — Rollout horizon `T` and number of environments `N`: constrained by `T > 2·Δ`
- **C3** — GAE lambda `λ`, learning rate `LR`, update epochs: read from KL diagnostics

### Evaluation
- Train on **small** grid → compare policy to V* (true optimum via value iteration)
- Train on **large** grid → final submission
- **Tuning table**: up to 20 structured runs, hypothesis-first
- **(Bonus)** Domain Randomization ablation

---

## Findings & Decisions

> *Updated as we work through the project.*

### Key Observations (pre-training)
- **"Never flap" outperforms random and "always flap"**: The effort cost (λ) is paid immediately, while survival reward accumulates — so flapping carelessly is worse than waiting.
- The **flap rate** is the single most useful diagnostic during training: too high → wasted energy; too low → crashes from gravity.

---

## Hyperparameters

> *To be filled in after calibration (C1–C3).*

| Parameter | Value | Justification |
|-----------|-------|---------------|
| γ (gamma) | TBD | |
| T (rollout) | TBD | |
| N (envs) | TBD | |
| λ (GAE) | TBD | |
| LR | TBD | |
| Epochs | TBD | |

---

## Results

> *To be filled in after training runs.*

| Run | Instance | Seeds | Mean Return | vs Optimal |
|-----|----------|-------|-------------|------------|
| Baseline | small | 3 | TBD | TBD |
| Final | large | 3 | TBD | TBD |

---

## Repository Structure

```
ETH-FDD-W4-flappy-bird-project/
├── w4_flappy_bird_project_student_cn.ipynb   # Main project notebook
├── FDD26-W4-FlappyBirdProject-cn.pdf         # Project brief (slides)
└── README.md                                  # This file
```

---

## References

- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2015) — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- ETH Zurich FDD Course, Week 4, August 2026
