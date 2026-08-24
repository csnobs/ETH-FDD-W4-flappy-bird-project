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

### Implementation Decisions

#### Task 1 — `build_observation`

- Use `obs_mode = "full"` with `n_preview` pipes ahead (not the minimal `(v, dy0)` baseline)
- Bird height `y` is **re-centred** on mid-grid `(C.Y-1)/2` before scaling — so 0 = middle, helping the network treat up/down symmetrically
- All features divided by fixed `scales(C)` constants (no running normaliser — avoids coupling runs to their own history)
- **`active` flag is essential**: without it, empty pipe slots (sentinel `dx = X`) are indistinguishable from genuinely distant obstacles
- Observation dim: `2 + 3 × n_preview` (y, v, then dx/dy/active per pipe)

```python
def _build_observation(state, C, cfg):
    s = scales(C)
    y = (state["y"] - (C.Y - 1) / 2) / s["y"]   # re-centred height
    v = state["v"] / s["v"]                        # scaled velocity

    if cfg.obs_mode == "minimal":
        dy0 = state["dy"][:, 0] / s["dy"]
        return np.stack([v, dy0], axis=1).astype(np.float32)

    if cfg.obs_mode != "full":
        raise ValueError(f"unknown observation mode {cfg.obs_mode!r}")

    P = min(cfg.n_preview, C.M)
    cols = [y, v]
    for i in range(P):
        cols.append(state["dx"][:, i] / s["dx"])          # distance to pipe i
        cols.append(state["dy"][:, i] / s["dy"])          # gap offset for pipe i
        cols.append(state["active"][:, i].astype(float))  # activity flag
    return np.stack(cols, axis=1).astype(np.float32)
```

#### Task 2 — `compute_gae`

- Implemented as a **backward loop** (t = T-1 → 0) because `A_t` depends on `A_{t+1}`
- The `(1 - done_t)` mask appears **twice**: once in the TD error (zeroes out next-state value on crash), once in the accumulation (prevents credit leaking across episode boundaries)
- Returns = advantages + values (critic regression target = full expected return, not just advantage)
- λ controls the "memory fade" — calibrated analytically in C3

```python
def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float64)
    last_gae   = np.zeros(N,      dtype=np.float64)

    for t in reversed(range(T)):
        next_value   = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta        = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae     = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns
```

#### Task 3 — `ppo_loss`

- Ratio ρ = `exp(log π_new - log π_old)` computed in log-space for numerical stability
- Pessimistic clipping: `torch.max(pg_1, pg_2)` — both terms carry a minus sign, so `max` picks the worse (less beneficial) branch, enforcing the leash
- Value loss: `0.5 * MSE(V, returns)` — the `0.5` matches the gradient scale convention
- Entropy bonus subtracted (rewarded) to prevent premature policy collapse
- KL estimator: Schulman's `mean((ρ - 1) - log ρ)` — always ≥ 0, lower variance than `-log ρ`; used in C3 to set learning rate

```python
def _ppo_loss(old_logprob, new_logprob, advantages, returns,
              new_value, entropy, cfg):
    log_ratio = new_logprob - old_logprob
    ratio     = log_ratio.exp()                                        # ρ = π_new/π_old

    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_1    = -advantages * ratio
    pg_2    = -advantages * ratio.clamp(1 - cfg.clip_coef,            # clipped branch
                                         1 + cfg.clip_coef)
    pg_loss = torch.max(pg_1, pg_2).mean()                            # pessimistic branch

    v_loss  = 0.5 * ((new_value - returns) ** 2).mean()               # critic MSE
    ent     = entropy.mean()
    loss    = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent     # combined loss

    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()           # Schulman's KL
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

    return loss, {
        "policy_loss":    pg_loss.item(),
        "value_loss":     v_loss.item(),
        "entropy":        ent.item(),
        "approx_kl":      approx_kl,
        "clip_fraction":  clip_frac,
    }
```

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
