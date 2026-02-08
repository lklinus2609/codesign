# PGHC Algorithm Verification Plan

This document outlines a systematic verification strategy for the Performance-Gated Hybrid Co-Design (PGHC) algorithm before deploying on the full G1 Humanoid.

## Executive Summary

**Problem**: Testing PGHC on a 30-DOF humanoid is slow (~hours per iteration) and debugging failures is nearly impossible---you cannot distinguish algorithm bugs from environment complexity.

**Solution**: A 4-level verification ladder, each level isolating specific algorithm components:

| Level | Environment | DOF | What It Tests | Time/Iter | Status |
|-------|-------------|-----|---------------|-----------|--------|
| 0 | Synthetic | 1 | Gradient & trust region correctness | ~1s | DONE |
| 1 | Cart-Pole | 2 | Envelope theorem validity | ~10s | DONE |
| 1.5 | Cart-Pole (Newton) | 2 | Newton/Warp integration, vectorized envs | ~5s | DONE |
| 2 | Ant | 27 | Multi-body dynamics, parametric morphology | ~30s | CREATED |
| 3 | G1 Humanoid | 37 | Full system | ~5min | BLOCKED |

**Key Insight**: If Level N fails, do not proceed to Level N+1. Each level is designed so that failure indicates a specific bug class.

---

## Level 0: Component Verification (No Physics)

### Purpose
Verify the mathematical correctness of individual components *before* introducing physics simulation complexity.

### 0A: Gradient Pipeline Verification

**Goal**: Confirm that BPTT gradients match finite-difference gradients.

**Setup**:
```python
# Synthetic differentiable function (no physics)
def synthetic_objective(theta):
    """Quadratic bowl: minimum at theta=0.5"""
    return (theta - 0.5)**2 + 0.1 * torch.sin(10 * theta)
```

**Test Procedure**:
1. Compute gradient via autograd: `grad_auto = torch.autograd.grad(obj, theta)`
2. Compute gradient via finite difference: `grad_fd = (f(theta+eps) - f(theta-eps)) / 2*eps`
3. Assert: `|grad_auto - grad_fd| / |grad_fd| < 0.01` (1% tolerance)

**Success Criterion**: Relative error < 1% for eps in {1e-4, 1e-5, 1e-6}

---

### 0B: Trust Region Mechanics

**Goal**: Verify the adaptive trust region accepts/rejects updates correctly.

**Test Cases**:

| Test | Initial theta | LR | Expected Behavior |
|------|---------------|----|--------------------|
| Accept good step | 0.0 | 0.1 | Accept, theta closer to optimum |
| Reject overshoot | 0.9 | 5.0 | Reject (overshoots), shrink LR |
| LR growth | 0.5 | 0.01 | Accept, small improvement, grow LR |
| At optimum | 1.0 | 0.1 | Accept (no change), stable |

**Success Criterion**: All test cases pass. After 100 iterations from theta=0, converges to theta=1.0 +/- 0.01.

---

### 0C: Quaternion Chain Rule Verification

**Goal**: Verify the quaternion parameterization gradients are correct.

**Success Criterion**: Analytical and autograd gradients match within 1e-6.

**Code**: `test_level0_verification.py`

---

## Level 1: Cart-Pole (PyTorch Physics)

### Purpose
Verify the envelope theorem approximation works when the policy converges. Cart-pole has known optimal solutions.

### Environment Specification

**State Space** (4D): `[x, theta, x_dot, theta_dot]`
**Action Space** (1D): Continuous force
**Morphology Parameter**: Pole length `L`

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Final L vs L* | Within 20% |
| Performance improvement | > 50% vs fixed suboptimal L |
| Gradient sign correct | 100% |

**Code**: `envs/cartpole/`, `codesign_cartpole.py`

---

## Level 1.5: Cart-Pole (Newton/Warp Vectorized)

### Purpose
Verify Newton/Warp integration works with vectorized GPU environments and finite-difference gradients.

### Environment Specification

**Physics**: Newton/Warp with MuJoCo solver
**Parallelism**: 2048 worlds on GPU
**Episode Length**: 500 steps (10 seconds at 50Hz)
**Force Limit**: +/-30 N
**Initial Angles**: +/-20 degrees
**Termination**: Cart out of bounds (|x| > 3.0m)

### PPO Configuration (Current)

| Parameter | Value |
|-----------|-------|
| Horizon | 32 steps |
| Mini-batches | 8 |
| PPO epochs | 5 |
| Clip ratio | 0.2 |
| Initial LR | 3e-4 |
| Max LR | 3e-4 |
| Desired KL | 0.005 |
| Entropy coeff | 0.005 |
| GAE lambda | 0.95 |
| Gamma | 0.99 |

### Reward Function

```
r = 1.0*(alive) - 2.0*(terminated) - theta^2 - 0.1*x^2
    - 0.01*action^2 - 0.1*(action - prev_action)^2
    - 0.01*|x_dot| - 0.005*|theta_dot|
```

### Convergence Criteria

**Inner loop**: Reward plateau (<1% relative change over window of 5 evaluations, sustained for 50 consecutive iterations, after 500 minimum iterations).

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Mean reward at convergence | > 450 |
| Mean episode length | 500 (full episode) |
| Reward std at convergence | < 30 |
| Outer loop triggers | Morphology updates visible |

**Code**: `envs/cartpole_newton/`, `codesign_cartpole_newton_vec.py`

---

## Level 2: Ant (Multi-Body Locomotion)

### Purpose
Test PGHC on a multi-body 3D locomotion task with parametric morphology using Newton/Warp physics.

### Environment Specification

**Morphology Parameters** (3D):
| Parameter | Range | Default |
|-----------|-------|---------|
| leg_length | [0.15, 0.4] | 0.28 |
| foot_length | [0.3, 0.7] | 0.57 |
| torso_radius | [0.15, 0.35] | 0.25 |

**State Space**: 27D
**Action Space**: 8 joint torques
**Objective**: Forward velocity + healthy bonus - control cost

### Success Criteria

| Metric | Threshold |
|--------|-----------|
| Environment steps without crash | 500+ |
| Finite gradients | 100% |
| Morphology change | Measurable |
| Performance vs baseline | >= baseline |

**Code**: `envs/ant/`, `codesign_ant.py`, `test_level2_ant.py`

---

## Level 3: G1 Humanoid (Target)

### Purpose
Full system validation on the target robot.

### Only Proceed Here If

- [x] Level 0: All gradient and trust region tests pass
- [x] Level 1: Cart-pole finds near-optimal length with PGHC
- [x] Level 1.5: Newton/Warp cart-pole vectorized environment works
- [ ] Level 2: Ant achieves stable locomotion with co-design

### Expected Challenges

| Challenge | Mitigation |
|-----------|------------|
| Slow iteration time | Reduce num_envs initially, add checkpointing |
| Frequent falls | Increase stability_threshold, more warmup iterations |
| High-dimensional morphology | Start with hip angles only (current setup) |
| Mode collapse | Multiple random seeds, curriculum learning |
| Gradient chain broken by numpy | Implement quaternion math as Warp kernels (see G1_DIFFERENTIABILITY.md) |

---

## Implementation Checklist

### Level 0: Math Verification --- DONE
- [x] Gradient pipeline verification (autograd vs finite difference)
- [x] Trust region mechanics (accept/reject logic)
- [x] Quaternion chain rule
- Code: `test_level0_verification.py`

### Level 1: Cart-Pole (PyTorch physics) --- DONE
- [x] Implement `ParametricCartPole` and `CartPoleEnv`
- [x] Implement PPO inner loop
- [x] Verify envelope theorem (autograd vs FD gradients match)
- [x] Run PGHC co-design, verify L converges toward optimum
- Code: `envs/cartpole/`, `codesign_cartpole.py`

### Level 1.5: Cart-Pole (Newton/Warp) --- DONE
- [x] Implement vectorized Newton cart-pole environment
- [x] Verify PPO training with vectorized envs
- [x] RSL-RL style episode tracking (no separate eval)
- [x] Reward plateau convergence detection
- [x] Conservative PPO (max LR 3e-4, desired_kl 0.005, 8 mini-batches)
- [x] Run PGHC with finite-difference gradients
- [x] Video recording with subprocess (OpenGL workaround)
- Code: `envs/cartpole_newton/`, `codesign_cartpole_newton_vec.py`

### Level 2: Ant (Newton/Warp) --- CREATED, NOT VALIDATED
- [x] Implement `ParametricAnt` with MJCF generation
- [x] Implement `AntEnv` with Newton physics
- [x] Implement finite-difference gradient computation
- [x] Write co-design script (`codesign_ant.py`)
- [ ] Run and validate PGHC on Ant (requires Newton GPU machine)
- Code: `envs/ant/`, `codesign_ant.py`, `test_level2_ant.py`

### Level 3: G1 Humanoid --- BLOCKED
- [x] Implement `ParametricG1Model` with joint angle parameterization
- [x] Implement `HybridAMPAgent` with outer loop
- [x] Implement differentiable rollout infrastructure
- [ ] Fix gradient chain (numpy breaks backprop --- see G1_DIFFERENTIABILITY.md)
- [ ] Run and validate PGHC on G1
- Code: `parametric_g1.py`, `hybrid_agent.py`, `train_hybrid.py`

---

## Appendix A: Null Test Protocol

**Purpose**: Verify the algorithm doesn't change morphology when it shouldn't.

**Setup**:
1. Use an environment where the default morphology is known to be optimal
2. Initialize at the optimal point
3. Run co-design

**Expected**:
- Gradients should be near-zero at optimum
- Morphology should stay within epsilon of initial
- No performance degradation

---

## Appendix B: Hyperparameter Reference

### Inner Loop (Policy Learning)
| Parameter | Cart-Pole (L1) | Cart-Pole Newton (L1.5) | Ant (L2) | G1 Humanoid (L3) |
|-----------|-----------------|--------------------------|----------|-------------------|
| `learning_rate` | 3e-4 | 3e-4 (max) | 3e-4 | 5e-5 |
| `num_envs` | 1 | 2048 | 1 | 2048 |
| `ppo_epochs` | 10 | 5 | 4 | 5 |
| `mini_batches` | - | 8 | 4 | 4 |
| `desired_kl` | - | 0.005 | 0.01 | 0.01 |
| `horizon` | 2048 | 32 | - | - |

### Outer Loop (Morphology)
| Parameter | Current | Tune At Level |
|-----------|---------|---------------|
| `design_lr` | 0.02 | Level 2 |
| `max_step` | 0.01 | Level 2 |
| `convergence` | Reward plateau (1%) | Level 1.5 |
| `diff_horizon` | 3 | Level 3 |

---

## Summary: Decision Tree

```
Level 0: Math Verification ------------ DONE
    |
Level 1: Cart-Pole (PyTorch) ---------- DONE
    |   Envelope theorem validated, PGHC converges
    |
Level 1.5: Cart-Pole (Newton/Warp) ---- DONE
    |   Vectorized envs, finite-diff gradients, RSL-RL tracking
    |
Level 2: Ant (Newton/Warp) ------------ YOU ARE HERE
    |   Created but not validated on GPU
    |   Need: Run codesign_ant.py on Newton machine
    |
Level 3: G1 Humanoid ------------------ BLOCKED
    |   Gradient chain broken by numpy conversions
    |   Need: Implement Warp kernels for quaternion math
    |
    └── Ready for thesis experiments
```

---

*Document Version: 3.0*
*Last Updated: 2026-02-08*
