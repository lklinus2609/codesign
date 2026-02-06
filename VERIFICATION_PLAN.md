# PGHC Algorithm Verification Plan

This document outlines a systematic verification strategy for the Performance-Gated Hybrid Co-Design (PGHC) algorithm before deploying on the full G1 Humanoid.

## Executive Summary

**Problem**: Testing PGHC on a 30-DOF humanoid is slow (~hours per iteration) and debugging failures is nearly impossible—you cannot distinguish algorithm bugs from environment complexity.

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
2. Compute gradient via finite difference: `grad_fd = (f(θ+ε) - f(θ-ε)) / 2ε`
3. Assert: `|grad_auto - grad_fd| / |grad_fd| < 0.01` (1% tolerance)

**What This Tests**:
- Autograd tape recording works correctly
- Gradient extraction logic is correct
- No sign errors or scaling bugs

**Success Criterion**:
- Relative error < 1% for ε ∈ {1e-4, 1e-5, 1e-6}

**Failure Modes**:
- Large error → Tape not recording properly, or wrong variable being differentiated
- Sign flip → Bug in gradient extraction or loss definition
- NaN → Division by zero or unstable computation

---

### 0B: Trust Region Mechanics

**Goal**: Verify the adaptive trust region accepts/rejects updates correctly.

**Setup**:
```python
# Known landscape with predictable behavior
def trust_region_test_objective(theta, noise_scale=0.0):
    """
    Quadratic with known minimum at theta=1.0
    Add optional noise to simulate stochastic evaluation
    """
    return (theta - 1.0)**2 + noise_scale * torch.randn(1)
```

**Test Cases**:

| Test | Initial θ | LR | Expected Behavior |
|------|-----------|----|--------------------|
| Accept good step | 0.0 | 0.1 | Accept, θ → 0.2 (closer to 1.0) |
| Reject overshoot | 0.9 | 5.0 | Reject (overshoots past 1.0), shrink LR |
| LR growth | 0.5 | 0.01 | Accept, small improvement → grow LR |
| At optimum | 1.0 | 0.1 | Accept (no change), stable |

**What This Tests**:
- Trust region violation detection (`D < -ξ * |J|`)
- Learning rate decay on rejection
- Learning rate growth on small improvement
- Boundary clamping behavior

**Success Criterion**:
- All test cases pass
- After 100 iterations from θ=0, converges to θ=1.0 ± 0.01

**Failure Modes**:
- Never accepts → ξ too small or bug in violation check
- Always accepts (even bad steps) → ξ too large
- Oscillates without converging → LR adaptation logic broken

---

### 0C: Quaternion Chain Rule Verification

**Goal**: Verify the quaternion parameterization gradients are correct.

**Setup**:
```python
def quaternion_from_angle(theta):
    """X-axis rotation: q = (cos(θ/2), sin(θ/2), 0, 0)"""
    return torch.stack([
        torch.cos(theta / 2),
        torch.sin(theta / 2),
        torch.zeros_like(theta),
        torch.zeros_like(theta)
    ])

def objective_through_quat(theta):
    """Some function of the quaternion"""
    q = quaternion_from_angle(theta)
    # Example: squared norm of rotated vector
    return (q[1]**2 + q[2]**2 + q[3]**2)  # sin²(θ/2)
```

**Test Procedure**:
1. Analytical gradient: `d/dθ sin²(θ/2) = sin(θ/2)cos(θ/2) = sin(θ)/2`
2. Autograd gradient: `torch.autograd.grad(obj, theta)`
3. Assert they match

**What This Tests**:
- Quaternion construction is correct
- Chain rule through quaternion parameterization works
- Matches the manual gradient extraction in `_extract_design_gradient()`

**Success Criterion**:
- Analytical and autograd gradients match within 1e-6

---

## Level 1: Inverted Pendulum (1D Analytical Baseline)

### Purpose
Verify the envelope theorem approximation works when the policy converges. Pendulum has analytical optimal solutions.

### Environment Specification

**State Space** (4D):
- `θ`: Pole angle from vertical (rad)
- `θ_dot`: Angular velocity (rad/s)
- `x`: Cart position (m) — if using cart-pole variant
- `x_dot`: Cart velocity (m/s)

**Action Space** (1D):
- Continuous torque or force

**Morphology Parameter**:
- **Option A**: Pole length `L ∈ [0.3, 1.5]` meters
- **Option B**: Pole mass `m ∈ [0.1, 1.0]` kg
- Recommend Option A (length) — more intuitive

### Analytical Baseline

For swing-up + balance, the optimal pole length depends on:
1. **Actuator torque limit** τ_max
2. **Control bandwidth** (how fast the controller can react)

**Approximate optimal length**:
```
L* ≈ (3 * τ_max) / (m * g)  [for swing-up with limited torque]
```

For a torque limit of τ_max = 2 Nm and mass m = 0.5 kg:
```
L* ≈ (3 * 2) / (0.5 * 9.81) ≈ 1.22 m
```

### Test Procedure

**Phase 1: Policy Training Only (No Co-Design)**
1. Fix L = 0.5m (suboptimal, too short for easy swing-up)
2. Train PPO policy until convergence
3. Record final performance (upright time, total reward)

**Phase 2: Enable Co-Design**
1. Initialize L = 0.5m
2. Enable outer loop with stability gating
3. Run until outer loop triggers 10+ times
4. Record trajectory of L over outer loop iterations

**Expected Behavior**:
- L should increase toward the analytical optimum (~1.2m)
- Performance should improve monotonically (with trust region)
- Gradient should point toward longer pole initially

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Final L vs L* | Within 20% | `|L_final - L*| / L* < 0.2` |
| Performance improvement | > 50% | vs fixed suboptimal L |
| Gradient sign correct | 100% | When L < L*, gradient should be positive |
| No trust region violations after warmup | < 10% | After first 5 outer loops |

### Failure Analysis

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| L doesn't change | Gradients are zero | Check tape recording, verify L has requires_grad |
| L goes wrong direction | Sign error in gradient | Check loss definition (minimize vs maximize) |
| L oscillates wildly | Trust region too loose | Decrease ξ or increase decay factor |
| Performance degrades | Envelope theorem violated | Policy not converged, increase min_iters |
| Converges to wrong L | Local minimum | Try multiple initializations |

### Files to Create

```
codesign/
├── envs/
│   └── pendulum/
│       ├── parametric_pendulum.py   # ParametricPendulumModel
│       ├── pendulum_env.py          # Gym-style environment
│       └── pendulum_physics.py      # Differentiable dynamics (Newton or custom)
├── config/
│   ├── pendulum_agent.yaml          # Simplified agent config (no AMP)
│   └── pendulum_env.yaml            # Environment parameters
└── scripts/
    └── verify_pendulum.py           # End-to-end verification script
```

---

## Level 2: Ant (Multi-Body Locomotion)

### Purpose
Test PGHC on a multi-body 3D locomotion task with parametric morphology using Newton/Warp physics. Ant has 4 legs with 8 actuated joints, providing a meaningful step up from cart-pole.

### Environment Specification

**Morphology Parameters** (3D):
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| leg_length | [0.15, 0.4] | 0.28 | Upper leg segment length |
| foot_length | [0.3, 0.7] | 0.57 | Lower leg segment length |
| torso_radius | [0.15, 0.35] | 0.25 | Torso sphere radius |

**State Space**: 27D (z, quaternion(4), joints(8), velocities(6), joint_velocities(8))
**Action Space**: 8 joint torques (4 hips + 4 ankles), range [-1, 1]
**Objective**: Forward velocity + healthy bonus - control cost

### Why Ant Tests Multi-Body Co-Design

The Ant is **complex enough to validate the full pipeline** while being forgiving:
- 4-legged → inherently stable (wide support polygon)
- 8 actuated joints → exercises parametric MJCF generation
- Morphology changes require model rebuild → tests finite difference pipeline
- Reward is forward velocity → clear optimization signal

### Test Procedure

**Phase 1: Environment Validation**
1. Verify MJCF generation with different parameters
2. Step environment with random actions, check stability
3. Verify observation/action dimensions

**Phase 2: Gradient Computation**
1. Compute dReturn/d(leg_length) via finite differences
2. Compute dReturn/d(foot_length) via finite differences
3. Verify gradients are finite and reasonable

**Phase 3: Full Co-Design**
1. Initialize with default morphology
2. Run PGHC with PPO inner loop
3. Observe morphology trajectory and performance

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Environment steps without crash | 500+ | Basic stability |
| Finite gradients | 100% | No NaN/Inf in gradient computation |
| Morphology change | Measurable | Should move from initial |
| Performance vs baseline | >= baseline | At least no worse |

### Implementation

- **Environment**: `envs/ant/ant_env.py` - Newton-based with MJCF generation
- **Co-design script**: `codesign_ant.py` - PGHC with PPO inner loop
- **Tests**: `test_level2_ant.py` - Environment and gradient verification

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

### Level 0: Math Verification — DONE
- [x] Gradient pipeline verification (autograd vs finite difference)
- [x] Trust region mechanics (accept/reject logic)
- [x] Quaternion chain rule
- Code: `test_level0_verification.py`

### Level 1: Cart-Pole (PyTorch physics) — DONE
- [x] Implement `ParametricCartPole` and `CartPoleEnv`
- [x] Implement PPO inner loop
- [x] Verify envelope theorem (autograd vs FD gradients match)
- [x] Run PGHC co-design, verify L converges toward optimum
- Code: `envs/cartpole/`, `codesign_cartpole.py`, `codesign_cartpole_simple.py`

### Level 1.5: Cart-Pole (Newton/Warp) — DONE
- [x] Implement vectorized Newton cart-pole environment
- [x] Verify PPO training with vectorized envs
- [x] Run PGHC with finite-difference gradients
- [x] Video recording with subprocess (OpenGL workaround)
- Code: `envs/cartpole_newton/`, `codesign_cartpole_newton_vec.py`

### Level 2: Ant (Newton/Warp) — CREATED, NOT VALIDATED
- [x] Implement `ParametricAnt` with MJCF generation
- [x] Implement `AntEnv` with Newton physics
- [x] Implement finite-difference gradient computation
- [x] Write co-design script (`codesign_ant.py`)
- [ ] Run and validate PGHC on Ant (requires Newton GPU machine)
- Code: `envs/ant/`, `codesign_ant.py`, `test_level2_ant.py`

### Level 3: G1 Humanoid — BLOCKED
- [x] Implement `ParametricG1Model` with joint angle parameterization
- [x] Implement `HybridAMPAgent` with outer loop
- [x] Implement differentiable rollout infrastructure
- [ ] Fix gradient chain (numpy breaks backprop — see G1_DIFFERENTIABILITY.md)
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
- Morphology should stay within ε of initial
- No performance degradation

**If This Fails**:
- Algorithm has bias toward certain morphologies
- Noise in gradient computation causes drift
- Need regularization toward initial morphology

---

## Appendix B: Minimal PPO Agent for Testing

For Levels 1-3, you don't need the full AMP machinery. Create a minimal agent:

```python
class MinimalCoDesignAgent:
    """Stripped-down agent for algorithm verification."""

    def __init__(self, env, config):
        self.env = env
        self.policy = SimplePPO(env.obs_dim, env.act_dim)
        self.parametric_model = env.parametric_model

        # Outer loop components (reuse from HybridAMPAgentBase)
        self.design_lr = config.design_lr
        self.trust_region_threshold = config.xi
        # ... etc

    def train_inner(self, num_iters):
        """Standard PPO training."""
        for _ in range(num_iters):
            trajectories = self.collect_rollouts()
            self.policy.update(trajectories)

    def outer_loop_update(self):
        """Reuse logic from HybridAMPAgentBase."""
        # Copy-paste _outer_loop_update() from hybrid_agent.py
        pass
```

This avoids MimicKit dependency while reusing your outer loop logic.

---

## Appendix C: Hyperparameter Reference

### Inner Loop (Policy Learning)
| Parameter | Cart-Pole | Cart-Pole (Newton) | Ant | G1 Humanoid |
|-----------|-----------|---------------------|-----|-------------|
| `learning_rate` | 3e-4 | 3e-4 | 3e-4 | 5e-5 |
| `num_envs` | 1 | 1024 | 1 | 2048 |
| `ppo_epochs` | 10 | 4 | 4 | 5 |
| `min_iters_for_stability` | 50 | 20 | 30 | 5000 |

### Outer Loop (Morphology)
| Parameter | Start | Tune At Level |
|-----------|-------|---------------|
| `design_lr_init` | 0.01 | Level 2 |
| `trust_region_threshold` | 0.1 | Level 2 |
| `lr_decay_factor` | 0.5 | Level 2 |
| `diff_horizon` | 3 | Level 3 |

---

## Summary: Decision Tree

```
Level 0: Math Verification ──────────── DONE ✓
    │
Level 1: Cart-Pole (PyTorch) ────────── DONE ✓
    │   Envelope theorem validated, PGHC converges
    │
Level 1.5: Cart-Pole (Newton/Warp) ──── DONE ✓
    │   Vectorized envs, finite-diff gradients, video recording
    │
Level 2: Ant (Newton/Warp) ──────────── YOU ARE HERE
    │   Created but not validated on GPU
    │   Need: Run codesign_ant.py on Newton machine
    │
Level 3: G1 Humanoid ────────────────── BLOCKED
    │   Gradient chain broken by numpy conversions
    │   Need: Implement Warp kernels for quaternion math
    │
    └── Ready for thesis experiments
```

---

*Document Version: 2.0*
*Last Updated: 2026-02-05*
