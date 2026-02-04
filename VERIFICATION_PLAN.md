# PGHC Algorithm Verification Plan

This document outlines a systematic verification strategy for the Performance-Gated Hybrid Co-Design (PGHC) algorithm before deploying on the full G1 Humanoid.

## Executive Summary

**Problem**: Testing PGHC on a 30-DOF humanoid is slow (~hours per iteration) and debugging failures is nearly impossible—you cannot distinguish algorithm bugs from environment complexity.

**Solution**: A 4-level verification ladder, each level isolating specific algorithm components:

| Level | Environment | DOF | What It Tests | Time/Iter |
|-------|-------------|-----|---------------|-----------|
| 0 | Synthetic | 1 | Gradient & trust region correctness | ~1s |
| 1 | Inverted Pendulum | 2 | Envelope theorem validity | ~10s |
| 2 | HalfCheetah | 18 | Trust region adaptation under instability | ~30s |
| 3 | Ant | 29 | 3D contact dynamics | ~1min |
| 4 | G1 Humanoid | 37 | Full system | ~5min |

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

## Level 2: HalfCheetah (2D Benchmark)

### Purpose
Test the adaptive trust region under realistic instability. HalfCheetah is sensitive to leg proportions—small changes can break the gait.

### Environment Specification

**Morphology Parameters** (4D):
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Torso length | [0.3, 0.8] | 0.5 | Affects CoM position |
| Thigh length | [0.1, 0.3] | 0.2 | Front and back legs symmetric |
| Shin length | [0.1, 0.3] | 0.2 | Critical for ground clearance |
| Foot length | [0.05, 0.2] | 0.1 | Affects contact stability |

**Objective**: Maximize forward velocity (simpler than CoT for this test)

### Why HalfCheetah Tests Trust Region

The HalfCheetah gait is **locally optimal but globally fragile**:
- A trained policy exploits the current morphology's resonance frequency
- Change leg length by 20% → Policy fails catastrophically
- This is exactly the scenario the trust region should handle

### Test Procedure

**Phase 1: Baseline**
1. Train policy on default morphology to convergence
2. Record velocity achieved (typically 5-8 m/s for well-tuned PPO)

**Phase 2: Co-Design with Aggressive LR**
1. Set initial `design_lr = 0.5` (intentionally too high)
2. Run co-design
3. **Expected**: Trust region should reject most updates, decay LR rapidly

**Phase 3: Co-Design with Conservative LR**
1. Set initial `design_lr = 0.01`
2. Run co-design for 50 outer loop iterations
3. Record morphology trajectory and performance

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Trust region activation | > 0 | Should trigger at least once |
| LR adaptation observed | Yes | Should see decay/growth in logs |
| Final velocity | > baseline | Improvement, even if small |
| No catastrophic policy failure | 0 occurrences | Should never drop below 50% baseline |

### Failure Analysis

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Trust region never triggers | LR too conservative or ξ too large | Increase design_lr or decrease ξ |
| Always rejects, LR → min | ξ too small or gradients too noisy | Increase ξ, add gradient smoothing |
| Policy collapses | Trust region not catching degradation | Bug in evaluation, or ξ way too large |

### Key Insight: β Tuning

This level is primarily for tuning the trust region hyperparameters:
- `design_lr_init` (β₀): Starting learning rate
- `lr_decay_factor`: How much to shrink on violation
- `lr_growth_factor`: How much to grow on success
- `trust_region_threshold` (ξ): Acceptable degradation

**Recommended Sweep**:
```yaml
design_lr_init: [0.001, 0.01, 0.1]
trust_region_threshold: [0.05, 0.1, 0.2]
lr_decay_factor: [0.3, 0.5, 0.7]
```

---

## Level 3: Ant (3D Bridge)

### Purpose
Introduce 3D contact dynamics before the full humanoid. Ant has 4 legs → inherently more stable than bipedal.

### Environment Specification

**Morphology Parameters** (4D symmetric):
| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| Hip offset | [0.1, 0.3] | 0.2 | Distance from torso center |
| Thigh length | [0.15, 0.35] | 0.25 | Upper leg segment |
| Shin length | [0.15, 0.35] | 0.25 | Lower leg segment |
| Foot radius | [0.02, 0.08] | 0.04 | Contact surface |

**Symmetry**: All 4 legs share the same parameters (reduces to 4 DOF from 16)

**Objective**: Forward velocity OR energy efficiency

### Why Ant Before Humanoid

| Challenge | Ant | Humanoid |
|-----------|-----|----------|
| 3D dynamics | ✓ | ✓ |
| Multiple contacts | 4 legs | 2 legs |
| Balance difficulty | Low (wide base) | High |
| Fall frequency | Rare | Common |
| Debug time | Moderate | Extreme |

### Test Procedure

**Phase 1: Verify Differentiable Rollout in 3D**
1. Run short rollouts (H=10 steps) with gradient computation
2. Verify no NaN gradients
3. Compare gradients with finite difference

**Phase 2: Full Co-Design**
1. Initialize with default morphology
2. Enable outer loop
3. Run for 30+ outer loop iterations

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| NaN-free gradients | 100% | Critical for 3D contact |
| Stable locomotion | > 90% episodes | Should rarely fall |
| Morphology change | Measurable | Should move from initial |
| Performance vs baseline | ≥ baseline | At least no worse |

### Known Challenge: Contact Gradient Instability

Your current implementation limits horizon to H=3 due to contact instabilities. For Ant:
- Expect similar issues with ground contact
- May need contact smoothing or shorter horizons
- This is a research problem, not just a bug

**Mitigation Options**:
1. Use randomized smoothing on contact forces
2. Limit BPTT to flight phases only
3. Use finite-difference for contact-heavy steps

---

## Level 4: G1 Humanoid (Target)

### Purpose
Full system validation on the target robot.

### Only Proceed Here If

- [ ] Level 0: All gradient and trust region tests pass
- [ ] Level 1: Pendulum finds near-optimal length
- [ ] Level 2: HalfCheetah improves with trust region active
- [ ] Level 3: Ant achieves stable 3D locomotion with co-design

### Expected Challenges

| Challenge | Mitigation |
|-----------|------------|
| Slow iteration time | Reduce num_envs initially, add checkpointing |
| Frequent falls | Increase stability_threshold, more warmup iterations |
| High-dimensional morphology | Start with hip angles only (current setup) |
| Mode collapse | Multiple random seeds, curriculum learning |

---

## Implementation Checklist

### Level 0 (Estimated: 1-2 days)
- [ ] Create `tests/test_gradient_pipeline.py`
- [ ] Create `tests/test_trust_region.py`
- [ ] Create `tests/test_quaternion_grad.py`
- [ ] All tests pass

### Level 1 (Estimated: 3-5 days)
- [ ] Implement `ParametricPendulumModel`
- [ ] Implement `PendulumEnv` (Gym-compatible)
- [ ] Create pendulum config files
- [ ] Integrate with simplified agent (no AMP)
- [ ] Run verification, document results

### Level 2 (Estimated: 3-5 days)
- [ ] Implement `ParametricHalfCheetahModel`
- [ ] Adapt MuJoCo HalfCheetah or use existing Newton model
- [ ] Create config files
- [ ] Run trust region tuning sweep
- [ ] Document optimal hyperparameters

### Level 3 (Estimated: 5-7 days)
- [ ] Implement `ParametricAntModel`
- [ ] Handle 3D contact in differentiable rollout
- [ ] Debug gradient instabilities
- [ ] Run full co-design, document results

### Level 4 (Ongoing)
- [ ] Transfer tuned hyperparameters from Levels 2-3
- [ ] Run on G1 Humanoid
- [ ] Iterate based on results

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
| Parameter | Pendulum | Cheetah | Ant | Humanoid |
|-----------|----------|---------|-----|----------|
| `learning_rate` | 3e-4 | 3e-4 | 3e-4 | 5e-5 |
| `num_envs` | 32 | 256 | 512 | 2048 |
| `ppo_epochs` | 10 | 10 | 10 | 5 |
| `min_iters_for_stability` | 500 | 2000 | 3000 | 5000 |

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
Start at Level 0
    │
    ├── All tests pass? ────No───→ Fix gradient/trust region bugs
    │         │
    │        Yes
    │         ↓
    ├── Level 1: Pendulum
    │         │
    │   Finds optimal L? ────No───→ Debug envelope theorem, check convergence
    │         │
    │        Yes
    │         ↓
    ├── Level 2: HalfCheetah
    │         │
    │   Trust region works? ──No──→ Tune ξ, β, decay factors
    │         │
    │        Yes
    │         ↓
    ├── Level 3: Ant
    │         │
    │   3D stable? ───────────No──→ Debug contact gradients, smoothing
    │         │
    │        Yes
    │         ↓
    └── Level 4: Humanoid
              │
        Ready for thesis experiments
```

---

*Document Version: 1.0*
*Last Updated: 2026-02-04*
*Author: Generated for PGHC Algorithm Verification*
