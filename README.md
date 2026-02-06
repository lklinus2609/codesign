# PGHC: Performance-Gated Hybrid Co-Design

Morphology and control co-optimization for legged robots using differentiable physics simulation.

## Overview

This project implements **Performance-Gated Hybrid Co-Design (PGHC)**, a bi-level optimization algorithm that jointly optimizes robot morphology and control policy:

- **Inner Loop**: Policy optimization (PPO) at fixed morphology
- **Outer Loop**: Morphology gradient descent using the envelope theorem

The key insight is that when the policy is optimized for a given morphology, we can compute `dReturn/dMorphology` by treating the policy as frozen (envelope theorem).

## Verification Ladder

We validate PGHC through a systematic progression of increasingly complex environments:

| Level | Environment | Physics Engine | Status |
|-------|-------------|----------------|--------|
| 0 | Math only | N/A | DONE |
| 1 | Cart-Pole | PyTorch (hand-coded) | DONE |
| 1.5 | Cart-Pole | Newton/Warp (vectorized) | DONE |
| 2 | Ant | Newton/Warp | CREATED |
| 3 | G1 Humanoid | Newton/Warp | BLOCKED (gradient chain) |

## Quick Start

### Level 0: Math Verification (No Physics)
```bash
python test_level0_verification.py
```
Verifies gradient computation, trust region logic, and quaternion chain rule.

### Level 1: Cart-Pole (PyTorch Physics)
```bash
# Train PPO on cart-pole
python train_cartpole_ppo.py

# Run PGHC co-design (full trust region)
python codesign_cartpole.py

# Run simplified PGHC (trusts gradient)
python codesign_cartpole_simple.py

# Find optimal pole length via grid search
python find_optimal_L.py
```

### Level 1.5 & 2: Newton-Based Environments
Requires Newton/Warp installation (see Installation below).

```bash
# Test Newton cart-pole
python test_level15_newton.py

# Test Ant environment
python test_level2_ant.py

# Run PGHC on Ant
python codesign_ant.py
```

### Level 3: G1 Humanoid
```bash
python train_hybrid.py \
    --agent_config config/hybrid_g1_agent.yaml \
    --env_config config/hybrid_g1_env.yaml
```

## Installation

### Basic (Levels 0-1)
```bash
conda create -n codesign python=3.10
conda activate codesign
pip install torch numpy
```

### Full (Levels 1.5+)
```bash
# Clone dependencies
git clone https://github.com/NVIDIA/newton.git ../newton
git clone https://github.com/NVlabs/MimicKit.git ../MimicKit

# Install Newton
pip install -e ../newton

# Create environment
conda env create -f environment.yml
conda activate codesign
```

## Project Structure

```
codesign/
├── envs/
│   ├── cartpole/
│   │   ├── cartpole_env.py        # Differentiable cart-pole (PyTorch)
│   │   └── ppo.py                 # PPO implementation
│   ├── cartpole_newton/
│   │   └── cartpole_newton_vec_env.py  # Vectorized Newton cart-pole (GPU)
│   ├── ant/
│   │   └── ant_env.py             # Newton-based Ant with parametric morphology
│   └── pendulum/                  # Simple pendulum (earlier work)
│
├── config/                        # Configuration files for G1
│
├── test_level0_verification.py    # Level 0: Math verification tests
├── test_level1_pendulum.py        # Level 1: Pendulum/cart-pole tests
├── test_level15_newton.py         # Level 1.5: Newton cart-pole tests
├── test_level2_ant.py             # Level 2: Ant environment tests
│
├── train_cartpole_ppo.py          # PPO training for cart-pole
├── codesign_cartpole.py           # Level 1: PGHC with trust region
├── codesign_cartpole_simple.py    # Level 1: Simplified PGHC
├── codesign_cartpole_newton_vec.py # Level 1.5: Vectorized PGHC
├── codesign_ant.py                # Level 2: PGHC for Ant morphology
├── find_optimal_L.py              # Grid search for optimal L*
│
├── parametric_g1.py               # Parametric G1 humanoid model
├── hybrid_agent.py                # Hybrid co-design agent (Level 3)
├── train_hybrid.py                # G1 training script
└── diff_rollout.py                # Differentiable rollout
```

## Algorithm Details

### PGHC Algorithm

```
Initialize: morphology θ, policy π

for outer_iteration in range(N):
    # INNER LOOP: Optimize policy at current morphology
    for inner_iteration in range(M):
        Collect rollouts with π in environment(θ)
        Update π using PPO

    # OUTER LOOP: Optimize morphology
    Compute gradient: g = dReturn/dθ (with π FROZEN)

    # Trust Region (optional)
    θ_candidate = θ + lr * g
    Train π at θ_candidate
    if Return(θ_candidate) > Return(θ) - ξ:
        Accept: θ = θ_candidate
    else:
        Reject: restore θ, decay lr
```

### Environment Specifications

**Cart-Pole (Level 1)**
- State: `[x, theta, x_dot, theta_dot]`
- Action: Force on cart `[-10, 10]` N
- Design: Pole length `L ∈ [0.3, 1.2]` m
- Physics: Mass scales with length (`m = ρ × 2L`)

**Ant (Level 2)**
- State: 27D (torso pose + joint angles + velocities)
- Action: 8 joint torques (4 hips + 4 ankles)
- Design: `leg_length`, `foot_length`, `torso_radius`
- Reward: Forward velocity + survival - control cost

**G1 Humanoid (Level 3)**
- State: Full body pose + velocities
- Action: Joint torques
- Design: Hip roll angle (leg splay)
- Uses MimicKit for motion priors

## Key Implementation Notes

### Envelope Theorem
When computing `dReturn/dθ`, the policy π must be FROZEN at its optimized state. The gradient flows through the physics dynamics, not the policy.

### Shaped Rewards
Binary rewards (e.g., +1 if balanced) have zero gradient. Use shaped rewards (e.g., `cos(theta)`) for gradient flow.

### Finite Difference Gradients
For Newton-based environments, morphology changes require model rebuilding. We use finite difference:
```
dReturn/dθ ≈ (Return(θ+ε) - Return(θ-ε)) / (2ε)
```

### Trust Region
The full PGHC uses accept/reject logic to validate gradient steps:
- Accept if performance improves (or doesn't degrade much)
- Reject and decay learning rate if performance drops significantly
- Simplified version trusts gradient with small step sizes

## Configuration

Key parameters in `config/hybrid_g1_agent.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_iters` | 10000 | Inner loop only before outer loop activates |
| `outer_loop_freq` | 200 | Outer loop runs every N inner iterations |
| `design_learning_rate` | 0.01 | Morphology optimization learning rate |
| `diff_horizon` | 3 | Differentiable rollout horizon |

## Monitoring

Training logs design parameters to TensorBoard/WandB:
- `Design_Theta`: Current morphology parameter
- `Outer_Loop_Loss`: Outer loop objective
- `Outer_Loop_Grad`: Design parameter gradient
- `Return`: Episode return

```bash
# TensorBoard
tensorboard --logdir output/

# Weights & Biases
python train_hybrid.py --logger wandb
```

## Known Limitations

1. **Newton contact instability**: Differentiable simulation has contact issues. Workaround: short horizon before ground contact.

2. **Model rebuild cost**: Changing morphology in Newton requires full model rebuild (expensive for gradient computation).

3. **Local optima**: PGHC finds local optima. Results depend on initialization.

## References

- Envelope theorem for bi-level optimization
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms"
- Newton: NVIDIA differentiable physics engine
- MimicKit: Adversarial Motion Priors for humanoid locomotion

## License

Research use only. See LICENSE for details.

## Acknowledgments

- **[NVIDIA Newton](https://github.com/NVIDIA/newton)** - Differentiable physics simulation
- **[MimicKit](https://github.com/NVlabs/MimicKit)** - Adversarial Motion Priors implementation
