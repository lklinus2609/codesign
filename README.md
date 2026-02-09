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
# Run PGHC co-design (full trust region)
python codesign_cartpole.py
```

### Level 1.5: Cart-Pole (Newton/Warp Vectorized)
```bash
# Run PGHC with vectorized GPU environments
python codesign_cartpole_newton_vec.py --wandb --num-worlds 2048

# Test Newton cart-pole
python test_level15_newton.py
```

### Level 2: Ant (Newton/Warp)
```bash
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
│   │   ├── cartpole_env.py             # Differentiable cart-pole (PyTorch)
│   │   └── ppo.py                      # PPO implementation
│   ├── cartpole_newton/
│   │   └── cartpole_newton_vec_env.py  # Vectorized Newton cart-pole (GPU)
│   └── ant/
│       └── ant_env.py                  # Newton-based Ant with parametric morphology
│
├── config/                             # Configuration files for G1
│   ├── hybrid_g1_agent.yaml
│   └── hybrid_g1_env.yaml
│
├── codesign_cartpole.py                # Level 1: PGHC with trust region
├── codesign_cartpole_newton_vec.py     # Level 1.5: Vectorized PGHC (main active script)
├── codesign_ant.py                     # Level 2: PGHC for Ant morphology
│
├── parametric_g1.py                    # Parametric G1 humanoid model
├── hybrid_agent.py                     # Hybrid co-design agent (Level 3)
├── train_hybrid.py                     # G1 training script
├── diff_rollout.py                     # Differentiable rollout
├── video_recorder.py                   # Headless video recording
│
├── test_level0_verification.py         # Level 0: Math verification tests
├── test_level15_newton.py              # Level 1.5: Newton cart-pole tests
├── test_level2_ant.py                  # Level 2: Ant environment tests
├── test_implementation.py              # Unit tests (G1)
├── test_checkpoint.py                  # Checkpoint/resume tests
├── test_gpu_memory.py                  # Memory profiling
├── validate_outer_loop.py              # End-to-end gradient validation
└── run_all_tests.py                    # Test runner
```

## Algorithm Details

### PGHC Algorithm

```
Initialize: morphology theta, policy pi, Adam optimizer for theta

for outer_iteration in range(N):
    # INNER LOOP: Optimize policy at current morphology
    for inner_iteration until convergence:
        Collect rollouts with pi in environment(theta)
        Update pi using PPO

    # OUTER LOOP: Optimize morphology
    Compute gradient: g = dReturn/d_theta (with pi FROZEN, vectorized FD eval)
    Adam step on theta using -g (gradient ascent via negated gradient)
    Clamp theta to physical bounds
```

### Level 1.5 Hyperparameters (Current)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_worlds` | 2048 | Parallel training environments |
| `num_eval_worlds` | 512 | Parallel eval environments for FD gradient |
| `horizon` | 32 | Rollout steps per update |
| `force_max` | 30 N | Cart force limit |
| `max_steps` | 500 | Episode length (10s at 50Hz) |
| `initial_L` | 0.6 m | Initial pole length |
| `design_lr` | 0.005 | Adam learning rate for design params |
| `desired_kl` | 0.005 | KL target for adaptive LR |
| `max_lr` | 3e-4 | Learning rate ceiling |
| `mini_batches` | 8 | Mini-batches per PPO epoch |
| Inner convergence | Reward plateau (<2% change, window=5) | Inner loop stopping criterion |
| Outer convergence | L range < 3mm over 5 iters | Outer loop stopping criterion |

### Reward Function (Level 1.5)

```
r = 1.0*(alive) - 2.0*(terminated) - theta^2 - 0.1*x^2 - 0.01*action^2
    - 0.1*(action - prev_action)^2 - 0.01*|x_dot| - 0.005*|theta_dot|
```

### Environment Specifications

**Cart-Pole (Level 1/1.5)**
- State: `[x, theta, x_dot, theta_dot]`
- Action: Force on cart `[-30, 30]` N
- Design: Pole length `L`
- Termination: Cart out of bounds (`|x| > 3.0m`)

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
When computing `dReturn/d_theta`, the policy pi must be FROZEN at its optimized state. The gradient flows through the physics dynamics, not the policy.

### Finite Difference Gradients
For Newton-based environments, morphology changes require model rebuilding. We use vectorized finite difference evaluation (512 parallel worlds per perturbation):
```
dReturn/d_theta = (Return(theta+eps) - Return(theta-eps)) / (2*eps)
```
Gradients are fed to an Adam optimizer (not clipped SGD), which normalizes magnitude automatically and uses momentum to smooth noisy estimates.

### RSL-RL Style Episode Tracking
Episode stats are tracked from training rollouts using per-world accumulators. No separate evaluation environment is needed.

## Monitoring

Training logs design parameters to WandB:

```bash
python codesign_cartpole_newton_vec.py --wandb --num-worlds 2048
```

## Known Limitations

1. **Newton contact instability**: Differentiable simulation has contact issues. Workaround: short horizon before ground contact.
2. **Model rebuild cost**: Changing morphology in Newton requires full model rebuild. Training env is freed before FD eval and rebuilt after to avoid GPU OOM.
3. **G1 gradient chain broken**: numpy in parametric_g1.py breaks warp's autograd. Needs Warp kernels for quaternion math.
4. **Local optima**: PGHC finds local optima. Results depend on initialization.

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
