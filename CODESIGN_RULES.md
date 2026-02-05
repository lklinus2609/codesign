# CODESIGN_RULES.md

This document is the **single source of truth** for coding decisions, style conventions, project organization, and development practices. All contributors (human or AI) must follow these rules.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [File Naming Conventions](#3-file-naming-conventions)
4. [Code Style](#4-code-style)
5. [Algorithm-to-Code Mapping](#5-algorithm-to-code-mapping)
6. [Configuration Conventions](#6-configuration-conventions)
7. [Testing Standards](#7-testing-standards)
8. [Verification Ladder](#8-verification-ladder)
9. [Git Workflow](#9-git-workflow)
10. [Documentation Standards](#10-documentation-standards)
11. [Dependency Management](#11-dependency-management)
12. [Decision Log](#12-decision-log)

---

## 1. Project Overview

**Project Name**: PGHC (Performance-Gated Hybrid Co-Design)

**Purpose**: Morphology and control co-optimization for humanoid robots using differentiable physics.

**Core Algorithm**: Algorithm 1 from the thesis - a two-loop optimization:
- **Inner Loop**: PPO + AMP for policy learning (MimicKit)
- **Outer Loop**: Gradient-based morphology optimization via BPTT (Newton)

**Key Dependencies**:
| Dependency | Purpose | Location |
|------------|---------|----------|
| Newton | Differentiable physics (Warp-based) | `../newton/` |
| MimicKit | AMP implementation + training infra | `../MimicKit/` |
| PyTorch | Neural networks, autograd | pip |
| Warp | GPU-accelerated simulation | pip (via Newton) |

---

## 2. Directory Structure

```
codesign/                          # Repository root
├── codesign/                      # Main package (this directory)
│   ├── __init__.py
│   │
│   ├── # === CORE ALGORITHM ===
│   ├── hybrid_agent.py            # PGHC algorithm (Algorithm 1)
│   ├── diff_rollout.py            # Differentiable rollout for outer loop
│   ├── parametric_g1.py           # G1 morphology parameterization
│   │
│   ├── # === TRAINING ===
│   ├── train_hybrid.py            # Main entry point
│   ├── video_recorder.py          # Headless video recording for wandb
│   │
│   ├── # === CONFIGURATION ===
│   ├── config/
│   │   ├── hybrid_g1_agent.yaml   # Agent hyperparameters
│   │   └── hybrid_g1_env.yaml     # Environment parameters
│   │
│   ├── # === TESTING ===
│   ├── run_all_tests.py           # Test runner
│   ├── test_implementation.py     # Unit tests
│   ├── test_checkpoint.py         # Checkpoint/resume tests
│   ├── test_gpu_memory.py         # Memory profiling
│   ├── validate_outer_loop.py     # End-to-end gradient validation
│   │
│   ├── # === VERIFICATION ENVIRONMENTS ===
│   ├── envs/                      # Simplified test environments
│   │   ├── __init__.py
│   │   ├── pendulum/              # Simple pendulum (early experiments)
│   │   │   ├── __init__.py
│   │   │   └── pendulum_env.py
│   │   ├── cartpole/              # Level 1: Cart-pole (current)
│   │   │   ├── __init__.py
│   │   │   ├── cartpole_env.py    # Differentiable physics
│   │   │   └── ppo.py             # PPO implementation
│   │   ├── cheetah/               # Level 2: HalfCheetah (planned, Warp)
│   │   └── ant/                   # Level 3: Ant (planned, Warp)
│   │
│   ├── # === DOCUMENTATION ===
│   ├── README.md                  # Quick start guide
│   ├── CODESIGN_RULES.md          # THIS FILE - conventions & decisions
│   └── VERIFICATION_PLAN.md       # Testing ladder documentation
│
├── output/                        # Training outputs (gitignored)
│   └── hybrid_g1/
│       ├── model.pt               # Latest checkpoint
│       ├── events.out.*           # TensorBoard logs
│       └── videos/                # Recorded videos
│
└── environment.yml                # Conda environment spec
```

### Rules for Adding New Files

1. **New environments** → `envs/<env_name>/`
2. **New parametric models** → `parametric_<robot>.py` in root
3. **New test files** → `test_<feature>.py` in root
4. **Configuration** → `config/<robot>_<type>.yaml`

---

## 3. File Naming Conventions

### Python Files

| Pattern | Purpose | Example |
|---------|---------|---------|
| `train_*.py` | Training entry points | `train_hybrid.py` |
| `test_*.py` | Test files (pytest compatible) | `test_implementation.py` |
| `validate_*.py` | End-to-end validation scripts | `validate_outer_loop.py` |
| `parametric_*.py` | Morphology parameterization | `parametric_g1.py` |
| `*_agent.py` | Agent implementations | `hybrid_agent.py` |
| `*_env.py` | Environment wrappers | `pendulum_env.py` |

### Configuration Files

| Pattern | Purpose | Example |
|---------|---------|---------|
| `*_agent.yaml` | Agent/algorithm hyperparameters | `hybrid_g1_agent.yaml` |
| `*_env.yaml` | Environment/task parameters | `hybrid_g1_env.yaml` |

### Class Naming

| Pattern | Purpose | Example |
|---------|---------|---------|
| `Parametric*Model` | Morphology parameterization | `ParametricG1Model` |
| `*Agent` | Training agent | `HybridAMPAgent` |
| `*AgentBase` | Mixin/base class | `HybridAMPAgentBase` |
| `*Env` | Gym-style environment | `PendulumEnv` |

---

## 4. Code Style

### Python Style

- **Formatter**: Black (line length 100)
- **Linter**: Flake8
- **Type hints**: Required for public APIs, optional for internal functions
- **Docstrings**: Google style

```python
def compute_gradient(
    model: ParametricModel,
    trajectory: Trajectory,
    horizon: int = 3,
) -> torch.Tensor:
    """Compute morphology gradient via BPTT.

    Args:
        model: Parametric model with design parameters.
        trajectory: Rollout trajectory for gradient computation.
        horizon: Number of timesteps for BPTT.

    Returns:
        Gradient tensor with shape (num_design_params,).

    Raises:
        ValueError: If trajectory length < horizon.
    """
```

### Import Order

```python
# 1. Standard library
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import torch
import warp as wp

# 3. Local/project
from parametric_g1 import ParametricG1Model
from diff_rollout import SimplifiedDiffRollout
```

### Variable Naming

| Convention | Use For | Example |
|------------|---------|---------|
| `snake_case` | Variables, functions, methods | `design_lr`, `compute_gradient()` |
| `PascalCase` | Classes | `HybridAMPAgent` |
| `SCREAMING_SNAKE` | Constants | `DEFAULT_HORIZON = 3` |
| `_leading_underscore` | Private/internal | `_sync_design_to_inner_model()` |

### Algorithm Variable Naming

Map thesis notation to code variables consistently:

| Thesis | Code Variable | Description |
|--------|---------------|-------------|
| θ (theta) | `design_theta` | Morphology parameters |
| φ (phi) | `policy_params` | Policy network weights |
| β (beta) | `design_lr` | Design learning rate |
| ξ (xi) | `trust_region_threshold` | Trust region threshold |
| H | `diff_horizon` | BPTT horizon |
| W | `stability_window` | Moving average window |
| δ_conv | `stability_threshold` | Convergence threshold |
| J(θ) | `objective` / `loss` | Outer loop objective |
| D | `improvement` | Performance change |

---

## 5. Algorithm-to-Code Mapping

### Algorithm 1: PGHC

```
ALGORITHM 1 (Thesis)              CODE LOCATION
─────────────────────────────────────────────────────────────────
Lines 1-5: Initialization         hybrid_agent.py: __init__()
Lines 6-11: Main loop             hybrid_agent.py: _train_iter()
Lines 12-18: Stability gating     hybrid_agent.py: _should_run_outer_loop()
Lines 19-23: Gradient computation hybrid_agent.py: _outer_loop_update()
                                  diff_rollout.py: forward(), compute_loss()
Lines 24-35: Trust region         hybrid_agent.py: _outer_loop_update()
```

### Key Methods

| Method | Algorithm Lines | Purpose |
|--------|-----------------|---------|
| `_train_iter()` | 6-35 | One iteration of hybrid training |
| `_should_run_outer_loop()` | 12-18 | Stability gate check |
| `_outer_loop_update()` | 19-35 | Full outer loop iteration |
| `_compute_objective()` | 20-22 | Differentiable rollout + loss |
| `_extract_design_gradient()` | 23 | Chain rule through quaternions |
| `_apply_trust_region_update()` | 24-35 | Adaptive LR update |
| `_sync_design_to_inner_model()` | (post-35) | Sync models after update |

---

## 6. Configuration Conventions

### YAML Structure

Agent configs follow this section order:

```yaml
# 1. Agent identification
agent_name: "..."

# 2. Inner loop (PPO/AMP) parameters
model:
  ...
optimizer:
  ...

# 3. Gating mode selection
gating_mode: "stability"  # or "fixed"

# 4. Stability gating parameters (if gating_mode: "stability")
stability_window: ...
stability_threshold: ...

# 5. Legacy fixed schedule (if gating_mode: "fixed")
warmup_iters: ...
outer_loop_freq: ...

# 6. Trust region parameters
design_learning_rate: ...
trust_region_threshold: ...

# 7. Differentiable rollout
diff_horizon: ...

# 8. Design parameter bounds
design_param_init: ...
design_param_min: ...
design_param_max: ...

# 9. Logging/checkpointing
log_design_history: ...
```

### Parameter Naming in YAML

- Use `snake_case` for all parameter names
- Group related parameters with comments
- Include units in comments where applicable

```yaml
# Trust region threshold xi (10% performance degradation allowed)
trust_region_threshold: 0.1

# Design parameter bounds (radians)
design_param_min: -0.5236  # -30 degrees
design_param_max: 0.5236   # +30 degrees
```

---

## 7. Testing Standards

### Test File Organization

```python
# test_*.py structure

class TestFeatureName:
    """Tests for [feature]."""

    def test_basic_functionality(self):
        """Test the happy path."""
        ...

    def test_edge_case_description(self):
        """Test specific edge case."""
        ...

    def test_error_handling(self):
        """Test that errors are raised appropriately."""
        ...
```

### Test Categories

| Prefix | Purpose | Run Frequency |
|--------|---------|---------------|
| `test_` | Unit tests | Every commit |
| `validate_` | End-to-end validation | Before PR merge |
| `benchmark_` | Performance benchmarks | Weekly/manual |

### Running Tests

```bash
# All unit tests
python run_all_tests.py

# Specific test file
pytest test_implementation.py -v

# End-to-end validation
python validate_outer_loop.py
```

---

## 8. Verification Ladder

**CRITICAL**: Before testing on the full humanoid, verify the algorithm on simpler environments.

### Progression

```
Level 0: Component Verification (no physics)         [DONE]
    ├── 0A: Gradient pipeline (finite diff vs autograd)
    ├── 0B: Trust region mechanics (synthetic objective)
    └── 0C: Quaternion chain rule
    Code: test_level0_verification.py

Level 1: Cart-Pole (hand-coded PyTorch physics)      [IN PROGRESS]
    ├── Design parameter: pole length L
    ├── Inner loop: PPO
    └── Verify envelope theorem, gradient correctness
    Code: envs/cartpole/, train_cartpole_ppo.py

Level 2: HalfCheetah (Warp/Newton physics)           [PLANNED]
    └── Tune trust region hyperparameters (beta, xi, decay)

Level 3: Ant (Warp/Newton physics)                   [PLANNED]
    └── Verify 3D contact handling

Level 4: G1 Humanoid (target)                        [PLANNED]
    └── Full system validation
```

### Gate Criteria

**Do not proceed to Level N+1 until Level N passes:**

| Level | Pass Criteria |
|-------|---------------|
| 0 | All gradient tests within 1% of finite difference |
| 1 | Finds pole length within 20% of analytical optimum |
| 2 | Trust region activates AND performance improves |
| 3 | Stable 3D locomotion with measurable morphology change |

See `VERIFICATION_PLAN.md` for detailed test procedures.

---

## 9. Git Workflow

### Branch Naming

| Pattern | Use For | Example |
|---------|---------|---------|
| `main` | Stable, tested code | - |
| `feature/<name>` | New features | `feature/pendulum-env` |
| `fix/<name>` | Bug fixes | `fix/gradient-sign` |
| `exp/<name>` | Experiments (may not merge) | `exp/contact-smoothing` |

### Commit Messages

```
<type>: <short description>

<optional body explaining why>

Types:
- feat: New feature
- fix: Bug fix
- refactor: Code restructure (no behavior change)
- test: Adding/updating tests
- docs: Documentation only
- config: Configuration changes
```

### What NOT to Commit

- `output/` directory (training artifacts)
- `*.pt` checkpoint files (too large)
- `__pycache__/`
- `.env` files with secrets
- Large data files (use git-lfs if needed)

---

## 10. Documentation Standards

### Required Documentation

| File | Purpose | Update When |
|------|---------|-------------|
| `README.md` | Quick start, installation | Dependencies or usage changes |
| `CODESIGN_RULES.md` | Conventions, decisions | Any convention change |
| `VERIFICATION_PLAN.md` | Testing methodology | Adding new verification levels |

### Code Comments

**DO comment**:
- Non-obvious algorithm steps (reference thesis section)
- Workarounds for known issues
- Magic numbers with units

```python
# Algorithm 1, Line 27: Check trust region violation
# D < -ξ * |J(φ_k)| indicates unacceptable performance degradation
if improvement < -self.trust_region_threshold * abs(current_objective):
    ...

# Workaround: SemiImplicit solver has contact instability after ~3 steps
# See GitHub issue #42 for tracking
diff_horizon = 3  # steps (before ground contact)
```

**DO NOT comment**:
- Obvious code (`i += 1  # increment i`)
- Commented-out code (delete it, use git history)
- TODOs without tracking (use GitHub issues)

---

## 11. Dependency Management

### Adding Dependencies

1. Add to `environment.yml` under appropriate section
2. Pin major.minor version (e.g., `torch>=2.0,<3.0`)
3. Document why the dependency is needed

```yaml
dependencies:
  # Core ML
  - pytorch>=2.0,<3.0          # Neural networks, autograd
  - numpy>=1.24                 # Array operations

  # Simulation (installed via pip for Newton compatibility)
  - pip:
    - warp-lang>=0.10           # GPU simulation (Newton dependency)
```

### Version Conflicts

If Newton or MimicKit require specific versions:
1. Document the conflict in `environment.yml` comments
2. Pin to the required version
3. Add to Decision Log below

---

## 12. Decision Log

Record significant technical decisions here with date and rationale.

### 2026-02-04: Verification Ladder Approach

**Decision**: Implement 5-level verification (Level 0 Math → Cart-Pole → Cheetah → Ant → Humanoid) before full humanoid testing.

**Rationale**: Debugging failures on 30-DOF humanoid is impractical. Each level isolates specific algorithm components:
- Level 0 tests math correctness (gradients, trust region, quaternion chain rule)
- Level 1 tests envelope theorem on cart-pole
- Level 2 tests trust region adaptation
- Level 3 tests 3D contact handling

**Reference**: `VERIFICATION_PLAN.md`

---

### 2026-02-04: Physics Engine Selection by Level

**Decision**: Use different physics backends per verification level:
- Level 0: No physics (pure math tests)
- Level 1: Hand-coded PyTorch (analytical cart-pole dynamics)
- Level 2+: NVIDIA Warp or Newton (differentiable multi-body physics)

**Rationale**:
- Level 1 needs differentiable dReturn/dL gradients; standard engines (MuJoCo, Bullet) not differentiable
- Hand-coded PyTorch sufficient for simple cart-pole (1 joint, 4 state variables)
- Level 2+ requires real physics engine for complex multi-body dynamics
- Warp/Newton provide GPU-accelerated differentiable simulation

**Code**: `envs/cartpole/cartpole_env.py` (hand-coded), Newton for Level 2+

---

### 2026-02-04: PPO as Inner Loop Policy Optimizer

**Decision**: Use PPO (Proximal Policy Optimization) for inner loop policy learning at all verification levels.

**Rationale**:
- Industry standard for continuous control
- Stable training with clipped surrogate objective
- Works with both simple (cart-pole) and complex (humanoid) environments
- Can later extend to PPO+AMP for motion imitation

**Code**: `envs/cartpole/ppo.py`

**Alternatives Considered**:
- LQR: Only works near equilibrium, no swing-up capability
- SAC: More sample efficient but more complex to implement
- Hand-tuned PD: Not generalizable to complex tasks

---

### 2026-02-04: Separate Inner/Outer Loop Models

**Decision**: Use two separate physics models - MuJoCo for inner loop, SemiImplicit for outer loop.

**Rationale**:
- Inner loop needs stable, fast simulation (MuJoCo)
- Outer loop needs differentiability (SemiImplicit)
- Syncing `joint_X_p` between them after each outer loop update

**Code**: `hybrid_agent.py: _sync_design_to_inner_model()`

---

### 2026-02-01: Short Differentiable Horizon (H=3)

**Decision**: Limit BPTT horizon to 3 timesteps.

**Rationale**: SolverSemiImplicit has numerical instability with ground contact forces. Robot makes contact around step 3-4, causing NaN gradients.

**Future Work**: Investigate contact smoothing or randomized smoothing for longer horizons.

**Code**: `config/hybrid_g1_agent.yaml: diff_horizon: 3`

---

### 2026-01-24: Stability Gating Over Fixed Schedule

**Decision**: Default to `gating_mode: "stability"` instead of fixed warmup + frequency.

**Rationale**: Fixed schedule doesn't guarantee policy convergence before morphology update. Stability gating enforces the Envelope Theorem assumption (frozen policy at optimum).

**Code**: `hybrid_agent.py: _should_run_outer_loop()`

---

### Template for New Decisions

```markdown
### YYYY-MM-DD: Decision Title

**Decision**: What was decided.

**Rationale**: Why this choice was made.

**Alternatives Considered**: (optional) Other options and why rejected.

**Code**: Where this is implemented.
```

---

## Appendix: Quick Reference

### Common Commands

```bash
# Training
python train_hybrid.py --agent_config config/hybrid_g1_agent.yaml --env_config config/hybrid_g1_env.yaml

# Testing
python run_all_tests.py
pytest test_implementation.py -v -k "test_gradient"

# Validation
python validate_outer_loop.py

# Resume from checkpoint
python train_hybrid.py --resume output/hybrid_g1/model.pt
```

### Key Hyperparameters to Tune

| Parameter | Start | Tune At |
|-----------|-------|---------|
| `design_lr` | 0.01 | Level 2 (Cheetah) |
| `trust_region_threshold` | 0.1 | Level 2 (Cheetah) |
| `stability_threshold` | 0.05 | Level 1 (Pendulum) |
| `diff_horizon` | 3 | Level 3 (Ant) |

### TensorBoard Metrics to Watch

| Metric | Collection | Healthy Range |
|--------|------------|---------------|
| `Design_Theta_Deg` | 3_Design | Should change over time |
| `Design_LR` | 3_Design | Should adapt (not stuck at min/max) |
| `TR_Accept_Rate` | 3_Design | 50-90% (not 0% or 100%) |
| `Stability_Delta_Rel` | 3_Design | Should decrease over inner loop |

---

*Document Version: 1.1*
*Last Updated: 2026-02-04*
*Maintainer: [Your Name]*
