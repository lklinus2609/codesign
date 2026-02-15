# Fix BPTT Physics Accuracy: Implicit Joint Attachment

## Problem

PGHC's core contribution is BPTT through differentiable physics for morphology optimization — no other humanoid co-design paper does this. All existing work (RoboCraft, MORPH, EA-CoRL, etc.) uses gradient-free methods (CMA-ES, ES, GA).

Newton's `SolverSemiImplicit` uses **explicit integration** for joint attachment penalty springs (`joint_attach_ke`, `joint_attach_kd`). The angular stability criterion is:

```
kd * dt / I_min < 2
```

The G1 robot has bodies with tiny inertia (ankle I=7e-6, waist I=4e-6), violating this criterion by 10,000-100,000x. The current workaround clamps all body inertias to 1.0 kg*m^2, which:

- Distorts 29/30 bodies by 100-250,000x
- Simulates a **physically wrong** robot
- Makes BPTT gradients **meaningless** for co-design

## Literature Review Summary

| Paper | Year | Outer Loop Method | Differentiable Physics? |
|-------|------|-------------------|------------------------|
| RoboCraft | 2025 | CMA-ES / ES / BO | No |
| MORPH | 2024 | CMA-ES + neural proxy | No (proxy only) |
| EA-CoRL | 2025 | CMA-ES | No |
| Luck et al. | 2019 | CMA-ES / PSO | No |
| David Ha | 2019 | ES / REINFORCE | No |
| Schaff et al. | 2019 | REINFORCE on GMM | No |
| Sartore et al. | 2023 | Genetic Algorithm | No |
| Ghansah et al. | 2023 | IPOPT (analytical model) | Analytical only |
| **PGHC (ours)** | **2025** | **Gradient descent (Adam)** | **Yes — BPTT through Newton** |

**PGHC is the only approach using BPTT through a physics engine for humanoid morphology optimization.** Switching to ES/FD/CMA-ES would eliminate this contribution.

---

## Path A: Implicit Joint Attachment Correction (Try First)

### The Math

The instability comes from explicit evaluation of stiff penalty springs:

```
v_new = v_old + dt * (ke * x_err + kd * v_err) / m
```

When `kd * dt / m >> 2`, the velocity overshoots and diverges exponentially.

An **implicit** treatment solves for the new velocity self-consistently:

```
v_new = v_old + dt * (ke * x_err_new + kd * v_err_new) / m
```

Substituting `x_err_new = x_err_old + dt * (v_new - v_parent)` and solving:

```
v_new = (m * v_predicted + dt * ke * x_err_old) / (m + dt^2 * ke + dt * kd)
```

The denominator `(m + dt^2*ke + dt*kd)` is **always positive** regardless of body mass/inertia, making this **unconditionally stable**. The formula is simple arithmetic — fully differentiable through `wp.Tape()`.

### Architecture

Create a new solver subclass `SolverSemiImplicitStable` — no modifications to existing Newton code:

```
SolverSemiImplicitStable.step():
  1. eval_body_joint_forces(ke=0, kd=0)   # PD targets + limits only, skip attachment
  2. eval_body_contact_forces(...)         # Contacts as normal
  3. integrate_bodies(...)                 # Integrate with non-attachment forces only
  4. apply_implicit_joint_attachment(...)   # NEW: correct velocities implicitly
```

Step 4 is a new Warp kernel that, for each joint:

1. Reads the post-integration (predicted) child and parent body states
2. Computes position error (child vs expected position from parent + joint transform)
3. Computes velocity error (child vs parent)
4. Applies implicit velocity correction to child body:
   - Linear: `v -= dt*(ke*x_err + kd*v_err) / (m + dt^2*ke + dt*kd)`
   - Angular: `w -= dt*(ke*ang_err + kd*w_err) / (I_eff + dt^2*ke + dt*kd)`
5. Re-integrates position from corrected velocity

### What Changes

**New file:** `newton/newton/_src/solvers/semi_implicit/solver_semi_implicit_stable.py`
- Subclass of `SolverSemiImplicit`
- Overrides `step()` with the 4-step flow above
- Contains the `apply_implicit_joint_attachment` Warp kernel

**Modified:** `newton/newton/solvers.py`
- Add re-export for `SolverSemiImplicitStable`

**Modified:** `codesign/g1_eval_worker.py`
- Change `SolverSemiImplicit` -> `SolverSemiImplicitStable`
- REMOVE all inertia/mass clamping code (lines 387-420)
- Use default ke=1e4, kd=100 (stable at any value now)

**Modified:** `codesign/test_solver_semi_implicit.py`
- Test new solver with REAL body inertias (no clamping)

### What Stays the Same

- `eval_body_joint_forces` kernel (called with ke=0, kd=0 for attachment; still computes PD targets, joint limits, joint forces)
- `integrate_bodies` kernel (unchanged)
- All contact force computation
- Body masses and inertias (**the whole point — no modifications**)

### Coupling Considerations

This treats each joint independently (Jacobi-style single pass). For a chain of bodies, coupling between joints may cause some drift but NOT explosion. If drift is observed, we can add 2-3 Gauss-Seidel sweeps (process joints root->leaves in the kinematic tree). For ~16 bodies per world this is trivial.

### Differentiability

All operations (division, multiplication, addition, quaternion ops) have Warp adjoints. The kernel is fully compatible with `wp.Tape()` for BPTT gradient computation.

### Verification

1. Run `test_solver_semi_implicit.py` with `SolverSemiImplicitStable` and **NO inertia clamping** — must pass 100 steps without NaN
2. Run full BPTT eval (`g1_eval_worker.py`) with 32 worlds — finite gradients, no NaN
3. Verify `wp.Tape()` backward pass produces non-zero, finite gradients

### Risks

- **Jacobi approximation accuracy**: Joints may drift slightly without coupling. Mitigated by Gauss-Seidel sweep if needed.
- **Angular error computation**: Quaternion errors need careful handling. The same math is in `eval_body_joints` — we mirror it.
- **Edge cases**: FREE joints (root), FIXED joints — need to handle all joint types.

---

## Path C: MuJoCo MJX (Fallback if Path A Fails)

### Overview

MuJoCo MJX is MuJoCo rewritten in JAX. It uses an implicit constraint solver that handles small inertias natively. JAX autodiff provides BPTT gradients.

### Pros

- Same MJCF format — reuse `g1.xml` directly
- Implicit constraint solver — no stability issues with small bodies
- Batched GPU simulation via `jax.vmap`
- Active development (latest release: Feb 2026)

### Cons

- CG solver's `jax.lax.while_loop` **blocks reverse-mode differentiation** unless solver iterations are fixed to a constant
- Contact gradients are noisy/inaccurate without DiffMJX modifications (June 2025 paper: "Hard Contacts with Soft Gradients")
- 10x slower than CPU MuJoCo for single environment
- Requires JAX — integration with PyTorch/MimicKit needs DLPack tensor bridges
- Significant refactoring: need to rewrite the BPTT loop in JAX
- Joint axis gradient optimization not documented anywhere (novel but unproven)

### Implementation Sketch

```python
import jax
import mujoco
from mujoco import mjx

# Load model (same MJCF as MimicKit)
mj_model = mujoco.MjModel.from_xml_path("g1.xml")
mjx_model = mjx.put_model(mj_model)

# Differentiable forward pass
def loss_fn(theta, actions, mjx_model):
    model = apply_theta_to_model(mjx_model, theta)  # Modify joint axes
    state = mjx.make_data(model)
    for t in range(horizon):
        state = set_actions(state, actions[t])
        state = mjx.step(model, state)
    return compute_cot(state)

# BPTT gradient via JAX autodiff
grad_fn = jax.grad(loss_fn)
gradient = grad_fn(theta, actions, mjx_model)
```

### Effort Estimate

- Rewrite BPTT loop in JAX: ~2-3 days
- Handle MJCF parameter modification in JAX: ~1 day
- Action bridge (PyTorch -> JAX for collected actions): ~0.5 days
- Testing and debugging gradient quality: ~2-3 days
- **Total: ~1-2 weeks**

---

## Execution Order

1. **Path A first** — Create `SolverSemiImplicitStable`, test with G1
2. **If Path A works** — Remove all inertia clamping, verify BPTT gradients, done
3. **If Path A fails** (coupling too severe, gradients NaN) — Fall back to Path C (MuJoCo MJX)
