# Cautious BFGS Upgrade Plan for PGHC Outer Loop

## Context

The outer loop in `codesign_g1_unified.py` currently uses Adam (lines 299-323) to update morphology parameters from FD gradients. Adam treats each parameter independently — it has no notion of cross-parameter coupling. When scaling beyond 6 oblique angles to include link lengths, motor params, etc., these parameters are heavily coupled (hip angle affects optimal knee angle, link length affects joint load profiles). We want BFGS because it builds a full inverse Hessian approximation from the history of past (step, gradient-change) pairs, learning the coupling structure with zero extra evaluations.

Full BFGS (not L-BFGS) is appropriate here because at 6 params a 6×6 matrix is trivial, and even at 20+ params a dense n×n matrix is negligible. Full BFGS gives a better Hessian estimate than limited-memory since it uses the entire history, not just the last m pairs.

## Scope

**Only** replacing what happens between "we have a gradient" and "we produce a parameter update." Everything else is untouched: inner loop, FD gradient estimation, stability gating, wandb, multi-GPU, video capture.

## File to modify

`codesign/codesign_g1_unified.py`

---

## Change 1: Replace `AdamOptimizer` class with `CautiousBFGS` class (lines 295-323)

Delete the `AdamOptimizer` class. Replace with:

```python
class CautiousBFGS:
    """Cautious BFGS for morphology optimization (gradient ascent).

    Maintains an explicit n×n inverse Hessian approximation H_k, updated
    each iteration via the BFGS formula. At 6-20 params this is trivial.

    Cautious: skips the Hessian update when s^T y <= 0 (negative/zero
    curvature from stochastic FD noise). H_k stays at its previous value
    rather than being corrupted by a bad update.
    """
```

**API** (split into two calls so bounds clipping and future trust region can intervene):

- `compute_direction(grad) -> direction`: Returns `H_k @ grad` — the inverse-Hessian-scaled ascent direction.
- `update(actual_s, grad_new)`: Called AFTER the step is taken (post-clip). Computes `y = grad_new - grad_prev`, applies cautious filter (`s^T y > eps`), then runs the standard BFGS inverse Hessian update formula on `H_k`. Stores `grad_new` as `prev_grad` for the next iteration.

**BFGS inverse Hessian update formula** (Sherman-Morrison, rank-2):
```
rho = 1 / (y^T s)
H_{k+1} = (I - rho * s * y^T) @ H_k @ (I - rho * y * s^T) + rho * s * s^T
```

**Initial state**: `H_0 = I` (identity matrix). After the first accepted `(s, y)` pair, rescale `H` by `gamma = (s^T y) / (y^T y)` before the first update. This one-time rescaling (standard practice) makes the initial Hessian approximation match the local curvature scale rather than being arbitrarily identity-scaled.

**Cautious filter**: If `s^T y <= 1e-10`, skip the BFGS update entirely. `H_k` stays as `H_{k-1}`. This is the key robustness feature for noisy FD gradients — prevents the Hessian from being corrupted by negative curvature estimates.

**Reset mechanism**: If the Hessian becomes poorly conditioned (eigenvalue ratio > 1e6, checked via `np.linalg.cond`), reset `H_k = I` and log a warning. This is a safety valve.

**State**: `H` (n×n float64 matrix), `prev_grad` (n, float64 or None), `initialized` (bool, tracks whether first scaling has been applied), `num_updates` (int, count of accepted BFGS updates), `num_skipped` (int, count of cautious rejections), `last_gamma` (float, most recent scaling factor for diagnostics).

## Change 2: Add argparse flag (lines ~1442 area)

No new flags needed (the `--lbfgs-history` flag from L-BFGS is not applicable). Just change the `--design-lr` default:

```
--design-lr        default 1.0  (was 0.005; BFGS self-scales via Hessian)
```

The `design_lr` acts as a fixed scaling factor on the BFGS direction. Default 1.0 means "trust the BFGS step fully."

## Change 3: Update the design update block in `pghc_worker` (lines 1310-1325)

Current code (rank 0 only):
```python
# lines 1316-1319
old_theta = theta.copy()
theta = design_optimizer.step(theta, grad_theta)
theta = np.clip(theta, theta_bounds[0], theta_bounds[1])
```

New code (rank 0 only):
```python
old_theta = theta.copy()
direction = bfgs.compute_direction(grad_theta)
theta = old_theta + args.design_lr * direction
theta = np.clip(theta, theta_bounds[0], theta_bounds[1])

# Record actual step (post-clip) for Hessian update
actual_s = theta - old_theta
bfgs.update(actual_s, grad_theta)
```

The `update` method internally computes `y = grad_theta - prev_grad`, applies the cautious filter, and updates `H_k` if accepted. The `actual_s` is post-clip so the Hessian learns the effective landscape including bound effects.

## Change 4: Update optimizer instantiation (line 1152)

```python
# OLD:
design_optimizer = AdamOptimizer(NUM_DESIGN_PARAMS, lr=args.design_lr)

# NEW:
bfgs = CautiousBFGS(NUM_DESIGN_PARAMS)
```

## Change 5: Update configuration printout (lines ~1207)

Change:
```python
print(f"  Design optimizer:  Adam (lr={args.design_lr})")
```
to:
```python
print(f"  Design optimizer:  Cautious BFGS (n={NUM_DESIGN_PARAMS}, lr={args.design_lr})")
```

## Change 6: Add wandb diagnostics (lines ~1374-1388)

Add to the existing `log_dict`:
```python
log_dict["outer/bfgs_updates"] = bfgs.num_updates
log_dict["outer/bfgs_skipped"] = bfgs.num_skipped
log_dict["outer/bfgs_cond"] = float(np.linalg.cond(bfgs.H))
```

- `bfgs_updates`: cumulative count of accepted Hessian updates (curvature pairs where `s^T y > 0`)
- `bfgs_skipped`: cumulative count of rejected pairs (cautious filter triggered)
- `bfgs_cond`: condition number of H — monitors Hessian health; if it spikes, the reset mechanism fires

---

## Design Decisions

**Why full BFGS over L-BFGS**: With n=6 (scaling to ~20), the n×n matrix is 6×6=36 doubles = 288 bytes. Negligible. Full BFGS uses the entire history of accepted pairs, not just the last m, giving a better Hessian approximation. L-BFGS would only make sense at n=100+.

**Why `design_lr` defaults to 1.0**: BFGS's `H_k @ grad` already produces a step with curvature-adapted magnitude. The `lr` is a safety multiplier on top. At 1.0, we trust BFGS fully. This replaces Adam's `lr=0.005` which was needed because Adam's normalized direction has unit scale.

**Why split `compute_direction` / `update`**: The Hessian must be updated with the *actually taken* step (post-clip), not the proposed step. Bounds clipping can shorten or redirect the step, and the Hessian should learn from what actually happened. This split also makes it trivial to add a trust region later — it sits between the two calls.

**Why condition number reset**: With noisy FD gradients, even the cautious filter can't prevent all bad updates. If `H` drifts to poor conditioning over many iterations, resetting to identity is a clean recovery. The BFGS update will quickly re-learn the local curvature from fresh pairs.

**First iteration behavior**: `H_0 = I`, so `direction = I @ grad = grad`. With `lr=1.0`, the first step is `theta + grad`, which is steepest ascent with unit step. This is reasonable — the Hessian hasn't been informed yet, so the step size depends on the gradient magnitude. If this is too aggressive, the user can set `--design-lr 0.1` or similar for conservatism on the first step.

## Verification

1. Run with `--outer-iters 3 --max-inner-iters 500` to verify the flow works end-to-end
2. Check wandb for `outer/bfgs_updates` incrementing (pairs being accepted) and `outer/bfgs_cond` staying reasonable (< 1e4 or so)
3. First outer iteration: direction = gradient (no history yet)
4. Second+ iterations: direction should differ from gradient (BFGS is rotating/scaling)
5. Print `H_k` after a few iterations to visually inspect the learned coupling structure (off-diagonal elements should be non-zero for coupled params like hip_pitch/knee)
6. If all cautious pairs are rejected, optimizer gracefully degrades to steepest ascent (H stays at I) — verify no crash
