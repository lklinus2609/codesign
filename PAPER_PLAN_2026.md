# GBC Paper Plan for CoRL/Humanoids 2026 — Honest Assessment & Pre-Submission Roadmap

**Author**: Linus Kim
**Last updated**: 2026-05-07
**Target venues**: CoRL 2026 (Sep deadline) | IEEE Humanoids 2026 (Nov deadline)

---

## 1. Brutally honest status

We have a working bilevel codesign pipeline (`codesign_g1_unified.py`) that runs SPSA and CMA-ES outer optimizers on G1 humanoid morphology, with multi-GPU training, AMP-style imitation, and (newly) soft-drop of low-SNR design axes. ~57 wandb runs over 6 weeks; ~10 are "meaningful" (completed > 5 outer iters). **Every one is N=1 seed.**

### The data we actually have (single-seed each)

| Run | Method | Final CoT | Outer iters | Wallclock |
|---|---|---|---|---|
| n2klgxkw | SPSA full plateau, no soft-drop | **0.118** | 24 | 17.4 hr |
| ps30cnin | SPSA full ablate-env, 60 seeds, no drop | 0.139 | 5 | 1.1 hr |
| q8l4opbu | SPSA full ablate-env, 30 seeds, no drop | 0.131 | 10 | 22.6 hr |
| zevmhbtp | SPSA full ablate-env, 60 seeds, **soft-drop ON** | 0.145 (best 0.138) | 10 | 4.7 hr |
| iv6l0enx | CMA-ES full constrained (sigma0=0.0087) | 0.127 | 21 | 15.9 hr |
| 2srymn8j | CMA-ES full unconstrained (sigma0=0.05) | 0.158 | 27 | 18.4 hr |
| imnt7ahm | CMA-ES full unconstrained (16384 envs) | 0.158 | 16 | 14.7 hr |
| rqlj8hqv | SPSA lower-scope (6 axes, old code) | 0.137 | 35 | 8.4 hr |

Plus various crashed/short runs and ~15 fixed-morphology baseline attempts.

### What the data tells us today

- SPSA edges CMA-ES under matched constraints by ~7% in final CoT (0.118 vs 0.127), single seed
- The pre-fix CMA-ES (unconstrained) gap was much larger (0.118 vs 0.158) but was operating outside the envelope-theorem-valid regime
- Soft-drop fires correctly (wrist_roll dropped first, shoulder_roll/yaw next — matches physical intuition)
- The bilevel codesign framework reduces CoT from ~0.34 (kickoff-converged baseline) to ~0.12-0.14 (after optimization) — but **this gap may include policy-improvement effects, not just morphology**

---

## 2. Reviewer-perspective stress test

The questions a CoRL/Humanoids reviewer asks (paraphrased):

1. **"How do I know this isn't lucky?"** — We have no defense. Single-seed runs.
2. **"Is the morphology actually better, or did the policy adapt?"** — Untested. Need fresh-policy-on-final-theta validation.
3. **"How does this compare to existing co-design methods?"** — Unaddressed. Ha 2017, Schaff 2019, Wang 2019, Belmonte-Baeza 2022.
4. **"Is the comparison fair?"** — Different runs use different inner regimes (plateau vs ablate-env), different env counts, different wallclock. No matched-budget comparison.
5. **"Which components matter?"** — Soft-drop, trust region, cautious BFGS — no clean ablation A/B runs.
6. **"Does it generalize?"** — Acknowledge as future work.

### Probability assessment

Plausible 3-seed multi-seed outcomes:

| Outcome | Probability | Paper status |
|---|---|---|
| SPSA clearly beats CMA-ES (gap > 2× pooled stderr, p<0.05) | ~25% | Strong |
| Numerical advantage but not statistically significant (gap ≈ stderr) | ~40% | Weak, needs reframing |
| Both essentially tied within noise | ~25% | Reframe as characterization |
| Baseline matches GBC at matched compute | ~10% | Paper killed |

**Combined "publishable headline result": ~40%.** A coin flip with extra steps.

---

## 3. Why I'm not optimistic about the current state

### 3a. Gap-to-stderr ratio

- Current single-seed gap: 0.009 absolute (7% relative)
- Expected stderr at 3 seeds: ~0.005-0.010 (educated guess for stochastic PPO + bilevel noise)
- **Gap ≈ 1× stderr → not statistically distinguishable with 3 seeds**

### 3b. SPSA's machinery is degraded

- **BFGS skip rate is 100%** in recent runs → BFGS preconditioning never engages → effectively running gradient descent
- **Trust region collapsed to 0.05° floor** in n2klgxkw; even with soft-drop (zevmhbtp), tr oscillates and rho still goes negative
- **rho test is structurally biased** in bilevel setting (mixes design effect with policy improvement)
- **If we "win," we win with degraded SPSA** — fixing it may widen the gap, but requires effort first

### 3c. Improvement-over-baseline claim is untested

- Don't know if bilevel framework beats fixed-morph PPO at matched compute
- Most CoT improvement might be policy refinement (40K+ inner PPO iters of training), not morphology
- A reviewer asking this single question kills the paper

### 3d. Soft-drop is engineering heuristic, not theory

- Reviewers will ask "why median over W=3?" and "what about group-LASSO?"
- Without sensitivity analysis, the contribution looks ad-hoc

---

## 4. Pipeline readiness — what must be fixed before committing 2 weeks of GPU time

| Issue | Severity | Fix cost | Why it matters |
|---|---|---|---|
| Rho test conflates policy & design improvement | **HIGH** | ~1 day code | Without this, trust region & BFGS never work in bilevel; SPSA is degraded |
| BFGS skip rate stuck at 100% | **HIGH** | ~1 day investigation | Without curvature info, SPSA loses its core advantage over CMA-ES |
| No final-design validation in pipeline | **HIGH** | ~0.5 day code + 6 hr compute | Without this, the "we made the morphology better" claim is undefended |
| Best-ever CoT not tracked | MEDIUM | ~1 hr code | Final-iter CoT can be a bad excursion (zevmhbtp 0.138 best vs 0.145 final) |
| No multi-seed launcher script | MEDIUM | ~0.5 day script | Need automated identical-budget runs |
| Mean SNR display includes dropped axes (cosmetic) | LOW | ~30 min code | Misleading wandb plots |

**Total: ~3-4 days of code work.** Skip these and we get statistically reliable evidence of a degraded SPSA tying CMA-ES.

---

## 5. SPSA-specific gradient-quality fix plan (priority order)

### Fix 1 — Frozen-policy rho test (HIGHEST PRIORITY)

**The bug**: Current `actual_improvement = center_reward[K] − prev_center_reward` mixes:
- Effect of theta change (theta_{K-1} → theta_K) at fixed policy — what the gradient should predict
- Effect of policy improvement (policy_{K-1} → policy_K) during iter K's inner training — uncorrelated with the design gradient

When inner training is far from convergence (which it always is, per `envelope_grad_norm ≈ 850`), policy improvement dominates the small design-step effect, and rho becomes meaningless.

**The fix**: at the START of outer iter K+1, BEFORE inner training begins, evaluate reward at theta_{K+1} with the current (frozen) policy_K. Use this as the actual-improvement reference.

**Concrete changes** (in `codesign_g1_unified.py`):

1. New helper function `eval_center_reward_frozen_policy(agent, env, engine, char_id, total_mass, theta_np, ...)` modeled on `compute_spsa_gradient_parallel` but with only the center partition (no perturbations). Returns `frozen_reward` scalar. ~50 LOC.

2. In outer loop, AFTER `update_training_joint_X_p(sim_model, theta, ...)` at line 2949 and BEFORE `inner_ctrl.train_until_converged(iter_dir)` at line 2587:
   ```python
   if args.mode == "spsa" and outer_iter > 0:
       frozen_baseline_reward = eval_center_reward_frozen_policy(...)
   ```

3. Replace the rho test at line 2702:
   ```python
   actual_improvement = frozen_baseline_reward - prev_center_reward  # was: center_reward - prev_center_reward
   ```

4. Continue setting `prev_center_reward = center_reward` (line 2718) — same as before, this is the "policy_K converged at theta_K" reward, the LHS of the rho test.

**Cost per outer iter**: 1 extra rollout of `num_spsa_seeds × eval_horizon` steps with center morphology. Estimate ~30-60 sec per outer iter (vs ~15-20 min for the full SPSA gradient eval).

**Expected result**: rho should reliably be in [0.5, 1.5] when gradient is good. Trust radius should grow. BFGS may engage if the curvature condition is satisfied.

### Fix 2 — Investigate BFGS 100% skip rate (HIGH PRIORITY)

**The mechanism**: cautious skip fires when `s^T y <= 1e-10` where:
- s = actual step taken (post-trust-region clip)
- y = `prev_grad - grad_new` (negated for ascent → minimization)

100% skip rate means s and y are essentially uncorrelated → either:
- (a) Step is so small (clamped to 0.05° floor) that it carries no info
- (b) Gradient noise is so large that y has random direction
- (c) Reward landscape is genuinely non-quadratic (linear, ridges, etc.)

**Diagnosis tasks**:

1. Log `s^T y / (||s|| * ||y||)` (cosine similarity) at every outer iter. If chronically near zero, gradient is uncorrelated with steps. If chronically negative, landscape may be saddle-like.

2. Check if Fix 1 (working trust region) makes BFGS engage. If trust radius grows to 0.5° and skip rate drops, BFGS just needed informative steps to work with.

3. If skip rate remains >50% even with Fix 1, consider:
   - (a) Lower threshold from 1e-10 to 1e-3 with damping (treat marginal as low-confidence update with reduced weight)
   - (b) **Replace BFGS with Adam** (per-axis adaptive learning rate, momentum smoothing — built for noisy gradients)
   - (c) Use SR1 with safeguard (better for indefinite Hessians)

**Recommended path**: implement Adam as alternative outer optimizer (`--design-optimizer {bfgs,adam}`), run head-to-head, pick the winner for paper. ~half day code.

```python
# Sketch of Adam outer step
m = beta1 * m + (1-beta1) * grad_theta
v = beta2 * v + (1-beta2) * grad_theta**2
m_hat = m / (1 - beta1**(outer_iter+1))
v_hat = v / (1 - beta2**(outer_iter+1))
step = adam_lr * m_hat / (np.sqrt(v_hat) + 1e-8)
```

For bilevel with noisy SPSA gradients, Adam is theoretically a better fit than BFGS. The choice "BFGS for sophistication" was an aesthetic mistake.

### Fix 3 — Best-ever CoT tracking (LOW EFFORT, HIGH READABILITY)

Current `outer/cot` is the current iter's value, which can be a bad excursion. Add `outer/best_cot` and `outer/best_theta_iter` to wandb log. ~5 LOC change near line 3050.

Both methods (SPSA, CMA-ES) should report best AND final separately. Current setup unfairly compares "SPSA's last bad iter" vs "CMA-ES's tracked-best."

### Fix 4 — Final-design validation scaffold (HIGH PRIORITY)

New script `validate_final_design.py` (~150 LOC):
1. Load a finished codesign run's final `theta_K`
2. Build a fresh agent + environment at `theta_K` (no warm start, no inner-loop history)
3. Run full PPO training from scratch for N inner iters (same budget as a single outer iter ≈ 5-10 min on multi-GPU)
4. Report final CoT
5. Repeat for `theta_0` (initial morphology) as the comparator

**This is the test that defends the morphology-actually-improved claim.** Without it, a reviewer can argue we just trained a better policy, not a better morphology.

### Fix 5 — Mean SNR cosmetic (LOW PRIORITY)

In wandb log block (~line 3092), change:
```python
finite_snr = grad_snr[np.isfinite(grad_snr)]
```
to:
```python
finite_snr = grad_snr[np.isfinite(grad_snr) & active_mask]  # exclude dropped axes
```
So mean SNR reflects only the axes still being optimized.

---

## 6. Pre-flight experiment plan (1 week, ~200 GPU-hr)

### Step 1: Code fixes (3-4 days)

In order:
1. **Fix 1 — frozen-policy rho** (1 day)
2. **Fix 4 — final-design validation script** (0.5 day)
3. **Fix 3 — best-ever CoT tracking** (1 hr)
4. **Fix 5 — SNR cosmetic** (30 min)
5. **Fix 2 part 1 — diagnose BFGS skip** (1 day): log cosine, run a short SPSA-with-fixes test to see if trust region recovers
6. **Multi-seed launcher script** (0.5 day): submits N seeds in parallel sbatch jobs, deposits results in named dirs

### Step 2: 3-seed pre-flight (~24 hr wallclock, ~150 GPU-hr)

ONE comparison only — does GBC beat fixed-morph baseline?

| Condition | Description |
|---|---|
| **Baseline-fixed** | PPO at θ=0 for matched wallclock budget, --baseline flag |
| **GBC-SPSA-fixed** | SPSA + soft-drop + frozen-policy rho fix + (BFGS or Adam, whichever won), resumed from kickoff, --ablate-env-fixed-iters 200 |

3 seeds (7, 11, 13) × 2 conditions × 4-6 hr per run × 8 GPU = ~150-200 GPU-hr.

### Step 3: Decision point (end of week 1)

| Result | Action |
|---|---|
| GBC mean − Baseline mean > 2× pooled stderr | **GREEN**: proceed with Tier 0a (CMA-ES + ablations) |
| Gap < 1× pooled stderr | **YELLOW**: diagnose root cause; reframe paper or fix more deeply |
| Gap is negative (baseline wins) | **RED**: paper claim wrong; fundamental rethink needed |

---

## 7. Conditional Tier 0a expansion (only if Step 3 = GREEN)

Add to the pre-flight 6 runs:

| Condition | Seeds | Cost |
|---|---|---|
| GBC-CMA-ES (constrained, matched compute) | 3 | ~75 GPU-hr |
| GBC-SPSA-no-soft-drop (ablation arm) | 3 | ~75 GPU-hr |
| Final-design validation: fresh PPO on each method's best θ | ~6 | ~75 GPU-hr |

**Tier 0a total** (including pre-flight): ~400-450 GPU-hr.

---

## 8. Conditional Tier 1-2 (only if Tier 0a results are strong)

### Tier 1 — Strong paper additions

- **Convergence-trajectory plot** (CoT vs wallclock with stderr bands) — uses Tier 0a data, just analysis script
- **Per-axis SNR / drop analysis** — uses Tier 0a data, analysis script
- **Soft-drop ablation** — already covered by GBC-SPSA-no-soft-drop arm

### Tier 2 — Defensible-paper additions

- **Component ablations** (1 seed each, ~30 GPU-hr): trust-region OFF, cautious-BFGS OFF, soft-drop OFF, all-guards OFF
- **Sensitivity sweep** (1 seed each, ~50 GPU-hr): num_spsa_seeds, spsa_epsilon, drop_window

### Tier 3 — Nice-to-have

- Pre-fix CMA-ES (already have data)
- Lower-scope (already have data)
- Inner-sample-budget instead of wallclock-budget plots

---

## 9. Fallback paper framings (if Step 3 = YELLOW or worse)

**Reframe A — Characterization paper**
> "We provide the first systematic comparison of gradient-based and gradient-free outer optimizers for bilevel humanoid co-design, identifying when each is preferable and characterizing failure modes."

Doesn't require a headline win. Publishable as a tools/characterization paper or workshop.

**Reframe B — Engineering paper**
> "Soft-drop: an automatic dimensionality reduction technique for SPSA-based bilevel optimization."

Contribution becomes the technique, not the head-to-head win. Standalone usefulness even without beating CMA-ES.

**Reframe C — Negative-result paper**
> "Why naive gradient-based bilevel co-design under-performs: a study of the bilevel non-stationarity problem in humanoid morphology optimization."

Higher difficulty to write but publishable at TMLR or NeurIPS workshops.

**Reframe D — Open-source framework paper**
> "GBC: an open-source pipeline for bilevel humanoid co-design with multi-GPU training, ablation flags, and reproducible benchmarks."

The MimicKit-as-library + multi-GPU + soft-drop + ablation flags is a real engineering contribution. Workshop-friendly.

---

## 10. What we do NOT need (despite professor's wishlist)

For CoRL/Humanoids 2026:
- ❌ Different robots (Spot, Atlas, Cassie)
- ❌ Different tasks (running, jumping, manipulation)
- ❌ Actuator parameters (motor inertia, gear ratios)
- ❌ Real hardware deployment

Each is 2-4 weeks of additional work. Acknowledge in "Future Work." A focused single-robot/single-task paper that nails the methodology beats a sprawling paper that demos 4 things badly.

---

## 11. Critical-path timeline

| Week | Task | Outcome |
|---|---|---|
| Week 1 | Code fixes 1-5; Tier 0 pre-flight (3 seeds, 2 conditions) | Decision point: GREEN / YELLOW / RED |
| Week 2 (if GREEN) | Tier 0a expansion (CMA-ES + ablation arms) | Headline numbers |
| Week 3 | Tier 1 additions (figures, soft-drop ablation) | Strong paper |
| Week 4 | Tier 2 additions (component ablation, sensitivity) | Defensible paper |
| Week 5-6 | Writing | First draft |
| Week 7-8 | Internal review, revisions | Submit |

Total: 8 weeks from now to submission. Compatible with both CoRL Sep 2026 and Humanoids Nov 2026 deadlines.

If Step 3 = YELLOW: stop, reframe, save 2-3 weeks of compute by not running Tier 0a-2.
If Step 3 = RED: pause, diagnose deeper, possibly pivot to Reframe C/D.

---

## 12. Open questions that may invalidate the plan

1. **What's the actual variance of PPO at fixed theta?** We estimated 0.005-0.010 stderr but never measured. If variance is 0.020+, the 0.009 gap is dwarfed by noise even with 10 seeds.
2. **Does the frozen-policy rho fix actually fix BFGS?** If trust region grows but BFGS still skips, we know the gradient noise is the real issue, not the rho miscalibration.
3. **What's a fair "matched compute budget"?** Inner-PPO sample count? Wallclock? Outer iter count? Different choices favor different methods.
4. **Will the prof accept a Reframe paper?** Need this conversation BEFORE Step 3 to avoid friction later.

---

## Appendix: Existing runs to keep / re-run / drop

| Run | Status |
|---|---|
| n2klgxkw (SPSA-30s-plateau) | Keep as historical reference; use in appendix |
| q8l4opbu (SPSA-30s-ablate, no drop) | Keep — useful as "no soft-drop" point if seeds add up |
| ps30cnin (SPSA-60s-ablate, no drop) | Keep — 30 vs 60 seed comparison data |
| zevmhbtp (SPSA-60s-ablate, soft-drop) | Pilot — re-run multi-seed with rho fix |
| iv6l0enx (cmaes3 constrained) | **Re-run with --ablate-envelope-gate** for fair head-to-head |
| 2srymn8j, imnt7ahm (unconstrained CMA-ES) | Keep for "why constraints matter" appendix figure |
| rqlj8hqv (lower scope) | Keep for appendix |
| wrp3uwkj (currently running baseline, 12.3 hr no progress) | **Kill** — stuck in plateau detector; re-run with --baseline --ablate-envelope-gate |
| All other crashed runs | Drop |
