#!/usr/bin/env python3
"""
Level 2: Ant Environment Verification

Tests:
1. Ant environment initialization and basic stepping
2. Parametric morphology changes
3. Gradient computation via finite difference
4. Performance benchmark

Run on machine with Newton/Warp installed:
    python test_level2_ant.py
"""

import numpy as np
import time

# Try to import Ant environment
try:
    from envs.ant import AntEnv, ParametricAnt
    NEWTON_AVAILABLE = True
except ImportError as e:
    print(f"Newton not available: {e}")
    NEWTON_AVAILABLE = False


def test_ant_basic():
    """Test basic Ant environment functionality."""
    print("\n" + "=" * 60)
    print("Test 1: Ant Environment Basic Functionality")
    print("=" * 60)

    if not NEWTON_AVAILABLE:
        print("SKIPPED: Newton not available")
        return False

    try:
        env = AntEnv()
        print(f"  Created environment")
        print(f"  Leg length: {env.parametric_model.leg_length:.3f} m")
        print(f"  Foot length: {env.parametric_model.foot_length:.3f} m")
        print(f"  Torso radius: {env.parametric_model.torso_radius:.3f} m")
        print(f"  Obs dim: {env.obs_dim}, Act dim: {env.act_dim}")

        obs = env.reset()
        print(f"  Reset successful, obs shape: {obs.shape}")

        # Run a few steps with random actions
        total_reward = 0
        for i in range(20):
            action = np.random.uniform(-1, 1, size=env.act_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                break

        print(f"  Stepped 20 times successfully")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final x position: {info['x']:.3f}")
        print(f"  Healthy: {info['healthy']}")
        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parametric_morphology():
    """Test changing morphology parameters."""
    print("\n" + "=" * 60)
    print("Test 2: Parametric Morphology Changes")
    print("=" * 60)

    if not NEWTON_AVAILABLE:
        print("SKIPPED: Newton not available")
        return False

    try:
        # Test different leg lengths
        leg_lengths = [0.2, 0.28, 0.35]

        for leg_len in leg_lengths:
            model = ParametricAnt(leg_length=leg_len)
            env = AntEnv(parametric_model=model)

            print(f"  Leg length = {leg_len:.2f}m:")
            print(f"    Created env, running 10 steps...")

            obs = env.reset()
            for _ in range(10):
                action = np.zeros(env.act_dim)  # No torque
                obs, reward, terminated, _, info = env.step(action)
                if terminated:
                    break

            print(f"    Final z: {obs[0]:.3f}, healthy: {info['healthy']}")

        print("PASSED")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_computation():
    """Test gradient computation via finite difference."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient Computation (Finite Difference)")
    print("=" * 60)

    if not NEWTON_AVAILABLE:
        print("SKIPPED: Newton not available")
        return False

    try:
        env = AntEnv()

        # Simple policy: random but deterministic
        np.random.seed(42)
        policy_weights = np.random.randn(env.obs_dim, env.act_dim) * 0.1

        def simple_policy(obs):
            return np.tanh(obs @ policy_weights)

        print("  Computing gradient dReturn/d(leg_length)...")
        start_time = time.time()
        mean_return, gradient = env.compute_gradient_wrt_morphology(
            simple_policy,
            param_name="leg_length",
            horizon=100,
            n_rollouts=3,
            eps=0.02,
        )
        elapsed = time.time() - start_time

        print(f"  Mean return: {mean_return:.2f}")
        print(f"  dReturn/d(leg_length): {gradient:.4f}")
        print(f"  Time: {elapsed:.2f}s")

        # Also test foot_length gradient
        print("\n  Computing gradient dReturn/d(foot_length)...")
        start_time = time.time()
        mean_return2, gradient2 = env.compute_gradient_wrt_morphology(
            simple_policy,
            param_name="foot_length",
            horizon=100,
            n_rollouts=3,
            eps=0.02,
        )
        elapsed2 = time.time() - start_time

        print(f"  Mean return: {mean_return2:.2f}")
        print(f"  dReturn/d(foot_length): {gradient2:.4f}")
        print(f"  Time: {elapsed2:.2f}s")

        print("PASSED: Gradient computation complete")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test simulation performance."""
    print("\n" + "=" * 60)
    print("Test 4: Performance Benchmark")
    print("=" * 60)

    if not NEWTON_AVAILABLE:
        print("SKIPPED: Newton not available")
        return False

    try:
        env = AntEnv()
        n_steps = 500
        obs = env.reset()

        start_time = time.time()
        for _ in range(n_steps):
            action = np.random.uniform(-1, 1, size=env.act_dim)
            obs, _, terminated, _, _ = env.step(action)
            if terminated:
                obs = env.reset()
        elapsed = time.time() - start_time

        steps_per_sec = n_steps / elapsed
        print(f"  {n_steps} steps in {elapsed:.2f}s")
        print(f"  {steps_per_sec:.0f} steps/second")

        if steps_per_sec > 50:
            print("PASSED: Performance acceptable")
        else:
            print("WARNING: Performance might be slow for training")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mjcf_generation():
    """Test MJCF generation with different parameters."""
    print("\n" + "=" * 60)
    print("Test 5: MJCF Generation")
    print("=" * 60)

    try:
        model = ParametricAnt(leg_length=0.3, foot_length=0.5, torso_radius=0.2)
        mjcf = model.generate_mjcf()

        print(f"  Generated MJCF with {len(mjcf)} characters")

        # Check key elements are present
        checks = [
            ("torso", "torso_geom" in mjcf),
            ("leg joints", "hip_1" in mjcf and "ankle_1" in mjcf),
            ("actuators", "<actuator>" in mjcf),
            ("4 legs", mjcf.count("hip_") == 4),
        ]

        all_passed = True
        for name, passed in checks:
            status = "OK" if passed else "MISSING"
            print(f"    {name}: {status}")
            all_passed = all_passed and passed

        if all_passed:
            print("PASSED")
        else:
            print("FAILED: Missing MJCF elements")
        return all_passed

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Level 2: Ant Environment Verification")
    print("=" * 60)

    if not NEWTON_AVAILABLE:
        print("\nNewton/Warp not installed in this environment.")
        print("Run this test on a machine with Newton installed.")
        print("\nRunning MJCF generation test (no Newton required)...")

    results = {
        "mjcf_generation": test_mjcf_generation(),
        "basic": test_ant_basic(),
        "parametric": test_parametric_morphology(),
        "gradient": test_gradient_computation(),
        "performance": test_performance(),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        if not NEWTON_AVAILABLE and name != "mjcf_generation":
            status = "SKIP"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    n_passed = sum(results.values())
    n_total = len(results)
    print(f"\n{n_passed}/{n_total} tests passed")

    if NEWTON_AVAILABLE and n_passed == n_total:
        print("\nLevel 2 COMPLETE - Ready for PGHC co-design on Ant!")


if __name__ == "__main__":
    main()
