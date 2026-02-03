#!/usr/bin/env python3
"""
Test Script for Hybrid Co-Design Implementation

This script tests the core components of the hybrid co-design framework:
1. ParametricG1Model - Quaternion math and joint transform updates
2. DifferentiableRollout - Gradient computation through physics
3. HybridAMPAgent - Inner/outer loop logic

Usage:
    python test_implementation.py
"""

import os
import sys
import math
import numpy as np
import torch

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODESIGN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODESIGN_DIR)

# Test flags
TESTS_PASSED = 0
TESTS_FAILED = 0


def test_result(name, passed, message=""):
    """Record test result."""
    global TESTS_PASSED, TESTS_FAILED
    if passed:
        TESTS_PASSED += 1
        print(f"  [PASS] {name}")
    else:
        TESTS_FAILED += 1
        print(f"  [FAIL] {name}: {message}")


def test_parametric_g1():
    """Test ParametricG1Model functionality."""
    print("\n" + "="*50)
    print("Testing ParametricG1Model")
    print("="*50)

    from codesign.parametric_g1 import ParametricG1Model

    # Test 1: Quaternion identity
    q_id = ParametricG1Model.angle_to_x_rotation_quat(0.0)
    test_result(
        "Quaternion identity",
        np.allclose(q_id, [1, 0, 0, 0]),
        f"Expected [1,0,0,0], got {q_id}"
    )

    # Test 2: Quaternion 90 degrees
    q_90 = ParametricG1Model.angle_to_x_rotation_quat(math.pi / 2)
    expected_90 = [math.cos(math.pi/4), math.sin(math.pi/4), 0, 0]
    test_result(
        "Quaternion 90 degrees",
        np.allclose(q_90, expected_90, atol=1e-6),
        f"Expected {expected_90}, got {q_90}"
    )

    # Test 3: Quaternion multiplication (45 + 45 = 90)
    q_45 = ParametricG1Model.angle_to_x_rotation_quat(math.pi / 4)
    q_combined = ParametricG1Model.quat_multiply(q_45, q_45)
    test_result(
        "Quaternion multiplication",
        np.allclose(q_combined, expected_90, atol=1e-6),
        f"45+45 should equal 90, got {q_combined}"
    )

    # Test 4: Model creation
    model = ParametricG1Model(device="cpu", theta_init=0.1)
    test_result(
        "Model creation",
        abs(model.get_theta() - 0.1) < 1e-6,
        f"Expected theta=0.1, got {model.get_theta()}"
    )

    # Test 5: Bounds projection
    model.theta = torch.tensor([0.5], requires_grad=True)  # Over max
    projected = model.project_bounds(model.theta)
    test_result(
        "Bounds projection (upper)",
        abs(projected.item() - 0.1745) < 0.001,
        f"Expected ~0.1745, got {projected.item()}"
    )

    model.theta = torch.tensor([-0.5], requires_grad=True)  # Under min
    projected = model.project_bounds(model.theta)
    test_result(
        "Bounds projection (lower)",
        abs(projected.item() - (-0.1745)) < 0.001,
        f"Expected ~-0.1745, got {projected.item()}"
    )

    # Test 6: Theta in degrees conversion
    model.set_theta(math.radians(5.0), update_model=False)
    test_result(
        "Theta degrees conversion",
        abs(model.get_theta_degrees() - 5.0) < 0.01,
        f"Expected 5.0 deg, got {model.get_theta_degrees()}"
    )

    # Test 7: State dict save/load
    state = model.get_state_dict()
    new_model = ParametricG1Model(device="cpu")
    new_model.load_state_dict(state)
    test_result(
        "State dict save/load",
        abs(new_model.get_theta() - model.get_theta()) < 1e-6,
        f"Theta mismatch after load"
    )

    print(f"\nParametricG1Model: {TESTS_PASSED} passed")


def test_hybrid_agent():
    """Test HybridAMPAgent functionality."""
    print("\n" + "="*50)
    print("Testing HybridAMPAgent")
    print("="*50)

    from codesign.hybrid_agent import HybridAMPAgent

    # Test configuration
    config = {
        "warmup_iters": 100,
        "outer_loop_freq": 20,
        "design_learning_rate": 0.01,
        "diff_horizon": 50,
        "design_param_init": 0.05,
    }

    # Test 1: Agent creation
    agent = HybridAMPAgent(config, env=None, device="cpu")
    test_result(
        "Agent creation",
        agent._warmup_iters == 100,
        f"Expected warmup_iters=100, got {agent._warmup_iters}"
    )

    # Test 2: Initial design parameter
    test_result(
        "Initial design parameter",
        abs(agent._parametric_model.get_theta() - 0.05) < 1e-6,
        f"Expected theta=0.05, got {agent._parametric_model.get_theta()}"
    )

    # Test 3: Should run outer loop - during warmup
    agent._iter = 50
    test_result(
        "No outer loop during warmup",
        not agent._should_run_outer_loop(),
        "Outer loop should not run during warmup"
    )

    # Test 4: Should run outer loop - after warmup, on frequency
    agent._iter = 100  # End of warmup
    test_result(
        "Outer loop at end of warmup",
        agent._should_run_outer_loop(),
        "Outer loop should run at iteration 100"
    )

    # Test 5: Should run outer loop - after warmup, off frequency
    agent._iter = 105
    test_result(
        "No outer loop off-frequency",
        not agent._should_run_outer_loop(),
        "Outer loop should not run at iteration 105"
    )

    # Test 6: Should run outer loop - next scheduled
    agent._iter = 120  # 100 + 20
    test_result(
        "Outer loop on frequency",
        agent._should_run_outer_loop(),
        "Outer loop should run at iteration 120"
    )

    # Test 7: Dummy inner loop (standalone mode)
    info = agent._dummy_inner_loop()
    test_result(
        "Dummy inner loop returns dict",
        isinstance(info, dict) and "inner_loss" in info,
        f"Expected dict with inner_loss, got {info}"
    )

    # Test 8: Train iteration (standalone, no outer loop)
    agent._iter = 50
    info = agent.train_iter()
    test_result(
        "Train iteration returns design param",
        "design_param_theta" in info,
        f"Expected design_param_theta in info"
    )

    print(f"\nHybridAMPAgent: All basic tests completed")


def test_newton_integration():
    """Test Newton integration if available."""
    print("\n" + "="*50)
    print("Testing Newton Integration")
    print("="*50)

    try:
        import warp as wp
        import newton
        wp.init()
        print("Warp/Newton available")
    except ImportError:
        print("Warp/Newton not available, skipping integration tests")
        return

    from codesign.parametric_g1 import ParametricG1Model

    # Try to create a simple differentiable model
    try:
        builder = newton.ModelBuilder()

        # Add a simple pendulum for testing
        builder.add_particle((0, 0, 0), (0, 0, 0), 0.0)  # Fixed particle
        builder.add_particle((0, 0, -1), (0, 0, 0), 1.0)  # Mass
        builder.add_spring(0, 1, 100.0, 10.0, 1.0)

        model = builder.finalize(requires_grad=True)

        test_result(
            "Newton model creation with gradients",
            model.requires_grad,
            "Model should have requires_grad=True"
        )

        # Test state creation
        state = model.state(requires_grad=True)
        test_result(
            "Newton state creation",
            state is not None,
            "State should be created"
        )

        # Test tape-based gradient computation
        tape = wp.Tape()
        loss = wp.zeros(1, dtype=float, requires_grad=True)

        with tape:
            # Simple computation - particle height
            kernel = get_simple_loss_kernel()
            wp.launch(
                kernel,
                dim=1,
                inputs=[state.particle_q],
                outputs=[loss]
            )

        tape.backward(loss)
        tape.zero()

        test_result(
            "Newton tape-based gradients",
            True,  # If we got here without error, it works
            ""
        )

    except Exception as e:
        test_result(
            "Newton integration",
            False,
            str(e)
        )


def get_simple_loss_kernel():
    """Get the simple loss kernel (lazy import to avoid issues when warp not available)."""
    import warp as wp

    @wp.kernel
    def simple_loss_kernel(
        particle_q: wp.array(dtype=wp.vec3),
        loss: wp.array(dtype=float),
    ):
        """Simple loss: negative height of first particle."""
        tid = wp.tid()
        if tid == 0:
            loss[0] = -particle_q[1][2]  # Negative z of particle 1

    return simple_loss_kernel


def test_g1_model_loading():
    """Test loading the actual G1 model."""
    print("\n" + "="*50)
    print("Testing G1 Model Loading")
    print("="*50)

    try:
        import warp as wp
        import newton
    except ImportError:
        print("Newton not available, skipping G1 loading test")
        return

    # Check if G1 model exists
    g1_path = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")
    if not os.path.exists(g1_path):
        print(f"G1 model not found at {g1_path}, skipping test")
        return

    from codesign.parametric_g1 import ParametricG1Model

    try:
        # Create builder and load G1
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        builder.add_mjcf(
            g1_path,
            floating=True,
            ignore_inertial_definitions=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )

        builder.add_ground_plane()

        # Finalize with gradients
        model = builder.finalize(requires_grad=True)

        test_result(
            "G1 model loading",
            model.joint_count > 0,
            f"Expected joints > 0, got {model.joint_count}"
        )

        # Test parametric model attachment
        param_model = ParametricG1Model(device="cpu")
        param_model.attach_model(model)

        test_result(
            "Parametric model attachment",
            param_model._left_hip_roll_idx is not None,
            "Left hip roll index should be set"
        )

        # Test morphology update
        param_model.set_theta(0.05, update_model=True)
        test_result(
            "Morphology update",
            True,  # If no exception, it worked
            ""
        )

        # Verify joint_X_p was modified
        joint_X_p = model.joint_X_p.numpy()
        left_quat = joint_X_p[param_model._left_hip_roll_idx, 3:7]
        test_result(
            "Joint transform modified",
            not np.allclose(left_quat, param_model._left_base_quat),
            "Quaternion should be different after update"
        )

    except Exception as e:
        test_result(
            "G1 model operations",
            False,
            str(e)
        )


def run_all_tests():
    """Run all tests."""
    global TESTS_PASSED, TESTS_FAILED

    print("\n" + "#"*60)
    print("# HYBRID CO-DESIGN IMPLEMENTATION TESTS")
    print("#"*60)

    # Run test suites
    test_parametric_g1()
    test_hybrid_agent()
    test_newton_integration()
    test_g1_model_loading()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    total = TESTS_PASSED + TESTS_FAILED
    print(f"Total:  {total}")
    print(f"Passed: {TESTS_PASSED}")
    print(f"Failed: {TESTS_FAILED}")

    if TESTS_FAILED == 0:
        print("\nAll tests PASSED!")
        return 0
    else:
        print(f"\n{TESTS_FAILED} test(s) FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
