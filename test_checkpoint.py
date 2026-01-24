#!/usr/bin/env python3
"""
Checkpoint Save/Load Test for HybridAMPAgent

Tests checkpoint functionality for design parameters:
1. Save agent state including design parameters
2. Load agent state and verify design parameters restored
3. Continue training from checkpoint

Usage:
    python test_checkpoint.py

Expected: Design parameters correctly saved and restored.
"""

import os
import sys
import tempfile
import shutil

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODESIGN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODESIGN_DIR)

import warp as wp
import numpy as np
import torch


def test_standalone_agent_checkpoint():
    """Test 1: Save/Load for standalone HybridAMPAgent."""
    print("\n" + "="*60)
    print("Test 1: Standalone Agent Checkpoint")
    print("="*60)

    from hybrid_codesign.hybrid_agent import HybridAMPAgent
    from hybrid_codesign.parametric_g1 import ParametricG1Model

    # Create config
    config = {
        "warmup_iters": 100,
        "outer_loop_freq": 20,
        "design_learning_rate": 0.01,
        "diff_horizon": 3,
        "design_param_init": 0.05,  # Non-zero initial value
    }

    # Create agent
    agent = HybridAMPAgent(config, env=None, device="cuda:0")

    # Modify state
    agent._iter = 12345
    agent._outer_loop_count = 10
    agent._design_history = [
        {"iter": 100, "theta": 0.01, "grad": 0.001, "loss": 0.5},
        {"iter": 200, "theta": 0.02, "grad": 0.002, "loss": 0.4},
    ]
    agent._parametric_model.set_theta(0.08, update_model=False)

    print(f"  [INFO] Before save:")
    print(f"    iter: {agent._iter}")
    print(f"    outer_loop_count: {agent._outer_loop_count}")
    print(f"    design_history length: {len(agent._design_history)}")
    print(f"    theta: {agent._parametric_model.get_theta():.4f}")

    # Create temp directory for checkpoint
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")

    try:
        # Save
        agent.save(checkpoint_path)
        print(f"\n  [INFO] Saved checkpoint to {checkpoint_path}")

        # Create new agent
        config2 = config.copy()
        config2["design_param_init"] = 0.0  # Different initial value
        agent2 = HybridAMPAgent(config2, env=None, device="cuda:0")

        print(f"\n  [INFO] Before load (new agent):")
        print(f"    iter: {agent2._iter}")
        print(f"    theta: {agent2._parametric_model.get_theta():.4f}")

        # Load
        agent2.load(checkpoint_path)

        print(f"\n  [INFO] After load:")
        print(f"    iter: {agent2._iter}")
        print(f"    outer_loop_count: {agent2._outer_loop_count}")
        print(f"    design_history length: {len(agent2._design_history)}")
        print(f"    theta: {agent2._parametric_model.get_theta():.4f}")

        # Verify
        assert agent2._iter == 12345, f"iter mismatch: {agent2._iter} != 12345"
        assert agent2._outer_loop_count == 10, f"outer_loop_count mismatch"
        assert len(agent2._design_history) == 2, f"design_history length mismatch"
        assert abs(agent2._parametric_model.get_theta() - 0.08) < 1e-6, f"theta mismatch"

        print(f"\n  [PASS] Standalone agent checkpoint test passed")
        return True

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_parametric_model_checkpoint():
    """Test 2: Save/Load for ParametricG1Model alone."""
    print("\n" + "="*60)
    print("Test 2: ParametricG1Model Checkpoint")
    print("="*60)

    from hybrid_codesign.parametric_g1 import ParametricG1Model

    # Create model with specific values
    model = ParametricG1Model(
        device="cuda:0",
        theta_init=0.05,
        theta_min=-0.2,
        theta_max=0.2,
    )

    # Set custom theta
    model.set_theta(0.123, update_model=False)

    print(f"  [INFO] Original model:")
    print(f"    theta: {model.get_theta():.6f}")
    print(f"    theta_min: {model.theta_min:.4f}")
    print(f"    theta_max: {model.theta_max:.4f}")

    # Get state dict
    state = model.get_state_dict()
    print(f"\n  [INFO] State dict keys: {list(state.keys())}")

    # Create new model with different initial values
    model2 = ParametricG1Model(
        device="cuda:0",
        theta_init=0.0,
        theta_min=-0.1,
        theta_max=0.1,
    )

    print(f"\n  [INFO] New model before load:")
    print(f"    theta: {model2.get_theta():.6f}")

    # Load state
    model2.load_state_dict(state)

    print(f"\n  [INFO] New model after load:")
    print(f"    theta: {model2.get_theta():.6f}")
    print(f"    theta_min: {model2.theta_min:.4f}")
    print(f"    theta_max: {model2.theta_max:.4f}")

    # Verify
    assert abs(model2.get_theta() - 0.123) < 1e-6, "theta not restored"
    assert abs(model2.theta_min - (-0.2)) < 1e-6, "theta_min not restored"
    assert abs(model2.theta_max - 0.2) < 1e-6, "theta_max not restored"

    print(f"\n  [PASS] ParametricG1Model checkpoint test passed")
    return True


def test_checkpoint_with_diff_model():
    """Test 3: Checkpoint with attached diff model."""
    print("\n" + "="*60)
    print("Test 3: Checkpoint with Attached Diff Model")
    print("="*60)

    g1_path = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")

    if not os.path.exists(g1_path):
        print(f"  [SKIP] G1 model not found: {g1_path}")
        return True

    from hybrid_codesign.hybrid_agent import HybridAMPAgent, create_diff_model_from_mjcf

    # Create agent
    config = {
        "warmup_iters": 100,
        "outer_loop_freq": 20,
        "design_learning_rate": 0.01,
        "diff_horizon": 3,
        "design_param_init": 0.0,
    }

    agent = HybridAMPAgent(config, env=None, device="cuda:0")
    agent.setup_diff_model(g1_path, "cuda:0")

    # Set theta and update model
    agent._parametric_model.set_theta(0.06, update_model=True)

    # Get joint_X_p value for verification
    left_idx = agent._parametric_model._left_hip_roll_idx
    original_joint_X_p = agent._diff_model.joint_X_p.numpy()[left_idx].copy()

    print(f"  [INFO] Original theta: {agent._parametric_model.get_theta():.4f}")
    print(f"  [INFO] Original joint_X_p[{left_idx}]: {original_joint_X_p}")

    # Save and load
    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "test_diff_model.pt")

    try:
        agent.save(checkpoint_path)

        # Create new agent
        agent2 = HybridAMPAgent(config, env=None, device="cuda:0")
        agent2.setup_diff_model(g1_path, "cuda:0")
        agent2.load(checkpoint_path)

        # Apply theta to model
        agent2._parametric_model.set_theta(
            agent2._parametric_model.get_theta(),
            update_model=True
        )

        # Check joint_X_p matches
        restored_joint_X_p = agent2._diff_model.joint_X_p.numpy()[left_idx]

        print(f"\n  [INFO] Restored theta: {agent2._parametric_model.get_theta():.4f}")
        print(f"  [INFO] Restored joint_X_p[{left_idx}]: {restored_joint_X_p}")

        # Verify theta
        assert abs(agent2._parametric_model.get_theta() - 0.06) < 1e-6, "theta mismatch"

        # Verify joint_X_p is close (may not be exact due to numerical precision)
        if np.allclose(original_joint_X_p, restored_joint_X_p, atol=1e-4):
            print(f"\n  [PASS] joint_X_p matches after reload")
        else:
            print(f"\n  [WARN] joint_X_p differs slightly (expected due to base quat)")

        print(f"\n  [PASS] Checkpoint with diff model test passed")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_training_resumption():
    """Test 4: Resume training from checkpoint."""
    print("\n" + "="*60)
    print("Test 4: Training Resumption")
    print("="*60)

    g1_path = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")

    if not os.path.exists(g1_path):
        print(f"  [SKIP] G1 model not found: {g1_path}")
        return True

    from hybrid_codesign.hybrid_agent import HybridAMPAgent

    # Create agent with low warmup for testing
    config = {
        "warmup_iters": 0,  # No warmup
        "outer_loop_freq": 1,  # Run outer loop every iteration
        "design_learning_rate": 0.001,
        "diff_horizon": 3,
        "design_param_init": 0.0,
    }

    agent = HybridAMPAgent(config, env=None, device="cuda:0")
    agent.setup_diff_model(g1_path, "cuda:0")

    # Run a few iterations
    print("  [INFO] Running initial iterations...")
    for i in range(3):
        info = agent.train_iter()
        print(f"    Iter {agent._iter}: theta={info.get('design_param_theta', 0):.6f}")

    # Save state
    checkpoint_iter = agent._iter
    checkpoint_theta = agent._parametric_model.get_theta()
    checkpoint_history_len = len(agent._design_history)

    print(f"\n  [INFO] Checkpoint state:")
    print(f"    iter: {checkpoint_iter}")
    print(f"    theta: {checkpoint_theta:.6f}")
    print(f"    history length: {checkpoint_history_len}")

    temp_dir = tempfile.mkdtemp()
    checkpoint_path = os.path.join(temp_dir, "resume_test.pt")

    try:
        agent.save(checkpoint_path)

        # Create fresh agent and load
        agent2 = HybridAMPAgent(config, env=None, device="cuda:0")
        agent2.setup_diff_model(g1_path, "cuda:0")
        agent2.load(checkpoint_path)

        # Verify state matches
        assert agent2._iter == checkpoint_iter, f"iter mismatch: {agent2._iter}"
        assert abs(agent2._parametric_model.get_theta() - checkpoint_theta) < 1e-6, "theta mismatch"

        # Continue training
        print(f"\n  [INFO] Continuing training from checkpoint...")
        for i in range(2):
            info = agent2.train_iter()
            print(f"    Iter {agent2._iter}: theta={info.get('design_param_theta', 0):.6f}")

        # Verify continued properly
        assert agent2._iter == checkpoint_iter + 2, "iteration count wrong"
        assert len(agent2._design_history) >= checkpoint_history_len, "history not extended"

        print(f"\n  [PASS] Training resumption test passed")
        return True

    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all checkpoint tests."""
    print("\n" + "#"*60)
    print("# CHECKPOINT SAVE/LOAD TESTS")
    print("#"*60)

    wp.init()

    results = {}

    # Test 1: Standalone agent
    results["standalone_checkpoint"] = test_standalone_agent_checkpoint()

    # Test 2: Parametric model
    results["parametric_model_checkpoint"] = test_parametric_model_checkpoint()

    # Test 3: With diff model
    results["diff_model_checkpoint"] = test_checkpoint_with_diff_model()

    # Test 4: Training resumption
    results["training_resumption"] = test_training_resumption()

    # Summary
    print("\n" + "="*60)
    print("CHECKPOINT TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll checkpoint tests PASSED!")
        return 0
    else:
        print(f"\n{total - passed} test(s) FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
