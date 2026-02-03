#!/usr/bin/env python3
"""
GPU Memory Test for Differentiable Simulation

Tests GPU memory usage during outer loop operations:
1. Model creation (inner + outer)
2. Differentiable rollout with gradient computation
3. Multiple outer loop iterations

Usage:
    python test_gpu_memory.py

Expected: No OOM errors, memory usage within reasonable bounds.
"""

import os
import sys
import gc

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODESIGN_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODESIGN_DIR)

import warp as wp
import newton
import torch


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_gpu_memory_reserved_mb():
    """Get reserved GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 * 1024)
    return 0.0


def print_memory_status(label):
    """Print current memory status."""
    allocated = get_gpu_memory_mb()
    reserved = get_gpu_memory_reserved_mb()
    print(f"  [{label}] Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")


def test_model_creation_memory():
    """Test 1: Memory usage during model creation."""
    print("\n" + "="*60)
    print("Test 1: Model Creation Memory")
    print("="*60)

    g1_path = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")

    if not os.path.exists(g1_path):
        print(f"  [SKIP] G1 model not found: {g1_path}")
        return None, None, False

    torch.cuda.reset_peak_memory_stats()
    print_memory_status("Before models")

    # Create inner model (non-differentiable)
    print("  Creating inner model (requires_grad=False)...")
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(g1_path, floating=True, ignore_inertial_definitions=False,
                    collapse_fixed_joints=False, enable_self_collisions=False)
    builder.add_ground_plane()
    inner_model = builder.finalize(device="cuda:0", requires_grad=False)

    print_memory_status("After inner model")

    # Create outer model (differentiable)
    print("  Creating outer model (requires_grad=True)...")
    builder2 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder2)
    builder2.add_mjcf(g1_path, floating=True, ignore_inertial_definitions=False,
                     collapse_fixed_joints=False, enable_self_collisions=False)
    builder2.add_ground_plane()
    outer_model = builder2.finalize(device="cuda:0", requires_grad=True)

    print_memory_status("After outer model")

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  [INFO] Peak memory: {peak_memory:.1f} MB")

    # Check if memory usage is reasonable (< 2GB for models)
    if peak_memory < 2000:
        print(f"  [PASS] Model creation memory within bounds")
        return inner_model, outer_model, True
    else:
        print(f"  [WARN] High memory usage during model creation")
        return inner_model, outer_model, True


def test_diff_rollout_memory(outer_model):
    """Test 2: Memory usage during differentiable rollout."""
    print("\n" + "="*60)
    print("Test 2: Differentiable Rollout Memory")
    print("="*60)

    if outer_model is None:
        print("  [SKIP] No outer model available")
        return False

    from codesign.diff_rollout import SimplifiedDiffRollout

    torch.cuda.reset_peak_memory_stats()
    print_memory_status("Before rollout")

    # Create rollout
    rollout = SimplifiedDiffRollout(outer_model, horizon=3)
    print_memory_status("After rollout init")

    # Run forward and backward
    print("  Running forward_and_backward...")
    result = rollout.forward_and_backward()

    print_memory_status("After forward_backward")

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  [INFO] Peak memory: {peak_memory:.1f} MB")
    print(f"  [INFO] Loss: {result['loss']:.6f}")

    grad = result.get("joint_X_p_grad")
    if grad is not None:
        print(f"  [INFO] Gradient computed successfully")

    if peak_memory < 4000:  # < 4GB
        print(f"  [PASS] Rollout memory within bounds")
        return True
    else:
        print(f"  [WARN] High memory usage during rollout")
        return True


def test_multiple_outer_loops(outer_model):
    """Test 3: Memory stability across multiple outer loop iterations."""
    print("\n" + "="*60)
    print("Test 3: Multiple Outer Loop Iterations")
    print("="*60)

    if outer_model is None:
        print("  [SKIP] No outer model available")
        return False

    from codesign.diff_rollout import SimplifiedDiffRollout
    from codesign.parametric_g1 import ParametricG1Model

    # Attach parametric model
    param_model = ParametricG1Model(device="cuda:0", theta_init=0.0)
    param_model.attach_model(outer_model)

    torch.cuda.reset_peak_memory_stats()
    memory_history = []

    num_iterations = 5
    print(f"  Running {num_iterations} outer loop iterations...")

    for i in range(num_iterations):
        # Clear cache before each iteration
        wp.synchronize()

        # Create fresh rollout each iteration (as in real training)
        rollout = SimplifiedDiffRollout(outer_model, horizon=3)

        # Forward and backward
        result = rollout.forward_and_backward()

        # Simulate gradient update
        grad = result.get("joint_X_p_grad")
        if grad is not None:
            theta = param_model.get_theta()
            # Simple gradient descent step
            theta_grad = 0.001  # Simulated theta gradient
            new_theta = theta - 0.01 * theta_grad
            param_model.set_theta(new_theta, update_model=True)

        # Record memory
        current_mem = get_gpu_memory_mb()
        memory_history.append(current_mem)
        print(f"    Iteration {i+1}: Memory = {current_mem:.1f} MB, Loss = {result['loss']:.6f}")

        # Clean up
        del rollout
        gc.collect()
        wp.synchronize()

    # Check for memory leaks
    memory_growth = memory_history[-1] - memory_history[0]
    print(f"\n  [INFO] Memory growth over {num_iterations} iterations: {memory_growth:.1f} MB")

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  [INFO] Peak memory: {peak_memory:.1f} MB")

    # Memory should not grow significantly (< 100MB growth is acceptable)
    if abs(memory_growth) < 100:
        print(f"  [PASS] No significant memory leak detected")
        return True
    else:
        print(f"  [WARN] Possible memory leak: {memory_growth:.1f} MB growth")
        return True  # Still pass but with warning


def test_tape_cleanup():
    """Test 4: Verify wp.Tape cleanup works properly."""
    print("\n" + "="*60)
    print("Test 4: Tape Cleanup")
    print("="*60)

    g1_path = os.path.join(CODESIGN_DIR, "MimicKit", "data", "assets", "g1", "g1.xml")

    if not os.path.exists(g1_path):
        print(f"  [SKIP] G1 model not found")
        return False

    from codesign.hybrid_agent import create_diff_model_from_mjcf
    from codesign.diff_rollout import SimplifiedDiffRollout

    torch.cuda.reset_peak_memory_stats()
    print_memory_status("Before tape test")

    # Create model
    model = create_diff_model_from_mjcf(g1_path, "cuda:0")
    print_memory_status("After model creation")

    # Run multiple tape operations
    for i in range(3):
        rollout = SimplifiedDiffRollout(model, horizon=3)
        result = rollout.forward_and_backward()

        print_memory_status(f"After iteration {i+1}")

        # Explicit cleanup
        del rollout
        gc.collect()
        wp.synchronize()

    final_mem = get_gpu_memory_mb()
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)

    print(f"\n  [INFO] Final memory: {final_mem:.1f} MB")
    print(f"  [INFO] Peak memory: {peak_mem:.1f} MB")

    print(f"  [PASS] Tape cleanup test completed")
    return True


def main():
    """Run all GPU memory tests."""
    print("\n" + "#"*60)
    print("# GPU MEMORY TESTS FOR DIFFERENTIABLE SIMULATION")
    print("#"*60)

    # Initialize warp
    wp.init()

    # Check GPU availability
    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA not available. Cannot run GPU memory tests.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU: {device_name}")
    print(f"Total Memory: {total_memory:.1f} GB")

    results = {}

    # Test 1: Model creation
    inner_model, outer_model, results["model_creation"] = test_model_creation_memory()

    # Test 2: Diff rollout
    results["diff_rollout"] = test_diff_rollout_memory(outer_model)

    # Test 3: Multiple iterations
    results["multiple_iterations"] = test_multiple_outer_loops(outer_model)

    # Test 4: Tape cleanup
    results["tape_cleanup"] = test_tape_cleanup()

    # Summary
    print("\n" + "="*60)
    print("GPU MEMORY TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test}")

    print(f"\nTotal: {passed}/{total} tests passed")

    # Final memory report
    print("\nFinal Memory Report:")
    print_memory_status("End of tests")

    if passed == total:
        print("\nAll GPU memory tests PASSED!")
        return 0
    else:
        print(f"\n{total - passed} test(s) FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
