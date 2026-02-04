#!/usr/bin/env python3
"""
Run All Hybrid Co-Design Tests

This script runs all test suites for the hybrid co-design implementation:
1. test_level0_verification.py - Level 0: Mathematical correctness (no physics)
2. test_implementation.py - Unit tests for components
3. validate_outer_loop.py - End-to-end outer loop validation
4. test_gpu_memory.py - GPU memory stability tests
5. test_checkpoint.py - Checkpoint save/load tests

Usage:
    python run_all_tests.py

Expected: All tests pass before GPU deployment.
"""

import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_test(name, script_path):
    """Run a test script and return success status."""
    print("\n" + "#"*70)
    print(f"# Running: {name}")
    print("#"*70 + "\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"\n>>> {name}: PASSED ({elapsed:.1f}s)")
        else:
            print(f"\n>>> {name}: FAILED (exit code {result.returncode}, {elapsed:.1f}s)")

        return success

    except Exception as e:
        print(f"\n>>> {name}: ERROR - {e}")
        return False


def main():
    """Run all test suites."""
    print("="*70)
    print(" HYBRID CO-DESIGN - FULL TEST SUITE")
    print("="*70)
    print(f"\nScript directory: {SCRIPT_DIR}")
    print(f"Python: {sys.executable}")

    tests = [
        ("Level 0 Verification", os.path.join(SCRIPT_DIR, "test_level0_verification.py")),
        ("Unit Tests", os.path.join(SCRIPT_DIR, "test_implementation.py")),
        ("Outer Loop Validation", os.path.join(SCRIPT_DIR, "validate_outer_loop.py")),
        ("GPU Memory Tests", os.path.join(SCRIPT_DIR, "test_gpu_memory.py")),
        ("Checkpoint Tests", os.path.join(SCRIPT_DIR, "test_checkpoint.py")),
    ]

    results = {}
    total_start = time.time()

    for name, path in tests:
        if os.path.exists(path):
            results[name] = run_test(name, path)
        else:
            print(f"\n>>> {name}: SKIPPED (file not found: {path})")
            results[name] = None

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print(" TEST SUITE SUMMARY")
    print("="*70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len(results)

    for name, result in results.items():
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"  {status} {name}")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Total time: {total_elapsed:.1f}s")

    if failed == 0 and passed > 0:
        print("\n" + "="*70)
        print(" ALL TESTS PASSED - READY FOR GPU DEPLOYMENT")
        print("="*70)
        return 0
    else:
        print("\n" + "="*70)
        print(" SOME TESTS FAILED - FIX BEFORE DEPLOYMENT")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
