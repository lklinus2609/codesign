"""
Generate modified G1 MJCF + env config for native MimicKit AMP testing.

Run from codesign/codesign/:
    python generate_test_files.py

This creates test_native_amp/ with:
    - g1_modified.xml   (theta=0, should be identical to stock g1.xml)
    - env_config.yaml   (points char_file to the modified xml)
    - meshes/           (symlink or copy of original meshes)

Then test with native MimicKit:
    cd ../MimicKit/mimickit
    python run.py --mode train --num_envs 4096 --engine_config data/engines/newton_engine.yaml --env_config ../../codesign/test_native_amp/env_config.yaml --agent_config data/agents/amp_g1_agent.yaml --visualize false --out_dir ../../codesign/test_native_amp/output/

Compare against stock:
    python run.py --mode train --num_envs 4096 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_env.yaml --agent_config data/agents/amp_g1_agent.yaml --visualize false --out_dir ../../codesign/test_stock_amp/output/
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MIMICKIT_DIR = (SCRIPT_DIR / ".." / "MimicKit").resolve()

BASE_MJCF = MIMICKIT_DIR / "data" / "assets" / "g1" / "g1.xml"
BASE_ENV_CONFIG = MIMICKIT_DIR / "data" / "envs" / "amp_g1_env.yaml"

OUT_DIR = SCRIPT_DIR / "test_native_amp"


def main():
    from g1_mjcf_modifier import G1MJCFModifier, NUM_DESIGN_PARAMS

    print(f"Base MJCF:      {BASE_MJCF}")
    print(f"Base env config: {BASE_ENV_CONFIG}")
    print(f"Output dir:      {OUT_DIR}")

    if not BASE_MJCF.exists():
        print(f"ERROR: Base MJCF not found: {BASE_MJCF}")
        sys.exit(1)
    if not BASE_ENV_CONFIG.exists():
        print(f"ERROR: Base env config not found: {BASE_ENV_CONFIG}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate modified MJCF with theta=0 (identical to stock)
    mod = G1MJCFModifier(str(BASE_MJCF))
    theta = np.zeros(NUM_DESIGN_PARAMS)

    modified_mjcf = OUT_DIR / "g1_modified.xml"
    mod.generate(theta, str(modified_mjcf))
    print(f"Generated: {modified_mjcf}")

    # Generate env config pointing to modified MJCF
    env_config_out = OUT_DIR / "env_config.yaml"
    mod.generate_env_config(
        str(modified_mjcf.resolve()),
        str(BASE_ENV_CONFIG),
        str(env_config_out),
    )
    print(f"Generated: {env_config_out}")

    # Symlink or copy meshes
    mesh_src = BASE_MJCF.parent / "meshes"
    mesh_dst = OUT_DIR / "meshes"
    if mesh_dst.exists():
        print(f"Meshes already exist: {mesh_dst}")
    elif mesh_src.exists():
        try:
            mesh_dst.symlink_to(mesh_src)
            print(f"Symlinked: {mesh_dst} -> {mesh_src}")
        except (OSError, NotImplementedError):
            shutil.copytree(str(mesh_src), str(mesh_dst))
            print(f"Copied meshes: {mesh_dst}")
    else:
        print(f"WARNING: No meshes dir found at {mesh_src}")

    # Print the test commands
    print("\n" + "=" * 70)
    print("FILES READY. Now run these commands to test:")
    print("=" * 70)

    print(f"\n--- Test 1: Native MimicKit with MODIFIED G1 (theta=0) ---")
    print(f"cd {MIMICKIT_DIR / 'mimickit'}")
    print(f'python run.py --mode train --num_envs 4096 --engine_config data/engines/newton_engine.yaml --env_config "{env_config_out}" --agent_config data/agents/amp_g1_agent.yaml --visualize false --out_dir "{OUT_DIR / "output"}"')

    print(f"\n--- Test 2: Native MimicKit with STOCK G1 ---")
    print(f"cd {MIMICKIT_DIR / 'mimickit'}")
    print(f'python run.py --mode train --num_envs 4096 --engine_config data/engines/newton_engine.yaml --env_config data/envs/amp_g1_env.yaml --agent_config data/agents/amp_g1_agent.yaml --visualize false --out_dir "{SCRIPT_DIR / "test_stock_amp" / "output"}"')

    print("\nIf both fail       -> MimicKit AMP + Newton is broken for G1")
    print("If stock works     -> g1_mjcf_modifier or env config is the problem")
    print("If both work       -> InnerLoopController wrapper is the problem")


if __name__ == "__main__":
    main()
