#!/usr/bin/env python3
"""
Hybrid Co-Design Training Script

This script implements the complete training pipeline for Algorithm 1:
Hybrid PPO Control + Differentiable Physics Morphology optimization.

Usage:
    # Basic training
    python train_hybrid.py --num_envs 4096 --max_samples 50000000

    # With custom config
    python train_hybrid.py --agent_config config/hybrid_g1_agent.yaml \
                          --env_config config/hybrid_g1_env.yaml

    # Resume from checkpoint
    python train_hybrid.py --resume output/hybrid_g1/model.pt

Reference: algorithm1.pdf (Algorithm 1)
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODESIGN_DIR = os.path.dirname(SCRIPT_DIR)
MIMICKIT_DIR = os.path.join(CODESIGN_DIR, "MimicKit", "mimickit")

sys.path.insert(0, MIMICKIT_DIR)
sys.path.insert(0, CODESIGN_DIR)

# Import Newton
try:
    import warp as wp
    import newton
    NEWTON_AVAILABLE = True
except ImportError:
    print("Warning: Newton not available")
    NEWTON_AVAILABLE = False

# Import MimicKit components
try:
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from util.logger import Logger
    import util.mp_util as mp_util
    import util.util as util
    MIMICKIT_AVAILABLE = True
except ImportError:
    print("Warning: MimicKit not available")
    MIMICKIT_AVAILABLE = False

# Import hybrid components
from codesign.parametric_g1 import ParametricG1Model
from codesign.hybrid_agent import HybridAMPAgent, HybridAMPAgentIntegrated
from codesign.diff_rollout import SimplifiedDiffRollout


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hybrid Co-Design Training")

    # Environment and agent configuration
    parser.add_argument("--agent_config", type=str,
                        default="codesign/config/hybrid_g1_agent.yaml",
                        help="Path to agent configuration file")
    parser.add_argument("--env_config", type=str,
                        default="codesign/config/hybrid_g1_env.yaml",
                        help="Path to environment configuration file")
    parser.add_argument("--engine_config", type=str,
                        default="MimicKit/data/engines/newton_engine.yaml",
                        help="Path to physics engine configuration file")

    # Training parameters
    parser.add_argument("--num_envs", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_samples", type=int, default=50000000,
                        help="Maximum training samples")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Computation device")

    # Output
    parser.add_argument("--out_dir", type=str, default="output/hybrid_g1",
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")

    # Resume training
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Enable visualization")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA (use CPU)")

    # Logging
    parser.add_argument("--logger", type=str, default="tb",
                        choices=["tb", "wandb"],
                        help="Logger type: 'tb' for TensorBoard, 'wandb' for Weights & Biases")

    # Video logging (wandb only)
    parser.add_argument("--video_interval", type=int, default=500,
                        help="Record video every N iterations (0 to disable, wandb only)")

    return parser.parse_args()


def load_config(config_path):
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def patch_env_config_paths(env_config_path):
    """
    Patch relative paths in env config to absolute paths.

    MimicKit expects paths relative to its own directory, but we run from
    codesign/. This function creates a temp config with absolute paths.

    Returns:
        Path to patched temporary config file
    """
    import tempfile

    config = load_config(env_config_path)

    # Paths that need to be patched
    path_keys = ["char_file", "motion_file"]

    for key in path_keys:
        if key in config:
            original_path = config[key]
            # Check if already absolute
            if os.path.isabs(original_path):
                continue
            # Try path as-is first (relative to cwd)
            if os.path.exists(original_path):
                config[key] = os.path.abspath(original_path)
            # Try with MimicKit prefix
            elif os.path.exists(os.path.join(CODESIGN_DIR, "MimicKit", original_path)):
                config[key] = os.path.join(CODESIGN_DIR, "MimicKit", original_path)
            # Try with CODESIGN_DIR prefix
            elif os.path.exists(os.path.join(CODESIGN_DIR, original_path)):
                config[key] = os.path.join(CODESIGN_DIR, original_path)
            else:
                print(f"Warning: Could not resolve path for {key}: {original_path}")

    # Write patched config to temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix="env_config_")
    with os.fdopen(temp_fd, 'w') as f:
        yaml.dump(config, f)

    return temp_path


def create_output_dir(out_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created output directory: {out_dir}")


def save_configs(agent_config, env_config, out_dir):
    """Save configuration files to output directory."""
    with open(os.path.join(out_dir, "agent_config.yaml"), 'w') as f:
        yaml.dump(agent_config, f)
    with open(os.path.join(out_dir, "env_config.yaml"), 'w') as f:
        yaml.dump(env_config, f)


def create_newton_model_for_diff(char_file, device):
    """
    Create a Newton model for differentiable simulation.

    This creates a separate model with requires_grad=True for the outer loop.

    Args:
        char_file: Path to character XML file
        device: Computation device

    Returns:
        Newton Model object
    """
    if not NEWTON_AVAILABLE:
        print("Newton not available, skipping differentiable model creation")
        return None

    print(f"Creating differentiable Newton model from {char_file}...")

    # Create builder
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # Load character
    builder.add_mjcf(
        char_file,
        floating=True,
        ignore_inertial_definitions=False,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
    )

    # Add ground plane
    builder.add_ground_plane()

    # Finalize with gradients enabled
    model = builder.finalize(device=device, requires_grad=True)

    print(f"Created model with {model.joint_count} joints, requires_grad=True")

    return model


def train_standalone(args, agent_config, env_config):
    """
    Standalone training loop (without full MimicKit integration).

    This is useful for testing the outer loop logic independently.
    """
    print("\n" + "="*60)
    print("STANDALONE TRAINING MODE")
    print("="*60 + "\n")

    device = "cpu" if args.no_cuda else args.device

    # Create output directory
    create_output_dir(args.out_dir)
    save_configs(agent_config, env_config, args.out_dir)

    # Create hybrid agent
    agent = HybridAMPAgent(agent_config, env=None, device=device)

    # Create differentiable Newton model
    char_file = env_config.get("char_file", "MimicKit/data/assets/g1/g1.xml")
    char_path = os.path.join(CODESIGN_DIR, char_file)

    if os.path.exists(char_path):
        diff_model = create_newton_model_for_diff(char_path, device)
        if diff_model is not None:
            agent.attach_newton_model(diff_model)
    else:
        print(f"Warning: Character file not found: {char_path}")

    # Resume from checkpoint if specified
    if args.resume:
        agent.load(args.resume)

    # Training loop
    print("\nStarting standalone training...")
    print(f"  Warmup iterations: {agent._warmup_iters}")
    print(f"  Outer loop frequency: {agent._outer_loop_freq}")
    print(f"  Design learning rate: {agent._design_lr}")
    print()

    start_time = time.time()
    max_iters = args.max_samples // agent_config.get("steps_per_iter", 32)

    for i in range(max_iters):
        # Run one training iteration
        info = agent.train_iter()

        # Logging
        if i % 100 == 0:
            elapsed = time.time() - start_time
            theta = info.get("design_param_theta", 0.0)
            theta_deg = info.get("design_param_degrees", 0.0)

            print(f"Iter {i:6d} | "
                  f"Time: {elapsed:6.1f}s | "
                  f"Theta: {theta:7.4f} rad ({theta_deg:6.2f} deg)")

            if "outer_loop_loss" in info:
                print(f"           | "
                      f"Outer Loss: {info['outer_loop_loss']:8.4f} | "
                      f"Outer Grad: {info['outer_loop_grad']:8.6f}")

        # Save checkpoint
        if i > 0 and i % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"model_{i:08d}.pt")
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(args.out_dir, "model_final.pt")
    agent.save(final_path)

    # Save design history
    history = agent.get_design_history()
    if history:
        history_path = os.path.join(args.out_dir, "design_history.yaml")
        with open(history_path, 'w') as f:
            yaml.dump(history, f)
        print(f"\nSaved design history to {history_path}")

    print("\nTraining complete!")
    print(f"Final theta: {agent._parametric_model.get_theta():.4f} rad "
          f"({agent._parametric_model.get_theta_degrees():.2f} deg)")


def train_integrated(args, agent_config, env_config):
    """
    Integrated training with full MimicKit.

    This uses the complete MimicKit training infrastructure.
    """
    print("\n" + "="*60)
    print("INTEGRATED TRAINING MODE (with MimicKit)")
    print("="*60 + "\n")

    # Use MimicKit's training infrastructure
    # This would typically call into MimicKit's run.py logic

    # Build environment
    device = "cpu" if args.no_cuda else args.device

    # Initialize MimicKit's multiprocessing utilities (required even for single GPU)
    mp_util.init(rank=0, num_procs=1, device=device, master_port=12355)
    engine_config = load_config(args.engine_config)

    # Patch env config paths for MimicKit compatibility
    patched_env_config = patch_env_config_paths(args.env_config)

    try:
        env = env_builder.build_env(
            patched_env_config,
            args.engine_config,
            args.num_envs,
            device,
            args.visualize
        )
    finally:
        # Clean up temp file
        if os.path.exists(patched_env_config):
            os.remove(patched_env_config)

    # Build hybrid agent (integrated with MimicKit's AMPAgent)
    agent = HybridAMPAgentIntegrated(agent_config, env, device)

    # Set up differentiable model for outer loop
    char_file = env_config.get("char_file", "data/assets/g1/g1.xml")
    # Try multiple path resolutions
    char_path = None
    for prefix in ["", "MimicKit/", os.path.join(CODESIGN_DIR, "MimicKit") + "/"]:
        candidate = os.path.join(CODESIGN_DIR, prefix, char_file) if prefix else os.path.join(CODESIGN_DIR, char_file)
        candidate = candidate.replace("//", "/")
        if os.path.exists(candidate):
            char_path = candidate
            break

    if char_path is None:
        # Fallback to MimicKit path
        char_path = os.path.join(CODESIGN_DIR, "MimicKit", char_file)

    if os.path.exists(char_path):
        agent.setup_diff_model(char_path, device)
        # Set inner model reference for sync
        if hasattr(env, '_engine') and hasattr(env._engine, '_sim_model'):
            agent.set_inner_model_reference(env._engine._sim_model)
    else:
        print(f"Warning: Could not find char file for diff model: {char_path}")

    # Create output directory
    create_output_dir(args.out_dir)
    save_configs(agent_config, env_config, args.out_dir)

    # Resume if specified
    if args.resume:
        agent.load(args.resume)

    # Set up video recording (wandb only)
    if args.logger == "wandb" and args.video_interval > 0:
        agent.setup_video_recording(video_interval=args.video_interval, fps=30)

    # Use MimicKit's training infrastructure
    print(f"\nStarting integrated training with MimicKit (logger: {args.logger})...")
    agent.train_model(
        max_samples=args.max_samples,
        out_dir=args.out_dir,
        save_int_models=True,
        logger_type=args.logger
    )

    print("\nIntegrated training complete!")


def main():
    """Main entry point."""
    args = parse_args()

    print("="*60)
    print("Hybrid Co-Design Training")
    print("Algorithm 1: PPO Control + Diff-Phys Morphology")
    print("="*60)

    # Load configurations
    agent_config = load_config(args.agent_config)
    env_config = load_config(args.env_config)

    print(f"\nAgent config: {args.agent_config}")
    print(f"Environment config: {args.env_config}")
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples}")

    # Set random seed
    seed = int(time.time() * 256) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seed: {seed}")

    # Initialize Warp
    if NEWTON_AVAILABLE:
        wp.init()
        print(f"Warp initialized, device: {wp.get_device()}")

    # Choose training mode based on available components
    if MIMICKIT_AVAILABLE and not args.no_cuda:
        train_integrated(args, agent_config, env_config)
    else:
        train_standalone(args, agent_config, env_config)


if __name__ == "__main__":
    main()
