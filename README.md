# Co-Design for Humanoid Robots

Hybrid morphology and control co-optimization using differentiable physics simulation.

This implementation combines:
- **Inner Loop**: Adversarial Motion Priors (AMP) via MimicKit for locomotion policy learning
- **Outer Loop**: NVIDIA Newton differentiable physics for morphology optimization

## Quick Start

```bash
# 1. Activate conda environment
conda activate codesign

# 2. Run tests to verify setup
python hybrid_codesign/run_all_tests.py

# 3. Start training
python hybrid_codesign/train_hybrid.py --agent_config hybrid_codesign/config/hybrid_g1_agent.yaml --env_config hybrid_codesign/config/hybrid_g1_env.yaml
```

## Installation

### Prerequisites
- NVIDIA GPU with CUDA support
- Conda package manager

### Setup

1. **Install dependencies in the same parent directory:**
   ```bash
   # Clone Newton
   git clone https://github.com/NVIDIA/newton.git

   # Clone MimicKit
   git clone https://github.com/NVlabs/MimicKit.git

   # Clone this repository
   git clone <this-repo-url> codesign
   ```

2. **Create conda environment:**
   ```bash
   cd codesign/hybrid_codesign
   conda env create -f environment.yml
   conda activate codesign
   ```

3. **Verify installation:**
   ```bash
   python run_all_tests.py
   ```

## Expected Folder Structure

```
parent_directory/
├── newton/                    # NVIDIA Newton (differentiable physics)
├── MimicKit/                  # MimicKit (AMP implementation)
│   └── data/
│       └── assets/
│           └── g1/
│               └── g1.xml     # G1 humanoid robot model
└── codesign/                  # This repository
    └── hybrid_codesign/
        ├── config/            # Configuration files
        ├── parametric_g1.py   # Parametric morphology model
        ├── diff_rollout.py    # Differentiable rollout
        ├── hybrid_agent.py    # Hybrid co-design agent
        ├── train_hybrid.py    # Training script
        └── ...
```

## Usage

```bash
# Basic training with default settings
python hybrid_codesign/train_hybrid.py --num_envs 4096 --max_samples 50000000

# Training with custom configs
python hybrid_codesign/train_hybrid.py \
    --agent_config hybrid_codesign/config/hybrid_g1_agent.yaml \
    --env_config hybrid_codesign/config/hybrid_g1_env.yaml

# Training with Weights & Biases logging
python hybrid_codesign/train_hybrid.py \
    --agent_config hybrid_codesign/config/hybrid_g1_agent.yaml \
    --num_envs 512 \
    --logger wandb

# Resume from checkpoint
python hybrid_codesign/train_hybrid.py --resume output/hybrid_g1/model.pt
```

### Logging Options
- `--logger tb` - TensorBoard (default)
- `--logger wandb` - Weights & Biases

### Video Recording (wandb only, headless)
- `--video_interval N` - Record video every N iterations (default: 500, 0 to disable)
- Uses true headless rendering via Newton's `ViewerGL(headless=True)` - no display required

```bash
# Record video every 1000 iterations
python hybrid_codesign/train_hybrid.py \
    --agent_config hybrid_codesign/config/hybrid_g1_agent.yaml \
    --logger wandb \
    --video_interval 1000

# Disable video recording
python hybrid_codesign/train_hybrid.py \
    --agent_config hybrid_codesign/config/hybrid_g1_agent.yaml \
    --logger wandb \
    --video_interval 0
```

## Configuration

Key parameters in `config/hybrid_g1_agent.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_iters` | 10000 | Inner loop iterations before outer loop activates |
| `outer_loop_freq` | 200 | Outer loop runs every N inner iterations |
| `design_learning_rate` | 0.01 | Learning rate for morphology optimization |
| `diff_horizon` | 3 | Differentiable rollout horizon |

## Monitoring

Training logs design parameters to TensorBoard/WandB under collection `3_Design`:
- `Design_Theta_Rad`: Hip roll angle in radians
- `Design_Theta_Deg`: Hip roll angle in degrees
- `Outer_Loop_Loss`: Outer loop objective
- `Outer_Loop_Grad`: Design parameter gradient

## Known Limitations

The differentiable simulation (SolverSemiImplicit) has contact instability issues. Current workaround uses a short horizon (3 steps) before ground contact. See `IMPLEMENTATION_PLAN.md` for details and future work.

## License

This project is provided for research purposes. See LICENSE for details.

## Acknowledgments

This project builds upon:

- **[NVIDIA Newton](https://github.com/NVIDIA/newton)** - Differentiable physics simulation
  Apache License 2.0 | Copyright NVIDIA Corporation

- **[MimicKit](https://github.com/NVlabs/MimicKit)** - Adversarial Motion Priors implementation
  Apache License 2.0 | Copyright NVIDIA Corporation
