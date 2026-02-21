# DIVERSED Training Guide

This document explains the two training approaches available in DIVERSED and when to use each one.

## Training Options

DIVERSED provides two different training scripts for different purposes:

### 1. `scripts/run_train.py` - Standard Training (RECOMMENDED)

**Use this for**: General DIVERSED model training with ensemble heads

**Features**:
- Command-line argument based (easier to use)
- Uses `configs/default_config.yaml` for configuration
- Standard DIVERSED training approach
- More straightforward setup

**Example Usage**:
```bash
python scripts/run_train.py \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --output_dir ./outputs/llama-diversed \
    --epochs 2 \
    --learning_rate 5e-4 \
    --batch_size 8
```

### 2. `rl_train.py` - Reinforcement Learning Training (ADVANCED)

**Use this for**: Advanced RL-based ensemble head training with specific reward functions

**Features**:
- Hydra configuration system (more complex but flexible)
- Uses `train_config/config.yaml` for configuration
- Reinforcement learning approach with custom reward functions
- Requires `.env` file with API tokens for wandb/huggingface
- More advanced hyperparameter management

**Example Usage**:
```bash
# First, create .env file with:
# WANDB_API_KEY=your_wandb_key
# HF_TOKEN=your_huggingface_token

# Then run:
python rl_train.py
# or
./run_rl.sh
```

## Which Should You Use?

### Use `scripts/run_train.py` if:
- ✅ You want to train DIVERSED models (most common use case)
- ✅ You prefer simple command-line arguments
- ✅ You're new to DIVERSED training
- ✅ You want the approach shown in the README

### Use `rl_train.py` if:
- ✅ You need advanced RL-based training
- ✅ You want to experiment with custom reward functions
- ✅ You need complex hyperparameter configurations
- ✅ You're doing research with specific RL requirements

## Recommendation

**For most users**: Start with `scripts/run_train.py` as it's simpler and covers the standard DIVERSED training workflow shown in the README.

**For researchers**: Use `rl_train.py` if you need the advanced RL capabilities for specific research requirements.

## Configuration Files

- `configs/default_config.yaml` - Used by `scripts/run_train.py`
- `train_config/config.yaml` - Used by `rl_train.py`

Both training approaches will produce ensemble head checkpoints that can be used with the `sd_en`/`diversed` inference methods.

## Output

Both training scripts will save trained models that can be used for inference with:
```bash
torchrun --nproc_per_node=8 src/speculative_decoding_dp.py \
    --method sd_en \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./path/to/your/trained/model
