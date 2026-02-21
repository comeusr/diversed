# DIVERSED: Dynamic Verification for Speculative Decoding

This repository contains the implementation of DIVERSED, a framework for improving speculative decoding through dynamic verification.

## Overview

DIVERSED introduces novel techniques for speculative decoding that improve both the efficiency and quality of text generation:

1. **Standard Speculative Decoding**: The baseline implementation as described in Leviathan et al. (2023)
2. **Lossy Speculative Decoding**: Another implementation of Leviathan et al. (2023)
3. **Static Ensemble Verification**: A verification approach that uses fixed weights to combine draft and target model distributions
4. **Dynamic Ensemble Verification (DIVERSED)**: A learned verification mechanism that dynamically combines the strengths of both draft and target models

## Installation

**Important**: This package includes a modified version of the transformers library that must be installed for the code to work properly.

```bash
# Clone the repository
git clone https://github.com/anonymous/diversed.git
cd diversed

# STEP 1: Install the modified transformers library first
cd transformers
pip install -e .
cd ..

# STEP 2: Install the main package dependencies
pip install -e .
```

### Troubleshooting Installation

If you encounter `ImportError: cannot import name 'AutoTokenizer' from 'transformers'`, it means the transformers library is not properly installed. Follow these steps:

```bash
# Check if transformers is installed
python -c "import transformers; print(transformers.__version__); print(transformers.__file__)"

# If the above fails or shows wrong path, reinstall:
cd diversed_code_release/transformers
pip uninstall transformers -y
pip install -e .

# Verify installation
python -c "import transformers; print('Transformers installed successfully:', transformers.__version__)"
```

### Alternative Installation (if above doesn't work)

```bash
# Install standard transformers first, then override with modified version
pip install transformers
cd diversed_code_release/transformers
pip install -e . --force-reinstall
```

### Verify Installation

After installation, run the verification script to check if everything is working:

```bash
python verify_installation.py
```

This script will check all dependencies and provide specific troubleshooting steps if any issues are found.

## Repository Structure

```
diversed_code_release/
├── configs/              # Configuration files
├── data/                 # Dataset directories (outputs will be saved here)
├── logs/                 # Log files will be saved here
├── scripts/              # Training and inference scripts
│   ├── run_dp.py         # Data parallel inference script
│   ├── run_inference.py  # General inference script
│   └── run_train.py      # Training script
├── src/                  # Source code
│   ├── models.py         # Model definitions
│   ├── speculative_decoding.py      # Speculative decoding implementation
│   ├── speculative_decoding_dp.py   # Data parallel speculative decoding
│   └── mydatasets/       # Dataset-specific utilities and prompts
├── train/                # Training utilities
│   ├── dataloader.py     # Data loading utilities
│   └── trainer.py        # Training loop implementation
├── transformers/         # Modified transformers library
└── utils/                # Utility functions
```

## Usage

### Configuration

The `configs/default_config.yaml` file contains all the configuration parameters for training and inference. You can modify this file or create your own configuration file.

### Available Methods

The following methods are available via the `--method` parameter:

- `auto`: Autoregressive decoding (baseline, no speculative decoding)
- `sd`: Standard Speculative Decoding
- `sd_lossy` / `lossy`: Lossy Speculative Decoding  
- `sd_static` / `static_en`: Static Ensemble Verification
- `sd_en` / `diversed`: Dynamic Ensemble Verification (DIVERSED)
- `spe_cas`: Speculative Cascading (experimental)

**Note**: Some scripts may use alternative method names:
- `static_en` is equivalent to `sd_static`
- `diversed` is equivalent to `sd_en`
- `lossy` is equivalent to `sd_lossy`
- `auto` enables autoregressive decoding without any draft model

### Method-Specific Parameters

- **For `sd_static`/`static_en`**: Use `--draft_ensemble_weights` to control the mixing weight between draft and target models (0.0 = pure target, 1.0 = pure draft)
- **For `sd_en`/`diversed`**: Requires a trained ensemble head model
- **For `auto`**: No draft model needed, uses only the target model
- **For `lossy`/`sd_lossy`**: Uses lossy speculative decoding with relaxed verification
- **For `spe_cas`**: Uses speculative cascading approach

### Additional Parameters

- `--assistant_schedule`: Controls draft token scheduling (`constant`, `heuristic`, `dynamic`)
- `--assistant_confidence_threshold`: Confidence threshold for assistant model (used with non-dynamic schedules)
- `--num_assistant_tokens`: Number of draft tokens to generate (default: 5)
- `--do_sample`: Whether to use sampling (`True` or `False`)
- `--temperature`: Sampling temperature (default: 0.0 for greedy decoding)

### Training

```bash
python scripts/run_train.py \
    --config configs/default_config.yaml \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --draft_model_name_or_path meta-llama/Llama-2-7b-hf \
    --output_dir ./outputs/llama2-7b-diversed
```

### Inference Examples

**Important**: Run all commands from the root directory of the repository (not from within subdirectories).

#### Autoregressive Decoding (Baseline)
```bash
python src/speculative_decoding_dp.py \
    --method auto \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/auto_baseline \
    --n_examples 100 \
    --max_tokens 128 \
    --temperature 0.0 \
    --do_sample False
```

#### Standard Speculative Decoding
```bash
python src/speculative_decoding_dp.py \
    --method sd \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/sd_results \
    --num_assistant_tokens 5 \
    --n_examples 100 \
    --max_tokens 128 \
    --assistant_schedule constant
```

#### Static Ensemble Verification
```bash
python src/speculative_decoding_dp.py \
    --method static_en \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/static_ensemble \
    --num_assistant_tokens 5 \
    --draft_ensemble_weights 0.3 \
    --n_examples 100 \
    --max_tokens 128
```

#### DIVERSED (Dynamic Ensemble)
```bash
python src/speculative_decoding_dp.py \
    --method sd_en \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./outputs/trained_diversed_model \
    --num_assistant_tokens 5 \
    --n_examples 100 \
    --max_tokens 128
```

#### Lossy Speculative Decoding
```bash
python src/speculative_decoding_dp.py \
    --method lossy \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/lossy_results \
    --num_assistant_tokens 5 \
    --n_examples 100 \
    --max_tokens 128
```

#### Speculative Cascading
```bash
python src/speculative_decoding_dp.py \
    --method spe_cas \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/spe_cas_results \
    --num_assistant_tokens 5 \
    --lenience 0.5 \
    --n_examples 100 \
    --max_tokens 128
```

#### Using the Simplified Scripts

For data parallel inference (recommended):

```bash
python scripts/run_dp.py \
    --model_path ./outputs/llama2-7b-diversed \
    --target_model meta-llama/Llama-2-7b-hf \
    --draft_model meta-llama/Llama-2-7b-hf \
    --dataset xsum \
    --method sd_en \
    --num_assistant_tokens 5
```

For single GPU inference:

```bash
python scripts/run_inference.py \
    --model_path ./outputs/llama2-7b-diversed \
    --target_model meta-llama/Llama-2-7b-hf \
    --draft_model meta-llama/Llama-2-7b-hf \
    --dataset xsum \
    --method sd_en \
    --num_assistant_tokens 5
```

### Supported Datasets

- `cnndm`: CNN/DailyMail summarization
- `xsum`: XSum summarization
- `wmt`: WMT translation
- `gsm8k`: GSM8K math problems
- `humaneval`: HumanEval code generation
- `mbpp`: MBPP code generation

### Output and Logging

- Generated outputs are saved to the `data/` directory
- Training and inference logs are saved to the `logs/` directory
- Model checkpoints are saved to the specified output directory

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{anonymous2025diversed,
  title={DIVERSED: Dynamic Verification for Speculative Decoding},
  author={Anonymous},
  booktitle={Anonymous Conference},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
