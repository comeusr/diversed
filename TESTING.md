# DIVERSED Testing Guide

This document describes how to test all the methods implemented in DIVERSED to ensure they work correctly.

## Quick Test

To quickly test all methods, run the comprehensive test script:

```bash
cd diversed_code_release
python test_all_methods.py
```

This script will test all documented methods and provide a summary of results.

## Manual Testing

You can also test individual methods manually using the commands below.

### Prerequisites

- Ensure you have activated your conda environment with PyTorch and CUDA support
- Make sure you're in the `diversed_code_release` directory
- Ensure the modified transformers library is installed (see README.md)

### Test Commands

#### 1. Autoregressive Decoding (Baseline)
```bash
torchrun --nproc_per_node=8 --master_port=29502 src/speculative_decoding_dp.py \
    --method auto \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/auto_test \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0
```

#### 2. Standard Speculative Decoding
```bash
torchrun --nproc_per_node=8 --master_port=29503 src/speculative_decoding_dp.py \
    --method sd \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/sd_test \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0
```

#### 3. Static Ensemble Verification
```bash
torchrun --nproc_per_node=8 --master_port=29504 src/speculative_decoding_dp.py \
    --method static_en \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/static_en_test \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0 \
    --draft_ensemble_weights 0.3
```

#### 4. Lossy Speculative Decoding
```bash
torchrun --nproc_per_node=8 --master_port=29505 src/speculative_decoding_dp.py \
    --method lossy \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/lossy_test \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0 \
    --lenience 0.5
```

#### 5. Speculative Cascading
```bash
torchrun --nproc_per_node=8 --master_port=29506 src/speculative_decoding_dp.py \
    --method spe_cas \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/spe_cas_test \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0 \
    --lenience 0.5
```

### Method Aliases

The following aliases are also supported and should work identically:

- `sd_lossy` → `lossy`
- `sd_static` → `static_en`
- `diversed` → `sd_en` (requires trained ensemble head)

### Expected Output

For successful tests, you should see:
1. Model loading messages
2. Generation progress with token counts
3. Final metrics saved to JSON files
4. No error messages

### Troubleshooting

**Port in use errors**: Change the `--master_port` value to a different number (e.g., 29507, 29508, etc.)

**CUDA out of memory**: Reduce `--n_examples` or use fewer GPUs with `--nproc_per_node`

**Import errors**: Ensure the modified transformers library is properly installed:
```bash
cd diversed_code_release/transformers
pip install -e .
```

**Template file warnings**: These are normal and don't affect functionality.

## Test Results Location

Test results are saved to:
- `./data/cnndm/{method}_test/{method}_*_generations.json` - Generated text
- `./data/cnndm/{method}_test/{method}_*_metrics.json` - Performance metrics

## Performance Verification

For lossy and speculative cascading methods, look for debug output showing:
- `[Num Accepted Tokens]: X, [Total Draft Tokens]: Y`

This indicates the speculative decoding is working correctly.

#### 6. Trained Ensemble Verification (sd_en)
```bash
torchrun --nproc_per_node=8 --master_port=29507 src/speculative_decoding_dp.py \
    --method sd_en \
    --target_model meta-llama/Llama-3.1-8B-Instruct \
    --draft_model meta-llama/Llama-3.2-1B-Instruct \
    --dataset cnndm \
    --model_path ./data/cnndm/sample_trained_checkpoint \
    --max_tokens 16 \
    --n_examples 1 \
    --do_sample False \
    --temperature 0.0
```

**Note**: A sample trained ensemble head checkpoint is included at `./data/cnndm/sample_trained_checkpoint/ensemble_head.bin` for testing purposes.

### Method Aliases

The following aliases are also supported and should work identically:

- `sd_lossy` → `lossy`
- `sd_static` → `static_en`
- `diversed` → `sd_en` (requires trained ensemble head)

### Expected Output

For successful tests, you should see:
1. Model loading messages
2. Generation progress with token counts
3. Final metrics saved to JSON files
4. No error messages

### Troubleshooting

**Port in use errors**: Change the `--master_port` value to a different number (e.g., 29507, 29508, etc.)

**CUDA out of memory**: Reduce `--n_examples` or use fewer GPUs with `--nproc_per_node`

**Import errors**: Ensure the modified transformers library is properly installed:
```bash
cd diversed_code_release/transformers
pip install -e .
```

**Template file warnings**: These are normal and don't affect functionality.

## Test Results Location

Test results are saved to:
- `./data/cnndm/{method}_test/{method}_*_generations.json` - Generated text
- `./data/cnndm/{method}_test/{method}_*_metrics.json` - Performance metrics

## Performance Verification

For lossy and speculative cascading methods, look for debug output showing:
- `[Num Accepted Tokens]: X, [Total Draft Tokens]: Y`

This indicates the speculative decoding is working correctly.

## Full Test Suite

The `test_all_methods.py` script tests:
- ✅ `auto` - Autoregressive baseline
- ✅ `sd` - Standard speculative decoding  
- ✅ `static_en` - Static ensemble verification
- ✅ `lossy` - Lossy speculative decoding
- ✅ `spe_cas` - Speculative cascading
- ✅ `sd_lossy` - Alias for lossy
- ✅ `sd_static` - Alias for static_en
- ✅ `sd_en` - Trained ensemble verification (includes sample checkpoint)
- ✅ `diversed` - Alias for `sd_en` (same as above)

## Continuous Integration

You can use this test script in CI/CD pipelines to ensure all methods continue working after code changes.
