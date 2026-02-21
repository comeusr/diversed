"""
Script for training the DIVERSED model.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import EnsembleWrapper
from train.dataloader import SFTDataLoader
from train.trainer import DiversedTrainer

# Import reward functions
from utils.rewards import (
    gsm8k_reward_func,
    cnndm_reward_func,
    xsum_reward_func,
    wmt_reward_func,
    humaneval_reward_func,
    mbpp_reward_func
)


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train DIVERSED model")
    
    # Model arguments
    parser.add_argument("--target_model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--draft_model", type=str, required=True, help="Draft model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # Training arguments
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "cnndm", "xsum", "wmt", "humaneval", "mbpp"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--n_examples", type=int, default=None, help="Number of examples to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    
    # DIVERSED specific arguments
    parser.add_argument("--reg_scale", type=float, default=5.0, help="Regularization scale")
    parser.add_argument("--target_draft_weight", type=float, default=0.6, help="Target draft weight")
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor")
    
    # Inference arguments
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in seconds")
    
    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_reward_function(dataset_name):
    """
    Get the reward function for a dataset.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        Reward function.
    """
    reward_functions = {
        "gsm8k": gsm8k_reward_func,
        "cnndm": cnndm_reward_func,
        "xsum": xsum_reward_func,
        "wmt": wmt_reward_func,
        "humaneval": humaneval_reward_func,
        "mbpp": mbpp_reward_func
    }
    
    return reward_functions[dataset_name]


def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load models
    print(f"Loading target model: {args.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"Loading draft model: {args.draft_model}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Create ensemble model
    model = EnsembleWrapper(target_model, draft_model, manual_place_head=True)
    
    # Create data loaders
    train_loader = SFTDataLoader(
        [args.dataset],
        tokenizer,
        split="train",
        microbatch_size=args.batch_size,
        batch_size=args.batch_size * args.gradient_accumulation_steps,
        n_examples=args.n_examples,
        n_epochs=args.epochs,
        max_length=2048,
        max_prompt_length=1024,
        seed=args.seed
    )
    
    eval_split = "validation" if args.dataset == "wmt" else "test"
    eval_loader = SFTDataLoader(
        [args.dataset],
        tokenizer,
        split=eval_split,
        microbatch_size=args.batch_size,
        batch_size=args.batch_size * args.gradient_accumulation_steps,
        n_examples=min(100, args.n_examples) if args.n_examples else 100,
        n_epochs=1,
        max_length=2048,
        max_prompt_length=1024,
        seed=args.seed
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=train_loader.num_training_steps - args.warmup_steps, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_steps])
    
    # Create config object
    config = argparse.Namespace()
    config.model = argparse.Namespace()
    config.model.max_tokens = args.max_tokens
    config.model.do_sample = args.do_sample
    config.model.temperature = args.temperature
    config.model.max_grad_norm = args.max_grad_norm
    config.model.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.model.reg_scale = args.reg_scale
    config.model.save_freqs = args.save_steps
    config.datasets = [args.dataset]
    config.minimum_log_interval_secs = args.log_interval
    config.local_run_dir = args.output_dir
    config.wandb = argparse.Namespace()
    config.wandb.enabled = args.wandb_project is not None
    
    # Get reward function
    reward_fn = get_reward_function(args.dataset)
    
    # Create trainer
    trainer = DiversedTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_iterator=train_loader,
        eval_iterator=eval_loader,
        config=config,
        seed=args.seed,
        reg_scale=args.reg_scale,
        log_ensemble_weight=True,
        target_w_draft=args.target_draft_weight,
        gamma=args.gamma
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print("Saving final model...")
    trainer.save(os.path.join(args.output_dir, "final"))
    
    # Evaluate model
    print("Evaluating model...")
    eval_metrics = trainer.evaluate()
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Close Weights & Biases
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
