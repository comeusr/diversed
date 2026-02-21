"""
Script for running inference with DIVERSED models.
"""

import os
import sys
import argparse
import torch
import time
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import EnsembleWrapper
from src.speculative_decoding import standard_speculative_decoding, static_ensemble_verification, diversed_decoding


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with DIVERSED model")
    
    # Model arguments
    parser.add_argument("--target_model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--draft_model", type=str, required=True, help="Draft model name or path")
    parser.add_argument("--ensemble_head", type=str, help="Path to trained ensemble head")
    
    # Inference arguments
    parser.add_argument("--method", type=str, required=True, choices=["auto", "sd", "static_en", "diversed"], 
                        help="Decoding method (auto: autoregressive, sd: standard speculative decoding, static_en: static ensemble, diversed: DIVERSED)")
    parser.add_argument("--static_weight", type=float, default=0.5, help="Static ensemble weight (only used with static_en method)")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--prompt_file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--output_file", type=str, help="Output file for generations")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--num_draft_tokens", type=int, default=5, help="Number of draft tokens to generate at once")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Benchmark arguments
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs")
    
    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(args):
    """
    Load prompts from command line or file.
    
    Args:
        args: Command line arguments.
        
    Returns:
        List of prompts.
    """
    if args.prompt:
        return [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Default prompts for testing
        return [
            "Summarize the following article:\n\nArticle: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and one of the most recognizable structures in the world. The Eiffel Tower is the most-visited paid monument in the world; 6.91 million people ascended it in 2015. The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.\n\nSummary:",
            "Question: John has 5 apples. He gives 2 apples to his friend and buys 3 more apples from the store. How many apples does John have now?\n\nAnswer:",
            "Translate the following German text to English:\n\nGerman: Die künstliche Intelligenz hat in den letzten Jahren enorme Fortschritte gemacht.\n\nEnglish:"
        ]


def run_inference(args, prompts):
    """
    Run inference with the specified model and method.
    
    Args:
        args: Command line arguments.
        prompts: List of prompts.
        
    Returns:
        List of generations.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load target model
    print(f"Loading target model: {args.target_model}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # For autoregressive decoding, we don't need the draft model
    if args.method != "auto":
        # Load draft model
        print(f"Loading draft model: {args.draft_model}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        draft_model = None
    
    # Load ensemble head for DIVERSED
    ensemble_head = None
    if args.method == "diversed":
        if not args.ensemble_head:
            raise ValueError("Ensemble head path must be provided for DIVERSED method")
        
        print(f"Loading ensemble head from: {args.ensemble_head}")
        # Create a temporary wrapper to load the ensemble head
        wrapper = EnsembleWrapper(target_model, draft_model, manual_place_head=True)
        wrapper.load_ensemble_head(args.ensemble_head)
        ensemble_head = wrapper.ensemble_head
    
    # Set generation parameters
    generation_config = target_model.generation_config
    generation_config.max_new_tokens = args.max_tokens
    generation_config.do_sample = args.do_sample
    generation_config.temperature = args.temperature
    generation_config.num_assistant_tokens = args.num_draft_tokens
    
    # Tokenize prompts
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
        input_ids_list.append(input_ids)
    
    # Run inference
    generations = []
    times = []
    
    for i, input_ids in enumerate(input_ids_list):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        # Run inference with the specified method
        start_time = time.time()
        
        if args.method == "auto":
            # Autoregressive decoding
            output_ids = target_model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature
            )
        elif args.method == "sd":
            # Standard speculative decoding
            output_ids = standard_speculative_decoding(
                input_ids,
                target_model,
                draft_model,
                generation_config=generation_config
            )
        elif args.method == "static_en":
            # Static ensemble verification
            output_ids = static_ensemble_verification(
                input_ids,
                target_model,
                draft_model,
                ensemble_weight=args.static_weight,
                generation_config=generation_config
            )
        elif args.method == "diversed":
            # DIVERSED
            output_ids = diversed_decoding(
                input_ids,
                target_model,
                draft_model,
                ensemble_head,
                generation_config=generation_config
            )
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        # Decode output
        generation = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        generations.append(generation)
        
        print(f"Generation: {generation}")
        print(f"Time: {times[-1]:.2f} seconds")
        print()
    
    # Print average time
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} seconds")
    
    return generations, times


def run_benchmark(args, prompts):
    """
    Run benchmark for different methods.
    
    Args:
        args: Command line arguments.
        prompts: List of prompts.
    """
    methods = ["auto", "sd", "static_en", "diversed"]
    results = {}
    
    for method in methods:
        if method == "diversed" and not args.ensemble_head:
            print(f"Skipping {method} because ensemble head is not provided")
            continue
        
        print(f"\n=== Running benchmark for {method} ===\n")
        args.method = method
        
        method_times = []
        for run in range(args.num_runs):
            print(f"Run {run+1}/{args.num_runs}")
            _, times = run_inference(args, prompts)
            method_times.append(sum(times) / len(times))
        
        avg_time = sum(method_times) / len(method_times)
        results[method] = avg_time
        print(f"\n{method} average time: {avg_time:.2f} seconds\n")
    
    # Print comparison
    print("\n=== Benchmark Results ===\n")
    print(f"{'Method':<10} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 30)
    
    auto_time = results.get("auto", float("inf"))
    for method, time in results.items():
        speedup = auto_time / time if method != "auto" else 1.0
        print(f"{method:<10} {time:<10.2f} {speedup:<10.2f}x")


def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load prompts
    prompts = load_prompts(args)
    
    if args.benchmark:
        run_benchmark(args, prompts)
    else:
        generations, _ = run_inference(args, prompts)
        
        # Save generations to file if specified
        if args.output_file:
            with open(args.output_file, "w") as f:
                for prompt, generation in zip(prompts, generations):
                    f.write(f"Prompt: {prompt}\n\n")
                    f.write(f"Generation: {generation}\n\n")
                    f.write("-" * 80 + "\n\n")


if __name__ == "__main__":
    main()
