"""
Script for running data parallel inference with DIVERSED models.
This is a modified version of the original speculative_decoding_dp.py script.
"""

import argparse
import json
import sys
import re
import traceback
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object
import torch
import time
import numpy as np
import evaluate
import sacrebleu
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataloader import SFTDataLoader
from src.models import EnsembleWrapper, EnsembleHead
from src.mydatasets.humaneval.evaluate import humaneval_reward_func, check_correctness, clean_pred
from src.mydatasets.mbpp.evaluate import mbpp_clean_pred, check_correctness_mbpp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_hidden_size(cfg):
    """
    Get the hidden size from a model configuration.
    
    Args:
        cfg: Model configuration.
        
    Returns:
        Hidden size.
    """
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    # Common HF models (Llama, Qwen, etc.)
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size


def cnndm_find_answer(text):
    """Extract summary from CNN/DailyMail format."""
    return re.split(r"\n\nArticle:", text)[0].strip()


def xsum_find_answer(text):
    """Extract summary from XSum format."""
    return re.split(r"\n\nDocument:", text)[0].strip()


def wmt_find_answer(text):
    """Extract translation from WMT format."""
    return re.split(r"\n\nEnglish:", text)[0].strip()


def truncate_to_n_words(text, n=32):
    """
    Truncate text to the first n words.
    
    Args:
        text: The text to truncate.
        n: Number of words to keep.
        
    Returns:
        Truncated text containing only the first n words.
    """
    words = text.split()
    if len(words) <= n:
        return text
    return ' '.join(words[:n])


def extract_first_answer_block(text):
    """Extract the first answer block from a text."""
    split_marker = "Question:"
    if split_marker in text:
        return text.split(split_marker, 1)[0].strip()
    return text.strip()


def find_answer(text):
    """Find the answer in a text for GSM8K."""
    match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
    if match:
        return round(float(match.group(1)))
    else:
        all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
        if all_m:
            return round(float(all_m[-1]))
    return "No answer found"


def reward_func(completions, ground_truth, **kwargs):
    """Reward function for GSM8K."""
    contents = [find_answer(c) for c in completions]
    ground_truth = [find_answer(gt) for gt in ground_truth]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def main(args):
    try:
        # Initialize accelerator for distributed training
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        
        # Get local rank for device placement
        local_rank = accelerator.local_process_index
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Process {accelerator.process_index} using device: {device}")
    
        # Set seed for reproducibility
        if args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        
        do_sample = (args.do_sample=="True")

        # Load target model with device_map="auto" for better memory management
        logger.info(f"Process {accelerator.process_index} loading target model: {args.target_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2",
            device_map={"": device},  # Map all modules to the specified device
        )

        if args.method == "sd" or args.method == "sd_lossy":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2",
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True

            logger.info(f"Number of assistant tokens: {args.num_assistant_tokens}")
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens

            if args.assistant_schedule != 'dynamic':
                logger.info(f"Assistant schedule: {args.assistant_schedule}")
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            ensemble_head = None
            draft_ensemble_weights = None
        elif args.method == "sd_en":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2",
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens
            
            if args.assistant_schedule != 'dynamic':
                logger.info(f"Assistant schedule: {args.assistant_schedule}")
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            target_hidden_size = get_hidden_size(model.config)
            draft_hidden_size = get_hidden_size(draft_model.config)

            ensemble_head = EnsembleHead(target_hidden_size=target_hidden_size, draft_hidden_size=draft_hidden_size)
            head_path = os.path.join(args.model_path, "ensemble_head.bin")
            logger.info(f"Loading ensemble head from {args.model_path}")
            ensemble_head.load_state_dict(torch.load(head_path))
            ensemble_head = ensemble_head.to(device)
            draft_ensemble_weights = None
        elif args.method == "sd_static":
            logger.info(f"Process {accelerator.process_index} loading draft model: {args.draft_model}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                args.draft_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="sdpa" if 'gemma' in args.target_model else "flash_attention_2",
                device_map={"": device},  # Map all modules to the specified device
            )
            draft_model.generation_config.do_sample = do_sample
            draft_model.generation_config.temperature = args.temperature
            draft_model.generation_config.is_assistant=True
            draft_model.generation_config.num_assistant_tokens=args.num_assistant_tokens

            if args.assistant_schedule != 'dynamic':
                draft_model.generation_config.num_assistant_tokens_schedule = args.assistant_schedule
                draft_model.generation_config.assistant_confidence_threshold = args.assistant_confidence_threshold
                draft_model.generation_config.min_length=int(1)

            draft_ensemble_weights = args.draft_ensemble_weights
            ensemble_head = None
        else:
            draft_model = None
            ensemble_head = None
            draft_ensemble_weights = None

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        
        # Set chat template
        template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "template.jinja")
        if not os.path.exists(template_path):
            logger.warning(f"Template file not found at {template_path}, using default template")
            template_content = """{% for message in messages %}
                                {% if message['role'] == 'user' %}
                                {{ message['content'] }}
                                {% elif message['role'] == 'assistant' %}
                                {{ message['content'] }}
                                {% endif %}
                                {% endfor %}"""
        else:
            with open(template_path, 'r') as f:
                template_content = f.read()
        
        tokenizer.chat_template = template_content
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Configure Gemma models if needed
        if 'gemma' in args.target_model:
            for m in (model, draft_model):
                if m is not None:
                    m.generation_config.cache_implementation = "dynamic"
                    m.config.cache_implementation = "dynamic"

        # Set up data loader
        data_iterator_kwargs = dict(
            process_index=accelerator.process_index,
            num_processes=accelerator.num_processes,
            max_length=2506,
            max_prompt_length=2048,
            seed=args.seed if args.seed is not None else 42,
            frac_unique_desirable=1.0,
            frac_unique_undesirable=1.0,
            control_tokens={},
        )

        dataloader = SFTDataLoader(
            [args.dataset], 
            tokenizer,
            split=args.split,
            microbatch_size=1,
            batch_size=args.gpu_count,
            n_examples=args.n_examples, 
            n_epochs=1,
            **data_iterator_kwargs
        )

        # Create output directories
        if accelerator.is_main_process:
            os.makedirs(args.model_path, exist_ok=True)    
        
        output_path = os.path.join(args.model_path, "{}_{}_{}_generations.json".format(args.method, args.draft_ensemble_weights, args.assistant_schedule))
        metrics_path = os.path.join(args.model_path, "{}_{}_{}_metrics.json".format(args.method, args.draft_ensemble_weights, args.assistant_schedule))
        
        all_completions, all_labels = [], []
        all_results = []

        all_metrics = {
            "generated": [],
            "total_time": [],
            "num_tokens": [],
            "num_tokens_per_sec": [],
        }

        # Prepare dataloader with accelerator
        dataloader = accelerator.prepare(dataloader)
        
        for idx, batch in enumerate(dataloader):
            try:
                logger.info(f"Process {accelerator.process_index} processing batch {idx}")
                
                # Move batch to the correct device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                input_ids = batch['original_prompt_input_ids']
                attn_mask = batch['original_prompt_attention_mask']
                
                # Standardize on using "problem" (singular) for consistency
                if "problems" in batch:
                    labels = batch["problems"]
                else:
                    labels = [answer[0]['content'] for answer in batch['target']]
                prompts = batch['original_prompt']

                # Clear CUDA cache to free up memory
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    start_time = time.time()
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=args.max_tokens,
                        do_sample=do_sample,
                        use_cache=True,
                        assistant_model=draft_model,
                        output_hidden_states=True,
                        temperature=args.temperature if args.temperature > 0 else 1.0,
                        num_beams=1,
                        ensemble_head=ensemble_head,
                        static_ensemble_draft_weight=draft_ensemble_weights,
                    )
                    end_time = time.time()
                    
                logger.info(f"Process {accelerator.process_index} generated {output_ids.shape[1] - input_ids.shape[1]} tokens in {end_time - start_time:.2f}s")

                all_metrics["total_time"].append(end_time-start_time)
                all_metrics["num_tokens"].append(output_ids[:, input_ids.shape[1]:].shape[-1])
                all_metrics["num_tokens_per_sec"].append(
                        all_metrics["num_tokens"][-1] / all_metrics["total_time"][-1]
                    )
                    
                generations = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

                all_completions.extend(generations)
                if "problems" not in batch:
                    all_labels.extend(labels)

                # Load evaluation metrics based on dataset
                if args.dataset == 'cnndm' or args.dataset == 'xsum':
                    rouge = evaluate.load('rouge')
                elif args.dataset == 'wmt':
                    all_bleu = []
                            
                # Process results for each example
                for p, g, t in zip(prompts, generations, labels):
                    if args.dataset == "gsm8k":
                        g = extract_first_answer_block(g)
                        pred = find_answer(g)
                        truth = find_answer(t)
                        label = (pred==truth)
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "pred": pred,
                            "answer": t,
                            "ground_truth": truth,
                            "label": label,
                            "model_path": args.model_path,
                            "seed": args.seed
                        })
                    elif args.dataset == "cnndm":
                        g = cnndm_find_answer(g)
                        metric = rouge.compute(predictions=[g], references=[t])
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "reference": t,
                            "metric": metric,
                        })
                    elif args.dataset == 'wmt':
                        g = wmt_find_answer(g)
                        metric = sacrebleu.sentence_bleu(g, [t], tokenize="13a", lowercase=True).score
                        all_bleu.append(metric)
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "reference": t,
                            "metric": metric,
                        })
                    elif args.dataset == "xsum":
                        g = xsum_find_answer(g)
                        # Truncate only the generated text to first 32 words for xsum task
                        g_truncated = truncate_to_n_words(g, 32)
                        metric = rouge.compute(predictions=[g_truncated], references=[t])
                        all_results.append({
                            "prompt": p[0],
                            "generation": g,
                            "generation_truncated": g_truncated,
                            "reference": t,
                            "metric": metric,
                        })
                    elif args.dataset == "humaneval":
                        # Handle the case where t might not have the expected structure
                        canonical_solution = t['canonical_solution'] if isinstance(t, dict) and 'canonical_solution' in t else None
                        metric = check_correctness(clean_pred(g), t)
                        all_results.append({
                            'prompt': p[0],
                            'generation': g,
                            "metric": metric,
                            "reference": canonical_solution
                        })
                        all_labels.append(metric)
                    elif args.dataset == "mbpp":
                        canonical_solution = t['code'] if isinstance(t, dict) and 'code' in t else None
                        metric = check_correctness_mbpp(g, t)
                        all_results.append({
                            'prompt': p[0],
                            'generation': g,
                            "metric": metric,
                            "reference": canonical_solution
                        })
                        all_labels.append(metric)

            except Exception as e:
                logger.error(f"Process {accelerator.process_index} encountered error in batch {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Wait for all processes to finish processing batches
        accelerator.wait_for_everyone()
        logger.info(f"Process {accelerator.process_index} finished processing all batches")
        
        # Gather results from all processes
        try:
            logger.info(f"Process {accelerator.process_index} gathering results")
            all_completions = gather_object(all_completions)
            all_labels = gather_object(all_labels)
            all_results = gather_object(all_results)
            for key in all_metrics:
                if key != "generated":
                    all_metrics[key] = accelerator.gather(torch.tensor(all_metrics[key], device=device)).cpu().numpy().tolist()
            logger.info(f"Process {accelerator.process_index} gathered results successfully")
        except Exception as e:
            logger.error(f"Process {accelerator.process_index} encountered error during gathering: {str(e)}")
            logger.error(traceback.format_exc())
            # Create empty results if gathering fails
            if accelerator.is_main_process:
                all_completions = []
                all_labels = []
                all_results = []
        
        # Only the main process should write to files
        if accelerator.is_main_process:
            # Dump all results at once as a JSON array
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

            # Compute and save metrics based on dataset
            if args.dataset == 'gsm8k':
                acc = sum(reward_func(all_completions, all_labels)) / len(all_labels)
                metrics = {"accuracy": acc}
            elif args.dataset == "cnndm":
                metrics = rouge.compute(predictions=all_completions, references=all_labels)
            elif args.dataset == "xsum":
                # Truncate only completions to first 32 words for xsum task
                truncated_completions = [truncate_to_n_words(completion, 32) for completion in all_completions]
                metrics = rouge.compute(predictions=truncated_completions, references=all_labels)
            elif args.dataset == "wmt":
                metrics = np.mean(all_bleu)
            elif args.dataset == "humaneval" or args.dataset == "mbpp":
                # Calculate accuracy for code generation datasets
                correct = 0
                total = 0
                for result in all_results:
                    if result.get("metric", False):
                        correct += 1
                    total += 1
                
                if total > 0:
                    accuracy = correct / total
                else:
                    accuracy = 0.0
                
                metrics = {"accuracy": accuracy}

            # Combine all metrics
            result_stats = {
                "performance": metrics,
                "num_tokens_per_sec": np.mean(all_metrics["num_tokens_per_sec"]),
                "total_time": np.mean(all_metrics["total_time"]),
                "num_tokens": np.mean(all_metrics["num_tokens"]),
            }
            
            # Save metrics
            with open(metrics_path, "w") as f:
                json.dump(result_stats, f, indent=2)

            logger.info(f"Saved generations to {output_path}")
            logger.info(f"Results: {result_stats}")
            logger.info(f"Saved metrics to {metrics_path}")

    except Exception as e:
        logger.error(f"Process {accelerator.process_index} encountered error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Initialize distributed environment variables if not already set
    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

    parser = argparse.ArgumentParser(description="Run data parallel inference with DIVERSED")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save model outputs")
    parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=128, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (train/test)")
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "cnndm", "xsum", "wmt", "humaneval", "mbpp"], help="Dataset name")
    parser.add_argument("--n_examples", type=int, default=100, help="Number of examples to process")
    parser.add_argument("--draft_model", type=str, help="Draft model name or path")
    parser.add_argument("--target_model", type=str, required=True, help="Target model name or path")
    parser.add_argument("--method", type=str, default="sd", choices=['auto', 'sd', 'sd_lossy', 'sd_static', 'static_en', 'sd_en', 'diversed'], help="Decoding method")
    parser.add_argument("--draft_ensemble_weights", type=float, default=0.5, help="Static ensemble weights for draft model (only for static_en)")
    parser.add_argument("--num_assistant_tokens", type=int, default=5, help="Number of assistant tokens")
    parser.add_argument("--do_sample", type=str, default="False", choices=["True", "False"], help="Whether to use sampling")
    parser.add_argument("--assistant_schedule", type=str, default="constant", choices=["constant", "heuristic", "dynamic"], help="Assistant schedule")
    parser.add_argument("--assistant_confidence_threshold", type=float, default=0, help="Assistant confidence threshold (only for non-dynamic schedule)")
    
    args = parser.parse_args()
    main(args)
