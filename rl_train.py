import re
import os
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from dotenv import load_dotenv

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate
import torch
import random
import wandb
import sacrebleu

from train.dataloader import SFTDataLoader
from train.models import EnsembleWrapper
from train.trainer import DiversedTrainer as ReinforceTrainer
from src.mydatasets.humaneval.evaluate import humaneval_reward_func
from src.mydatasets.mbpp.evaluate import mbpp_reward_func

# Create a wrapper for humaneval_reward_func that adapts the input format
def humaneval_reward_func_wrapper(completions, ground_truth, **kwargs):
    # Check if ground_truth contains the expected problem structure
    if ground_truth and isinstance(ground_truth[0], dict) and "test" in ground_truth[0] and "entry_point" in ground_truth[0]:
        # If ground_truth already has the correct format, use it directly
        return humaneval_reward_func(completions, ground_truth)
    else:
        # If ground_truth doesn't have the expected format, try to extract problem info from batch
        if "problem" in kwargs:
            problems = kwargs["problem"]
            return humaneval_reward_func(completions, problems)
        else:
            # Fallback: return zero rewards if we can't get the problem info
            print("Warning: humaneval_reward_func called without proper problem information")
            return [0.0] * len(completions)

# Create a wrapper for mbpp_reward_func that adapts the input format
def mbpp_reward_func_wrapper(completions, ground_truth, **kwargs):
    # Check if ground_truth contains the expected problem structure
    if ground_truth and isinstance(ground_truth[0], dict) and "test_list" in ground_truth[0]:
        # If ground_truth already has the correct format, use it directly
        return mbpp_reward_func(completions, ground_truth)
    else:
        # If ground_truth doesn't have the expected format, try to extract problem info from batch
        if "problem" in kwargs:
            problems = kwargs["problem"]
            return mbpp_reward_func(completions, problems)
        else:
            # Fallback: return zero rewards if we can't get the problem info
            print("Warning: mbpp_reward_func called without proper problem information")
            return [0.0] * len(completions)

# Make the wrapped functions available in globals
globals()["humaneval_reward_func"] = humaneval_reward_func_wrapper
globals()["mbpp_reward_func"] = mbpp_reward_func_wrapper


def extract_first_answer_block(text):
    split_marker = "Question:"
    if split_marker in text:
        return text.split(split_marker, 1)[0].strip()
    return text.strip()

def find_answer(text):
    match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
    if match:
        return round(float(match.group(1)))
    else:
        all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
        if all_m:
            return round(float(all_m[-1]))
    return "No answer found"

    
def gsm8k_reward_func(completions, ground_truth, **kwargs):

    contents = [find_answer(extract_first_answer_block(completion)) for completion in completions]
    ground_truth = [find_answer(truth) for truth in ground_truth]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]

def wmt_find_answer(text):
    return re.split("\n\nEnglish:", text)[0].strip()

def wmt_reward_func(completions, ground_truth):
    contents = [wmt_find_answer(complete) for complete in completions]
    reward=[]
    for i, (content, gt) in enumerate(zip(contents, ground_truth)):
        reward.append(sacrebleu.sentence_bleu(content, [gt]).score)
    return reward


def cnndm_find_answer(text):
    return re.split("\n\nArticle:", text)[0]
    
def cnndm_reward_func(completions, ground_truth):

    rouge = evaluate.load("rouge")
    
    results = []
    for completion, gt in zip(completions, ground_truth):
        completion = cnndm_find_answer(completion)
        results.append(rouge.compute(predictions=[completion], references=[gt])['rouge2'])
    
    return results


def xsum_find_answer(text):
    return re.split("\n\nDocument:", text)[0]
    
def xsum_reward_func(completions, ground_truth):

    rouge = evaluate.load("rouge")
    
    results = []
    for completion, gt in zip(completions, ground_truth):
        completion = xsum_find_answer(completion)
        results.append(rouge.compute(predictions=[completion], references=[gt])['rouge2'])
    
    return results


@hydra.main(version_base=None, config_path="train_config", config_name="config")
def main(config: DictConfig):

    if config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = config.cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=config.cache_dir,
            name=config.exp_name,
        )
        
        config_path = os.path.join(config.local_run_dir, 'config.yaml')
        os.makedirs(config.local_run_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f)

        print('=' * 80)
        print(f'Writing to {config.local_run_dir}')
        print('=' * 80)

    tokenizer = AutoTokenizer.from_pretrained(config.model.target_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    special_tokens = []
    # Check if the tokenizer has a chat template and set a default one if it doesn't
    if not tokenizer.chat_template:
        with open("train_config/template.jinja") as f:
            tokenizer.chat_template = f.read()

        print("Default chat template set.")

    control_tokens = list(config.loss.get("control_tokens", {}).values())
    special_tokens.extend(control_tokens)

    num_tokens_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    data_iterator_kwargs = dict(
        process_index=0,
        num_processes=1,
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        seed=config.seed,
        frac_unique_desirable=config.frac_unique_desirable,
        frac_unique_undesirable=config.frac_unique_undesirable,
        control_tokens=config.loss.get("control_tokens", {}),
    )

    train_iterator = SFTDataLoader(
        config.datasets, 
        tokenizer,
        split='train',
        microbatch_size=config.model.batch_size,
        n_examples=config.n_examples,
        n_epochs=config.n_epochs,
        **data_iterator_kwargs
    )
    
    eval_iterator = SFTDataLoader(
        config.datasets, 
        tokenizer,
        split='validation' if config.datasets[0] == 'wmt' else 'test',
        microbatch_size=config.model.eval_microbatch_size,
        n_examples=config.n_eval_examples, 
        n_epochs=1,
        **data_iterator_kwargs
    )

    device0 = torch.device("cuda:0")

    print("Debuging the attention implementation: ", config.model.attn_implementation)

    target_model = AutoModelForCausalLM.from_pretrained(config.model.target_name_or_path, 
                                                  torch_dtype=torch.bfloat16, 
                                                  trust_remote_code=True, 
                                                  attn_implementation=config.model.attn_implementation,
                                                  device_map='auto'
                                                  )

    draft_model = AutoModelForCausalLM.from_pretrained(config.model.draft_name_or_path, 
                                                  torch_dtype=torch.bfloat16, 
                                                  trust_remote_code=True, 
                                                  attn_implementation=config.model.attn_implementation,
                                                  device_map='auto'
                                                  )


    model = EnsembleWrapper(target_model, draft_model, True)

    for name, param in model.target_model.named_parameters():
        param.requires_grad=False

    for name, param in model.draft_model.named_parameters():
        param.requires_grad=False

    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=train_iterator.num_training_steps - config.warmup_steps, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[config.warmup_steps])

    # Define Reward Functions
    reward_func = globals()["{}_reward_func".format(config.datasets[0])]

    trainer = ReinforceTrainer(
         model=model, 
         tokenizer=tokenizer, 
         reward_fn=reward_func, 
         optimizer=optimizer,
         scheduler=scheduler,
         train_iterator=train_iterator,
         eval_iterator=eval_iterator,
         config=config
    )

    # 7. Train
    trainer.train()
    
    # 8. Save
    trainer.save(
        os.path.join(config.local_run_dir, 'FINAL'), 
        metrics={'counter': trainer.example_counter}
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
