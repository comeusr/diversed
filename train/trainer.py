"""
Trainer for DIVERSED (DynamIc VErification RElaxed SpEculative Decoding).
This module implements the reinforcement learning training procedure for the ensemble head.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Callable, Any, Tuple

from transformers import AutoTokenizer

# Constants
EPS = 1e-8


class RunningRewardBaseline:
    """
    Simple running average baseline for reward normalization.
    """
    def __init__(self, momentum=0.9):
        """
        Initialize the running reward baseline.
        
        Args:
            momentum: Momentum factor for the running average.
        """
        self.momentum = momentum
        self.baseline = None  # Scalar

    def update(self, rewards: torch.Tensor):
        """
        Update the baseline with new rewards.
        
        Args:
            rewards: Tensor of shape [batch_size] containing rewards.
        """
        mean_reward = rewards.mean().item()
        if self.baseline is None:
            self.baseline = mean_reward
        else:
            self.baseline = (
                self.momentum * self.baseline + (1 - self.momentum) * mean_reward
            )

    def get_advantages(self, rewards: torch.Tensor):
        """
        Compute advantages by subtracting the baseline from rewards.
        
        Args:
            rewards: Tensor of shape [batch_size] containing rewards.
            
        Returns:
            Tensor of shape [batch_size] containing advantages.
        """
        if self.baseline is None:
            return rewards
        return rewards - self.baseline


class DiversedTrainer:
    """
    Trainer for the DIVERSED model using reinforcement learning.
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 reward_fn, 
                 optimizer,
                 scheduler,
                 train_iterator, 
                 eval_iterator, 
                 config,
                 seed=42,
                 reg_scale=5,
                 log_ensemble_weight=True,
                 target_w_draft=0.6,
                 gamma=1.0
                ):
        """
        Initialize the DIVERSED trainer.
        
        Args:
            model: The ensemble model.
            tokenizer: Tokenizer for the model.
            reward_fn: Function to compute rewards.
            optimizer: Optimizer for training.
            scheduler: Learning rate scheduler.
            train_iterator: Iterator for training data.
            eval_iterator: Iterator for evaluation data.
            config: Configuration object.
            seed: Random seed.
            reg_scale: Regularization scale.
            log_ensemble_weight: Whether to log ensemble weights.
            target_w_draft: Target weight for the draft model.
            gamma: Discount factor.
        """
        self.seed = seed

        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.batch_counter = 0
        self.example_counter = 0

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reward_fn = reward_fn
        self.train_iterator = train_iterator
        self.eval_iterator = eval_iterator
        self.reg_scale = reg_scale
        self.gamma = gamma
        self.config = config
        self.log_ensemble_weights = log_ensemble_weight
        self.target_w_draft = target_w_draft

    def _get_batch_metric(self, batch, baseline_tracker=None):
        """
        Process a batch and compute metrics.
        
        Args:
            batch: Batch of data.
            baseline_tracker: Optional baseline tracker for reward normalization.
            
        Returns:
            Tuple of (loss, metrics).
        """
        input_ids = batch['original_prompt_input_ids']
        attn_mask = batch['original_prompt_attention_mask']
        target = [answer[0]['content'] for answer in batch['target']]

        # Generate outputs with the model
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.config.model.max_tokens,
                do_sample=self.config.model.do_sample,
                use_cache=True,
                temperature=self.config.model.temperature,
                num_beams=1,
            )
            if hasattr(self.model, 'reset_cache'):
                self.model.reset_cache()

        # Extract generated text
        reply_ids = output_ids[:, input_ids.shape[-1]:]
        generations = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)

        # Compute rewards
        with torch.no_grad():
            # Pass problem information to the reward function if available
            if 'problem' in batch:
                rewards = torch.Tensor(self.reward_fn(generations, target, problem=batch['problem']))
            else:
                rewards = torch.Tensor(self.reward_fn(generations, target))

        # Get logits from model for computing log-probs
        input_ids = output_ids[:, :-1]
        target_ids = output_ids[:, 1:]
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, log_ensemble_weights=False, return_draft_logits=True)
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()

        # Slice out generation logits and target ids
        gen_start = batch['prompt_input_ids'].shape[-1] - 1
        logits_gen = outputs.logits[:, gen_start:, :]         # [B, T_gen, V]
        target_ids_gen = target_ids[:, gen_start:]            # [B, T_gen]
        draft_logits = outputs.draft_logits[:, gen_start:, :] # [B, T_gen, V]
        w_draft = outputs.w_draft[:, gen_start:, :]
        w_target = outputs.w_target[:, gen_start:, :]

        # Create attention mask to ignore pad tokens
        gen_attention_mask = (target_ids_gen != self.tokenizer.pad_token_id).float()  # [B, T_gen]
        
        # Compute token-level log probs
        log_probs = F.log_softmax(logits_gen, dim=-1)                                   # [B, T_gen, V]
        
        # Move tensors to the same device
        target_ids_gen = target_ids_gen.to(log_probs.device)
        gen_attention_mask = gen_attention_mask.to(log_probs.device)
        
        token_log_probs = torch.gather(log_probs, 2, target_ids_gen.unsqueeze(-1)).squeeze(-1)  # [B, T_gen]
        
        # Apply padding mask and sum
        masked_log_probs = token_log_probs * gen_attention_mask                        # [B, T_gen]
        sequence_log_probs = masked_log_probs.sum(dim=1)                               # [B]

        # Compute regularization
        w_draft = w_draft.to(log_probs.device)
        w_target = w_target.to(log_probs.device)

        # Log ensemble weights if requested
        if self.log_ensemble_weights:
            w_draft_mean = (w_draft.squeeze(-1) * gen_attention_mask).sum() / gen_attention_mask.sum()
            w_target_mean = (w_target.squeeze(-1) * gen_attention_mask).sum() / gen_attention_mask.sum()
            
            print({
                "Draft_weight": w_draft_mean.item(),
                "Target_weight": w_target_mean.item()
            })
            wandb.log({
                "Draft_weight": w_draft_mean.item(),
                "Target_weight": w_target_mean.item()
            }, commit=False)
      
        # Compute entropy regularization
        w = torch.cat([w_draft, w_target], dim=-1)  # [B, T_gen, 2]
        entropy = - (w + EPS) * torch.log(w + EPS)  # [B, T_gen, 2]
        entropy = entropy.sum(dim=-1)               # [B, T_gen]
        
        # Mask out padding
        entropy = entropy * gen_attention_mask      # [B, T_gen]
        
        # Sum and average over batch
        entropy_reg = entropy.mean()     # scalar

        # Compute REINFORCE loss
        rewards = torch.tensor(rewards, dtype=torch.float32, device=log_probs.device)
        if baseline_tracker is None:
            advantages = rewards - 1
        else:
            advantages = baseline_tracker.get_advantages(rewards)
            baseline_tracker.update(rewards)

        reward_loss = -(sequence_log_probs * advantages).mean()
        
        # Compute Total Variance between draft and ensemble distributions
        draft_probs = F.softmax(draft_logits, dim=-1)      # [B, T_gen, V]
        ensemble_probs = F.softmax(logits_gen, dim=-1)     # [B, T_gen, V]
        
        # Total Variation distance: 0.5 * sum(|P - Q|) over vocabulary
        tv_distance = 0.5 * torch.abs(draft_probs - ensemble_probs).sum(dim=-1)  # [B, T_gen]
        
        # Compute 1 - TV distance, mask out padding tokens, and sum over sequence
        one_minus_tv = (1 - tv_distance) * gen_attention_mask  # [B, T_gen]
        reg_loss = -one_minus_tv.sum(dim=1).mean()  # sum over sequence, mean over batch

        # Compute dynamic scale
        reward_value = abs(reward_loss.detach().item())
        reg_value = reg_loss.detach().item()
        
        if reward_value > EPS:
            scale = 0.05 * reward_value / (reg_value + EPS)
        else:
            scale = self.config.model.reg_scale  # fallback when reward is zero
        
        # Total loss
        loss = reward_loss + scale * reg_loss
       
        # Logging
        metric = {
            "loss": loss.item(),
            "reward_mean": rewards.mean().item(),
            "log_prob_mean": sequence_log_probs.mean().item(),
        }

        return loss, metric

    def train(self):
        """
        Train the DIVERSED model.
        """
        grad_accum_steps = self.config.model.gradient_accumulation_steps

        # Initialize baseline tracker for certain tasks
        task_list = ['cnndm', 'wmt', 'xsum']
        if self.config.datasets[0] in task_list:
            baseline_tracker = RunningRewardBaseline()
        else:
            baseline_tracker = None

        for epoch in range(self.config.global_epochs):
            last_log = None
            batch_metrics = {}

            for batch_idx, batch in enumerate(self.train_iterator):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()

                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Compute loss and metrics
                loss, metrics = self._get_batch_metric(batch, baseline_tracker)
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                loss.backward()

                # Update metrics
                for k, v in metrics.items():
                    if k not in batch_metrics:
                        batch_metrics[k] = []
                    batch_metrics[k].append(v)

                # Update parameters after accumulation steps
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Clip gradients
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.model.max_grad_norm)
                    if "grad_norm" not in batch_metrics:
                        batch_metrics["grad_norm"] = []
                    batch_metrics["grad_norm"].append(grad_norm.item())
                    
                    # Update parameters
                    self.optimizer.step()
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"Learning Rate: {current_lr}")
                    self.optimizer.zero_grad()
    
                    end_time.record()
                    torch.cuda.synchronize()
                    step_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                    
                    examples_per_second = self.config.model.batch_size / step_time
                    if "examples_per_second" not in batch_metrics:
                        batch_metrics["examples_per_second"] = []
                    batch_metrics["examples_per_second"].append(examples_per_second)
                    
                    self.batch_counter += 1
                    self.example_counter += self.config.model.batch_size
    
                    # Log metrics periodically
                    if last_log is None or (torch.cuda.Event(enable_timing=True).record() - last_log > self.config.minimum_log_interval_secs):
                        mean_train_metrics = {}
                        for k, v in batch_metrics.items():
                            if len(v) > 0:
                                mean_train_metrics[k] = sum(v) / len(v)
    
                        mean_train_metrics['counters/examples'] = self.example_counter
                        mean_train_metrics['counters/updates'] = self.batch_counter
                        print(f'Train stats after {self.example_counter} examples: {mean_train_metrics}')
    
                        if self.config.wandb.enabled:
                            wandb.log(mean_train_metrics, step=self.example_counter)
    
                        last_log = torch.cuda.Event(enable_timing=True)
                        last_log.record()
                        batch_metrics = {}
                    else:
                        print(f'Skipping logging after {self.example_counter} examples to avoid logging too frequently')

                # Save checkpoint periodically
                if (batch_idx + 1) % self.config.model.save_freqs == 0:
                    self.save(
                        os.path.join(self.config.local_run_dir, str(self.example_counter)), 
                        metrics={'counter': self.example_counter}
                    )

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = {}):
        """
        Save the model to disk.
        
        Args:
            output_dir: Directory to save the model to.
            metrics: Metrics to save with the model.
        """
        print(f"Saving...")
        if output_dir is None:
            output_dir = os.path.join(self.config.local_run_dir, f'step-{self.example_counter}')

        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            metrics['counter'] = self.example_counter
            json.dump(metrics, f)
        
        print(f"Saving model...")
        print(output_dir)

        # Save model
        self.model.save_pretrained(output_dir)

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dict of evaluation metrics.
        """
        self.model.eval()
        eval_metrics = {}
        
        for batch in tqdm(self.eval_iterator, desc="Evaluating"):
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():
                _, batch_metrics = self._get_batch_metric(batch)
            
            # Update metrics
            for k, v in batch_metrics.items():
                if k not in eval_metrics:
                    eval_metrics[k] = []
                eval_metrics[k].append(v)
        
        # Compute mean metrics
        mean_eval_metrics = {}
        for k, v in eval_metrics.items():
            if len(v) > 0:
                mean_eval_metrics[k] = sum(v) / len(v)
        
        return mean_eval_metrics
