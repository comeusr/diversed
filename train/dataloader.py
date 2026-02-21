"""
Data loading utilities for DIVERSED (DynamIc VErification RElaxed SpEculative Decoding).
"""

import torch
import random
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset
from transformers import AutoTokenizer


class DataLoader:
    """
    Base class for data loading.
    """
    def __init__(self, tokenizer, batch_size=8, n_examples=None, seed=42):
        """
        Initialize the data loader.
        
        Args:
            tokenizer: Tokenizer for the model.
            batch_size: Batch size.
            n_examples: Number of examples to load.
            seed: Random seed.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def __iter__(self):
        """
        Return an iterator over the dataset.
        """
        raise NotImplementedError
    
    def __len__(self):
        """
        Return the number of batches.
        """
        raise NotImplementedError


class SFTDataLoader(DataLoader):
    """
    Data loader for supervised fine-tuning.
    """
    def __init__(
        self,
        datasets: List[str],
        tokenizer: AutoTokenizer,
        split: str = "train",
        microbatch_size: int = 8,
        batch_size: int = 32,
        n_examples: Optional[int] = None,
        n_epochs: int = 1,
        process_index: int = 0,
        num_processes: int = 1,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        seed: int = 42,
        frac_unique_desirable: float = 1.0,
        frac_unique_undesirable: float = 1.0,
        control_tokens: Dict[str, str] = {},
    ):
        """
        Initialize the SFT data loader.
        
        Args:
            datasets: List of dataset names.
            tokenizer: Tokenizer for the model.
            split: Dataset split to use.
            microbatch_size: Micro-batch size.
            batch_size: Batch size.
            n_examples: Number of examples to load.
            n_epochs: Number of epochs.
            process_index: Process index for distributed training.
            num_processes: Number of processes for distributed training.
            max_length: Maximum sequence length.
            max_prompt_length: Maximum prompt length.
            seed: Random seed.
            frac_unique_desirable: Fraction of unique desirable examples.
            frac_unique_undesirable: Fraction of unique undesirable examples.
            control_tokens: Control tokens for the model.
        """
        super().__init__(tokenizer, microbatch_size, n_examples, seed)
        
        self.datasets = datasets
        self.split = split
        self.microbatch_size = microbatch_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.process_index = process_index
        self.num_processes = num_processes
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.frac_unique_desirable = frac_unique_desirable
        self.frac_unique_undesirable = frac_unique_undesirable
        self.control_tokens = control_tokens
        
        # Load datasets
        self.data = []
        for dataset_name in self.datasets:
            self._load_dataset(dataset_name)
        
        # Shuffle data
        random.shuffle(self.data)
        
        # Limit number of examples if specified
        if self.n_examples is not None:
            self.data = self.data[:self.n_examples]
        
        # Calculate number of training steps
        self.num_training_steps = len(self.data) * self.n_epochs // (self.batch_size * self.num_processes)
        
    def _load_dataset(self, dataset_name: str):
        """
        Load a dataset.
        
        Args:
            dataset_name: Name of the dataset to load.
        """
        if dataset_name == "gsm8k":
            self._load_gsm8k()
        elif dataset_name == "cnndm":
            self._load_cnndm()
        elif dataset_name == "xsum":
            self._load_xsum()
        elif dataset_name == "wmt":
            self._load_wmt()
        elif dataset_name == "humaneval":
            self._load_humaneval()
        elif dataset_name == "mbpp":
            self._load_mbpp()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_gsm8k(self):
        """
        Load the GSM8K dataset.
        """
        dataset = load_dataset("gsm8k", "main")
        split_data = dataset[self.split]
        
        for item in split_data:
            prompt = f"Question: {item['question']}\n\nAnswer:"
            answer = item['answer']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": answer}],
                "problem": item
            })
    
    def _load_cnndm(self):
        """
        Load the CNN/DailyMail dataset.
        """
        dataset = load_dataset("cnn_dailymail", "3.0.0")
        split_data = dataset[self.split]
        
        for item in split_data:
            prompt = f"Summarize the following article:\n\nArticle: {item['article']}\n\nSummary:"
            summary = item['highlights']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": summary}]
            })
    
    def _load_xsum(self):
        """
        Load the XSum dataset.
        """
        dataset = load_dataset("xsum")
        split_data = dataset[self.split]
        
        for item in split_data:
            prompt = f"Write a concise summary of the following document:\n\nDocument: {item['document']}\n\nSummary:"
            summary = item['summary']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": summary}]
            })
    
    def _load_wmt(self):
        """
        Load the WMT dataset.
        """
        dataset = load_dataset("wmt16", "de-en")
        split_data = dataset[self.split if self.split != "test" else "test"]
        
        for item in split_data:
            prompt = f"Translate the following German text to English:\n\nGerman: {item['translation']['de']}\n\nEnglish:"
            translation = item['translation']['en']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": translation}]
            })
    
    def _load_humaneval(self):
        """
        Load the HumanEval dataset.
        """
        dataset = load_dataset("openai_humaneval")
        split_data = dataset["test"]  # HumanEval only has a test split
        
        for item in split_data:
            prompt = f"Write a Python function to solve the following problem:\n\n{item['prompt']}"
            solution = item['canonical_solution']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": solution}],
                "problem": item
            })
    
    def _load_mbpp(self):
        """
        Load the MBPP dataset.
        """
        dataset = load_dataset("mbpp")
        split_data = dataset[self.split if self.split != "test" else "validation"]  # MBPP uses validation as test
        
        for item in split_data:
            prompt = f"Write a Python function to solve the following problem:\n\n{item['text']}\n\nYour solution should pass these tests:\n{item['test_list']}"
            solution = item['code']
            
            self.data.append({
                "prompt": prompt,
                "target": [{"role": "assistant", "content": solution}],
                "problem": item
            })
    
    def _prepare_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch of examples.
        
        Args:
            examples: List of examples.
            
        Returns:
            Dict containing the prepared batch.
        """
        batch = {
            "prompt_text": [],
            "original_prompt": [],
            "target": []
        }
        
        # Extract data from examples
        for example in examples:
            batch["prompt_text"].append(example["prompt"])
            batch["original_prompt"].append(example["prompt"])
            batch["target"].append(example["target"])
            
            # Add problem information if available
            if "problem" in example:
                if "problems" not in batch:
                    batch["problems"] = []
                batch["problems"].append(example["problem"])
        
        # Tokenize prompts
        prompt_tokenized = self.tokenizer(
            batch["prompt_text"],
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )
        
        batch["prompt_input_ids"] = prompt_tokenized["input_ids"]
        batch["prompt_attention_mask"] = prompt_tokenized["attention_mask"]
        
        # Prepare combined inputs (prompt + target)
        batch["original_prompt_text"] = batch["prompt_text"]
        batch["original_prompt_input_ids"] = batch["prompt_input_ids"]
        batch["original_prompt_attention_mask"] = batch["prompt_attention_mask"]
        
        # Tokenize combined inputs
        combined_texts = []
        for i, prompt in enumerate(batch["prompt_text"]):
            target_text = batch["target"][i][0]["content"]
            combined_texts.append({"role": "user", "content": prompt})
            combined_texts.append({"role": "assistant", "content": target_text})
        
        combined_tokenized = self.tokenizer.apply_chat_template(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # apply_chat_template returns input_ids directly when return_tensors="pt"
        batch["target_combined_input_ids"] = combined_tokenized
        # Create attention mask manually
        batch["target_combined_attention_mask"] = (combined_tokenized != self.tokenizer.pad_token_id).long()
        
        # Create labels for training
        labels = combined_tokenized.clone()
        
        # Set labels to -100 for prompt tokens (we only want to compute loss on target tokens)
        for i in range(len(labels)):
            prompt_length = batch["prompt_input_ids"][i].shape[0]
            labels[i, :prompt_length] = -100
        
        batch["target_labels"] = labels
        
        return batch
    
    def __iter__(self):
        """
        Return an iterator over the dataset.
        """
        # Create batches
        batches = []
        for epoch in range(self.n_epochs):
            # Shuffle data for each epoch
            epoch_data = self.data.copy()
            random.shuffle(epoch_data)
            
            # Create batches
            for i in range(0, len(epoch_data), self.microbatch_size):
                batch_data = epoch_data[i:i+self.microbatch_size]
                if len(batch_data) == self.microbatch_size:  # Only yield full batches
                    batches.append(batch_data)
        
        # Shuffle batches
        random.shuffle(batches)
        
        # Yield batches
        for batch_data in batches:
            yield self._prepare_batch(batch_data)
    
    def __len__(self):
        """
        Return the number of batches.
        """
        return len(self.data) * self.n_epochs // self.microbatch_size
