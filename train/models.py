"""
Contains the classes necessary for implementing DIVERSED (DynamIc VErification RElaxed SpEculative Decoding).
"""
import gc
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Union, Optional

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput
from transformers.cache_utils import Cache


@dataclass
class EnsembleModelOutWithPast(ModelOutput):
    """
    Output class for the Ensemble Model.
    
    Args:
        loss: Language modeling loss (for training).
        logits: Prediction scores of the language modeling head.
        past_key_values: Cache for fast auto-regressive decoding.
        hidden_states: Hidden states of the model.
        attentions: Attention weights.
        w_draft: Weight assigned to the draft model distribution.
        w_target: Weight assigned to the target model distribution.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    w_draft: Optional[torch.FloatTensor] = None
    w_target: Optional[torch.FloatTensor] = None


class EnsembleHead(nn.Module):
    """
    The EnsembleHead class implements a head that returns a 2D tensor for each output token,
    representing the weights to assign to the draft and target model distributions.
    """

    def __init__(self, target_hidden_size, draft_hidden_size, config=None, **kwargs):
        """
        Initialize the ensemble head.
        
        Args:
            target_hidden_size: Hidden size of the target model.
            draft_hidden_size: Hidden size of the draft model.
            config: Optional configuration object.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.model.summary_dropout_prob

        self.config = config
        self.target_hidden_size = target_hidden_size
        self.draft_hidden_size = draft_hidden_size
        self.hidden_size = target_hidden_size + draft_hidden_size
        self.device = None

        self.summary = nn.Linear(self.hidden_size, 2)
        self._equal_init()
        self.summary.to(torch.bfloat16)

    def _equal_init(self):
        """Initialize the weights of the ensemble head."""
        with torch.no_grad():
            w = torch.randn(self.hidden_size) * 0.01
            self.summary.weight.copy_(w.repeat(2, 1))
            self.summary.bias.zero_()

    def forward(self, hidden_states):
        """
        Forward pass of the ensemble head.
        
        Args:
            hidden_states: Hidden states from the concatenated draft and target models.
            
        Returns:
            Tensor of shape (batch_size, sequence_length, 2) representing the weights
            to assign to the draft and target model distributions.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            hidden_states = hidden_states.to(dtype=self.summary.weight.dtype)
            output = self.summary(hidden_states)
        return output


class EnsembleWrapper(nn.Module, GenerationMixin):
    """
    Wrapper class that combines a target model and a draft model with an ensemble head
    to implement DIVERSED.
    """

    def __init__(self, 
                 target_model, 
                 draft_model, 
                 manual_place_head=False,
                 config=None, 
                 static_draft_weights: Optional[float] = None,
                 **kwargs):
        """
        Initialize the ensemble wrapper.
        
        Args:
            target_model: The target language model.
            draft_model: The draft language model.
            manual_place_head: Whether to manually place the ensemble head.
            config: Optional configuration object.
            static_draft_weights: Optional static weights for the draft model.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.static_draft_weights = static_draft_weights

        self.target_hidden_size = self._get_hidden_size(target_model.config)
        self.draft_hidden_size = self._get_hidden_size(draft_model.config)

        if manual_place_head:
            self.ensemble_head = EnsembleHead(self.target_hidden_size, self.draft_hidden_size).to(self.draft_model.device)
            self.ensemble_head.device = self.draft_model.device
        else:
            self.ensemble_head = EnsembleHead(self.target_hidden_size, self.draft_hidden_size)

        # Prevent base-model bypass; keep prefix but point it back to self
        self.base_model_prefix = getattr(target_model, "base_model_prefix", "model")         
        if self.base_model_prefix == "model":                                                 
            self.model = self  # so getattr(self, "model", self) returns the WRAPPER  

        self.generation_config = target_model.generation_config
        self.config = target_model.config
        self.main_input_name = target_model.main_input_name
        self._supports_cache_class = target_model._supports_cache_class
        self.device = target_model.device
        self.dtype = target_model.dtype

        self.tp_size = getattr(target_model, "tp_size", getattr(draft_model, "tp_size", 1))

        self._supports_static_cache = getattr(
            target_model, "_supports_static_cache", False
        )

        self._target_past_key_values = None
        self._draft_past_key_values = None

    def get_compiled_call(self, compile_config):
        """Get compiled call for the model."""
        return self.forward

    def _get_hidden_size(self, cfg):
        """Get the hidden size from the model configuration."""
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            return cfg.text_config.hidden_size
        # Common HF models (Llama, Qwen, etc.)
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size

    def reset_cache(self):
        """Reset the key-value cache."""
        self._target_past_key_values = None
        self._draft_past_key_values = None

    def forward(
        self,
        input_ids=None,
        target_past_key_values=None,
        draft_past_key_values=None,
        attention_mask=None,
        use_cache=True,
        log_ensemble_weights=False,
        return_draft_logits=False,
        **kwargs,
    ):
        """
        Forward pass of the ensemble wrapper.
        
        Args:
            input_ids: Input token IDs.
            target_past_key_values: Past key-values for the target model.
            draft_past_key_values: Past key-values for the draft model.
            attention_mask: Attention mask.
            use_cache: Whether to use the key-value cache.
            log_ensemble_weights: Whether to log the ensemble weights.
            **kwargs: Additional keyword arguments.
            
        Returns:
            EnsembleModelOutWithPast: Output of the ensemble model.
        """
        with torch.no_grad():
            # Get draft model outputs
            draft_output = self.draft_model(input_ids=input_ids.to(self.draft_model.device), 
                                          past_key_values=self._draft_past_key_values, 
                                          attention_mask=attention_mask,
                                          use_cache=use_cache,
                                          output_hidden_states=True)
            if use_cache:
                self._draft_past_key_values = draft_output.past_key_values
            draft_last_hidden = draft_output.hidden_states[-1].detach().to(self.ensemble_head.device)
            draft_logits = draft_output.logits.detach().to(self.ensemble_head.device)
            if not return_draft_logits:
                del draft_logits
            
            # Get target model outputs
            target_output = self.target_model(input_ids=input_ids.to(self.target_model.device), 
                                            past_key_values=self._target_past_key_values,
                                            attention_mask=attention_mask,
                                            use_cache=use_cache,
                                            output_hidden_states=True)
            if use_cache:
                self._target_past_key_values = target_output.past_key_values
            target_last_hidden = target_output.hidden_states[-1].detach().to(self.ensemble_head.device)
            target_logits = target_output.logits.detach().to(self.ensemble_head.device)
            del target_output

        # Determine ensemble weights
        if self.static_draft_weights is not None:
            # Use static weights
            shape = list(draft_logits.shape)
            shape[-1] = 1
            w_draft = (torch.ones(shape)*self.static_draft_weights).to(self.ensemble_head.device)
            w_target = 1-w_draft
        else:
            # Use dynamic weights from ensemble head
            ensemble_input = torch.cat([draft_last_hidden, target_last_hidden], dim=-1)
            ensemble_weights = F.softmax(self.ensemble_head(ensemble_input), dim=-1)

            w_draft = ensemble_weights[..., 0].unsqueeze(-1)  # [B, T, 1]
            w_target = ensemble_weights[..., 1].unsqueeze(-1)

        # Combine logits using weights
        draft_logits = draft_logits.to(w_draft.dtype)
        target_logits = target_logits.to(w_target.dtype)

        logits = draft_logits.mul(w_draft)
        logits.add_(target_logits.mul(w_target))

        del draft_logits, target_logits

        loss = None

        return EnsembleModelOutWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            draft_logits=draft_logits if return_draft_logits else None,
            w_draft=w_draft,
            w_target=w_target,
        )

    def save_pretrained(
        self,
        save_directory: str,
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        **kwargs
    ):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model to.
            is_main_process: Whether this is the main process.
            state_dict: Optional state dictionary.
            save_function: Function to use for saving.
            **kwargs: Additional keyword arguments.
        """
        if not is_main_process:
            return
        
        os.makedirs(save_directory, exist_ok=True)
    
        # Save ensemble head weights
        head_path = os.path.join(save_directory, "ensemble_head.bin")
        save_function(self.ensemble_head.state_dict(), head_path)
    
        # Save config (store paths to base models too)
        config_to_save = {
            "target_model_path": getattr(self.target_model, "name_or_path", "unknown"),
            "draft_model_path": getattr(self.draft_model, "name_or_path", "unknown"),
            "target_hidden_size": self.ensemble_head.target_hidden_size,
            "draft_hidden_size": self.ensemble_head.draft_hidden_size,
        }
        with open(os.path.join(save_directory, "ensemble_config.json"), "w") as f:
            json.dump(config_to_save, f)

    def load_ensemble_head(self, load_directory):
        """
        Load the ensemble head from a directory.
        
        Args:
            load_directory: Directory to load the ensemble head from.
        """
        head_path = os.path.join(load_directory, "ensemble_head.bin")
        self.ensemble_head.load_state_dict(torch.load(head_path, map_location="cpu"))
