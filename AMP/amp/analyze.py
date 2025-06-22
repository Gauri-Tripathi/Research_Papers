from config.config import *
from transformers import (AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed)
from collections import defaultdict
from typing import Callable,Dict,List,Any
import warnings
import torch
import numpy as np
from tqdm import tqdm




class HookManager:
    """Manages PyTorch forward hooks for neural analysis."""
    
    def __init__(self):
        self._hooks = []

    def register_hook(self, handle):
        self._hooks.append(handle)

    def deregister_all(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()



class AttentionHeadAnalyzer:
    """Handles attention head importance analysis."""
    
    def __init__(self, model, arch_name: str):
        self._model = model
        self._arch_name = arch_name
        self._head_count = model.config.num_attention_heads
        self._importance_scores = defaultdict(list)
        
        if arch_name not in ARCHITECTURE_MAP:
            raise ValueError(f"Unsupported architecture '{arch_name}'.")
        
        self._arch_config = ARCHITECTURE_MAP[arch_name]
        self._head_dimension = self._arch_config["head_dim"]
        
        print(f"Model has {self._head_count} heads with dimension {self._head_dimension}.")
    
    def get_importance_scores(self) -> Dict[int, List]:
        return {k: v.copy() for k, v in self._importance_scores.items()}
    
    def get_arch_config(self) -> Dict:
        return self._arch_config.copy()
    
    def get_head_count(self) -> int:
        return self._head_count
    
    def create_hook_function(self, layer_index: int) -> Callable:
        def hook_function(module, model_input, model_output):
            input_tensor = model_input[0].float()
            batch_size, seq_len, _ = input_tensor.shape
            hidden_dim = module.weight.shape[0]

           
            reshaped_input = input_tensor.reshape(-1, self._head_count, self._head_dimension)
            weight_tensor = module.weight.T.reshape(
                self._head_count, self._head_dimension, hidden_dim
            ).float()
            
  
            head_outputs = torch.einsum('bhd,hdo->bho', reshaped_input, weight_tensor)
            head_score = head_outputs.abs().sum(dim=(0, 2)) / (batch_size * seq_len)

  
            if torch.isinf(head_score).any():
                warnings.warn(f"Infinite values detected in layer {layer_index}.")

            self._importance_scores[layer_index].append(head_score.detach().cpu())
        
        return hook_function
    
    def setup_hooks(self, hook_manager: HookManager):
        """Setup hooks on attention projection layers."""
        attention_projection_name = self._arch_config["attn_proj"]
        
        for layer_index, model_layer in enumerate(self._model.model.layers):
            attention_module = model_layer.self_attn
            
            if not hasattr(attention_module, attention_projection_name):
                raise AttributeError(
                    f"Layer {layer_index} does not have attribute '{attention_projection_name}'."
                )
            
            projection_layer = getattr(attention_module, attention_projection_name)
            hook_handle = projection_layer.register_forward_hook(
                self.create_hook_function(layer_index)
            )
            hook_manager.register_hook(hook_handle)
    
    def compute_average_scores(self) -> Dict[int, np.ndarray]:
        mean_scores_by_layer = {}
        for layer_index, score_list in self._importance_scores.items():
            stacked_scores = torch.stack(score_list, dim=0)
            mean_score = stacked_scores.mean(dim=0)
            mean_scores_by_layer[layer_index] = mean_score.numpy()
        return mean_scores_by_layer
    
    def print_results(self, mean_scores: Dict[int, np.ndarray]):
        print("\nAttention Head Importance Results:")
        for layer_index in sorted(mean_scores.keys()):
            layer_importance = mean_scores[layer_index]
            importance_string = ", ".join([f"{score:.2f}" for score in layer_importance])
            print(f"Layer {layer_index} mean head importance: [{importance_string}]")
    
    def cleanup(self):

        self._importance_scores.clear()
