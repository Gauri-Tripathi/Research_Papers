import torch
from typing import List
from config.config import ARCHITECTURE_MAP

def apply_attention_head_pruning(attention_layer, head_indices_to_prune: List[int], arch: str):
    arch_config = ARCHITECTURE_MAP[arch]
    attn_output_name = arch_config["attention_output"]

    v_proj_weight = attention_layer.v_proj.weight
    num_heads = attention_layer.num_heads
    head_dim = v_proj_weight.shape[0] // num_heads

    head_indices_to_keep = sorted(set(range(num_heads)) - set(head_indices_to_prune))
    dim_indices_to_keep = []
    for idx in head_indices_to_keep:
        dim_indices_to_keep.extend(range(idx * head_dim, (idx+1)*head_dim))

    # prune q, k, v
    attention_layer.q_proj.weight = torch.nn.Parameter(attention_layer.q_proj.weight.data[dim_indices_to_keep, :])
    attention_layer.k_proj.weight = torch.nn.Parameter(attention_layer.k_proj.weight.data[dim_indices_to_keep, :])
    attention_layer.v_proj.weight = torch.nn.Parameter(attention_layer.v_proj.weight.data[dim_indices_to_keep, :])

    # prune output projection
    attn_output = getattr(attention_layer, attn_output_name)
    attn_output.weight = torch.nn.Parameter(attn_output.weight.data[:, dim_indices_to_keep])
    attn_output.in_features = len(dim_indices_to_keep)

    # update metadata
    new_num_heads = len(head_indices_to_keep)
    new_hidden = new_num_heads * head_dim
    attention_layer.num_heads = new_num_heads
    attention_layer.hidden_size = new_hidden
    attention_layer.q_proj.out_features = new_hidden
    attention_layer.k_proj.out_features = new_hidden
    attention_layer.v_proj.out_features = new_hidden

def apply_mlp_neuron_pruning(mlp_module, neuron_indices_to_prune: List[int], arch: str):
    arch_config = ARCHITECTURE_MAP[arch]
    intermediate_name = arch_config["mlp_intermediate"]
    intermediate_layer = getattr(mlp_module, intermediate_name)
    intermediate_size = intermediate_layer.out_features

    keep_indices = sorted(set(range(intermediate_size)) - set(neuron_indices_to_prune))
    new_size = len(keep_indices)

    if arch == "llama":
        mlp_module.gate_proj.out_features = new_size
        mlp_module.gate_proj.weight = torch.nn.Parameter(mlp_module.gate_proj.weight.data[keep_indices, :])
        mlp_module.up_proj.out_features = new_size
        mlp_module.up_proj.weight = torch.nn.Parameter(mlp_module.up_proj.weight.data[keep_indices, :])
        mlp_module.down_proj.in_features = new_size
        mlp_module.down_proj.weight = torch.nn.Parameter(mlp_module.down_proj.weight.data[:, keep_indices])

    elif arch == "phi":
        mlp_module.fc1.out_features = new_size
        mlp_module.fc1.weight = torch.nn.Parameter(mlp_module.fc1.weight.data[keep_indices, :])
        mlp_module.fc2.in_features = new_size
        mlp_module.fc2.weight = torch.nn.Parameter(mlp_module.fc2.weight.data[:, keep_indices])
