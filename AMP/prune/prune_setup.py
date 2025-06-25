import numpy as np
from config.config import ARCHITECTURE_MAP

class PruneSetup:
    def __init__(self, transformer_layer, architecture: str):
        self._layer = transformer_layer
        self._arch = architecture
        if architecture not in ARCHITECTURE_MAP:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def get_pruning_indices(self, scores: np.ndarray, n_to_select: int):
        return np.argsort(scores)[:n_to_select]

    def get_num_attention_heads(self):
        return self._layer.self_attn.num_heads

    def get_mlp_intermediate_size(self):
        name = ARCHITECTURE_MAP[self._arch]["mlp_intermediate"]
        return getattr(self._layer.mlp, name).out_features