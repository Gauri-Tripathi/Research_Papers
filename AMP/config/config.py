from dataclasses import dataclass,field
from typing import Optional
from collections import defaultdict
import random 
import pandas as pd 
from datasets import Dataset
from transformers import  pipeline

ARCHITECTURE_MAP = {
    "llama": {
        "head_dim": 128,
        "attn_proj": "o_proj",
        "mlp_proj": "down_proj",
    },
    "phi": {
        "head_dim": 64,
        "attn_proj": "dense",
        "mlp_proj": "fc2",
    }
}



@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    prompt_text: str
    max_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class AnalysisConfig:
    """Configuration for neural importance analysis."""
    model_name: str
    arch: str
    task: str  # "heads" or "mlps"
    dataset_path: str
    max_prompts: Optional[int] = None
    random_subset: bool = False
    device: str = "cuda:0"

