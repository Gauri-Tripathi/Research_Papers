from dataclasses import dataclass
from typing import Optional

ARCHITECTURE_MAP = {
    "llama": {
        "head_dim": 128,
        "attn_proj": "o_proj",
        "mlp_proj": "down_proj",
        "mlp_intermediate": "gate_proj",
        "attention_output": "o_proj"
    },
    "phi": {
        "head_dim": 64,
        "attn_proj": "dense",
        "mlp_proj": "fc2",
        "mlp_intermediate": "fc1",
        "attention_output": "dense"
    }
}

@dataclass
class GenerationConfig:
    prompt_text: str
    max_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class AnalysisConfig:
    model_name: str
    arch: str
    task: str
    dataset_path: str
    max_prompts: Optional[int] = None
    random_subset: bool = False
    device: str = "cuda:0"

@dataclass
class PruningConfig:
    model_path: str
    save_dir: str
    arch: str
    device: str = "cuda"
    seed: int = 42
    iteration: int = 1
    dataset_path: str = "./Data.csv"
    max_evaluation_prompts: int = 50
