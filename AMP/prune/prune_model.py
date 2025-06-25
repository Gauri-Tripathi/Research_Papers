import os
import gc
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from amp.evaluate_scores import NeuralImportanceAnalyzer
from prune.prune_setup import PruneSetup
from prune.pruning import apply_attention_head_pruning, apply_mlp_neuron_pruning
from config.config import PruningConfig, ARCHITECTURE_MAP, AnalysisConfig

class ModelPruningExecutor:
    def __init__(self, cfg: PruningConfig):
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._arch_cfg = ARCHITECTURE_MAP[cfg.arch]
        self._setup = None
        self._load()

    def _load(self):
        self._model = AutoModelForCausalLM.from_pretrained(self._cfg.model_path, torch_dtype="auto", trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self._cfg.model_path, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        device = torch.device(self._cfg.device if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._setup = PruneSetup(self._model.model.layers[0], self._cfg.arch)

    def evaluate_importance(self):
        base = AnalysisConfig(
            model_name=self._cfg.model_path, arch=self._cfg.arch,
            dataset_path=self._cfg.dataset_path, max_prompts=self._cfg.max_evaluation_prompts,
            random_subset=False, device=self._cfg.device, task="heads")
        heads = NeuralImportanceAnalyzer(base).run_analysis()
        base.task = "mlps"
        mlps = NeuralImportanceAnalyzer(base).run_analysis()
        return heads, mlps
    def report_pruned_counts(self, head_indices_dict, neuron_indices_dict):
        total_heads_pruned = 0
        total_neurons_pruned = 0

        print("\n=== Pruning Report ===")
        for layer_idx in sorted(head_indices_dict.keys()):
            heads = len(head_indices_dict[layer_idx])
            neurons = len(neuron_indices_dict[layer_idx])
            total_heads_pruned += heads
            total_neurons_pruned += neurons
            print(f"Layer {layer_idx}: pruned {heads} heads, {neurons} neurons")

        print(f"\nTotal heads pruned: {total_heads_pruned}")
        print(f"Total neurons pruned: {total_neurons_pruned}\n")


    def prune_and_save(self):
        total_params = sum(p.numel() for p in self._model.parameters())
        params_layer = sum(p.numel() for p in self._model.model.layers[0].parameters())
        prune_ratio = (self._cfg.iteration * params_layer) / total_params
        print(f"Pruning ratio: {prune_ratio:.4f}")

        heads_scores, mlps_scores = self.evaluate_importance()

        heads_pruned = {}
        neurons_pruned = {}

        for idx, layer in enumerate(self._model.model.layers):
            n_heads = self._setup.get_num_attention_heads()
            n_to_prune = max(1, round(prune_ratio * n_heads))
            head_idx = self._setup.get_pruning_indices(heads_scores[idx], n_to_prune)
            apply_attention_head_pruning(layer.self_attn, head_idx, self._cfg.arch)
            heads_pruned[idx] = head_idx

            inter_size = self._setup.get_mlp_intermediate_size()
            n_neurons = max(1, round(prune_ratio * inter_size))
            neuron_idx = self._setup.get_pruning_indices(mlps_scores[idx], n_neurons)
            apply_mlp_neuron_pruning(layer.mlp, neuron_idx, self._cfg.arch)
            neurons_pruned[idx] = neuron_idx

        self.report_pruned_counts(heads_pruned, neurons_pruned)

        save_dir = os.path.join(self._cfg.save_dir, f'prune_iter_{self._cfg.iteration}')
        os.makedirs(save_dir, exist_ok=True)
        self._model.save_pretrained(save_dir)
        self._tokenizer.save_pretrained(save_dir)
    

    def cleanup(self):
        del self._model, self._tokenizer
        gc.collect()
        torch.cuda.empty_cache()

def main():
    cfg = PruningConfig(
        model_path="microsoft/phi-1_5",
        save_dir="./pruned_models",
        arch="phi",
        device="cuda",
        seed=42,
        iteration=1
    )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    executor = ModelPruningExecutor(cfg)
    executor.prune_and_save()
    executor.cleanup()

if __name__ == "__main__":
    main()
