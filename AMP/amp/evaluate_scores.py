from analyze import HookManager, AttentionHeadAnalyzer
from config.config import *
from collections import defaultdict
from typing import Callable, Dict, List, Any
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from data_loader import DatasetLoader

class TextGenerator:

    def __init__(self, model, tokenizer, device: str = "cuda:0"):
        print("Initializing text generation pipeline...")
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("Pipeline initialized.")

    def generate(self, config: GenerationConfig) -> str:
        """Generate text using the configured pipeline."""
        generation_output = self._pipeline(
            config.prompt_text,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            eos_token_id=self._pipeline.tokenizer.eos_token_id,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            do_sample=True,
            num_return_sequences=1,
        )
        return generation_output[0]['generated_text']


class MLPNeuronAnalyzer(AttentionHeadAnalyzer):
    """Handles MLP neuron importance analysis."""
    
    def __init__(self, model, arch_name: str):
        super().__init__(model, arch_name)
    
    def create_hook_function(self, layer_index: int) -> Callable:
        """Create hook function for recording MLP neuron activations."""
        def hook_function(module, model_input, model_output):
            activations = model_input[0].float()
            absolute_activations = activations.abs()
            mean_activations = absolute_activations.mean(dim=(0, 1)).detach().cpu().numpy()
            self._importance_scores[layer_index].append(mean_activations)
        
        return hook_function
    
    def setup_hooks(self, hook_manager: HookManager):
        """Setup hooks on MLP projection layers."""
        mlp_projection_name = self._arch_config["mlp_proj"]
        
        for layer_index, model_layer in enumerate(self._model.model.layers):
            mlp_module = model_layer.mlp
            
            if not hasattr(mlp_module, mlp_projection_name):
                raise AttributeError(
                    f"MLP layer {layer_index} does not have attribute '{mlp_projection_name}'."
                )
            
            projection_layer = getattr(mlp_module, mlp_projection_name)
            hook_handle = projection_layer.register_forward_hook(
                self.create_hook_function(layer_index)
            )
            hook_manager.register_hook(hook_handle)
    
    def compute_average_scores(self) -> Dict[int, List[float]]:
        """Compute average MLP neuron importance scores."""
        mean_activations_by_layer = {}
        for layer_index, activation_list in self._importance_scores.items():
            stacked_activations = np.stack(activation_list, axis=0)
            mean_activation = stacked_activations.mean(axis=0)
            mean_activations_by_layer[layer_index] = mean_activation.tolist()
        return mean_activations_by_layer
    
    def print_results(self, mean_scores: Dict[int, List[float]]):
        """Print MLP neuron analysis results."""
        print("\nMLP Neuron Importance Results:")
        for layer_index in sorted(mean_scores.keys()):
            layer_importance = mean_scores[layer_index]
            # Show only first 10 neurons for brevity
            importance_string = ", ".join([f"{score:.4f}" for score in layer_importance[:10]])
            print(f"Layer {layer_index} mean neuron importance (first 10): [{importance_string}]")

class AnalyzerFactory:
    """Factory for creating different types of analyzers."""
    
    @staticmethod
    def create_analyzer(task: str, model, arch_name: str):
        """Create appropriate analyzer based on task type."""
        if task == "heads":
            return AttentionHeadAnalyzer(model, arch_name)
        elif task == "mlps":
            return MLPNeuronAnalyzer(model, arch_name)
        else:
            raise ValueError(f"Unknown task: {task}")
        
class ModelManager:
    """Manages model and tokenizer loading."""
    
    def __init__(self, model_name: str, device: str = "cuda:0"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self._model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name).to(self._device)
        print("Model loaded successfully.")
    
    def get_model(self):
        """Get the loaded model."""
        return self._model
    
    def get_tokenizer(self):
        """Get the loaded tokenizer."""
        return self._tokenizer

class NeuralImportanceAnalyzer:
    """Main orchestrator class for neural importance analysis."""
    
    def __init__(self, config: AnalysisConfig, model_manager=None, text_generator=None, analyzer=None, dataset_loader=None):
        self._config = config
        self._hook_manager = HookManager()
        
        self._model_manager = model_manager or ModelManager(config.model_name, config.device)
        self._text_generator = text_generator or TextGenerator(
            self._model_manager.get_model(),
            self._model_manager.get_tokenizer(),
            config.device
        )
        self._analyzer = analyzer or AnalyzerFactory.create_analyzer(
            config.task,
            self._model_manager.get_model(),
            config.arch
        )
        self._dataset_loader = dataset_loader or DatasetLoader(config.dataset_path)
    
    def run_analysis(self) -> Dict[int, Any]:
        print(f"Starting {self._config.task} importance analysis...")
        
        try:
            self._analyzer.setup_hooks(self._hook_manager)
            
            prompt_list = self._dataset_loader.load_prompts(
                max_count=self._config.max_prompts,
                use_random_subset=self._config.random_subset
            )
            print(f"Processing {len(prompt_list)} prompts.")
            
            print("Collecting importance scores...")
            for prompt in tqdm(prompt_list, desc="Processing prompts"):
                generation_config = GenerationConfig(prompt_text=prompt)
                _ = self._text_generator.generate(generation_config)
            
            print("Finished collecting scores.")
            
            mean_scores = self._analyzer.compute_average_scores()
            self._analyzer.print_results(mean_scores)
            
            return mean_scores
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self._hook_manager.deregister_all()
        self._analyzer.cleanup()