from config.config import AnalysisConfig
from evaluate_scores import NeuralImportanceAnalyzer


def main():
    config = AnalysisConfig(
        model_name="microsoft/phi-1_5",
        arch="phi",
        task="mlps",  # "heads" or "mlps"
        dataset_path="./Data.csv",
        max_prompts=10,
        random_subset=True,
        device="cuda:0"
    )
    
   
    analyzer = NeuralImportanceAnalyzer(config)
    results = analyzer.run_analysis()
    
    return results


if __name__ == "__main__":
    main()



