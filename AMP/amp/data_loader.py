import random
from typing import List, Optional, Union
from pathlib import Path

import pandas as pd
from datasets import Dataset


class DatasetLoader:
    def __init__(self, data_source: Union[str, Dataset] = "./Data.csv"):
        self._data_source = data_source

    def load_prompts(self, max_count: Optional[int] = None, use_random_subset: bool = False) -> List[str]:
        if isinstance(self._data_source, str):
            return self._load_from_csv(self._data_source, max_count, use_random_subset)
        elif isinstance(self._data_source, Dataset):
            return self._load_from_dataset(self._data_source, max_count, use_random_subset)
        else:
            raise ValueError("data_source must be a string (file path) or Dataset object")

    def _load_from_csv(self, file_path: str, max_count: Optional[int], use_random_subset: bool) -> List[str]:
        dataframe = pd.read_csv(file_path)
        hf_dataset = Dataset.from_pandas(dataframe)
        return self._load_from_dataset(hf_dataset, max_count, use_random_subset)

    def _load_from_dataset(self, dataset: Dataset, max_count: Optional[int], use_random_subset: bool) -> List[str]:
        prompt_list = []
        for data_row in dataset:
            instruction_text = data_row.get("instruction", "")
            input_context = data_row.get("input", "")
            response_text = data_row.get("Response", "")

            if input_context:
                formatted_prompt = (
                    f"### Instruction:\n{instruction_text}\n\n"
                    f"### Input:\n{input_context}\n\n"
                    f"### Response:\n{response_text}"
                )
            else:
                formatted_prompt = (
                    f"### Instruction:\n{instruction_text}\n\n"
                    f"### Response:\n{response_text}"
                )
            prompt_list.append(formatted_prompt)

        if max_count is not None and max_count < len(prompt_list):
            if use_random_subset:
                return random.sample(prompt_list, max_count)
            return prompt_list[:max_count]

        return prompt_list

    def set_data_source(self, data_source: Union[str, Dataset]):
        self._data_source = data_source

    def get_data_source(self) -> Union[str, Dataset]:
        return self._data_source
