"""Extracts results from "results*.jsons" and outputs tab-based results ready to copy-paste into google sheets.

Usage example:

```bash
MODELS_DIR="model_results_dir_path"
TASK_NAME="ogx_truthfulqax_mc2"
python extract_results.py $MODELS_DIR --task_name $TASK_NAME
```
Use --english_only flag for non ogx tasks.
"""
import json
import logging
import pathlib

import fire
import numpy as np

LOGGER = logging.getLogger(__file__)

BASE_MODELS = [
    "utter-project/EuroLLM-9B",
    "google/gemma-2-9b",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    "ibm-granite/granite-3.1-8b-base",
    "CohereForAI/aya-23-8B",
    "mistralai/Mistral-7B-v0.3",
    "allenai/OLMo-2-1124-7B",
    "occiglot/occiglot-7b-eu5",
    "BSC-LT/salamandra-7b",
]

INSTRUCT_MODELS = [
    "utter-project/EuroLLM-9B-Instruct",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "ibm-granite/granite-3.1-8b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-2-1124-7B-Instruct",
    "CohereForAI/aya-expanse-8b",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-8B-Instruct-2410",
    "occiglot/occiglot-7b-eu5-instruct",
    "BSC-LT/salamandra-7b-instruct",
    "Aleph-Alpha/Pharia-1-LLM-7B-control-hf",
    "openGPT-X/Teuken-7B-instruct-research-v0.4",
    "openGPT-X/Teuken-7B-instruct-commercial-v0.4",
]

LANGUAGES = [
    'bg', 'cs', 'da', 'de', "el", 'es', 'et', 'fi', 'fr', 'hu', 'it', 'lt', 'lv', 'nl', 'pl', 'pt-pt', 'ro', 'sk', 'sl',
    'sv'
]

METRIC = {
    "ogx_arcx_easy": "acc_norm,none",
    "ogx_arcx_challenge": "acc_norm,none",
    "ogx_hellaswagx": "acc_norm,none",
    "ogx_mmlux": "acc,none",
    "ogx_gsm8kx": "acc,none",
    "ogx_truthfulqax": "acc,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "hellaswag": "acc_norm,none",
    "gsm8k": "exact_match,strict-match",
    "truthfulqa_mc2": "acc,none",
}


def main(path: str, task_name: str = "ogx_arcx_challenge", english_only: bool = False):
    outputs = []
    results_path = pathlib.Path(path)
    metric_name = METRIC[task_name]
    for model in BASE_MODELS + INSTRUCT_MODELS:
        output = [model, ""]  # Include model name and average field
        model_name = model.replace("/", "__")
        result_file = list((results_path / model_name).rglob("results*.json"))
        if len(result_file) == 0:
            LOGGER.warning(f"Could not find a results file for model {model}, skipping.")
            outputs.append([])  # Empty line for missing models
            continue

        assert len(result_file) == 1, "Multiple results files."
        result_file = result_file[0]

        with open(result_file) as f:
            results = json.load(f)["results"]

        if english_only:
            score = results[task_name][metric_name]
            output.append(str(score))
        else:
            for lang in LANGUAGES:
                if task_name == "ogx_mmlux":
                    score = []
                    for result_name in results:
                        if result_name.startswith(f"{task_name}_{lang}-"):
                            score.append(results[result_name][metric_name])
                    score = np.average(score)
                else:
                    score = results[f"{task_name}_{lang}"][metric_name]
                output.append(str(score))

        outputs.append(output)

    for output in outputs:
        print("\t".join(output))


if __name__ == '__main__':
    fire.Fire(main)
