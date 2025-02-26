import json
import logging
import pathlib

import fire

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


def main(path: str, task_name: str = "ogx_arcx_challenge", metric_name: str = "acc_norm,none"):
    outputs = []
    results_path = pathlib.Path(path)
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

        for lang in LANGUAGES:
            score = results[f"{task_name}_{lang}"][metric_name]
            output.append(str(score))

        outputs.append(output)

    for output in outputs:
        print("\t".join(output))


if __name__ == '__main__':
    fire.Fire(main)
