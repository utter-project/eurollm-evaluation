#!/usr/bin/env bash

rdir="./results"

task="$1"
shots="$2"
rdir="$3"

base_models=(
    "utter-project/EuroLLM-9B"
    "google/gemma-2-9b"
    "meta-llama/Llama-3.1-8B"
    "Qwen/Qwen2.5-7B"
    "ibm-granite/granite-3.1-8b-base"
    "CohereForAI/aya-23-8B"
    "mistralai/Mistral-7B-v0.3"
    "allenai/OLMo-2-1124-7B"
    "occiglot/occiglot-7b-eu5"
    "BSC-LT/salamandra-7b"
)

instruct_models=(
    "utter-project/EuroLLM-9B-Instruct"
    "google/gemma-2-9b-it"
    "meta-llama/Llama-3.1-8B-Instruct"
    "ibm-granite/granite-3.1-8b-instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "allenai/OLMo-2-1124-7B-Instruct"
    "CohereForAI/aya-expanse-8b"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Ministral-8B-Instruct-2410"
    "occiglot/occiglot-7b-eu5-instruct"
    "BSC-LT/salamandra-7b-instruct"
    "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"
    "openGPT-X/Teuken-7B-instruct-research-v0.4"
    "openGPT-X/Teuken-7B-instruct-commercial-v0.4"
)

for model in "${base_models[@]}" "${instruct_models[@]}"; do
    arguments="--num_fewshot $shots"

    model_name=$(echo $model | sed 's/\//__/g')



    if [[ " ${base_models[@]} " =~ " ${model} " ]]; then
        :
    else
        arguments="$arguments --apply_chat_template"
        if [[ "$shots" != "0" ]]; then
            arguments="$arguments --fewshot_as_multiturn"
        fi
    fi


    odir=$rdir/$task

    echo "------------------------------------------------------"
    if [ -e $odir/$model_name/results*json ]; then
        echo "Skipping $model $task"
        echo $arguments
        continue
    fi

    echo "Processing $model $task with arguments: '$arguments'"
    accelerate launch --main_process_port 18001 -m lm_eval \
                    --model hf \
                    --model_args pretrained=$model,dtype="bfloat16",trust_remote_code=True,nccl_timeout=3600 \
                    --trust_remote_code \
                    --tasks $task \
                    --batch_size auto \
                    --bootstrap_iters 100 \
                    --output_path $odir $arguments
done
