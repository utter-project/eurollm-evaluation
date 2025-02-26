# EuroLLM Evaluation


## European LLM Leaderboard

Leaderboard details: https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard

### Installation
```bash
git clone https://github.com/OpenGPTX/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 43468b998f0cc1db09d8dd3a252470cc63728eec
pip install -e .
```

### Fix chat template
Some models might have dict instead of string chat template and it will fail when using the lm-evaluation-harness from above.
To avoid that replace the `chat_template` function in the `lm_eval/lm_eval/models/huggingface.py`.

```python
def chat_template(self) -> str:
    if self.tokenizer.chat_template is not None:
        if isinstance(self.tokenizer.chat_template, dict):
            return self.tokenizer.chat_template["default"]  # Will throw error if there is no default template.
        return self.tokenizer.chat_template
    return self.tokenizer.default_chat_template
```

### Running eval

```bash
bash europeanllm_leaderboard_evaluation.sh $task $shots $rdir
```

#### Configuration:

`rdir` - output directory path

Tasks & Shots
- Hellaswag
  - `task="ogx_hellaswagx_*"`
  - `shots="10"`
- Arc Easy
  - `task="ogx_arcx_easy_*"`
  - `shots="25"`
- Arc Challenge
  - `task="ogx_arcx_challenge_*"`
  - `shots="25"`
- GSM8k
  - `task="ogx_gsm8kx_*"`
  - `shots="5"`
- TruthfulQA
  - `task="ogx_truthfulqax_mc2_*"`
  - `shots="0"`
- MMLU
  - `task="ogx_mmlux_*-*"`
  - `shots="5"`
