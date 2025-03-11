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
Some models might have dict instead of string chat template, and it will fail when using the lm-evaluation-harness from above.
To avoid that replace the `chat_template` function in the `lm_eval/models/huggingface.py` or `lm_eval/models/vllm_causallms.py`.

```python
def chat_template(self) -> str:
    if self.tokenizer.chat_template is not None:
        if isinstance(self.tokenizer.chat_template, dict):
            return self.tokenizer.chat_template["default"]  # Will throw error if there is no default template.
        return self.tokenizer.chat_template
    return self.tokenizer.default_chat_template
```

### Fix Gemma-2
#### Fix required for all the tasks
Additionally, we ported Gemma-2 models fix described here:
https://github.com/EleutherAI/lm-evaluation-harness/pull/2049

In `lm_eval/models/huggingface.py` replace (for vLLM eval see the link above):

```python
if getattr(self.config, "model_type", None) == "gemma":
```
with
```python
if getattr(self.config, "model_type", None) in ["gemma", "gemma2"]:
```

#### Fix required for the tasks with `description` field in the data
In lm-evaluation-harness the `description` field of the tasks, if exists, will be used as the system prompt.
This can cause models that do not support system prompts (e.g. `gemma-2-9b-instruct`) to crash.
This MR [Fix chat template; fix leaderboard math](https://github.com/EleutherAI/lm-evaluation-harness/pull/2475) suggests to change the `apply_chat_template` function in `lm_eval/models/huggingface.py` to the one below:
```
    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history, tokenize=False, add_generation_prompt=True
            )

        return chat_templated
```

Please note that with this change now you need to also import `jinja2` in `lm_eval/models/huggingface.py` by adding the following line to the beginning of your `huggingface.py`:
```
import jinja2
```




### Running eval

```bash
bash europeanllm_leaderboard_evaluation.sh $task $shots $rdir
```

#### Configuration:

`rdir` - output directory path

Tasks & Shots
- Hellaswag
  - `task="ogx_hellaswagx_*"` or `task="hellaswag"`
  - `shots="10"`
- Arc Easy
  - `task="ogx_arcx_easy_*"`  or `task="arc_easy"`
  - `shots="25"`
- Arc Challenge
  - `task="ogx_arcx_challenge_*"` or `task="arc_challenge"`
  - `shots="25"`
- GSM8k
  - `task="ogx_gsm8kx_*"` or `task=gsm8k`
  - `shots="5"`
- TruthfulQA
  - `task="ogx_truthfulqax_mc2_*"` or `task="truthfulqa_mc2"`
  - `shots="0"`
- MMLU
  - `task="ogx_mmlux_*-*"` or `task="mmlu"`
  - `shots="5"`
