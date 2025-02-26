# EuroLLM Evaluation


## EuropeanLLM Leaderboard

### Installation
```bash
git clone https://github.com/OpenGPTX/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 43468b998f0cc1db09d8dd3a252470cc63728eec
pip install -e .
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
