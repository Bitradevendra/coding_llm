# coding_llm

`coding_llm` is a distributed LLM training project. The main source code lives inside the `llm_project` folder and includes training, inference, monitoring, and model-merging utilities.

## Install

```bash
cd llm_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Use

Train with DeepSpeed:

```bash
cd llm_project
deepspeed --hostfile hostfile train.py
```

Generate from a checkpoint:

```bash
cd llm_project
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

## How It Works

- `train.py` starts distributed training.
- `generate.py` runs inference.
- `model_merging.py` handles model combination experiments.
- `monitoring.py` and `utils.py` handle logs and checkpoints.
