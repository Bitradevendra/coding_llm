# Coding LLM

A distributed LLM training playground built for ambitious experiments: model training, checkpointing, monitoring, inference, and model-merging in one repo.

## Why This Project Exists

`coding_llm` is for builders who want to move past toy notebooks and into real distributed training structure. The repository centers on the `llm_project` core, where model orchestration, DeepSpeed workflows, and checkpoint-aware utilities live.

## What It Does

- trains transformer-style language models in distributed setups
- supports DeepSpeed launch workflows
- includes checkpoint-aware inference and model-merging utilities
- provides monitoring and training coordination helpers

## Project Structure

```text
coding_llm/
|-- llm_project/
|   |-- train.py
|   |-- generate.py
|   |-- model.py
|   |-- model_merging.py
|   |-- monitoring.py
|   |-- distributed_training.py
|   |-- data_management.py
|   `-- requirements.txt
`-- README.md
```

## Requirements

- Python 3.8+
- CUDA-capable machine recommended
- DeepSpeed-compatible PyTorch environment

## Installation

```bash
cd llm_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

Distributed training:

```bash
cd llm_project
deepspeed --hostfile hostfile train.py
```

Run generation from a checkpoint:

```bash
cd llm_project
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

Run model merging:

```bash
cd llm_project
python model_merging.py
```

## How It Works

- `train.py` bootstraps the distributed training lifecycle.
- `distributed_training.py` and `data_management.py` handle execution and data movement.
- `model.py` contains the model-side building blocks.
- `monitoring.py` and `utils.py` support metrics, logging, and checkpoints.
- `generate.py` lets you step from training into actual prompt-driven output.

## Best Fit

This repo is best for engineers who want a code-first training scaffold they can extend into more serious infrastructure over time.
