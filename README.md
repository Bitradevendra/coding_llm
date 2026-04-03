# coding_llm

`coding_llm` is a distributed LLM training project. The core source lives in the `llm_project` folder and includes training, generation, monitoring, and model-merging utilities.

## Overview

This repository is organized around large-model experimentation with DeepSpeed, checkpointing, inference, and distributed training support.

## Project Structure

```text
coding_llm/
|-- llm_project/
|   |-- train.py
|   |-- generate.py
|   |-- model.py
|   |-- model_merging.py
|   |-- monitoring.py
|   |-- requirements.txt
|   `-- ...
`-- README.md
```

## Requirements

- Python 3.8+
- NVIDIA GPU recommended
- CUDA and PyTorch environment compatible with DeepSpeed

## Installation

```bash
cd llm_project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running The Project

Distributed training:

```bash
cd llm_project
deepspeed --hostfile hostfile train.py
```

Generation from a checkpoint:

```bash
cd llm_project
python generate.py --checkpoint_path ./checkpoints/<checkpoint>.pt
```

Model merging:

```bash
cd llm_project
python model_merging.py
```

## How It Works

- `train.py` starts the distributed training loop.
- `model.py` defines model components.
- `data_management.py` and `distributed_training.py` handle data and cluster setup.
- `monitoring.py` and `utils.py` cover logging, metrics, and checkpoints.
- `generate.py` performs prompt-based inference from saved weights.
