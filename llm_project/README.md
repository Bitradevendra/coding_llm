# Distributed LLM Training Project

This project provides a comprehensive framework for training large language models (LLMs) on a distributed setup with multiple nodes and GPUs.

## Project Structure

```
llm_project/
├── README.md
├── requirements.txt
├── config.py
├── data_management.py
├── distributed_training.py
├── model.py
├── monitoring.py
├── optimizations.py
├── training_coordinator.py
├── train.py
├── model_merging.py
└── utils.py
```

## Setup

1.  **Install Dependencies:**
    Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Training:**
    Adjust training parameters, model configurations, and DeepSpeed settings in `config.py`.

3.  **Prepare Datasets:**
    Modify `data_management.py` to load your custom datasets. The current implementation uses dummy data for demonstration.

## Running the Training

This project uses `deepspeed` to launch distributed training across multiple nodes.

1.  **Create a Hostfile:**
    On your master node, create a file named `hostfile` that lists the IP addresses of your training machines and the number of GPUs (`slots`) each machine will use.

    ```
    # hostfile
    <master_node_ip> slots=1
    <worker_node_ip> slots=1
    ```
    Replace `<master_node_ip>` and `<worker_node_ip>` with the actual IP addresses of your two computers.

2.  **Launch the Training:**
    From your master node, run the following command in the terminal from the `llm_project` directory. This command will start the training process on both machines.

    ```bash
    deepspeed --hostfile hostfile train.py
    ```
    DeepSpeed will handle the distribution of the code and data to the worker node.

## Model Merging

The `model_merging.py` script provides a utility for combining different pre-trained models. You can adapt this script to merge models you have trained.

To use it, you can run it directly to see the example usage:
```bash
python model_merging.py
```

## Inference and Generation

After training your model, you can interact with it using the `generate.py` script. This script loads a trained checkpoint and lets you provide prompts to get responses from the model.

1.  **Find a Checkpoint:**
    Locate a checkpoint file (`.pt`) in the `llm_project/checkpoints/` directory that was saved during training.

2.  **Run the Generation Script:**
    Execute the following command, replacing `<path_to_your_checkpoint.pt>` with the actual path to your checkpoint file.

    ```bash
    python generate.py --checkpoint_path ./checkpoints/<path_to_your_checkpoint.pt>
    ```

3.  **Interact with the Model:**
    Once the script is running, you can type your prompts in the terminal and press Enter to get a response from the LLM.

## Functionality Overview

-   **`config.py`**: Contains all configuration classes for the project, including DeepSpeed, 3D parallelism, and training parameters.
-   **`data_management.py`**: Handles distributed data loading and management.
-   **`distributed_training.py`**: Manages the setup of the distributed environment (DDP, NCCL).
-   **`model.py`**: Defines the model architecture, including memory-efficient attention and checkpointed transformer blocks.
-   **`monitoring.py`**: Includes tools for performance monitoring and logging.
-   **`optimizations.py`**: Implements various optimization techniques like gradient compression, adaptive batch sizing, and communication-computation overlap.
-   **`training_coordinator.py`**: Orchestrates the overall training process, including federated learning and knowledge distillation.
-   **`utils.py`**: Contains utility functions for checkpointing, fault tolerance, and communication topology optimization.
