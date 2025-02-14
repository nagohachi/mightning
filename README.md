# Mightning
Minimal Lightning-like distributed dataparallel wrapper for PyTorch.

# Feature
- Single-node, multi-GPU distributed dataparallel
- W&b logging of loss, metric values
- Saving checkpoints according to metrics

## Requirements
- [uv](https://docs.astral.sh/uv/) 0.5.3 or later

## Installation
- for CPU
    ```sh
    uv sync --extra cpu 
    ```
- for CUDA 11.x
    ```sh
    uv sync --extra cu118
    ```
- for CUDA 12.x
    ```
    uv sync --extra cu124
    ```
