# Mightning
Minimal [Lightning](https://lightning.ai/)-like PyTorch wrapper for single-node multi-gpu training, implemented with pure distributed dataparallel

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

## Examples
| Model | Data | Metric | Dev | Test | Command | Remarks |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ResNet50 | [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) | Accuracy | 98.57 | - | `./examples/train.sh` | trained on 4 * RTX4090 |
