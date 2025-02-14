# Mightning
Minimal [Lightning](https://lightning.ai/)-like PyTorch wrapper for single-node multi-gpu training, implemented with pure distributed dataparallel

## Features
- Single-node, multi-GPU distributed dataparallel
- W&B logging of loss, metric values
- Saving checkpoints according to metrics
- 16-bit mixed precision

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
| ResNet50 | [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html) | Accuracy | 98.57 | - | `./examples/resnet/train.sh` | trained on 4 * RTX4090 |

## Todos
- [ ] Support 1 GPU, batch_size 1 inference in `Trainer.test`, while train on more than 1 GPUs
- [ ] Support running ddp without `torchrun`
- [ ] Support `Tensorboard` logger
- [ ] Support multi-node
