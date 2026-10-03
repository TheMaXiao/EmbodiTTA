# EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems

This repository provides the EmbodiTTA implementation for continuous ImageNet-C and CIFAR-10-C evaluation, with candidate construction from ImageNet or CIFAR-10 training data.

## Installation

Use Python 3.10 or newer. Install a PyTorch build for your system from the [official selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data And Checkpoints

Set dataset and checkpoint paths in [`configs/default.json`](configs/default.json). ImageNet data should use `ImageFolder` class directories. CIFAR-10 candidate construction expects the extracted torchvision CIFAR-10 training data; CIFAR-10-C expects its `.npy` corruption arrays and `labels.npy`.

Datasets and checkpoints are not included. CIFAR-10 workflows require a trusted serialized model checkpoint. ImageNet candidate construction uses torchvision's pretrained ResNet-50 by default.

## Quick Start

Edit the paths in `configs/default.json`, then run from this directory:

```bash
python build_candidates.py --dataset imagenet --config configs/default.json
python build_candidates.py --dataset cifar10 --config configs/default.json
python run_imagenetc.py --config configs/default.json
python run_cifar10c.py --config configs/default.json
```

The scripts in `scripts/` provide equivalent Bash launch commands. Command-line options can override config values; use `--help` to see available options. Candidate pools and evaluation results are saved to the configured output paths.

## Citation

Xiao Ma, Young D. Kwon, and Dong Ma, “EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems,” *IEEE Internet of Things Journal*, 2026. [https://doi.org/10.1109/JIOT.2026.3738781](https://doi.org/10.1109/JIOT.2026.3738781).