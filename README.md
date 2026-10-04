# EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems

This repository provides the EmbodiTTA implementation for continuous ImageNet-C and CIFAR-10-C evaluation. Source-domain BatchNorm candidates can be obtained in two ways: built offline from labeled source (training) data, or constructed online from the test stream itself.

## Installation

Use Python 3.10 or newer. Install a PyTorch build for your system from the [official selector](https://pytorch.org/get-started/locally/), then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data And Checkpoints

Set dataset and checkpoint paths in [`configs/default.json`](configs/default.json). ImageNet data should use `ImageFolder` class directories. CIFAR-10 candidate construction expects the extracted torchvision CIFAR-10 training data; CIFAR-10-C expects its `.npy` corruption arrays and `labels.npy`.

Prebuilt candidate pools are included at `checkpoints/cifar10_resnet50_candidates.pth` and `checkpoints/imagenet_resnet50_candidates.pth`. CIFAR-10 evaluation still requires your CIFAR-10-C data and a trusted serialized model checkpoint. ImageNet-C evaluation requires ImageNet-C data; by default it builds the candidate pool online from the test stream (set `online_candidates` to `false` to load the included pool instead). Candidate construction scripts remain available if you want to rebuild either pool from labeled training data.

The included `checkpoints/imagenet_resnet50_candidates.pth` was regenerated so its BatchNorm statistics are consistent with the default torchvision ResNet-50 weights (the original pool was trained with different weights and loading it into the default model destroyed accuracy). If your `--checkpoint` differs, rebuild the pool (Option A) or use the online option (Option B) so the candidates always match the evaluated model.

### Two ways to build the candidate pool

The candidate pool is a set of source-domain BatchNorm states that adaptation can fall back to. Both options below produce the same `list[dict[str, Tensor]]` format.

**Option A - offline from labeled source (training) data.** Cluster source features with KMeans and fine-tune BatchNorm per cluster using `build_candidates.py`:

```bash
python build_candidates.py --dataset imagenet --config configs/default.json
python run_imagenetc.py --config configs/default.json --no-online-candidates
```

This requires `build_candidates.imagenet.source_root` (or `--source-root`) to point at an ImageNet `ImageFolder`. The prebuilt `checkpoints/*_candidates.pth` files are loaded when `online_candidates` is `false`.

**Option B - online from the test domains (no source data needed).** Enable `"online_candidates": true` (the default). The pool starts with the source model's BatchNorm state and, after every adaptation trigger, the adapted BatchNorm state is appended as a new candidate. Later triggers select the nearest candidate among the source state and all previously adapted states, which lets the model fall back to earlier/source domains. This is the recommended option when labeled source data is unavailable.

## Quick Start

Edit the paths in `configs/default.json`, then run from this directory:

```bash
# ImageNet-C (online candidates, no source training data required)
python run_imagenetc.py --config configs/default.json

# CIFAR-10-C (requires CIFAR-10-C data and a trusted serialized model checkpoint)
python run_cifar10c.py --config configs/default.json
```

The scripts in `scripts/` provide equivalent Bash launch commands. Command-line options can override config values; use `--help` to see available options. Candidate pools and evaluation results are saved to the configured output paths.

### Shift detection initialization

The shift detector is initialized with two values from the config: `reference_entropy`, the mean entropy of the source model on the source training set, and the trigger threshold (`fixed_threshold` for ImageNet-C, `threshold` for CIFAR-10-C). For ImageNet-C these default to `0.54` and `0.3`, and for CIFAR-10-C to `0.13` and `0.06`. These defaults were computed with our own model checkpoint on the corresponding source training data, so they depend on the specific model and dataset; please re-estimate them for your own checkpoint and training set.


## Citation

Xiao Ma, Young D. Kwon, and Dong Ma, “EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems,” *IEEE Internet of Things Journal*, 2026. [https://doi.org/10.1109/JIOT.2026.3738781](https://doi.org/10.1109/JIOT.2026.3738781).
