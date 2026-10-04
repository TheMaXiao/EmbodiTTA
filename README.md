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

### Detector and adaptation options

| Config key / flag | Default | Description |
| --- | --- | --- |
| `online_candidates` / `--online-candidates` | `true` | Build the pool online from the source model and each adapted model (Option B). Set to `false`/`--no-online-candidates` to load `candidate_pool` instead (Option A). |
| `candidate_pool` / `--candidate-pool` | `checkpoints/imagenet_resnet50_candidates.pth` | Prebuilt pool used when `online_candidates` is false. |
| `reference_entropy` / `--reference-entropy` | `0.54` | Fixed source/base entropy. When set, calibration is skipped. Set to `null` to calibrate on clean `imagenet_val_root`. |
| `fixed_threshold` / `--fixed-threshold` | `0.3` | Shift-detector threshold used with `reference_entropy`. |
| `absolute_entropy_threshold` / `--absolute-entropy-threshold` | `6.0` | Force an adaptation trigger whenever the smoothed entropy exceeds this value; the reference entropy is also capped at this value. |
| `adaptation_entropy_delta` / `--adaptation-entropy-delta` | `0.02` | Skip adaptation when the incoming window's pre-adaptation mean entropy is within this delta of the last adapted window's mean entropy (avoids redundant triggers). |
| `ema_momentum` / `--ema-momentum` | `0.998` | Momentum of the entropy EMA used by the shift detector. |
| `samples_per_corruption` / `--samples-per-corruption` | `null` (all) | Optionally evaluate a random subset per corruption; `--sample-seed` controls the sampling seed. |

For the 10k-samples-per-corruption setting used in our experiments:

```bash
python run_imagenetc.py --config configs/default.json \
  --samples-per-corruption 10000 --sample-seed 0 \
  --output outputs/imagenetc_emboditta_10k.json
```

Results are written as JSON and include `trigger_positions`, `skipped_positions`, and per-adaptation reports (`candidate_index`, `updated_batches`, `mean_entropy`).

## Citation

Xiao Ma, Young D. Kwon, and Dong Ma, “EmbodiTTA: Resource-Efficient Test-Time Adaptation for Embodied Visual Systems,” *IEEE Internet of Things Journal*, 2026. [https://doi.org/10.1109/JIOT.2026.3738781](https://doi.org/10.1109/JIOT.2026.3738781).
