import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from emboditta.core import (
    EMADomainShiftDetector,
    adapt_bn_decoupled,
    extract_bn_state,
    load_candidate_pool,
    select_source_candidate,
)
from emboditta.config import parse_configured_args, require_values
from emboditta.models import load_trusted_model


DEFAULT_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)


class CIFAR10CStream(Dataset):
    def __init__(
        self,
        root: str,
        corruptions: tuple[str, ...],
        severity: int = 5,
        samples_per_corruption: int = 10000,
        seed: int = 0,
    ) -> None:
        if severity not in range(1, 6):
            raise ValueError("CIFAR-10-C severity must be between 1 and 5")
        if samples_per_corruption < 1 or samples_per_corruption > 10000:
            raise ValueError("samples_per_corruption must be between 1 and 10000")
        self.root = Path(root)
        self.corruptions = corruptions
        self.samples_per_corruption = samples_per_corruption
        self.severity_offset = (severity - 1) * 10000
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.247, 0.243, 0.261),
                ),
            ]
        )
        self.labels = np.load(self.root / "labels.npy", mmap_mode="r")
        rng = np.random.default_rng(seed)
        self.sample_indices = [
            rng.permutation(10000)[:samples_per_corruption]
            for _ in corruptions
        ]
        self.array_paths = []
        for corruption in corruptions:
            path = self.root / f"{corruption}.npy"
            if not path.is_file():
                raise FileNotFoundError(path)
            self.array_paths.append(path)
        self._arrays = None

    def __len__(self) -> int:
        return len(self.corruptions) * self.samples_per_corruption

    def __getitem__(self, index: int):
        domain_index, position = divmod(index, self.samples_per_corruption)
        sample_index = int(self.sample_indices[domain_index][position])
        if self._arrays is None:
            self._arrays = {}
        if domain_index not in self._arrays:
            self._arrays[domain_index] = np.load(self.array_paths[domain_index], mmap_mode="r")
        image = Image.fromarray(
            np.asarray(self._arrays[domain_index][self.severity_offset + sample_index])
        )
        return self.transform(image), int(self.labels[sample_index])


def compute_baseline(model, dataset, device, batch_size, workers, samples_per_corruption):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    correct_by_domain = np.zeros(len(dataset.corruptions), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            start = batch_index * batch_size
            offsets = range(start, start + len(labels))
            for offset, prediction, label in zip(offsets, predictions, labels):
                domain_index = offset // samples_per_corruption
                correct_by_domain[domain_index] += int(prediction == label)
    return correct_by_domain


def evaluate(args):
    device = torch.device(args.device)
    model = load_trusted_model(args.model_checkpoint, device, args.model_module_root)
    baseline_model = copy.deepcopy(model).eval()
    candidates = load_candidate_pool(args.candidate_pool)
    candidates.append(extract_bn_state(model))

    corruptions = tuple(item.strip() for item in args.corruptions.split(",") if item.strip())
    dataset = CIFAR10CStream(
        args.cifar10c_root,
        corruptions,
        severity=args.severity,
        samples_per_corruption=args.samples_per_corruption,
        seed=args.seed,
    )
    detector = EMADomainShiftDetector(
        reference_entropy=args.reference_entropy,
        stable_ema_std=0.0,
        momentum=args.ema_momentum,
        smoothing_window=args.entropy_window,
    )
    detector.ema_entropy = args.reference_entropy
    detector.threshold = args.threshold

    baseline_correct = compute_baseline(
        baseline_model,
        dataset,
        device,
        args.baseline_batch_size,
        args.workers,
        args.samples_per_corruption,
    )
    method_correct = np.zeros(len(corruptions), dtype=np.int64)
    method_counts = np.zeros(len(corruptions), dtype=np.int64)
    trigger_positions = []
    adaptation_reports = []
    iterator = iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    stream_index = 0

    while stream_index < len(dataset):
        images, labels = next(iterator)
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        triggered = detector.update(logits)[0]
        enough_samples = stream_index + args.adaptation_samples < len(dataset)
        if not triggered or not enough_samples:
            domain_index = stream_index // args.samples_per_corruption
            method_correct[domain_index] += int((logits.argmax(dim=1).cpu() == labels).sum().item())
            method_counts[domain_index] += labels.numel()
            stream_index += 1
            continue

        trigger_positions.append(stream_index)
        domain_index = stream_index // args.samples_per_corruption
        method_correct[domain_index] += int((logits.argmax(dim=1).cpu() == labels).sum().item())
        method_counts[domain_index] += labels.numel()
        window_images = []
        window_labels = []
        for _ in range(args.adaptation_samples):
            next_images, next_labels = next(iterator)
            window_images.append(next_images)
            window_labels.append(next_labels)
        image_window = torch.cat(window_images, dim=0)
        labels_window = torch.cat(window_labels, dim=0)
        previous_logits = None
        if args.consistency_weight > 0:
            with torch.no_grad():
                previous_logits = torch.cat(
                    [model(batch.to(device)).cpu() for batch in image_window.split(args.previous_batch_size)],
                    dim=0,
                )
        candidate_index = select_source_candidate(
            model,
            image_window,
            candidates,
            device,
            feature_name=args.candidate_feature,
            batch_size=args.candidate_batch_size,
            stats_momentum=args.candidate_momentum,
        )
        report = adapt_bn_decoupled(
            model,
            image_window,
            previous_logits,
            device,
            stats_batch_size=args.stats_batch_size,
            parameter_batch_size=args.parameter_batch_size,
            stats_momentum=args.bn_momentum,
            learning_rate=args.learning_rate,
            entropy_margin=args.entropy_margin,
            consistency_weight=args.consistency_weight,
        )
        with torch.no_grad():
            adapted_predictions = torch.cat(
                [
                    model(batch.to(device)).argmax(dim=1).cpu()
                    for batch in image_window.split(args.eval_batch_size)
                ],
                dim=0,
            )
        first_adapted_index = stream_index + 1
        for offset, prediction, label in zip(
            range(first_adapted_index, first_adapted_index + len(labels_window)),
            adapted_predictions,
            labels_window,
        ):
            domain_index = offset // args.samples_per_corruption
            method_correct[domain_index] += int(prediction == label)
            method_counts[domain_index] += 1
        adaptation_reports.append(
            {
                "stream_index": stream_index,
                "candidate_index": candidate_index,
                **report,
            }
        )
        print(
            f"trigger={stream_index} candidate={candidate_index} "
            f"adapted_samples={len(labels_window)} mean_entropy={report['mean_entropy']:.4f} "
            f"bn_momentum={report['stats_momentum']:.4f}"
        )
        detector.rebaseline(report["mean_entropy"])
        stream_index += 1 + len(labels_window)

    domain_results = []
    for index, corruption in enumerate(corruptions):
        count = int(method_counts[index])
        domain_results.append(
            {
                "corruption": corruption,
                "samples": count,
                "baseline_accuracy": int(baseline_correct[index]) / args.samples_per_corruption,
                "emboditta_accuracy": int(method_correct[index]) / count if count else 0.0,
            }
        )
    total = int(method_counts.sum())
    result = {
        "severity": args.severity,
        "samples_per_corruption": args.samples_per_corruption,
        "sample_count": total,
        "baseline_accuracy": int(baseline_correct.sum()) / len(dataset),
        "emboditta_accuracy": int(method_correct.sum()) / total if total else 0.0,
        "trigger_count": len(trigger_positions),
        "trigger_positions": trigger_positions,
        "initial_reference_entropy": args.reference_entropy,
        "final_reference_entropy": detector.reference_entropy,
        "used_threshold": detector.threshold,
        "domains": domain_results,
        "adaptations": adaptation_reports,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Run on-demand EmbodiTTA on a CIFAR-10-C stream.")
    parser.add_argument("--cifar10c-root", help="Directory containing CIFAR-10-C .npy files")
    parser.add_argument("--model-checkpoint", help="Trusted serialized CIFAR-10 model")
    parser.add_argument("--candidate-pool", help="CIFAR-10 source-derived BN candidate list")
    parser.add_argument("--model-module-root", help="Optional sys.path entry needed by the serialized model")
    parser.add_argument("--output", default="outputs/cifar10c_emboditta.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", default=",".join(DEFAULT_CORRUPTIONS))
    parser.add_argument("--samples-per-corruption", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-momentum", type=float, default=0.995)
    parser.add_argument("--entropy-window", type=int, default=10)
    parser.add_argument("--reference-entropy", type=float, default=0.13293485641437627)
    parser.add_argument("--threshold", type=float, default=0.06)
    parser.add_argument("--adaptation-samples", type=int, default=256)
    parser.add_argument("--candidate-feature", default="layer1.0.bn1.running_mean")
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--candidate-momentum", type=float, default=0.3)
    parser.add_argument("--previous-batch-size", type=int, default=64)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--parameter-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--bn-momentum", type=float, default=None,
                        help="BN running-stats momentum; default auto = stats_batch_size / adaptation_samples")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--entropy-margin", type=float, default=0.4 * np.log(10))
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--baseline-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parse_configured_args(parser, "run_cifar10c")
    require_values(parser, args, ("cifar10c_root", "model_checkpoint", "candidate_pool"))
    evaluate(args)


if __name__ == "__main__":
    main()