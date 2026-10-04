import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from emboditta.core import (
    EMADomainShiftDetector,
    adapt_bn_decoupled,
    calibrate_shift_detector,
    entropy_from_logits,
    extract_bn_state,
    load_candidate_pool,
    select_source_candidate,
)
from emboditta.config import parse_configured_args, require_values
from emboditta.data import IMAGENETC_CORRUPTIONS, load_clean_imagenet, load_imagenetc_stream
from emboditta.models import load_resnet50


def evaluate(args):
    device = torch.device(args.device)
    model = load_resnet50(device, checkpoint=args.checkpoint)
    if args.online_candidates:
        # Start the pool with the source model and grow it with each adapted model.
        candidates = [extract_bn_state(model)]
    else:
        candidates = load_candidate_pool(args.candidate_pool)
    if args.reference_entropy is not None:
        # Use the provided source/base entropy and a fixed threshold instead of
        # calibrating on clean data.
        detector = EMADomainShiftDetector(
            reference_entropy=args.reference_entropy,
            stable_ema_std=0.0,
            momentum=args.ema_momentum,
            threshold_scale=1.0,
            absolute_entropy_threshold=args.absolute_entropy_threshold,
        )
        if args.fixed_threshold is not None:
            detector.threshold = float(args.fixed_threshold)
        detector.ema_entropy = float(args.reference_entropy)
    else:
        clean_set = load_clean_imagenet(args.imagenet_val_root)
        clean_loader = DataLoader(
            clean_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
        )
        detector = calibrate_shift_detector(
            model,
            clean_loader,
            device,
            momentum=args.ema_momentum,
            threshold_scale=args.threshold_scale,
            max_samples=args.calibration_samples,
            absolute_entropy_threshold=args.absolute_entropy_threshold,
        )

    corruptions = tuple(name.strip() for name in args.corruptions.split(",") if name.strip())
    stream = load_imagenetc_stream(
        args.imagenetc_root,
        severity=args.severity,
        corruptions=corruptions,
        samples_per_corruption=args.samples_per_corruption,
        sample_seed=args.sample_seed,
    )
    stream_loader = DataLoader(stream, batch_size=1, shuffle=False, num_workers=args.workers)
    stream_iterator = iter(stream_loader)
    correct = 0
    total = 0
    trigger_positions = []
    adaptation_reports = []
    skipped_positions = []
    last_adapted_window_entropy = None
    model.eval()

    while True:
        try:
            images, labels = next(stream_iterator)
        except StopIteration:
            break
        with torch.no_grad():
            logits = model(images.to(device))
        shifted = detector.update(logits)[0]
        if not shifted:
            correct += int((logits.argmax(dim=1).cpu() == labels).sum().item())
            total += labels.numel()
            continue

        window_images = [images]
        window_labels = [labels]
        previous_logits = [logits.detach().cpu()]
        for _ in range(args.adaptation_samples - 1):
            try:
                next_images, next_labels = next(stream_iterator)
            except StopIteration:
                break
            with torch.no_grad():
                next_logits = model(next_images.to(device))
            window_images.append(next_images)
            window_labels.append(next_labels)
            previous_logits.append(next_logits.detach().cpu())

        image_window = torch.cat(window_images, dim=0)
        previous_window_logits = torch.cat(previous_logits, dim=0)
        labels_window = torch.cat(window_labels, dim=0)
        # Gate redundant adaptations: skip when the incoming window's (free) pre-adaptation
        # entropy barely differs from the entropy of the last adapted window.
        window_entropy = float(entropy_from_logits(previous_window_logits).mean().item())
        if (
            args.adaptation_entropy_delta is not None
            and last_adapted_window_entropy is not None
            and abs(window_entropy - last_adapted_window_entropy) < args.adaptation_entropy_delta
        ):
            window_predictions = previous_window_logits.argmax(dim=1)
            correct += int((window_predictions == labels_window).sum().item())
            total += labels_window.numel()
            skipped_positions.append(total - labels_window.numel())
            print(
                f"Skipped adaptation at stream index {skipped_positions[-1]}; "
                f"window entropy={window_entropy:.4f} vs last adapted={last_adapted_window_entropy:.4f}"
            )
            continue
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
            previous_window_logits,
            device,
            stats_batch_size=args.stats_batch_size,
            parameter_batch_size=args.parameter_batch_size,
            stats_momentum=args.bn_momentum,
            learning_rate=args.learning_rate,
            entropy_margin=args.entropy_margin,
            consistency_weight=args.consistency_weight,
        )
        if args.online_candidates:
            # Keep the post-adaptation BN state as a new candidate for later triggers.
            candidates.append(extract_bn_state(model))
        with torch.no_grad():
            window_predictions = torch.cat(
                [model(batch.to(device)).argmax(dim=1).cpu() for batch in image_window.split(args.eval_batch_size)]
            )
        correct += int((window_predictions == labels_window).sum().item())
        total += labels_window.numel()
        trigger_positions.append(total - labels_window.numel())
        adaptation_reports.append(
            {
                "stream_index": trigger_positions[-1],
                "candidate_index": candidate_index,
                **report,
            }
        )
        last_adapted_window_entropy = float(report["mean_entropy"])
        detector.rebaseline(report["mean_entropy"])
        print(
            f"Adapted at stream index {trigger_positions[-1]} from candidate {candidate_index}; "
            f"window accuracy={float((window_predictions == labels_window).float().mean()):.4f}"
        )

    result = {
        "severity": args.severity,
        "corruptions": corruptions,
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "trigger_count": len(trigger_positions),
        "trigger_positions": trigger_positions,
        "skip_count": len(skipped_positions),
        "skipped_positions": skipped_positions,
        "adaptation_entropy_delta": args.adaptation_entropy_delta,
        "adaptations": adaptation_reports,
        "detector_reference_entropy": detector.reference_entropy,
        "detector_threshold": detector.threshold,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate EmbodiTTA on a continuous ImageNet-C stream.")
    parser.add_argument("--imagenetc-root")
    parser.add_argument("--imagenet-val-root", help="Clean ImageNet validation ImageFolder root")
    parser.add_argument("--candidate-pool")
    parser.add_argument("--online-candidates", action=argparse.BooleanOptionalAction, default=False,
                        help="Build the candidate pool online: seed it with the source model and append the "
                             "adapted BN state after every trigger (use --no-online-candidates to load --candidate-pool)")
    parser.add_argument("--checkpoint", help="Optional ResNet-50 checkpoint; otherwise download torchvision weights")
    parser.add_argument("--output", default="outputs/imagenetc_emboditta.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--severity", type=int, default=5)
    parser.add_argument("--corruptions", default=",".join(IMAGENETC_CORRUPTIONS))
    parser.add_argument("--samples-per-corruption", type=int)
    parser.add_argument("--sample-seed", type=int, default=0,
                        help="Seed for randomly subsampling each corruption when samples-per-corruption is set")
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--reference-entropy", type=float, default=None,
                        help="Fixed source/base entropy; when set, calibration is skipped")
    parser.add_argument("--fixed-threshold", type=float, default=None,
                        help="Fixed shift-detector threshold used with --reference-entropy")
    parser.add_argument("--absolute-entropy-threshold", type=float, default=None,
                        help="Force an adaptation trigger whenever the smoothed entropy exceeds this value; "
                             "also caps the reference entropy at this value")
    parser.add_argument("--adaptation-entropy-delta", type=float, default=None,
                        help="Skip adaptation when the incoming window's pre-adaptation mean entropy is within "
                             "this delta of the last adapted window's mean entropy")
    parser.add_argument("--adaptation-samples", type=int, default=512)
    parser.add_argument("--ema-momentum", type=float, default=0.995)
    parser.add_argument("--threshold-scale", type=float, default=5.0)
    parser.add_argument("--candidate-feature", default="layer1.0.bn1.running_mean")
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--candidate-momentum", type=float, default=0.3)
    parser.add_argument("--stats-batch-size", type=int, default=16)
    parser.add_argument("--parameter-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--bn-momentum", type=float, default=None,
                        help="BN running-stats momentum; default auto = stats_batch_size / adaptation_samples")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--entropy-margin", type=float, default=0.1 * math.log(1000))
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parse_configured_args(parser, "run_imagenetc")
    require_values(parser, args, ("imagenetc_root",))
    if not args.online_candidates:
        require_values(parser, args, ("candidate_pool",))
    if args.reference_entropy is None:
        require_values(parser, args, ("imagenet_val_root",))
    evaluate(args)


if __name__ == "__main__":
    main()