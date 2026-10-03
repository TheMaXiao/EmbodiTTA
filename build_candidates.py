import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from emboditta.config import parse_configured_args, require_values
from emboditta.core import extract_bn_state
from emboditta.data import cifar10_transform, imagenet_transform
from emboditta.models import load_resnet50, load_trusted_model


def extract_features(model, dataset, device, layer_name, batch_size, max_samples):
    features = []
    captured = []
    layer = model.get_submodule(layer_name)

    def save_activation(_module, _inputs, output):
        pooled = output.mean(dim=(-2, -1)) if output.ndim == 4 else output
        captured.append(pooled.detach().cpu())

    hook = layer.register_forward_hook(save_activation)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    seen = 0
    model.eval()
    try:
        with torch.no_grad():
            for images, _ in loader:
                captured.clear()
                model(images.to(device))
                features.append(captured[-1])
                seen += images.shape[0]
                if seen >= max_samples:
                    break
    finally:
        hook.remove()
    if not features:
        raise ValueError("No source samples were available for candidate construction")
    return torch.cat(features, dim=0)[:max_samples]


def configure_bn_for_training(model, momentum):
    model.requires_grad_(False)
    parameters = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            module.track_running_stats = True
            module.momentum = momentum
            parameters.extend((module.weight, module.bias))
    return parameters


def train_candidate(base_model, dataset, device, stats_batch_size, train_batch_size, epochs, learning_rate):
    model = copy.deepcopy(base_model).to(device)
    parameters = configure_bn_for_training(model, momentum=0.1)
    stats_loader = DataLoader(dataset, batch_size=stats_batch_size, shuffle=False, num_workers=4)
    model.train()
    with torch.no_grad():
        for images, _ in stats_loader:
            model(images.to(device))

    train_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    model.eval()
    for _ in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return extract_bn_state(model)


def main():
    parser = argparse.ArgumentParser(description="Build an EmbodiTTA BN candidate pool from labeled source data.")
    parser.add_argument("--dataset", choices=("imagenet", "cifar10"), default="imagenet")
    parser.add_argument("--source-root", help="ImageFolder root (ImageNet) or torchvision CIFAR-10 data root")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint", help="Optional ResNet-50 checkpoint; defaults to torchvision ImageNet weights")
    parser.add_argument("--model-module-root", help="Source directory needed to import the serialized CIFAR model")
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--feature-layer", default="layer1.0.bn1")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--stats-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_configured_args(
        parser,
        lambda preview: f"build_candidates.{preview.dataset}",
    )
    require_values(parser, args, ("source_root", "output"))
    if args.dataset == "imagenet":
        dataset = datasets.ImageFolder(args.source_root, transform=imagenet_transform())
        model = load_resnet50(args.device, checkpoint=args.checkpoint)
    else:
        require_values(parser, args, ("checkpoint",))
        dataset = datasets.CIFAR10(
            args.source_root,
            train=True,
            download=False,
            transform=cifar10_transform(),
        )
        model = load_trusted_model(args.checkpoint, args.device, args.model_module_root)
    sample_count = min(len(dataset), args.max_samples)
    if sample_count < args.num_candidates:
        parser.error("--max-samples must be at least --num-candidates")
    torch.manual_seed(0)
    sample_indices = torch.randperm(len(dataset))[:sample_count].tolist()
    dataset = Subset(dataset, sample_indices)
    features = extract_features(
        model,
        dataset,
        args.device,
        args.feature_layer,
        args.feature_batch_size,
        sample_count,
    )
    assignments = KMeans(n_clusters=args.num_candidates, random_state=0, n_init=10).fit_predict(
        features.numpy().astype(np.float32, copy=False)
    )
    candidates = []
    for candidate_id in range(args.num_candidates):
        indices = np.flatnonzero(assignments == candidate_id).tolist()
        if not indices:
            continue
        candidates.append(
            train_candidate(
                model,
                Subset(dataset, indices),
                args.device,
                args.stats_batch_size,
                args.train_batch_size,
                args.epochs,
                args.learning_rate,
            )
        )
        print(f"Built candidate {candidate_id + 1}/{args.num_candidates} ({len(indices)} source samples)")

    payload = {
        "format_version": 1,
        "feature_layer": args.feature_layer,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved {len(candidates)} candidates to {output}")


if __name__ == "__main__":
    main()