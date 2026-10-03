import math
from collections import deque
from collections.abc import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probabilities = logits.log_softmax(dim=1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=1)


class EMADomainShiftDetector:
    def __init__(
        self,
        reference_entropy: float,
        stable_ema_std: float,
        momentum: float = 0.995,
        threshold_scale: float = 5.0,
        min_interval: int = 0,
        smoothing_window: int = 1,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if stable_ema_std < 0:
            raise ValueError("stable_ema_std must be non-negative")
        if threshold_scale <= 0:
            raise ValueError("threshold_scale must be positive")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")
        self.reference_entropy = float(reference_entropy)
        self.threshold = max(float(stable_ema_std) * threshold_scale, 1e-8)
        self.momentum = momentum
        self.initial_momentum = momentum
        self.min_interval = max(0, int(min_interval))
        self.smoothing_window = smoothing_window
        self.ema_entropy: float | None = None
        self._ema_window: deque[float] = deque(maxlen=smoothing_window)
        self.samples_since_trigger = self.min_interval
        self._recovery_remaining = 0
        self._recovery_seen = 0

    @classmethod
    def calibrate(
        cls,
        model: nn.Module,
        loader: Iterable,
        device: torch.device | str,
        momentum: float = 0.995,
        threshold_scale: float = 5.0,
        max_samples: int = 1000,
        smoothing_window: int = 1,
    ) -> "EMADomainShiftDetector":
        detector = cls(
            0.0,
            0.0,
            momentum,
            threshold_scale,
            smoothing_window=smoothing_window,
        )
        ema_values: list[float] = []
        model.eval()
        seen = 0
        with torch.no_grad():
            for images, *_ in loader:
                logits = model(images.to(device))
                for entropy in entropy_from_logits(logits).detach().cpu().tolist():
                    ema_values.append(detector._update_ema(float(entropy)))
                    seen += 1
                    if seen >= max_samples:
                        break
                if seen >= max_samples:
                    break
        if not ema_values:
            raise ValueError("The calibration loader did not yield any samples")
        values = torch.tensor(ema_values, dtype=torch.float64)
        detector.reference_entropy = float(values.mean())
        stable_ema_std = float(values.std(unbiased=False))
        detector.threshold = max(stable_ema_std * threshold_scale, 1e-8)
        detector.samples_since_trigger = detector.min_interval
        return detector

    def _update_ema(self, entropy: float) -> float:
        if self.ema_entropy is None:
            self.ema_entropy = entropy
        else:
            self.ema_entropy = (
                self.momentum * self.ema_entropy + (1.0 - self.momentum) * entropy
            )
        return self.ema_entropy

    def update(self, logits: torch.Tensor) -> list[bool]:
        events: list[bool] = []
        for entropy in entropy_from_logits(logits).detach().cpu().tolist():
            ema = self._update_ema(float(entropy))
            self._ema_window.append(ema)
            if self._recovery_remaining:
                self._recovery_seen += 1
                self._recovery_remaining -= 1
                if self._recovery_seen == 100:
                    self.momentum = 0.999
                if self._recovery_remaining == 0:
                    self.reference_entropy = ema
                    self.samples_since_trigger = 0
                events.append(False)
                continue
            self.samples_since_trigger += 1
            smoothed_ema = sum(self._ema_window) / len(self._ema_window)
            shifted = abs(smoothed_ema - self.reference_entropy) > self.threshold
            triggered = shifted and self.samples_since_trigger >= self.min_interval
            if triggered:
                self.samples_since_trigger = 0
            events.append(triggered)
        return events

    def begin_recovery(
        self,
        stable_samples: int = 2000,
        initial_entropy: float | None = None,
    ) -> None:
        if stable_samples < 1:
            raise ValueError("stable_samples must be positive")
        self._recovery_remaining = stable_samples
        self._recovery_seen = 0
        self.momentum = self.initial_momentum
        self._ema_window.clear()
        if initial_entropy is not None:
            self.ema_entropy = float(initial_entropy)

    def rebaseline(self, entropy: float) -> None:
        self.reference_entropy = float(entropy)
        self.ema_entropy = float(entropy)
        self.momentum = self.initial_momentum
        self._ema_window.clear()
        self.samples_since_trigger = self.min_interval
        self._recovery_remaining = 0
        self._recovery_seen = 0


def calibrate_shift_detector(
    model: nn.Module,
    loader: Iterable,
    device: torch.device | str,
    momentum: float = 0.995,
    threshold_scale: float = 5.0,
    max_samples: int = 1000,
    smoothing_window: int = 1,
) -> EMADomainShiftDetector:
    return EMADomainShiftDetector.calibrate(
        model,
        loader,
        device,
        momentum=momentum,
        threshold_scale=threshold_scale,
        max_samples=max_samples,
        smoothing_window=smoothing_window,
    )


def extract_bn_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for attribute in ("weight", "bias", "running_mean", "running_var"):
                value = getattr(module, attribute, None)
                if value is not None:
                    state[f"{name}.{attribute}"] = value.detach().cpu().clone()
    if not state:
        raise ValueError("The model does not contain tracked BatchNorm layers")
    return state


def load_bn_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    missing: list[str] = []
    with torch.no_grad():
        for name, destination in (*parameters.items(), *buffers.items()):
            if name not in state:
                continue
            if destination.shape != state[name].shape:
                raise ValueError(f"Shape mismatch for BN tensor {name}")
            destination.copy_(state[name].to(device=destination.device, dtype=destination.dtype))
        for name in state:
            if name not in parameters and name not in buffers:
                missing.append(name)
    if missing:
        raise ValueError(f"Candidate contains tensors absent from the model: {missing[:3]}")


def _configure_bn_affine(model: nn.Module, momentum: float) -> list[nn.Parameter]:
    model.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.weight is None or module.bias is None:
                raise ValueError("All BatchNorm layers must have affine parameters")
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            module.track_running_stats = True
            module.momentum = momentum
            trainable.extend((module.weight, module.bias))
    if not trainable:
        raise ValueError("The model does not contain BatchNorm affine parameters")
    return trainable


def select_source_candidate(
    model: nn.Module,
    images: torch.Tensor,
    candidates: Sequence[dict[str, torch.Tensor]],
    device: torch.device | str,
    feature_name: str = "layer1.0.bn1.running_mean",
    batch_size: int = 16,
    stats_momentum: float = 0.3,
) -> int:
    if not candidates:
        raise ValueError("At least one BN candidate is required")
    if images.shape[0] == 0:
        raise ValueError("Cannot select a candidate from an empty window")
    bn_names = {
        f"{name}.running_mean"
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if feature_name not in bn_names:
        raise ValueError(f"BN feature {feature_name!r} is not present in this model")
    _configure_bn_affine(model, stats_momentum)
    model.train()
    with torch.no_grad():
        for batch in images.split(batch_size):
            model(batch.to(device))
    current_mean = dict(model.named_buffers())[feature_name]
    distances = []
    for candidate in candidates:
        if feature_name not in candidate:
            raise ValueError(f"Candidate is missing {feature_name!r}")
        reference = candidate[feature_name].to(device=current_mean.device, dtype=current_mean.dtype)
        distances.append(torch.linalg.vector_norm(current_mean - reference))
    selected = int(torch.stack(distances).argmin().item())
    load_bn_state(model, candidates[selected])
    model.eval()
    return selected


def _directional_consistency_loss(
    current_logits: torch.Tensor,
    previous_logits: torch.Tensor,
) -> torch.Tensor:
    current_direction = F.normalize(current_logits, p=2, dim=1)
    previous_direction = F.normalize(previous_logits, p=2, dim=1)
    direction = F.normalize(current_direction - previous_direction, p=2, dim=1)
    return -(current_logits * direction).sum(dim=1).mean()


def adapt_bn_decoupled(
    model: nn.Module,
    images: torch.Tensor,
    previous_logits: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    stats_batch_size: int = 16,
    parameter_batch_size: int = 1,
    stats_momentum: float | None = None,
    learning_rate: float = 1e-6,
    entropy_margin: float | None = None,
    consistency_weight: float = 0.0,
) -> dict[str, float | int]:
    if images.shape[0] == 0:
        raise ValueError("Cannot adapt on an empty window")
    if stats_momentum is None:
        # Scale the momentum so the whole window contributes roughly one full
        # round of BN-statistics updates: momentum ~= stats_bs / num_samples.
        stats_momentum = min(stats_batch_size / images.shape[0], 0.99)
    if consistency_weight > 0:
        if previous_logits is None:
            raise ValueError("previous_logits are required when consistency_weight > 0")
        if images.shape[0] != previous_logits.shape[0]:
            raise ValueError("images and previous_logits must contain the same samples")
    parameters = _configure_bn_affine(model, stats_momentum)
    model.train()
    with torch.no_grad():
        for batch in images.split(stats_batch_size):
            model(batch.to(device))

    optimizer = torch.optim.SGD(
        parameters,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    model.eval()
    if entropy_margin is None:
        with torch.no_grad():
            entropy_margin = 0.1 * math.log(model(images[:1].to(device)).shape[1])
    updated_batches = 0
    entropy_total = 0.0
    use_consistency = consistency_weight > 0
    for start in range(0, images.shape[0], parameter_batch_size):
        end = min(start + parameter_batch_size, images.shape[0])
        batch = images[start:end].to(device)
        logits = model(batch)
        entropy = entropy_from_logits(logits)
        entropy_total += float(entropy.detach().sum().item())
        keep = entropy < entropy_margin
        if not bool(keep.any()):
            continue
        selected_entropy = entropy[keep]
        weights = torch.exp(entropy_margin - selected_entropy.detach())
        loss = (selected_entropy * weights).mean()
        if use_consistency:
            previous = previous_logits[start:end].to(device)
            loss = loss + consistency_weight * _directional_consistency_loss(logits, previous)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        updated_batches += 1
    model.eval()
    return {
        "updated_batches": updated_batches,
        "mean_entropy": entropy_total / images.shape[0],
        "stats_momentum": stats_momentum,
    }


def load_candidate_pool(path: str, map_location: str | torch.device = "cpu") -> list[dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    candidates = payload.get("candidates", payload) if isinstance(payload, dict) else payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file must contain a non-empty list of BN state dictionaries")
    if not all(isinstance(candidate, dict) for candidate in candidates):
        raise ValueError("Each candidate must be a BN state dictionary")
    return candidates