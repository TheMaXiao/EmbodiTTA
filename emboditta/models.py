from pathlib import Path
import sys

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def load_trusted_model(
    checkpoint: str,
    device: torch.device | str,
    module_root: str | None = None,
) -> nn.Module:
    if module_root:
        sys.path.insert(0, str(Path(module_root).resolve()))
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError("The supplied checkpoint must contain a serialized model")
    return model.to(device).eval()


def load_resnet50(
    device: torch.device | str,
    checkpoint: str | None = None,
    use_torchvision_weights: bool = True,
) -> nn.Module:
    weights = ResNet50_Weights.DEFAULT if checkpoint is None and use_torchvision_weights else None
    model = resnet50(weights=weights)
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, nn.Module):
            model = payload
        else:
            state = payload
            if isinstance(payload, dict):
                for key in ("state_dict", "model", "model_state_dict"):
                    if key in payload and isinstance(payload[key], dict):
                        state = payload[key]
                        break
            if not isinstance(state, dict):
                raise ValueError("Checkpoint must contain a model or state dictionary")
            state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state)
    return model.to(device).eval()