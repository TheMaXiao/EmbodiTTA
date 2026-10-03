from pathlib import Path

from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms


IMAGENETC_CORRUPTIONS = (
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


def imagenet_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def cifar10_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.247, 0.243, 0.261),
            ),
        ]
    )


def load_imagenetc_stream(
    root: str,
    severity: int = 5,
    corruptions: tuple[str, ...] = IMAGENETC_CORRUPTIONS,
    samples_per_corruption: int | None = None,
):
    if severity not in range(1, 6):
        raise ValueError("ImageNet-C severity must be between 1 and 5")
    transform = imagenet_transform()
    domain_datasets = []
    for corruption in corruptions:
        domain_path = Path(root) / corruption / str(severity)
        if not domain_path.is_dir():
            raise FileNotFoundError(f"ImageNet-C domain not found: {domain_path}")
        dataset = datasets.ImageFolder(domain_path, transform=transform)
        if samples_per_corruption is not None:
            count = min(samples_per_corruption, len(dataset))
            dataset = Subset(dataset, range(count))
        domain_datasets.append(dataset)
    if not domain_datasets:
        raise ValueError("At least one ImageNet-C corruption is required")
    return ConcatDataset(domain_datasets)


def load_clean_imagenet(root: str):
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {root_path}")
    return datasets.ImageFolder(root_path, transform=imagenet_transform())