"""Data pipeline module for MNIST dataset with augmentation and splits."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.1,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders with augmentation.

    Args:
        data_dir: Local path to download/load dataset.
        batch_size: Mini-batch size.
        val_split: Proportion of training data reserved for validation.
        random_seed: Seed for split reproducibility.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    full_train = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    # Deterministic train/validation split
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(random_seed)

    train_subset, val_subset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem
    )

    return train_loader, val_loader, test_loader
