"""Convolutional Neural Network (CNN) for MNIST classification.

This script loads the MNIST dataset, trains a simple CNN architecture
for 2-3 epochs, and evaluates its accuracy and loss on the test set.
"""

import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    """A lightweight Convolutional Neural Network for digit classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_data_loaders(
    batch_size: int = 64, data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """Download and prepare MNIST train and test data loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        predictions = output.argmax(dim=1, keepdim=False)
        correct += (predictions == target).sum().item()
        total += target.size(0)

        if batch_idx % 200 == 0 or batch_idx == len(loader):
            current_loss = running_loss / total
            current_acc = (correct / total) * 100.0
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx}/{len(loader)}] - "
                f"Loss: {current_loss:.4f} | Accuracy: {current_acc:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluate model performance on the test dataset."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item() * data.size(0)
            predictions = output.argmax(dim=1, keepdim=False)
            correct += (predictions == target).sum().item()
            total += target.size(0)

    avg_loss = test_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


def main() -> int:
    """Main execution function."""
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    epochs = 3
    batch_size = 64
    learning_rate = 0.001

    print("[*] Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)
    print(
        f"[*] Training samples: {len(train_loader.dataset)}, "
        f"Test samples: {len(test_loader.dataset)}"
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n[*] Starting training loop (3 epochs)...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            device=device,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        print(
            f"--> Epoch {epoch}/{epochs} Complete - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
        )

    print("[*] Evaluating on test set...")
    test_loss, test_accuracy = evaluate(
        model=model, device=device, loader=test_loader, criterion=criterion
    )

    print("=" * 50)
    print("FINAL TEST EVALUATION RESULTS:")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.2f}%")
    print("=" * 50)

    # Save model weights
    save_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[*] Model weights saved to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
