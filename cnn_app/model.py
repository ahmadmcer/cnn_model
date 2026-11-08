"""Residual Convolutional Neural Network architecture for MNIST classification.

Implements residual skip connections, batch normalization, dropout,
and feature map extraction hooks for Grad-CAM explainability.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """A 2D Residual block with identity/projection skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class ResCNN(nn.Module):
    """Deep Residual CNN designed for handwritten digit classification."""

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3) -> None:
        super().__init__()
        # Initial projection: 1x28x28 -> 32x28x28
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32 channels with residual refinement + downsampling -> 32x14x14
        self.res1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 32 -> 64 channels + residual refinement + downsampling -> 64x7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.res2 = ResidualBlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: 64 -> 128 channels (Target layer for Grad-CAM)
        self.target_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling and Dense Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes),
        )

        # Hook storage for Grad-CAM gradients and activations
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

    def activations_hook(self, grad: torch.Tensor) -> None:
        """Capture gradients of target convolutional layer."""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = self.init_conv(x)
        x = self.res1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.res2(x)
        x = self.pool2(x)

        x = self.target_conv(x)

        # Register hook if gradients are being tracked
        if x.requires_grad:
            self.activations = x
            x.register_hook(self.activations_hook)

        feat = self.global_pool(x)
        out = self.classifier(feat)
        return out

    def get_activations_gradient(self) -> Optional[torch.Tensor]:
        """Return the captured gradients from target layer."""
        return self.gradients

    def get_activations(self) -> Optional[torch.Tensor]:
        """Return the captured forward activations from target layer."""
        return self.activations
