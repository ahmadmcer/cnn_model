"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module."""

import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cnn_app.model import ResCNN


class GradCAMExplainer:
    """Computes Grad-CAM heatmaps for interpreting ResCNN decisions."""

    def __init__(self, model: ResCNN, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    def generate_heatmap(
        self, input_tensor: torch.Tensor, target_class: int
    ) -> np.ndarray:
        """Compute 2D Grad-CAM heatmap for a single image tensor (1, 1, 28, 28)."""
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        self.model.zero_grad()

        # Score for target class
        score = output[0, target_class]
        score.backward()

        # Retrieve gradients and activations
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations()

        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture Grad-CAM gradients or activations.")

        # Global average pool the gradients over spatial dimensions (H, W)
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weight activations by pooled gradients
        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Channel-wise mean of weighted activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize heatmap to [0, 1]
        heatmap_np = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np /= max_val

        return heatmap_np

    def save_explanations(
        self,
        samples: List[Tuple[torch.Tensor, int]],
        output_path: str = "outputs/gradcam_explain.png",
    ) -> str:
        """Generate and save side-by-side visualization of digits and Grad-CAM heatmaps."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        num_samples = len(samples)

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.2, 5))

        for idx, (img_tensor, true_label) in enumerate(samples):
            # Forward pass to get prediction
            with torch.no_grad():
                logits = self.model(img_tensor.unsqueeze(0).to(self.device))
                pred_label = logits.argmax(dim=1).item()
                confidence = F.softmax(logits, dim=1)[0, pred_label].item() * 100.0

            # Compute heatmap for predicted class
            heatmap = self.generate_heatmap(img_tensor.unsqueeze(0), pred_label)

            # Original image
            img_disp = img_tensor.squeeze().cpu().numpy()
            axes[0, idx].imshow(img_disp, cmap="gray")
            axes[0, idx].set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)")
            axes[0, idx].axis("off")

            # Heatmap overlay
            axes[1, idx].imshow(img_disp, cmap="gray")
            axes[1, idx].imshow(heatmap, cmap="jet", alpha=0.55, interpolation="bilinear")
            axes[1, idx].set_title("Grad-CAM")
            axes[1, idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path
