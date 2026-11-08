"""Evaluation metrics, confusion matrix, and reporting module."""

import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


class EvaluationMetrics:
    """Computes comprehensive classification metrics and confusion matrices."""

    def __init__(self, num_classes: int = 10) -> None:
        self.num_classes = num_classes

    def compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate NxN confusion matrix where rows=true, cols=pred."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm

    def evaluate_model(
        self, model: torch.nn.Module, loader: DataLoader, device: torch.device
    ) -> Tuple[Dict[str, float], np.ndarray, List[Dict[str, float]]]:
        """Run full evaluation on a data loader and compute detailed metrics."""
        model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item() * data.size(0)

                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        cm = self.compute_confusion_matrix(y_true, y_pred)
        total_samples = len(y_true)
        overall_accuracy = (np.diag(cm).sum() / total_samples) * 100.0
        avg_loss = total_loss / total_samples

        # Per-class metrics
        class_reports: List[Dict[str, float]] = []
        precisions: List[float] = []
        recalls: List[float] = []
        f1_scores: List[float] = []

        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            support = int(cm[c, :].sum())

            precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            class_reports.append(
                {
                    "class": c,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )

        summary = {
            "loss": avg_loss,
            "accuracy": overall_accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1_scores)),
        }

        return summary, cm, class_reports

    def print_classification_report(
        self, summary: Dict[str, float], class_reports: List[Dict[str, float]]
    ) -> None:
        """Render a formatted classification report table in terminal."""
        print("\n" + "=" * 65)
        print("          COMPREHENSIVE CLASSIFICATION REPORT")
        print("=" * 65)
        print(f"{'Class':>6} | {'Precision (%)':>14} | {'Recall (%)':>12} | {'F1-Score (%)':>14} | {'Support':>8}")
        print("-" * 65)
        for r in class_reports:
            print(
                f"  {r['class']:>4} | {r['precision']:>13.2f}% | {r['recall']:>11.2f}% | "
                f"{r['f1']:>13.2f}% | {int(r['support']):>8}"
            )
        print("-" * 65)
        print(
            f"{'Macro Avg':>6} | {summary['macro_precision']:>13.2f}% | "
            f"{summary['macro_recall']:>11.2f}% | {summary['macro_f1']:>13.2f}% |"
        )
        print(f"{'Overall Acc':>6} | {summary['accuracy']:>13.2f}% | Loss: {summary['loss']:.4f}")
        print("=" * 65 + "\n")

    def save_confusion_matrix_plot(
        self, cm: np.ndarray, output_path: str = "outputs/confusion_matrix.png"
    ) -> str:
        """Render and save annotated confusion matrix heatmap as PNG."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                val = cm[i, j]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

        fig.colorbar(cax)
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels([str(i) for i in range(self.num_classes)])
        ax.set_yticklabels([str(i) for i in range(self.num_classes)])
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("ResCNN MNIST Confusion Matrix", fontsize=14, pad=15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=180)
        plt.close(fig)
        return output_path
