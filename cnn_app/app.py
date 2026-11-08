"""Comprehensive Deep Residual CNN Application CLI for MNIST.

Orchestrates training with cosine learning rate schedule, full multiclass
evaluation, confusion matrix generation, Grad-CAM explainability, and latency benchmarking.
"""

import argparse
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from cnn_app.benchmark import benchmark_inference, count_parameters, print_benchmark_report
from cnn_app.dataset import get_data_loaders
from cnn_app.gradcam import GradCAMExplainer
from cnn_app.metrics import EvaluationMetrics
from cnn_app.model import ResCNN


CHECKPOINT_PATH = "res_cnn_mnist.pth"


def train_model(
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    checkpoint_out: str = CHECKPOINT_PATH,
) -> ResCNN:
    """Train ResCNN model with validation tracking and learning rate decay."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Initializing training pipeline on device: {device}")
    train_loader, val_loader, _ = get_data_loaders(batch_size=batch_size)

    model = ResCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))

    metrics_calc = EvaluationMetrics()
    best_val_acc = 0.0

    print(f"[*] Training configuration: {epochs} Epochs, Batch size {batch_size}, Initial LR {learning_rate}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

            if batch_idx % 250 == 0 or batch_idx == len(train_loader):
                batch_loss = running_loss / total
                batch_acc = (correct / total) * 100.0
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch [{epoch}/{epochs}] Step [{batch_idx:>3}/{len(train_loader)}] "
                    f"- Loss: {batch_loss:.4f} | Train Acc: {batch_acc:.2f}% | LR: {current_lr:.6f}"
                )

        # Validation check
        val_summary, _, _ = metrics_calc.evaluate_model(model, val_loader, device)
        print(
            f"--> Epoch [{epoch}/{epochs}] Result: "
            f"Val Loss: {val_summary['loss']:.4f} | Val Accuracy: {val_summary['accuracy']:.2f}%"
        )

        if val_summary["accuracy"] > best_val_acc:
            best_val_acc = val_summary["accuracy"]
            torch.save(model.state_dict(), checkpoint_out)
            print(f"    [+] New best model saved to {checkpoint_out} (Acc: {best_val_acc:.2f}%)")

        print("-" * 65)

    print(f"[*] Training complete. Best Validation Accuracy: {best_val_acc:.2f}%\n")
    return model


def evaluate_model(checkpoint_path: str = CHECKPOINT_PATH) -> None:
    """Run full test evaluation, print report, and generate confusion matrix plot."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders()

    model = ResCNN().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[*] Loaded model weights from {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")

    metrics_calc = EvaluationMetrics()
    print("[*] Running comprehensive test evaluation on 10,000 samples...")
    summary, cm, class_reports = metrics_calc.evaluate_model(model, test_loader, device)

    # Display ASCII Classification report table
    metrics_calc.print_classification_report(summary, class_reports)

    # Save visual confusion matrix
    cm_path = "outputs/confusion_matrix.png"
    metrics_calc.save_confusion_matrix_plot(cm, output_path=cm_path)
    print(f"[*] Confusion matrix heatmap saved to: {cm_path}")


def explain_predictions(checkpoint_path: str = CHECKPOINT_PATH) -> None:
    """Generate Grad-CAM heatmaps for sample test digits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(batch_size=1)

    model = ResCNN().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with current weights.")

    explainer = GradCAMExplainer(model, device)

    # Select 6 distinct digit samples
    samples = []
    seen_labels = set()
    for img, label in test_loader:
        lbl = label.item()
        if lbl not in seen_labels:
            seen_labels.add(lbl)
            samples.append((img.squeeze(0), lbl))
        if len(samples) >= 6:
            break

    print("[*] Generating Grad-CAM visual explanations...")
    out_path = "outputs/gradcam_explain.png"
    explainer.save_explanations(samples, output_path=out_path)
    print(f"[*] Grad-CAM visualization saved to: {out_path}")


def benchmark_model(checkpoint_path: str = CHECKPOINT_PATH) -> None:
    """Run model parameter profiling and latency benchmarking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResCNN().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    param_info = count_parameters(model)
    print("[*] Benchmarking inference latency (500 iterations)...")
    bench_results = benchmark_inference(model, device)
    print_benchmark_report(bench_results, param_info)


def run_all_pipeline() -> int:
    """End-to-end full execution pipeline demonstrating all capabilities."""
    print("\n" + "=" * 65)
    print("      DEEP RESIDUAL CNN - FULL DEMO & VERIFICATION PIPELINE")
    print("=" * 65)

    # 1. Train model
    train_model(epochs=2, batch_size=64, learning_rate=0.001)

    # 2. Comprehensive evaluation & confusion matrix
    evaluate_model(CHECKPOINT_PATH)

    # 3. Grad-CAM visual interpretability
    explain_predictions(CHECKPOINT_PATH)

    # 4. Latency and parameter benchmark
    benchmark_model(CHECKPOINT_PATH)

    print("=" * 65)
    print("[SUCCESS] ALL MODULES EXECUTED SUCCESSFULLY WITHOUT ERRORS.")
    print("=" * 65 + "\n")
    return 0


def main() -> int:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Deep Residual CNN Application for MNIST."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run-all",
        choices=["train", "eval", "explain", "benchmark", "run-all"],
        help="Pipeline command to execute (default: run-all).",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate.")

    args = parser.parse_args()

    if args.command == "train":
        train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
    elif args.command == "eval":
        evaluate_model()
    elif args.command == "explain":
        explain_predictions()
    elif args.command == "benchmark":
        benchmark_model()
    else:
        return run_all_pipeline()

    return 0


if __name__ == "__main__":
    sys.exit(main())
