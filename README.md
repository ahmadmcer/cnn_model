app.py
# Deep Residual CNN Application for MNIST

A modular, production-grade Deep Residual Convolutional Neural Network (ResCNN) application implemented in PyTorch for classifying handwritten digits using the MNIST dataset.

---

## Architecture & Features

### 1. Residual Convolutional Network (`cnn_app/model.py`)
- **Residual Blocks**: Residual connections with dual 3x3 convolutions, batch normalization, and ReLU activations to mitigate vanishing gradients.
- **Hierarchical Feature Maps**: 32 -> 64 -> 128 channel progressions.
- **Global Average Pooling**: Minimizes parameter counts while preventing overfitting.
- **Grad-CAM Saliency Hooks**: Built-in gradient and activation hooks for visual interpretability.

### 2. Explainability & Analysis (`cnn_app/gradcam.py`)
- Computes Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the exact spatial regions influencing prediction decisions.
- Saves multi-sample comparative heatmaps to `outputs/gradcam_explain.png`.

### 3. Detailed Multiclass Metrics (`cnn_app/metrics.py`)
- Full confusion matrix calculation.
- Per-class Precision, Recall, and F1-Score reporting.
- High-resolution annotated confusion matrix heatmap saved to `outputs/confusion_matrix.png`.

### 4. Performance Benchmarking (`cnn_app/benchmark.py`)
- Hardware profiling: Parameter count analysis.
- Latency profiling across 500 iterations (Mean, Median P50, P95, P99).
- Inference throughput calculation (frames / samples per second).

---

## Installation & Setup

```powershell
pip install -r requirements.txt
```

---

## CLI Usage

### Run End-to-End Pipeline (Training, Evaluation, Grad-CAM, & Benchmark)
```powershell
python -m cnn_app.app run-all
```

### Standalone Subcommands
- **Train Model**:
  ```powershell
  python -m cnn_app.app train --epochs 2 --batch-size 64 --lr 0.001
  ```
- **Evaluate on Test Set & Save Confusion Matrix**:
  ```powershell
  python -m cnn_app.app eval
  ```
- **Generate Grad-CAM Heatmaps**:
  ```powershell
  python -m cnn_app.app explain
  ```
- **Benchmark Latency & Throughput**:
  ```powershell
  python -m cnn_app.app benchmark
  ```

---

## License

This project is licensed under the MIT License.
