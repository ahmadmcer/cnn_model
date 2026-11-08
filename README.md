# Simple Convolutional Neural Network (CNN) for MNIST

A lightweight, modular, and efficient Convolutional Neural Network (CNN) implemented in PyTorch for classifying handwritten digits using the MNIST dataset.

---

## Architecture Overview

The network processes 28x28 grayscale digit images through the following pipeline:

1. **Convolutional Block 1**:
   - `Conv2d(1, 32, kernel_size=3, padding=1)`
   - `BatchNorm2d(32)`
   - `ReLU()`
   - `MaxPool2d(kernel_size=2, stride=2)` -> Output shape: `(32, 14, 14)`

2. **Convolutional Block 2**:
   - `Conv2d(32, 64, kernel_size=3, padding=1)`
   - `BatchNorm2d(64)`
   - `ReLU()`
   - `MaxPool2d(kernel_size=2, stride=2)` -> Output shape: `(64, 7, 7)`

3. **Classification Head**:
   - `Flatten()`
   - `Linear(64 * 7 * 7, 128)` + `ReLU()`
   - `Dropout(p=0.25)` (regularization)
   - `Linear(128, 10)` (logits for digits 0-9)

---

## Setup & Installation

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git

### 2. Environment Setup
Create a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

---

## Usage

Run the training and evaluation script:

```powershell
python main.py
```

### What `main.py` Does:
1. Automatically downloads and normalizes the MNIST dataset to `./data/`.
2. Trains the `SimpleCNN` model for 3 epochs using the Adam optimizer and CrossEntropyLoss.
3. Evaluates test set performance and reports loss and accuracy metrics.
4. Saves the trained weights to `mnist_cnn.pth`.

---

## Expected Results

After 3 epochs of training:
- **Test Loss**: ~0.0300 - 0.0400
- **Test Accuracy**: > 98.8%

---

## License

This project is licensed under the MIT License.
