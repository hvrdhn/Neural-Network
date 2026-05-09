# Neural-Network from Scratch — MNIST Classifier

A fully connected neural network built from scratch using only NumPy, trained on the MNIST handwritten digit dataset. Achieves ~97% test accuracy.

---

## Architecture

```
Input (784) → Hidden (128) → Hidden (64) → Output (10)
```

| Layer | Size | Activation |
|-------|------|------------|
| Input | 784 (28×28 pixels) | Mone |
| Hidden 1 | 128 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 10 (digits 0–9) | Softmax |

---

## Requirements

```
numpy
torch
torchvision
matplotlib
```

Install with:
```bash
pip install numpy torch torchvision matplotlib
```

---

## Usage

```bash
python NN3.py
```

The script will:
1. Download MNIST automatically (into `./data/`)
2. Train for 20 epochs using mini-batch SGD
3. Print accuracy after each epoch

---

## Key Optimizations

| What | Naive | Optimized |
|------|-------|-----------|
| Backprop | Python loop over each sample | Vectorized over full batch — one matrix op |
| Evaluation | 10,000 separate `feedforward()` calls | Single matrix multiply |
| Data storage | List of (784, 1) arrays | Pre-stacked (784, 60000) matrix |
| Weight update | Creates new arrays each step | In-place `-=` update |

The big-O complexity class is unchanged — the constant factor drops ~10–20x because NumPy hands off to optimized BLAS (OpenBLAS/MKL) routines instead of running Python loops.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Mini-batch size | 32 |
| Learning rate | 0.01 |
| Weight init | He initialization (`√(2/n)`) |
| Bias init | Small random (`0.1 × N(0,1)`) |

---

## File Structure

```
.
├── NN3.py          # Main script
├── README.md       # This file
└── data/           # MNIST dataset (auto-downloaded)
    └── MNIST/
```

---

## Reference

Based on Michael Nielsen's *Neural Networks and Deep Learning* (neuralnetworksanddeeplearning.com), with vectorized mini-batch backprop and He weight initialization added for performance.
