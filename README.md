
# 🧠 CIFAR-10 Image Classification with CNNs (PyTorch)

This project trains and evaluates convolutional neural networks (CNNs) on the CIFAR-10 dataset using PyTorch. It explores the trade-off between accuracy and overfitting through dropout regularization and hyperparameter tuning.

## 📁 Project Structure

```
cifar10-cnn/
├── main.py
├── model.py
├── utils.py
├── README.md
└── requirements.txt
```

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the training:
```bash
python main.py
```

## 📦 Requirements

- Python 3.7+
- torch
- torchvision
- matplotlib
- numpy

## 📌 Features

- CNN with dropout regularization
- CIFAR-10 dataset from torchvision
- Accuracy and loss plotted across epochs
- Train/test evaluation with confusion matrix
