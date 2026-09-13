# ECE 491E Lab 1

## Exploring Neural Network Design: Modifying Architecture, Loss Functions, and Datasets

This repository contains the code and experimental results for **ECE 491E Lab 1**. The project begins with the official PyTorch *Learn the Basics* tutorial and then modifies the dataset, neural network architecture, and loss function.

The project consists of four tasks:

1. Complete the PyTorch tutorial and demonstrate the trained model's predictions.
2. Replace the original dataset with CIFAR-10, normalize the data, and create an 80/20 training/testing split.
3. Replace the tutorial model with a deeper feedforward neural network.
4. Replace the classification loss function with Mean Squared Error (MSE) loss.

---

## Repository Structure

```text
ECE491E_Lab1/
│
├── README.md
│
├── Task1/
│   ├── Task1_Tensors.ipynb
│   ├── Task1_Datasets&DataLoaders.ipynb
│   ├── Task1_transforms.ipynb
│   ├── Task1_buildmodel_tutorial.ipynb
│   ├── Task1_autogradqs_tutorial.ipynb
│   ├── Task1_optimization.ipynb
│   └── Task1_saveloadrun.ipynb
│
└── Task2/
    └── Task2_CIFAR10.ipynb
```

Tasks 3 and 4 will be added as the project progresses.

---

# Task 1 — PyTorch Tutorial

**Status: Complete**

Task 1 follows the official PyTorch *Learn the Basics* tutorial. The purpose of this task was to gain experience with the fundamental PyTorch workflow, including tensors, datasets and DataLoaders, transforms, neural network construction, automatic differentiation, optimization, and model persistence.

All Task 1 notebooks are located in the [`Task1/`](Task1/) directory.

### Notebooks

| Notebook                                                                       | Description                                                                   |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| [`Task1_Tensors.ipynb`](Task1/Task1_Tensors.ipynb)                             | Introduction to PyTorch tensors and tensor operations.                        |
| [`Task1_Datasets&DataLoaders.ipynb`](Task1/Task1_Datasets%26DataLoaders.ipynb) | Working with PyTorch `Dataset` and `DataLoader`.                              |
| [`Task1_transforms.ipynb`](Task1/Task1_transforms.ipynb)                       | Applying transformations to datasets.                                         |
| [`Task1_buildmodel_tutorial.ipynb`](Task1/Task1_buildmodel_tutorial.ipynb)     | Building a neural network using `nn.Module`.                                  |
| [`Task1_autogradqs_tutorial.ipynb`](Task1/Task1_autogradqs_tutorial.ipynb)     | Automatic differentiation using `torch.autograd`.                             |
| [`Task1_optimization.ipynb`](Task1/Task1_optimization.ipynb)                   | Training and evaluating the FashionMNIST model and demonstrating predictions. |
| [`Task1_saveloadrun.ipynb`](Task1/Task1_saveloadrun.ipynb)                     | Saving and loading PyTorch models.                                            |

### Model

The FashionMNIST tutorial model uses the following architecture:

```text
784 → 512 → 512 → 10
```

where the 784 inputs correspond to a flattened 28 × 28 grayscale image and the 10 outputs correspond to the FashionMNIST classes.

### Training and Results

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Dataset                 | FashionMNIST     |
| Batch Size              | 64               |
| Loss Function           | CrossEntropyLoss |
| Optimizer               | SGD              |
| Learning Rate           | 0.001            |
| Epochs                  | 10               |
| Final Test Accuracy     | **70.6%**        |
| Final Average Test Loss | **0.789814**     |

### Prediction Demonstration

A **Prediction Demonstration** was added to the optimization notebook to satisfy the project requirement to display ground-truth labels alongside model predictions.

The demonstration retrieves test images from the FashionMNIST DataLoader, performs inference using the trained model, and displays ten samples captioned with both their ground-truth and predicted classes.

---

# Task 2 — CIFAR-10 Dataset

**Status: Complete**

Task 2 replaces FashionMNIST with the **CIFAR-10** dataset and prepares the data for the modified neural network used in the subsequent tasks.

The implementation is available in:

[`Task2_CIFAR10.ipynb`](Task2/Task2_CIFAR10.ipynb)

### Dataset Preprocessing

CIFAR-10 contains **60,000 RGB images** with dimensions of 32 × 32 pixels distributed among 10 classes.

The Task 2 preprocessing pipeline performs the following operations:

1. Loads the CIFAR-10 dataset using `torchvision.datasets.CIFAR10`.
2. Converts the images to PyTorch tensors using `ToTensor()`.
3. Normalizes all three RGB channels using a mean and standard deviation of 0.5.
4. Combines the original CIFAR-10 training and testing portions.
5. Creates a new reproducible 80/20 training/testing split.
6. Creates PyTorch DataLoaders using a batch size of 64.
7. Verifies the image dimensions and visualizes samples from the processed dataset.

### Dataset Split

| Dataset   |    Samples | Percentage |
| --------- | ---------: | ---------: |
| Training  |     48,000 |        80% |
| Testing   |     12,000 |        20% |
| **Total** | **60,000** |   **100%** |

A fixed random seed of `42` is used when performing the split so that the same training and testing sets can be reproduced.

### Input Dimensions

Each CIFAR-10 image has the tensor dimensions:

```text
[3, 32, 32]
```

representing three RGB channels and a 32 × 32 pixel image.

When flattened for use in the feedforward neural network:

```text
3 × 32 × 32 = 3072 features
```

Therefore, **3,072 input features** will be used as the input size of the model developed in Task 3.

---

# Task 3 — Modified Neural Network

**Status: Not Started**

Task 3 will replace the tutorial network with a deeper feedforward neural network using the CIFAR-10 input produced in Task 2.

Required architecture:

```text
3072
  ↓
Linear(3072, 128)
  ↓
ReLU
  ↓
Dropout(0.3)
  ↓
Linear(128, 64)
  ↓
ReLU
  ↓
Dropout(0.3)
  ↓
Linear(64, 1)
  ↓
Regression Output
```

---

# Task 4 — Modified Loss Function

**Status: Not Started**

Task 4 will replace the classification loss function used in Task 1:

```python
nn.CrossEntropyLoss()
```

with the regression loss:

```python
nn.MSELoss()
```

The differences between the two loss functions and the implications of using a regression loss with CIFAR-10 will be discussed in the project report.

---

# Report

The accompanying project report is written in LaTeX using the **NeurIPS 2022 template**.

The report documents the implementation, methodology, results, and visualizations produced throughout the four project tasks.

---

# Author

**Luis Hernandez**
Department of Electrical and Computer Engineering
University of Hawaiʻi at Mānoa
