# ECE 491E Lab 1

## Exploring Neural Network Design: Modifying Architecture, Loss Functions, and Datasets

This repository contains the code, experiments, and results for **ECE 491E Lab 1**. The project explores the fundamentals of neural network development using PyTorch and investigates how changing the dataset, model architecture, and loss function affects the neural network workflow.

The project consists of four tasks:

1. Complete the PyTorch *Learn the Basics* tutorial and demonstrate the trained model's predictions.
2. Replace the original dataset with CIFAR-10 and preprocess the data.
3. Replace the tutorial model with a deeper feedforward neural network.
4. Replace the classification loss function with Mean Squared Error (MSE) loss.

---

## Repository Structure

```text
ECE491E_Lab1/
│
├── README.md
│
└── Task1/
    ├── Task1_Tensors.ipynb
    ├── Task1_Datasets&DataLoaders.ipynb
    ├── Task1_transforms.ipynb
    ├── Task1_buildmodel_tutorial.ipynb
    ├── Task1_autogradqs_tutorial.ipynb
    ├── Task1_optimization.ipynb
    └── Task1_saveloadrun.ipynb
```

Additional directories will be added as Tasks 2–4 are completed.

---

# Task 1 — PyTorch Tutorial

**Status: Complete**

Task 1 follows the official PyTorch *Learn the Basics* tutorial. The purpose of this task was to gain experience with the fundamental components of the PyTorch workflow rather than to design a new neural network.

The tutorials cover tensors, datasets and DataLoaders, transforms, neural network construction, automatic differentiation, optimization, and saving and loading models.

All Task 1 notebooks are organized in the [`Task1/`](Task1/) directory.

## Task 1 Notebooks

| Notebook                                                                       | Description                                                                                  |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| [`Task1_Tensors.ipynb`](Task1/Task1_Tensors.ipynb)                             | Introduction to PyTorch tensors and tensor operations.                                       |
| [`Task1_Datasets&DataLoaders.ipynb`](Task1/Task1_Datasets%26DataLoaders.ipynb) | Working with PyTorch `Dataset` and `DataLoader`.                                             |
| [`Task1_transforms.ipynb`](Task1/Task1_transforms.ipynb)                       | Applying transformations to datasets.                                                        |
| [`Task1_buildmodel_tutorial.ipynb`](Task1/Task1_buildmodel_tutorial.ipynb)     | Building a neural network using `nn.Module`.                                                 |
| [`Task1_autogradqs_tutorial.ipynb`](Task1/Task1_autogradqs_tutorial.ipynb)     | Automatic differentiation using `torch.autograd`.                                            |
| [`Task1_optimization.ipynb`](Task1/Task1_optimization.ipynb)                   | Training and evaluating the FashionMNIST neural network and demonstrating model predictions. |
| [`Task1_saveloadrun.ipynb`](Task1/Task1_saveloadrun.ipynb)                     | Saving and loading PyTorch models.                                                           |

---

## Task 1 Model

The PyTorch tutorial uses the **FashionMNIST** dataset. Each FashionMNIST image is a 28 × 28 grayscale image belonging to one of ten clothing classes.

The tutorial neural network uses the following architecture:

```text
28 × 28 Image
     │
     ▼
   Flatten
     │
     ▼
Linear(784, 512)
     │
     ▼
    ReLU
     │
     ▼
Linear(512, 512)
     │
     ▼
    ReLU
     │
     ▼
Linear(512, 10)
     │
     ▼
10 FashionMNIST Classes
```

---

## Training Configuration

The optimization portion of the tutorial was run with the following configuration:

| Parameter     | Value            |
| ------------- | ---------------- |
| Dataset       | FashionMNIST     |
| Batch Size    | 64               |
| Loss Function | CrossEntropyLoss |
| Optimizer     | SGD              |
| Learning Rate | 0.001            |
| Epochs        | 10               |

---

## Task 1 Results

After training for 10 epochs, the model achieved:

* **Test Accuracy:** 70.6%
* **Average Test Loss:** 0.789814

These results were obtained from the completed optimization tutorial.

---

## Prediction Demonstration

The project instructions additionally require a figure displaying the model's **ground-truth labels and predictions**.

To satisfy this requirement, a **Prediction Demonstration** section was added to [`Task1_optimization.ipynb`](Task1/Task1_optimization.ipynb).

This section was an addition to the optimization tutorial and performs the following steps:

1. Retrieves a batch of images and ground-truth labels from the FashionMNIST test DataLoader.
2. Places the trained model in evaluation mode.
3. Performs inference without calculating gradients.
4. Determines the predicted class from the model output.
5. Displays ten test images.
6. Captions each image with its **Ground Truth** and **Prediction**.

This visualization provides a qualitative demonstration of the trained model's performance in addition to the numerical test accuracy.

---

# Remaining Tasks

## Task 2 — Customize Dataset

**Status: Not Started**

The FashionMNIST dataset will be replaced with **CIFAR-10**. The dataset will be normalized and divided into an **80% training / 20% testing** split.

---

## Task 3 — Customize Model

**Status: Not Started**

The tutorial model will be replaced with the required deeper feedforward neural network:

```text
Input
  │
  ▼
Linear(Input Size, 128)
  │
  ▼
ReLU
  │
  ▼
Dropout(0.3)
  │
  ▼
Linear(128, 64)
  │
  ▼
ReLU
  │
  ▼
Dropout(0.3)
  │
  ▼
Linear(64, 1)
  │
  ▼
Regression Output
```

---

## Task 4 — Customize Loss Function

**Status: Not Started**

The classification loss function used in Task 1:

```python
nn.CrossEntropyLoss()
```

will be replaced with:

```python
nn.MSELoss()
```

The differences between Cross-Entropy Loss and Mean Squared Error Loss will be discussed in the accompanying report.

---

# Report

The project report is written in **LaTeX using the NeurIPS 2022 template**.

The report documents the implementation and results of each task, including:

* PyTorch tutorial implementation
* FashionMNIST training and evaluation
* Ground-truth and prediction visualization
* CIFAR-10 preprocessing
* Modified neural network architecture
* Loss-function modifications
* Experimental results and visualizations

---

# Author

**Luis Hernandez**
Department of Electrical and Computer Engineering
University of Hawaiʻi at Mānoa
