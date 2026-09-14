# ECE 491E Lab 1

## Exploring Neural Network Design: Modifying Architecture, Loss Functions, and Datasets

This repository contains the code and experimental results for **ECE 491E Lab 1**. The project begins with the official PyTorch *Learn the Basics* tutorial and then explores modifications to the dataset, neural network architecture, and loss function.

### Project Tasks

1. **Task 1:** Complete the PyTorch tutorial and demonstrate the trained model's predictions.
2. **Task 2:** Replace FashionMNIST with CIFAR-10, normalize the dataset, and create an 80/20 training/testing split.
3. **Task 3:** Replace the tutorial model with a deeper feedforward neural network.
4. **Task 4:** Replace the classification loss function with Mean Squared Error (MSE) loss.

---

## Repository Structure

```text
ECE491E_Lab1/
│
├── README.md
│
├── Task1/
│   └── PyTorch tutorial notebooks
│
├── Task2/
│   └── Task2_CIFAR10.ipynb
│
└── Task3/
    └── Task3_CustomModel.ipynb
```

Tasks 1–3 are complete. Task 4 will be added as the project progresses.

---

# Task 1 — PyTorch Tutorial

**Status: Complete**

Task 1 follows the official PyTorch *Learn the Basics* tutorial. The purpose of this task was to gain experience with the fundamental PyTorch workflow, including tensors, datasets and DataLoaders, transforms, neural network construction, automatic differentiation, optimization, and model persistence.

All Task 1 notebooks are located in the [`Task1/`](Task1/) directory.

### Notebooks

| Notebook | Description |
|---|---|
| [`Task1_Tensors.ipynb`](Task1/Task1_Tensors.ipynb) | Introduction to PyTorch tensors and tensor operations. |
| [`Task1_Datasets&DataLoaders.ipynb`](Task1/Task1_Datasets%26DataLoaders.ipynb) | Working with PyTorch `Dataset` and `DataLoader`. |
| [`Task1_transforms.ipynb`](Task1/Task1_transforms.ipynb) | Applying transformations to datasets. |
| [`Task1_buildmodel_tutorial.ipynb`](Task1/Task1_buildmodel_tutorial.ipynb) | Building a neural network using `nn.Module`. |
| [`Task1_autogradqs_tutorial.ipynb`](Task1/Task1_autogradqs_tutorial.ipynb) | Automatic differentiation using `torch.autograd`. |
| [`Task1_optimization.ipynb`](Task1/Task1_optimization.ipynb) | Training and evaluating the FashionMNIST model and demonstrating predictions. |
| [`Task1_saveloadrun.ipynb`](Task1/Task1_saveloadrun.ipynb) | Saving and loading PyTorch models. |

### Model

The FashionMNIST tutorial model uses the architecture:

```text
784 → 512 → 512 → 10
```

The 784 inputs correspond to a flattened 28 × 28 grayscale image, while the 10 outputs correspond to the FashionMNIST classes.

### Training and Results

| Parameter | Value |
|---|---|
| Dataset | FashionMNIST |
| Batch Size | 64 |
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |
| Learning Rate | 0.001 |
| Epochs | 10 |
| Final Test Accuracy | **70.6%** |
| Final Average Test Loss | **0.789814** |

### Prediction Demonstration

A **Prediction Demonstration** was added to the optimization notebook to satisfy the project requirement to display ground-truth labels alongside model predictions.

The demonstration retrieves test images from the FashionMNIST DataLoader, performs inference using the trained model, and displays ten samples captioned with both their ground-truth and predicted classes.

---

# Task 2 — CIFAR-10 Dataset

**Status: Complete**

Task 2 replaces FashionMNIST with **CIFAR-10** and prepares the dataset for the modified neural network.

The implementation is available in [`Task2_CIFAR10.ipynb`](Task2/Task2_CIFAR10.ipynb).

### Dataset Preprocessing

CIFAR-10 contains **60,000 RGB images** with dimensions of 32 × 32 pixels distributed among 10 classes.

The preprocessing pipeline:

1. Loads CIFAR-10 using `torchvision.datasets.CIFAR10`.
2. Converts the images to PyTorch tensors using `ToTensor()`.
3. Normalizes all three RGB channels using a mean and standard deviation of 0.5.
4. Combines the original CIFAR-10 training and testing portions.
5. Creates a reproducible 80/20 training/testing split using a random seed of `42`.
6. Creates PyTorch DataLoaders with a batch size of 64.
7. Verifies the image dimensions and visualizes samples from the processed dataset.

### Dataset Split

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 48,000 | 80% |
| Testing | 12,000 | 20% |
| **Total** | **60,000** | **100%** |

### Input Dimensions

Each CIFAR-10 image has dimensions:

```text
[3, 32, 32]
```

When flattened for use in a feedforward neural network:

```text
3 × 32 × 32 = 3072 features
```

Therefore, the model developed in Task 3 requires **3,072 input features**.

---

# Task 3 — Modified Neural Network Architecture

**Status: Complete**

Task 3 replaces the neural network architecture used in the PyTorch tutorial with a deeper feedforward neural network designed for the CIFAR-10 input prepared in Task 2.

The implementation is available in the [`Task3/`](Task3/) directory.

### Model Architecture

The model first flattens each `[3, 32, 32]` CIFAR-10 image into 3,072 input features.

The resulting architecture is:

```text
CIFAR-10 Image [3, 32, 32]
          │
          ▼
       Flatten
          │
          ▼
 Linear(3072, 128)
          │
          ▼
         ReLU
          │
          ▼
   Dropout(p=0.3)
          │
          ▼
  Linear(128, 64)
          │
          ▼
         ReLU
          │
          ▼
   Dropout(p=0.3)
          │
          ▼
    Linear(64, 1)
          │
          ▼
  Regression Output
```

The two hidden layers contain 128 and 64 neurons. ReLU activation introduces nonlinearity after each hidden linear layer, while dropout with a probability of `0.3` provides regularization.

The final layer contains a single output neuron as required for the regression output.

### Model Verification

A synthetic batch matching the dimensions of CIFAR-10 data was passed through the network:

```text
Input shape:  [64, 3, 32, 32]
Output shape: [64, 1]
```

The successful forward pass confirms that the network accepts CIFAR-10 images and produces one output value for each input sample.

### Model Parameters

| Parameter | Value |
|---|---:|
| Input Features | 3,072 |
| Hidden Layer 1 | 128 neurons |
| Hidden Layer 2 | 64 neurons |
| Output Neurons | 1 |
| Dropout Rate | 0.3 |
| Total Parameters | 401,665 |
| Trainable Parameters | 401,665 |

The completed Task 3 architecture can be summarized as:

```text
3072 → 128 → 64 → 1
```

The model is now prepared for the regression loss function and training procedure introduced in Task 4.

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

The completed model will then be trained and evaluated using the CIFAR-10 dataset prepared in Task 2 and the architecture developed in Task 3.

---

# Report

The accompanying project report is written in **LaTeX using the NeurIPS 2022 template**.

The report documents the methodology, implementation, experimental results, and visualizations produced throughout the four project tasks.

---

# Author

**Luis Hernandez**  
Department of Electrical and Computer Engineering  
University of Hawaiʻi at Mānoa
