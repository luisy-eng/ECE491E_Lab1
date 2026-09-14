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
│   └── PyTorch tutorial notebooks
│
├── Task2/
│   └── Task2_CIFAR10.ipynb
│
└── Task3/
    └── Task3_CustomModel.ipynb
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

````markdown
# Task 3 — Modified Neural Network Architecture

**Status: Complete**

Task 3 replaces the neural network architecture used in the original PyTorch
tutorial with a deeper feedforward neural network designed for the CIFAR-10
input prepared in Task 2.

The implementation is available in the [`Task3/`](Task3/) directory.

## Input Features

Each CIFAR-10 image has dimensions:

```text
[3, 32, 32]
````

representing three RGB channels and a 32 × 32 pixel image.

Because the model is a fully connected feedforward neural network, each image
is flattened before entering the first linear layer:

```text
3 × 32 × 32 = 3072 input features
```

## Model Architecture

The modified model is implemented as a subclass of PyTorch's `nn.Module`.

The required architecture is:

```text
CIFAR-10 Image [3, 32, 32]
          │
          ▼
       Flatten
          │
          ▼
        3072
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

The two hidden layers contain 128 and 64 neurons respectively. ReLU activation
is applied after each hidden linear layer to introduce nonlinearity.

A dropout probability of `0.3` is applied after each hidden layer to reduce
overfitting during training.

The final layer contains a single neuron as required for the regression output
used in the project.

## Model Verification

The model was tested using a synthetic batch matching the dimensions of a
CIFAR-10 batch:

```text
Input shape:  [64, 3, 32, 32]
Output shape: [64, 1]
```

The successful forward pass confirms that the network accepts CIFAR-10 images
and produces one output value for each input image.

## Model Parameters

The completed network contains:

| Parameter            |       Value |
| -------------------- | ----------: |
| Input Features       |       3,072 |
| Hidden Layer 1       | 128 neurons |
| Hidden Layer 2       |  64 neurons |
| Output Neurons       |           1 |
| Dropout Rate         |         0.3 |
| Total Parameters     |     401,665 |
| Trainable Parameters |     401,665 |

All model parameters are trainable.

## Task 3 Result

Task 3 successfully produced the required deeper feedforward neural network:

```text
3072 → 128 → 64 → 1
```

with ReLU activation and dropout regularization after each hidden layer.

The model architecture is now prepared for the regression loss function and
training procedure introduced in Task 4.

````

Then change the top repository structure from:

```text
ECE491E_Lab1/
│
├── README.md
├── Task1/
│   └── ...
│
└── Task2/
    └── Task2_CIFAR10.ipynb
````

to:

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

Use your **actual Task 3 notebook filename** there if it's different.

Finally, change this line:

```markdown
Tasks 3 and 4 will be added as the project progresses.
```

to:

```markdown
Task 4 will be added as the project progresses.
```

And leave your existing Task 4 section as:

```markdown
# Task 4 — Modified Loss Function

**Status: Not Started**
```

That keeps the README appropriately concise: the **LaTeX report explains the theory in depth**, while the README tells someone looking at GitHub what you implemented, where to find it, and what results to expect.



---

# Report

The accompanying project report is written in LaTeX using the **NeurIPS 2022 template**.

The report documents the implementation, methodology, results, and visualizations produced throughout the four project tasks.

---

# Author

**Luis Hernandez**
Department of Electrical and Computer Engineering
University of Hawaiʻi at Mānoa
