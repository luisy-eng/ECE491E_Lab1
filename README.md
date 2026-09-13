# ECE 491E Lab 1

## Exploring Neural Network Design: Modifying Architecture, Loss Functions, and Datasets

This repository contains the code and experimental results for **ECE 491E Lab 1**. The project explores the PyTorch neural network workflow and then modifies the dataset, neural network architecture, and loss function.

The project is divided into four tasks:

1. Complete the PyTorch *Learn the Basics* tutorials and demonstrate model predictions.
2. Replace the original dataset with CIFAR-10 and preprocess the data.
3. Modify the neural network to use a deeper feedforward architecture.
4. Replace the classification loss function with Mean Squared Error (MSE) loss.

---

## Task 1 — PyTorch Tutorial

**Status: Complete**

Task 1 follows the official PyTorch *Learn the Basics* tutorial. The purpose of this task was to become familiar with the basic PyTorch workflow, including tensors, datasets, transforms, neural network construction, automatic differentiation, optimization, and model persistence.

### Task 1 Notebooks

| Notebook                            | Description                                                       |
| ----------------------------------- | ----------------------------------------------------------------- |
| `Task1_tensorqs_tutorial.ipynb`     | Introduction to PyTorch tensors and tensor operations.            |
| `Task1_data_tutorial.ipynb`         | Working with datasets and DataLoaders.                            |
| `Task1_transforms_tutorial.ipynb`   | Applying transformations to datasets.                             |
| `Task1_buildmodel_tutorial.ipynb`   | Building a neural network using `nn.Module`.                      |
| `Task1_autogradqs_tutorial.ipynb`   | Introduction to automatic differentiation using `torch.autograd`. |
| `Task1_optimization_tutorial.ipynb` | Training and evaluating the FashionMNIST neural network.          |
| `Task1_saveloadrun_tutorial.ipynb`  | Saving and loading PyTorch models.                                |

> **Note:** Notebook filenames in the repository should be used as the authoritative names if they differ slightly from those shown above.

### Task 1 Model

The tutorial uses the FashionMNIST dataset and a fully connected neural network with the following architecture:

```text
28 × 28 Image
     ↓
   Flatten
     ↓
Linear(784, 512)
     ↓
    ReLU
     ↓
Linear(512, 512)
     ↓
    ReLU
     ↓
Linear(512, 10)
     ↓
10 FashionMNIST Classes
```

### Training Configuration

The optimization tutorial was run using:

| Parameter     |            Value |
| ------------- | ---------------: |
| Dataset       |     FashionMNIST |
| Batch Size    |               64 |
| Loss Function | CrossEntropyLoss |
| Optimizer     |              SGD |
| Learning Rate |            0.001 |
| Epochs        |               10 |

### Results

After 10 epochs, the trained model achieved:

* **Test Accuracy:** 70.6%
* **Average Test Loss:** 0.789814

### Prediction Demonstration

An additional **Prediction Demonstration** was added to the optimization notebook to satisfy the project requirement to display model predictions alongside their corresponding ground-truth labels.

The demonstration:

1. Obtains test images and their ground-truth labels from the FashionMNIST test DataLoader.
2. Passes the images through the trained model.
3. Selects the class with the highest model output as the prediction.
4. Displays ten test images.
5. Labels each image with both its **Ground Truth** and **Prediction**.

The resulting figure is also used in the project report.

---

## Task 2 — CIFAR-10 Dataset

**Status: Not Started**

Task 2 will replace the original dataset with CIFAR-10.

The dataset will be:

* Converted to tensors.
* Normalized.
* Split into 80% training and 20% testing sets.

Implementation and experimental details will be added after completion.

---

## Task 3 — Modified Neural Network

**Status: Not Started**

Task 3 will replace the tutorial model with a deeper feedforward neural network.

Required architecture:

```text
Input
  ↓
Linear(Input Size, 128)
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

Implementation and experimental results will be added after completion.

---

## Task 4 — Modified Loss Function

**Status: Not Started**

Task 4 will replace the classification loss function used in Task 1:

```python
nn.CrossEntropyLoss()
```

with the regression loss:

```python
nn.MSELoss()
```

The differences between the two loss functions and their effects on the model will be discussed in the accompanying report.

---

## Project Report

The accompanying report is written in LaTeX using the **NeurIPS 2022 template**.

The report discusses:

* Task 1 PyTorch tutorial results
* CIFAR-10 preprocessing
* Modified neural network architecture
* Cross-Entropy Loss and Mean Squared Error Loss
* Training and evaluation results
* Model prediction visualizations

---

## Repository

Project repository:

`https://github.com/luisy-eng/ECE491E_Lab1`

---

## Author

**Luis Hernandez**
Department of Electrical and Computer Engineering
University of Hawaiʻi at Mānoa
