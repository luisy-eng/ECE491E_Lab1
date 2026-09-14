# ECE 491E Lab 1

## Exploring Neural Network Design: Modifying Architecture, Loss Functions, and Datasets

This repository contains the implementation and experimental results for **ECE 491E Lab 1**.

The project begins with the official PyTorch *Learn the Basics* tutorial and then modifies the dataset, neural network architecture, and loss function. The completed project demonstrates the transition from the original FashionMNIST classification example to a custom feedforward neural network using CIFAR-10 and Mean Squared Error (MSE) loss.

---

## Project Tasks

| Task | Description | Status |
|---|---|---|
| **Task 1** | Complete the PyTorch tutorial and demonstrate model predictions | Complete |
| **Task 2** | Replace FashionMNIST with CIFAR-10 and create an 80/20 data split | Complete |
| **Task 3** | Replace the tutorial model with a deeper feedforward neural network | Complete |
| **Task 4** | Replace Cross-Entropy Loss with Mean Squared Error Loss and train the modified model | Complete |

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
├── Task2/
│   └── Task2_CIFAR10.ipynb
│
├── Task3/
│   └── Task3_CustomModel.ipynb
│
└── Task 4/
    └── Task4_MSELoss.ipynb
```

---

# Task 1 — PyTorch Tutorial

**Status: Complete**

Task 1 follows the official PyTorch *Learn the Basics* tutorial. The purpose of this task was to gain experience with the fundamental PyTorch workflow before modifying the dataset and neural network in the subsequent tasks.

The tutorials cover:

- Tensors and tensor operations
- Datasets and DataLoaders
- Data transformations
- Neural network construction
- Automatic differentiation
- Model optimization
- Saving and loading models

All Task 1 notebooks are available in the [`Task1/`](Task1/) directory.

## Task 1 Notebooks

| Notebook | Description |
|---|---|
| [`Task1_Tensors.ipynb`](Task1/Task1_Tensors.ipynb) | Introduction to PyTorch tensors and tensor operations |
| [`Task1_Datasets&DataLoaders.ipynb`](Task1/Task1_Datasets%26DataLoaders.ipynb) | Working with PyTorch `Dataset` and `DataLoader` |
| [`Task1_transforms.ipynb`](Task1/Task1_transforms.ipynb) | Applying transformations to datasets |
| [`Task1_buildmodel_tutorial.ipynb`](Task1/Task1_buildmodel_tutorial.ipynb) | Building a neural network using `nn.Module` |
| [`Task1_autogradqs_tutorial.ipynb`](Task1/Task1_autogradqs_tutorial.ipynb) | Automatic differentiation using `torch.autograd` |
| [`Task1_optimization.ipynb`](Task1/Task1_optimization.ipynb) | Training, testing, and prediction demonstration |
| [`Task1_saveloadrun.ipynb`](Task1/Task1_saveloadrun.ipynb) | Saving and loading PyTorch models |

## Task 1 Model

The FashionMNIST tutorial model uses the following architecture:

```text
28 × 28 Grayscale Image
          │
          ▼
        Flatten
          │
          ▼
         784
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

This can be summarized as:

```text
784 → 512 → 512 → 10
```

## Task 1 Training and Results

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

A **Prediction Demonstration** was added to the optimization notebook to satisfy the project requirement to display model predictions together with their corresponding ground-truth labels.

The demonstration:

1. Retrieves images and labels from the FashionMNIST test DataLoader.
2. Performs inference using the trained model.
3. Determines the predicted class for each image.
4. Displays ten test samples.
5. Captions each image with its ground-truth and predicted class.

This addition provides a qualitative demonstration of the model's performance in addition to the numerical test accuracy.

---

# Task 2 — CIFAR-10 Dataset

**Status: Complete**

Task 2 replaces the FashionMNIST dataset with **CIFAR-10** and prepares the new dataset for use with the modified neural network.

The implementation is available in:

[`Task2_CIFAR10.ipynb`](Task2/Task2_CIFAR10.ipynb)

## CIFAR-10 Preprocessing

CIFAR-10 contains **60,000 RGB images** with dimensions of 32 × 32 pixels distributed among 10 classes.

The preprocessing pipeline:

1. Loads CIFAR-10 using `torchvision.datasets.CIFAR10`.
2. Converts images to PyTorch tensors using `ToTensor()`.
3. Normalizes the red, green, and blue channels.
4. Uses a mean of `(0.5, 0.5, 0.5)`.
5. Uses a standard deviation of `(0.5, 0.5, 0.5)`.
6. Combines the original CIFAR-10 training and testing portions.
7. Creates a new reproducible 80/20 training/testing split.
8. Creates training and testing DataLoaders using a batch size of 64.

A fixed random seed of `42` is used during the split so that the same training and testing sets can be reproduced.

## Dataset Split

| Dataset | Samples | Percentage |
|---|---:|---:|
| Training | 48,000 | 80% |
| Testing | 12,000 | 20% |
| **Total** | **60,000** | **100%** |

The resulting DataLoaders contain:

```text
Training batches: 750
Testing batches:  188
```

## CIFAR-10 Input Dimensions

Each CIFAR-10 image has the tensor dimensions:

```text
[3, 32, 32]
```

representing three RGB channels and a 32 × 32 pixel image.

When flattened:

```text
3 × 32 × 32 = 3072
```

Therefore, the modified feedforward neural network requires **3,072 input features**.

---

# Task 3 — Modified Neural Network Architecture

**Status: Complete**

Task 3 replaces the neural network used in the original PyTorch tutorial with a deeper feedforward architecture designed for the CIFAR-10 input prepared in Task 2.

The implementation is available in the [`Task3/`](Task3/) directory.

## Model Architecture

The network first flattens each `[3, 32, 32]` CIFAR-10 image into a vector containing 3,072 features.

The modified architecture is:

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

The architecture can be summarized as:

```text
3072 → 128 → 64 → 1
```

ReLU activation is applied after each hidden linear layer, while dropout with a probability of `0.3` is applied after each hidden layer for regularization.

## Model Verification

A synthetic batch matching the dimensions of the CIFAR-10 input was passed through the completed network.

```text
Input shape:  [64, 3, 32, 32]
Output shape: [64, 1]
```

The successful forward pass confirms that the network accepts CIFAR-10 images and produces one output for each input sample.

## Model Parameters

| Parameter | Value |
|---|---:|
| Input Features | 3,072 |
| Hidden Layer 1 | 128 neurons |
| Hidden Layer 2 | 64 neurons |
| Output Neurons | 1 |
| Dropout Rate | 0.3 |
| Total Parameters | 401,665 |
| Trainable Parameters | 401,665 |

---

# Task 4 — Mean Squared Error Loss

**Status: Complete**

Task 4 combines the CIFAR-10 dataset from Task 2 with the neural network developed in Task 3 and replaces the classification loss used in Task 1 with **Mean Squared Error (MSE) loss**.

The implementation is available in:

[`Task4_MSELoss.ipynb`](Task%204/Task4_MSELoss.ipynb)

## Classification to Regression

Task 1 used:

```python
nn.CrossEntropyLoss()
```

because FashionMNIST was treated as a multi-class classification problem.

Task 4 replaces this with:

```python
nn.MSELoss()
```

The Task 3 model contains only one output neuron. Therefore, the CIFAR-10 class indices from `0` through `9` are treated as numerical regression targets.

The integer labels are converted into floating-point tensors and reshaped to match the `[batch_size, 1]` model output before calculating MSE.

> **Note:** CIFAR-10 is normally a classification dataset. Treating its categorical class indices as numerical regression targets imposes an artificial numerical ordering on the classes. This formulation is used to satisfy the regression architecture and loss-function requirements of the project.

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 |
| Training Samples | 48,000 |
| Testing Samples | 12,000 |
| Batch Size | 64 |
| Model | 3072 → 128 → 64 → 1 |
| Hidden Activation | ReLU |
| Dropout | 0.3 |
| Loss Function | MSELoss |
| Optimizer | SGD |
| Learning Rate | 0.001 |
| Epochs | 10 |
| Compute Device | NVIDIA T4 GPU |

## Training Results

| Epoch | Training MSE | Testing MSE |
|---:|---:|---:|
| 1 | 9.221878 | 7.504057 |
| 2 | 7.627019 | 7.205754 |
| 3 | 7.332627 | 7.133181 |
| 4 | 7.184160 | 7.024420 |
| 5 | 7.045232 | 6.905049 |
| 6 | 6.939080 | 6.848081 |
| 7 | 6.847244 | 6.791628 |
| 8 | 6.805724 | 6.761819 |
| 9 | 6.707950 | 6.702089 |
| 10 | **6.627063** | **6.663068** |

The training MSE decreased from approximately **9.22 to 6.63**, while the testing MSE decreased from approximately **7.50 to 6.66** over the ten training epochs.

The similar final training and testing losses indicate that there was not a large train-test divergence during this experiment.

These MSE values should not be interpreted as classification accuracy. MSE measures the numerical difference between the continuous model output and the numerical class index.

## Regression Prediction Demonstration

The trained model produces a continuous numerical value for each image.

For visualization, the continuous output is:

1. Rounded to the nearest integer.
2. Limited to the valid CIFAR-10 class range of `0–9`.
3. Mapped back to the corresponding CIFAR-10 class name.

The rounded class is used only for interpreting and displaying predictions. Training uses the original continuous output produced by the network.

---

# Final Project Pipeline

The completed project progresses from the original PyTorch tutorial to the modified regression experiment as follows:

```text
Task 1
FashionMNIST Classification
784 → 512 → 512 → 10
CrossEntropyLoss
        │
        ▼
Task 2
CIFAR-10 Dataset
60,000 RGB Images
80% Train / 20% Test
        │
        ▼
Task 3
Modified Architecture
3072 → 128 → 64 → 1
ReLU + Dropout(0.3)
        │
        ▼
Task 4
Regression Training
MSELoss + SGD
        │
        ▼
Final Test MSE
6.663068
```

---

# Report

The accompanying project report is written in **LaTeX using the NeurIPS 2022 template**.

The report provides a more detailed discussion of:

- The PyTorch tutorial workflow
- FashionMNIST classification
- CIFAR-10 preprocessing and normalization
- The 80/20 dataset split
- The modified neural network architecture
- ReLU activation and dropout
- Cross-Entropy Loss versus Mean Squared Error Loss
- Model training and evaluation
- Training and testing loss
- Regression prediction results
- Limitations of treating CIFAR-10 class labels as regression targets

---

# Tools

The project was implemented using:

- Python
- PyTorch
- TorchVision
- Matplotlib
- Jupyter Notebook / Google Colab
- CUDA GPU acceleration for Task 4

---

# Author

**Luis Hernandez**  
Department of Electrical and Computer Engineering  
University of Hawaiʻi at Mānoa
