# LAB-4: Analysis of Vanishing and Exploding Gradient Problems

## Overview
This lab investigates the vanishing and exploding gradient problems in deep neural networks by implementing networks from scratch and analyzing gradient behavior with different activation functions and network depths.

---

## Lab Objectives

1. **Understand gradient computation** through manual implementation of forward and backward propagation
2. **Study gradient flow** in networks with varying depths (2 to 15+ layers)
3. **Compare activation functions** (Sigmoid, Tanh, ReLU) and their impact on gradient propagation
4. **Identify conditions** causing vanishing or exploding gradients
5. **Learn practical solutions** to gradient problems in deep learning

---

## Files Structure

```
lab4_gradient_analysis/
│
├── task1_dataset_generation.py      # Dataset creation and preprocessing
├── task2_two_layer_nn.py            # Shallow network from scratch
├── task3_4_deep_networks.py         # Deep networks with PyTorch
├── task5_comparative_study.py       # Analysis and visualizations
├── run_all_tasks.py                 # Master script to run everything
└── README.md                        # This file
```

---

## Task Descriptions

### Task 1: Dataset Generation and Preprocessing
**File:** `task1_dataset_generation.py`

**What it does:**
- Generates a non-linearly separable moon-shaped dataset (1000 samples)
- Splits data into training (80%) and testing (20%) sets
- Normalizes features using StandardScaler (mean=0, std=1)
- Saves preprocessed data for subsequent tasks
- Creates visualizations of original and normalized datasets

**Output:**
- `task1_original_dataset.png` - Visualization of raw data
- `task1_normalized_dataset.png` - Visualization after normalization
- `preprocessed_data.npz` - Saved data for other tasks

---

### Task 2: Two-Layer Neural Network (From Scratch)
**File:** `task2_two_layer_nn.py`

**What it does:**
- Implements a 2-layer neural network manually (no automatic differentiation)
- Architecture: Input(2) → Hidden(10) → Output(1)
- Trains 3 separate networks using Sigmoid, Tanh, and ReLU activations
- Manually computes gradients using the chain rule
- Tracks gradient norms during training (1000 epochs)

**Key Concepts:**
- **Forward Propagation**: Computing outputs layer by layer
- **Backward Propagation**: Computing gradients using chain rule
- **Weight Updates**: Gradient descent optimization
- **Gradient Monitoring**: Tracking gradient magnitudes

**Output:**
- `task2_loss_curves.png` - Training and test loss for each activation
- `task2_accuracy_curves.png` - Training and test accuracy
- `task2_gradient_norms.png` - Gradient behavior comparison (linear and log scale)

**Understanding the Code:**
```python
# Forward pass example
Z1 = X @ W1 + b1           # Linear combination
A1 = activation(Z1)        # Apply activation function
Z2 = A1 @ W2 + b2          # Second layer linear
A2 = sigmoid(Z2)           # Output (always sigmoid for binary)

# Backward pass example (chain rule)
dZ2 = A2 - y               # Gradient at output
dW2 = A1.T @ dZ2 / m       # Gradient for W2
dZ1 = dZ2 @ W2.T * activation'(Z1)  # Chain rule application
dW1 = X.T @ dZ1 / m        # Gradient for W1
```

---

### Task 3 & 4: Deep Neural Networks with Gradient Monitoring
**File:** `task3_4_deep_networks.py`

**What it does:**
- Implements deep networks using PyTorch
- Tests 3 different depths: 5, 10, and 15 hidden layers
- For each depth, trains 3 networks (Sigmoid, Tanh, ReLU)
- Total: 9 different network configurations
- Extracts layer-wise gradients during training
- Monitors gradient flow through all layers

**Network Architectures:**
```
5-layer:  Input(2) → [Hidden(20)] × 5 → Output(1)
10-layer: Input(2) → [Hidden(20)] × 10 → Output(1)
15-layer: Input(2) → [Hidden(20)] × 15 → Output(1)
```

**Key Concepts:**
- **Deep Networks**: Understanding how depth affects learning
- **Layer-wise Gradients**: Monitoring each layer separately
- **Activation Impact**: How choice of activation scales with depth
- **Gradient Vanishing**: Observing exponential decay in deep networks

**Output:**
- `task3_gradient_norms_depth_X.png` - For each depth (X = 5, 10, 15)
- `task4_layerwise_gradients_X.png` - For each activation function
- `task4_depth_comparison_X.png` - Comparing depths for each activation

---

### Task 5: Comparative Study and Interpretation
**File:** `task5_comparative_study.py`

**What it does:**
- Creates comprehensive visualizations comparing all results
- Analyzes theoretical vs. experimental gradient behavior
- Generates decision flowcharts for activation selection
- Provides detailed mathematical explanations
- Produces summary tables and recommendations

**Output:**
- `task5_comprehensive_analysis.png` - Multi-panel analysis
- `task5_decision_flowchart.png` - Activation selection guide
- `task5_comparison_table.png` - Feature comparison table
- `LAB4_SUMMARY_REPORT.txt` - Complete text summary

---

## Running the Lab

### Option 1: Run All Tasks at Once
```bash
python run_all_tasks.py
```

### Option 2: Run Tasks Individually
```bash
python task1_dataset_generation.py
python task2_two_layer_nn.py
python task3_4_deep_networks.py
python task5_comparative_study.py
```

---

## Requirements

### Python Libraries
```
numpy
matplotlib
scikit-learn
torch
seaborn
```

### Installation
```bash
pip install numpy matplotlib scikit-learn torch seaborn
```

---

## Key Findings You Should Observe

### Vanishing Gradient Problem:

1. **Sigmoid (Worst)**
   - Gradient decreases exponentially with depth
   - After 10 layers: gradient ≈ 0.25^10 ≈ 0.000001
   - Network cannot learn effectively

2. **Tanh (Better)**
   - Less severe than sigmoid
   - After 10 layers: gradient ≈ 0.5^10 ≈ 0.001
   - Still problematic for very deep networks

3. **ReLU (Best)**
   - Gradient remains constant (≈ 1.0)
   - No vanishing gradient problem
   - Enables training of very deep networks

---

## Report Writing Guidelines

Your lab report should include:

1. **Introduction**: Brief overview of vanishing/exploding gradients
2. **Methodology**: Description of experiments (include architecture diagrams)
3. **Results**: All plots with detailed captions
4. **Analysis**: 
   - Compare gradient behavior across activations
   - Explain why ReLU performs better
   - Discuss depth impact
5. **Conclusion**: Summary of key findings and practical recommendations
6. **Code**: Clean, commented code for all tasks

### Important Points to Address:

- Why does sigmoid cause vanishing gradients?
- How does network depth affect the problem?
- Why is ReLU the preferred activation for deep networks?
- What are the trade-offs of each activation function?
- When might you still use sigmoid or tanh?

---

## Practical Tips

1. **Default Choice**: Use ReLU for hidden layers
2. **Monitor Training**: Always plot loss and accuracy curves
3. **Check Gradients**: If training stalls, check gradient magnitudes
4. **Start Simple**: Begin with shallow networks, increase depth gradually
5. **Use Modern Tools**: Batch normalization helps with very deep networks

---

**End of README**
