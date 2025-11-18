# Micrograd

A minimal autograd engine and neural network library implemented from scratch in Python. This project demonstrates the fundamentals of automatic differentiation, backpropagation, and neural network training by building a simple but complete deep learning framework.

## Overview

Micrograd is an educational implementation based on [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd). It captures the core concepts with some modifications and enhancements while not being a direct copy or as feature rich as the original. Key features include:

- **Automatic Differentiation**: Build computational graphs and compute gradients automatically
- **Neural Network Components**: Ready-to-use Neuron, Layer, and MLP classes
- **Visualization**: Graph visualization of computational graphs before and after backpropagation
- **Training**: Built-in gradient descent training for neural networks

## Features

### Core Components

- **`Node`**: The fundamental building block that represents a value in a computational graph with automatic gradient computation
- **`Neuron`**: A single neuron with weights, bias, and tanh activation
- **`Layer`**: A layer of neurons that processes inputs
- **`MLP`**: Multi-Layer Perceptron for building feedforward neural networks

### Operations Supported

The `Node` class supports various mathematical operations:
- Addition (`+`), Subtraction (`-`), Multiplication (`*`), Division (`/`)
- Power (`**`), Negation (`-`)
- Exponential (`exp()`), Hyperbolic Tangent (`tanh()`)

All operations automatically build the computational graph and support backpropagation.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) as the package manager.

```bash
# Install dependencies
uv sync

# Run examples
uv run python main.py
uv run python -m micrograd.neuron
```

## Quick Start

### Basic Usage

```python
from micrograd.node import Node

# Create input nodes
x1 = Node(2.0, label="x1")
x2 = Node(0.0, label="x2")

# Create weight and bias nodes
w1 = Node(-3.0, label="w1")
w2 = Node(1.0, label="w2")
b = Node(6.8813735870195432, label="b")

# Build computational graph
n = x1 * w1 + x2 * w2 + b
o = n.tanh()

# Backpropagate gradients
o.backpropagate(visualize=True)  # visualize=True generates graphviz diagrams
```

### Building a Neural Network

```python
from micrograd.node import Node
from micrograd.neuron import MLP

# Create input data
inputs = [
    [Node(2.0), Node(3.0), Node(-1.0)],
    [Node(3.0), Node(-1.0), Node(0.5)],
    [Node(0.5), Node(1.0), Node(1.0)],
    [Node(1.0), Node(1.0), Node(-1.0)],
]

targets = [1.0, -1.0, -1.0, 1.0]

# Create MLP: 3 inputs -> 4 neurons -> 4 neurons -> 1 output
mlp = MLP(3, [4, 4, 1])

# Train the network
mlp.train(
    inputs=inputs,
    targets=targets,
    learning_rate=0.1,
    num_rounds=500,
    verbose=True
)
```

## Examples

### Example 1: Basic Forward and Backward Pass

Run the basic example that demonstrates forward pass and backpropagation:

```bash
uv run python main.py
```

This example:
- Creates a simple computational graph (x1*w1 + x2*w2 + b, then tanh)
- Performs backpropagation
- Generates visualization graphs (requires graphviz)

### Example 2: Neural Network Training

The `neuron.py` module includes a complete training example:

```bash
uv run python -m micrograd.neuron
```

This trains an MLP on a small dataset using gradient descent.

### PyTorch Comparison

Compare micrograd with PyTorch's autograd system:

```bash
uv run python -m micrograd.pytorch_example
```

## Project Structure

```
micrograd/
├── src/
│   └── micrograd/
│       ├── __init__.py
│       ├── node.py          # Core Node class with autograd
│       ├── graph.py          # Graph visualization utilities
│       ├── neuron.py         # Neuron, Layer, and MLP classes
│       └── pytorch_example.py # PyTorch comparison example
├── main.py                   # Basic usage example
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Dependencies

- **graphviz**: For computational graph visualization
- **numpy**: Numerical operations (for future extensions)
- **torch**: For PyTorch comparison examples
- **pre-commit**: Code quality tools

## How It Works

### Automatic Differentiation

Micrograd uses reverse-mode automatic differentiation (backpropagation):

1. **Forward Pass**: Operations build a computational graph where each `Node` stores its value and references to its input nodes
2. **Backward Pass**: Starting from the output node, gradients are propagated backward through the graph using the chain rule


### Computational Graph

Each operation creates a new `Node` that:
- Stores the computed value (`data`)
- Remembers its parent nodes (`_prev`)
- Stores its gradient (`_grad`)
- Defines a backward function (`_backward`) that computes gradients for its parents

## Credits

Special thanks to Andrej Karpathy for creating the original micrograd project and making it available as an educational resource.

## License

Educational project - feel free to use and modify for learning purposes :)

