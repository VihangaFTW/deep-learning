# Deep Learning

A collection of deep learning starter projects.

## Projects

### [micrograd](./micrograd/)

A minimal autograd engine and neural network library implemented from scratch in Python. This project is based on [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and demonstrates the fundamentals of automatic differentiation and backpropagation by building a simple neural network framework.

**Features:**

- Automatic differentiation engine
- Computational graph visualization
- Neural network components (Neuron, Layer, MLP)
- Gradient descent training
- A simple PyTorch comparison example

See the [micrograd README](./micrograd/README.md) for detailed documentation.

## Requirements

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) package manager

## Getting Started

Each project directory contains its own `pyproject.toml` and can be set up independently:

```bash
uv sync
uv run ...
```
