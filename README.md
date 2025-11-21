# Deep Learning

A collection of deep learning starter projects exploring fundamental concepts from scratch because I got bored with gpt wrappers.

Each project includes explicit credits and acknowledgments.

## Projects

### [micrograd](./micrograd/)

A minimal autograd engine and neural network library implemented from scratch in Python. This project serves as my first hands-on experience with deep learning neural networks.

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
cd to-directory
uv sync
uv run ...
```
