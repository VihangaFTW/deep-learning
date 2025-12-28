# Deep Learning

A collection of deep learning starter projects exploring fundamental concepts from scratch because I got bored with gpt wrappers.

Each project includes explicit credits and acknowledgments.

## Projects

### [Autodiff](./autodiff/)

A minimal autograd engine implemented from scratch in Python. This project serves as my first hands-on experience with deep learning neural networks.

**Features:**

- Automatic differentiation engine
- Computational graph visualization
- Neural network components (Neuron, Layer, MLP)
- Gradient descent training
- A simple PyTorch comparison example

See the [autodiff README](./autodiff/README.md) for detailed documentation.

### [Spoticore](./spoticore/)

A character-level language model trained on Spotify lyrics, implementing progressively sophisticated neural architectures from a basic bigram model to a WaveNet-inspired MLP.

**Features:**

- Statistical bigram model with frequency counting
- Neural bigram model with character embeddings
- Multi-layer perceptron with context windows
- Optimizations like Kaiming initialization and Batch Normalization
- WaveNet-inspired architecture with custom PyTorch-like modules
- Text generation capabilities

See the [spoticore README](./spoticore/README.md) for detailed documentation.

## Requirements

- Python >= 3.14
- [uv](https://github.com/astral-sh/uv) package manager

## Getting Started

Each project directory contains its own `pyproject.toml` and can be set up independently:

```bash
cd to-project-directory
uv sync
uv run ...
```
