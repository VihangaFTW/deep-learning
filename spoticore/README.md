# Spoticore

A character-level language model trained on Spotify lyrics, implementing progressively sophisticated neural architectures from a basic bigram model to a WaveNet-inspired MLP.

## Overview

This project implements a series of increasingly complex language models, starting with simple bigram statistics and evolving into deep neural networks with modern optimization techniques.

Each _stage_ builds upon previous work and is structured to demonstrate fundamental concepts in natural language processing and deep learning.

## Project Structure

---

### Core Files

- [`constants.py`](./src/spoticore/constants.py) - Centralized configuration constants (seeds, hyperparameters, paths)
- [`reader.py`](./src/spoticore/reader.py) - CSV data processing utilities for lyrics dataset
- [`spotify_lyrics.csv`](./src/spoticore/spotify_lyrics.csv) - Dataset containing 57,650 Spotify song lyrics

### Implementation Stages

#### Stage 0: Bigram Models

**Stage 0.0** ([`spoticore_p0_0.py`](./src/spoticore/spoticore_p0_0.py))

- Statistical bigram model using frequency counting
- Bigram probability distributions with add-one smoothing
- Text generation via probabilistic sampling
- Visualization with heatmaps
- Negative log-likelihood evaluation

**Stage 0.1** ([`spoticore_p0_1.py`](./src/spoticore/spoticore_p0_1.py))

- Neural network bigram model with character embeddings
- Dense vector representations instead of one-hot encoding
- Forward/backward pass implementation from scratch
- L2 regularization for weight decay
- Gradient descent optimization

#### Stage 1: Multi-Layer Perceptron

**Stage 1.0** ([`spoticore_p1_0.ipynb`](./src/spoticore/spoticore_p1_0.ipynb))

- MLP architecture handling context windows (block size > 1)
- Embedding layer → Hidden layer (tanh) → Output layer (softmax)
- Mini-batch stochastic gradient descent
- Development set validation
- Learning rate tuning experiments
- Based on [Bengio et al., 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

#### Stage 2: Optimization Techniques

**Stage 2.0** ([`spoticore_p2_0.ipynb`](./src/spoticore/spoticore_p2_0.ipynb))

- Kaiming initialization for stable weight initialization
- Batch normalization to reduce internal covariate shift
- Running statistics with exponentially weighted moving average (EWMA)
- Tanh saturation analysis
- Hidden layer size experiments
- References: [He et al., 2015](https://arxiv.org/pdf/1502.01852), [Ioffe & Szegedy, 2015](https://arxiv.org/pdf/1502.03167)

**Stage 2.1 & 2.2** ([`spoticore_p2_1.ipynb`](./src/spoticore/spoticore_p2_1.ipynb), [`spoticore_p2_2.ipynb`](./src/spoticore/spoticore_p2_2.ipynb))

- Further optimization refinements
- Additional architectural explorations
- A basic Pytorchified version of _stage_ 2
- A complete manual backward pass showcase
- Some new performance tuning and analysis techniques

#### Stage 3: WaveNet Architecture

**Stage 3.0** ([`spoticore_p3_0.ipynb`](./src/spoticore/spoticore_p3_0.ipynb))

- WaveNet-inspired architecture with hierarchical receptive fields
- A smol (but respectable) PyTorch-like module system via Embedding, Linear, FlattenConsecutive, Tanh and Sequential classes. Yes, it's inspired by Pytorch's `torch.nn` module.

Note:

> This project is a WIP. New stages will be added soon!

## Checklist of concepts I learnt

- Pytorch tensor manipulation
- Character-level language modeling
- Vocabulary construction and character encoding
- Bigram statistics and probability distributions
- Neural network embeddings
- Gradient descent and backpropagation
- Regularization techniques (L2, batch normalization)
- Weight initialization strategies
- Mini-batch training
- Model evaluation and text generation
- Hierarchical neural architectures

## Requirements

- Python >= 3.14
- PyTorch >= 2.9.1
- NumPy >= 2.3.5
- Matplotlib >= 3.10.7
- Jupyter (ipykernel >= 7.1.0)

## Acknowledgement

This project is based on Andrej Kaparthy's [Makemore](https://github.com/karpathy/makemore). All credits goes to him :).
