# LiminalGPT

Character-level GPT implementation using decoder-only transformer architecture.

## Architecture

### Model Components

**Transformer Block:**

- Multi-head self-attention mechanism
- Position-wise feedforward network (expansion factor: 4x)
- Pre-norm architecture with layer normalization
- Residual connections around both sub-layers

**Attention Head:**

- Scaled dot-product attention: `softmax(QK^T / sqrt(d_k))V`
- Causal masking with lower triangular matrix
- Query/Key/Value projections without bias
- Dropout on attention weights

**Embeddings:**

- Token embeddings: `vocab_size → embd_dims`
- Positional embeddings: `block_size → embd_dims`
- Additive combination of token and position encodings

### Hyperparameters

Default configuration in `liminal_p2.py`:

```
vocab_size: dataset-dependent
embd_dims: 32
n_heads: 4
block_size: 8 (context window)
ffn_layer_scale: 4
drop_rate: 0.2
n_layers: 20
learning_rate: 1e-3
batch_size: 32
```

### Training Details

- Optimizer: AdamW
- Loss: Cross-entropy on next-token prediction
- Data split: 90% train, 10% validation
- Evaluation: Average loss over multiple batches
- Generation: Multinomial sampling from softmax probabilities

## Files

- `liminal_p1.py`: Bigram baseline (single-layer, no attention)
- `liminal_p2.py`: Full transformer implementation
- `liminal_p2_attention.ipynb`: Attention mechanism exploration
- `type_checker.py`: Type verification utility for Linear layers

## Design Decisions

**Pre-norm vs Post-norm:**
Uses pre-norm (normalize before sub-layer) for better gradient flow in deep networks.

**Causal Masking:**
Lower triangular mask ensures position `i` only attends to positions `≤ i`, maintaining autoregressive property.

**Head Size Calculation:**
`head_size = embd_dims / n_heads` splits embedding dimension across heads, allowing parallel processing of different representation subspaces.

**Projection After Multi-Head:**
Linear projection after concatenating head outputs allows learned interactions between different attention heads.

**Dropout Placement:**
Applied after attention weights, after projection, and in feedforward network for regularization.

## Performance

Validation loss progression (32 embd_dims, block_size=8):

| Heads | Blocks | FFN | Residual | Layer Norm | Dropout | Val Loss | Notes           |
| ----- | ------ | --- | -------- | ---------- | ------- | -------- | --------------- |
| 1     | 1      | ✗   | ✗        | ✗          | ✗       | 2.3310   | -               |
| 4     | 1      | ✗   | ✗        | ✗          | ✗       | 2.1903   | -               |
| 4     | 1      | ✓   | ✗        | ✗          | ✗       | 2.1735   | -               |
| 4     | 4      | ✓   | ✗        | ✗          | ✗       | 2.3096   | gradient issues |
| 4     | 4      | ✓   | ✓        | ✗          | ✗       | 1.9157   | -               |
| 4     | 4      | ✓   | ✓        | ✓          | ✗       | 1.8945   | -               |
| 4     | 4      | ✓   | ✓        | ✓          | ✓       | 1.8919   | -               |

## Usage

```bash
uv sync
uv run python src/gpt/liminal_p2.py
```

Generates text to `output.txt` after training.
