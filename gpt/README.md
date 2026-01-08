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

Current configuration in `liminal_p2.py`:

```
vocab_size: dataset-dependent
embd_dims: 384
n_heads: 6
block_size: 256 (context window)
ffn_layer_scale: 4
drop_rate: 0.2
n_layers: 6
learning_rate: 3e-4
batch_size: 64
max_iters: 5000
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

### Small Model (32 embd_dims, block_size=8)

Validation loss progression:

| Heads | Blocks | FFN | Residual | Layer Norm | Dropout | Val Loss | Notes            |
| ----- | ------ | --- | -------- | ---------- | ------- | -------- | ---------------- |
| 1     | 1      | ✗   | ✗        | ✗          | ✗       | 2.3310   | -                |
| 4     | 1      | ✗   | ✗        | ✗          | ✗       | 2.1903   | -                |
| 4     | 1      | ✓   | ✗        | ✗          | ✗       | 2.1735   | -                |
| 4     | 4      | ✓   | ✗        | ✗          | ✗       | 2.3096   | gradient issues  |
| 4     | 4      | ✓   | ✓        | ✗          | ✗       | 1.9157   | -                |
| 4     | 4      | ✓   | ✓        | ✓          | ✗       | 1.8945   | possible overfit |
| 4     | 4      | ✓   | ✓        | ✓          | ✓       | 2.0000   | reduced overfit  |

### Large Model (10.8M) (384 embd_dims, block_size=256, 6 heads, 6 layers)

Latest run (5000 iterations):

| Iteration | Train Loss | Val Loss | Notes                                   |
| --------- | ---------- | -------- | --------------------------------------- |
| 0         | 4.3995     | 4.3870   | -                                       |
| 500       | 1.5181     | 1.5560   | -                                       |
| 1000      | 1.3142     | 1.4578   | -                                       |
| 1500      | 1.0790     | 1.3850   | Best validation loss                    |
| 2000      | 0.8604     | 1.4080   | Overfitting begins                      |
| 2500      | 0.6539     | 1.4955   | Train/val gap widening                  |
| 3000      | 0.4723     | 1.6318   | Significant overfitting                 |
| 3500      | 0.3232     | 1.7740   | Model memorizing training data          |
| 4000      | 0.2161     | 1.9384   | -                                       |
| 4500      | 0.1530     | 2.0811   | Severe overfitting, better text quality |

**Generated output:**

```txt
The driver seates the table TV screen.

Eventual screen Denny’s traight that close to fill by the man’s apearance,
she sleeps soundly in the corresponding to the area total smiles. The fluorescent 
lamps a man’s feeling, like an old but well-stel between by swunt to it of something. 
He can loses are hand, says her body wanted for the table. But the woman is stread lose 
to see me than Eri Asai.
```


**Observations:**

- Model shows significant overfitting after iteration 1500.
- Training loss continues decreasing while validation loss increases.
- Generated text quality improved despite overfitting (more coherent words/structures).
- Suggests need for stronger regularization or early stopping around iteration 1500.

## Usage

```bash
uv sync
uv run python src/gpt/liminal_p2.py
```

Generates text to `output.txt` after training.
