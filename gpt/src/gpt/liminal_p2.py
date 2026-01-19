"""
LiminalGPT: A decoder-only transformer for character-level language modeling.

This module implements a GPT-style language model trained on character-level text data.
The architecture includes:
- Token and positional embeddings.
- Multi-head self-attention with causal masking.
- Position-wise feedforward networks.
- Layer normalization and residual connections.
- Dropout regularization.

The model is trained using next-token prediction with cross-entropy loss and
can generate text autoregressively by sampling from the learned probability distribution.

Usage:
    Run as a script to train on a text corpus and generate samples:
    $ python liminal_p2.py
"""

from dataclasses import dataclass
from enum import StrEnum 
from typing import Final
import torch
import torch.nn as nn
import torch.nn.functional as F

from type_checker import apply_module


SEED: Final[int] = 3252
TXT_PATH: Final[str] = "./book.txt"


class BatchType(StrEnum):
    """
    Enumeration for different types of data batches used in training.

    Attributes:
        train: Training dataset batch type.
        val: Validation dataset batch type.
    """

    train = "training_dataset"
    val = "validation_dataset"


@dataclass(frozen=True)
class TokenStore:
    """
    Immutable container for storing tokenized training and validation datasets.

    Attributes:
        train: Tokenized training data as a 1D tensor.
        val: Tokenized validation data as a 1D tensor.
    """

    train: torch.Tensor
    val: torch.Tensor


@dataclass(frozen=True)
class ModelParams:
    """
    Configuration parameters for the LiminalGPT model architecture.

    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens).
        embd_dims: Dimensionality of token embeddings and hidden states.
        n_heads: Number of attention heads in multi-head attention.
        block_size: Maximum sequence length (context window).
        ffn_layer_scale: Scale factor for feedforward network hidden layer size.
        drop_rate: Dropout probability for regularization.
        n_layers: Number of transformer blocks in the model.
    """

    vocab_size: int
    embd_dims: int = 32
    n_heads: int = 4
    block_size: int = 8
    ffn_layer_scale: int = 4
    drop_rate: float = 0.2
    n_layers: int = 5

    @property
    def head_size(self) -> int:
        """Calculate head size from embedding dimensions."""
        return self.embd_dims // self.n_heads


# model hyperparameters
batch_size = 64
block_size = 256

max_iters = 5000
learning_rate = 3e-4

eval_iters = 200
eval_interval = 500

embd_dims = 384

n_heads = 6
n_layers = 6
ffn_layer_scale = 4
drop_rate = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


# define token encoder & decoder
def _encode(s: str, char2tok: dict[str, int]) -> torch.Tensor:
    """
    Encode a string into a tensor of token indices.

    Args:
        s: Input string to encode.
        char2tok: Dictionary mapping characters to token indices.

    Returns:
        1D tensor of token indices with dtype torch.long.
    """
    return torch.tensor([char2tok[char] for char in s], dtype=torch.long)


def _decode(sequences: torch.Tensor, tok2char: dict[int, str]) -> str:
    """
    Decode a tensor of token indices back into a string.

    Args:
        sequences: Tensor of token indices, typically of shape (B, T) where B is batch size
                   and T is sequence length.
        tok2char: Dictionary mapping token indices to characters.

    Returns:
        Decoded string formed by concatenating all characters corresponding to token indices.
    """
    return "".join(tok2char[int(tok.item())] for seq in sequences for tok in seq)


def _save_generated_text(text: str, filepath: str) -> None:
    """
    Save generated text to a file and print confirmation.

    Args:
        text: The text content to write to the file.
        filepath: Path to the output file.

    Returns:
        None. Writes the text to the specified filepath and prints a confirmation message.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Generated text saved to: {filepath}")


# define mini batch data loader
def _get_batch(
    btype: BatchType, tokens: TokenStore
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random mini-batch of input-target pairs from the dataset.

    This function samples random starting positions in the dataset and extracts
    sequences of length block_size. For each input sequence, the target sequence
    is offset by one token to facilitate next-token prediction.

    Args:
        btype: Type of batch to generate (train or validation).
        tokens: TokenStore containing training and validation token tensors.

    Returns:
        A tuple containing:
        - inputs: Tensor of shape (batch_size, block_size) with input token sequences.
        - targets: Tensor of shape (batch_size, block_size) with target token sequences,
                   offset by one position from inputs.
    """
    match btype:
        case BatchType.train:
            toks = tokens.train
        case BatchType.val:
            toks = tokens.val

    starts = torch.randint(len(toks) - block_size, (batch_size,))
    inputs = torch.stack([toks[start : start + block_size] for start in starts], dim=0)
    targets = torch.stack(
        [toks[start + 1 : start + block_size + 1] for start in starts], dim=0
    )

    return inputs, targets


@torch.inference_mode()
def estimate_loss(
    model: nn.Module, tokens: TokenStore, device: str
) -> dict[str, torch.Tensor]:
    """
    Estimate average loss on training and validation datasets.

    This function evaluates the model's performance by computing the average loss
    over multiple batches for both training and validation datasets. The model is
    temporarily set to evaluation mode during loss estimation to disable dropout
    and other training-specific behaviors.

    Args:
        model: The neural network model to evaluate.
        tokens: TokenStore containing training and validation token tensors.
        device: Device string ('cuda' or 'cpu') indicating where to run computations.

    Returns:
        Dictionary mapping BatchType to average loss tensors. Keys are BatchType.train
        and BatchType.val, values are scalar tensors representing average losses.

    Note:
        The function uses torch.inference_mode() for efficiency and ensures the model
        is set back to training mode via try-finally, even if an error occurs.
    """
    model.eval()
    # try-finally ensures model.train() called even if there's an error
    # duing loss estimation so that training loop is not affected.
    try:
        avg_losses: dict[str, torch.Tensor] = {}
        for btype in BatchType:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = _get_batch(btype, tokens)
                _, loss = model(X.to(device), Y.to(device))
                losses[k] = loss.item()
            avg_losses[btype] = losses.mean()
        return avg_losses
    finally:
        model.train()


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism for parallel attention computation.

    This module runs multiple attention heads in parallel, allowing the model
    to jointly attend to information from different representation subspaces.
    Each head operates on a different portion of the embedding dimensions,
    and their outputs are concatenated and projected back to the original
    embedding dimension.

    The projection layer after concatenation allows different heads' learned
    representations to interact and combine through matrix multiplication,
    rather than keeping them isolated.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(
                params.embd_dims, params.block_size, params.head_size, params.drop_rate
            )
            for _ in range(params.n_heads)
        )
        # ? after concat, each head's output sits in its own isolated part of the embd dims.
        # ? The projection layer lets different heads' outputs interact and combine their
        # ? learned representations through matrix multiplication.
        self.proj = apply_module(nn.Linear(params.embd_dims, params.embd_dims))
        self.dropout = nn.Dropout(params.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to input embeddings.

        Args:
            x: Input tensor of shape (B, T, embd_dims) where B is batch size,
               T is sequence length, and embd_dims is embedding dimension.

        Returns:
            Output tensor of shape (B, T, embd_dims) after multi-head attention,
            projection, and dropout.
        """
        # each head output: (B, T, head_size)
        # concat result: (B,T,embd_dims) as head_size = embd_dims/n_heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # (B,T,C) @ (B,C,C) + (B,C,C) --> (B,T,C)
        out = self.dropout(self.proj(out))
        return out


class Head(nn.Module):
    """
    Single head of scaled dot-product self-attention with causal masking.

    This module implements one attention head using the query-key-value mechanism:
    - Queries and keys are used to compute attention weights (affinities between tokens).
    - Scaled dot-product prevents gradient issues as head_size grows.
    - Causal (triangular) masking ensures tokens can only attend to previous positions.
    - Values are weighted by attention scores to produce the output.

    The attention mechanism allows each token to focus on relevant context
    from earlier positions in the sequence.
    """

    def __init__(
        self, embd_dims: int, block_size: int, head_size: int, drop_rate: float
    ) -> None:
        super().__init__()
        self.head_size = head_size

        self.key = apply_module(nn.Linear(embd_dims, head_size, bias=False))
        self.query = apply_module(nn.Linear(embd_dims, head_size, bias=False))
        self.value = apply_module(nn.Linear(embd_dims, head_size, bias=False))
        self.tril_mask: torch.Tensor
        # register mask as buffer as we need to move the mask between cpu/gpu
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product attention with causal masking.

        This method performs the following steps:
        1. Projects input to queries, keys, and values.
        2. Computes attention scores using scaled dot-product (Q @ K^T / sqrt(head_size)).
        3. Applies causal mask to prevent attending to future tokens.
        4. Applies softmax to get attention weights.
        5. Applies dropout to attention weights.
        6. Computes weighted sum of values.

        Args:
            x: Input tensor of shape (B, T, embd_dims) where B is batch size,
               T is sequence length, and embd_dims is embedding dimension.

        Returns:
            Output tensor of shape (B, T, head_size) representing the attention-weighted
            values for this head.
        """
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)  # (B,T,head_size)
        # (B,T,head_size) @ (B,head_size, T) --> (B,T,T)
        w = q @ k.transpose(-2, -1)
        scaled_w: torch.Tensor = w * self.head_size**-0.5
        # slice mask to match actual sequence length T
        masked_w = scaled_w.masked_fill(self.tril_mask[:T, :T] == 0, float("-inf"))
        att_mat = masked_w.softmax(-1)
        # apply dropout regularization
        att_mat = self.dropout(att_mat)
        # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        weighted_att_mat = att_mat @ v
        return weighted_att_mat


class FeedForward(nn.Module):
    """
    Position-wise feedforward network with expansion and projection.

    This module applies a two-layer feedforward network to each token independently:
    1. Expansion: Projects embeddings to a larger hidden dimension (embd_dims * layer_scale)
       to provide a larger "workspace" for computing complex non-linear transformations.
    2. ReLU activation: Introduces non-linearity.
    3. Projection: Compresses back to original embedding dimension, forcing the network
       to distill the most useful features.
    4. Dropout: Applied for regularization by randomly zeroing elements.

    This expansion-and-projection pattern allows the model to learn rich representations
    while maintaining a consistent embedding dimension throughout the architecture.
    """

    def __init__(self, embd_dims: int, layer_scale: int, drop_rate: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # ? expand hidden layer size to givethe network a larger "workspace"
            # ? to compute complex non-linear transformations
            nn.Linear(embd_dims, embd_dims * layer_scale),
            nn.ReLU(),
            # ? projection layer forces the network to compress that information
            # ? into the most useful features.
            nn.Linear(layer_scale * embd_dims, embd_dims),
            # ? randomly zeroing individual elements throughout the (B, T, C) tensor
            # ? means that each token has some of its features temporarily removed
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feedforward transformation to input.

        Args:
            x: Input tensor of shape (B, T, embd_dims) where B is batch size,
               T is sequence length, and embd_dims is embedding dimension.

        Returns:
            Output tensor of shape (B, T, embd_dims) after feedforward transformation.
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block combining multi-head attention and feedforward network.

    This module implements a standard transformer block with:
    - Multi-head self-attention for capturing token relationships.
    - Feedforward network for token-wise transformations.
    - Layer normalization (pre-norm) before each sub-layer.
    - Residual connections around each sub-layer.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(params)
        self.ffn = FeedForward(
            params.embd_dims, params.ffn_layer_scale, params.drop_rate
        )
        self.ln1 = nn.LayerNorm(params.embd_dims)
        self.ln2 = nn.LayerNorm(params.embd_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with pre-norm and residual connections.

        The computation follows the pre-norm architecture:
        1. x = x + self_attention(layer_norm(x))
        2. x = x + feedforward(layer_norm(x))

        Args:
            x: Input tensor of shape (B, T, embd_dims) where B is batch size,
               T is sequence length, and embd_dims is embedding dimension.

        Returns:
            Output tensor of shape (B, T, embd_dims) after attention, feedforward,
            and residual connections.
        """
        # normalize feature vector per token
        # add residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LiminalGPT(nn.Module):
    """
    A GPT-style language model for character-level text generation.

    This model implements a decoder-only transformer architecture with:
    - Token and positional embeddings.
    - Multiple stacked transformer blocks.
    - Language modeling head for next-token prediction.

    The model uses masked self-attention to ensure autoregressive generation,
    where each token can only attend to previous tokens in the sequence.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.params = params

        self.token_embd_dims_table = nn.Embedding(params.vocab_size, params.embd_dims)
        self.position_embd_dims_table = nn.Embedding(
            params.block_size, params.embd_dims
        )
        self.blocks = nn.Sequential(*[Block(params) for _ in range(params.n_layers)])
        self.final_ln = nn.LayerNorm(params.embd_dims)
        # lm_head is the final projection layer that converts model's
        # internal representations back into predictions over vocabulary
        self.lm_head = apply_module(nn.Linear(params.embd_dims, params.vocab_size))

    def forward(
        self, input_toks: torch.Tensor, target_toks: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for language modeling with optional loss computation.

        The forward pass consists of:
        1. Token embeddings + positional embeddings.
        2. Processing through transformer blocks.
        3. Final layer normalization.
        4. Projection to vocabulary logits.
        5. Cross-entropy loss computation (if targets provided).

        Args:
            input_toks: Input token indices of shape (B, T) where B is batch size
                        and T is sequence length.
            target_toks: Target token indices of shape (B, T) for loss computation,
                         or None during generation.

        Returns:
            A tuple containing:
            - logits: Predicted token logits of shape (B, T, vocab_size) during generation,
                      or (B*T, vocab_size) during training.
            - loss: Cross-entropy loss if target_toks provided, None otherwise.
        """
        B, T = input_toks.shape

        tok_embds = self.token_embd_dims_table(input_toks)  # (B, T, embd_dims)
        pos_embds = self.position_embd_dims_table(
            torch.arange(T, device=device)
        )  # (T, embd_dims)

        input_embds = tok_embds + pos_embds  # (B, T, embd_dims)

        acts = self.blocks(input_embds)  # (B,T,embd_dims)
        # normalize feature vectors per token
        acts = self.final_ln(acts)
        # (B,T,embd_dims) @ (embd_dims, vocab_size) --> (B,T,vocab_size)
        logits = self.lm_head(acts)

        if target_toks is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits, targets = (
                logits.reshape(B * T, C),
                target_toks.reshape(
                    B * T,
                ),
            )
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, context: torch.Tensor, max_new_toks: int) -> torch.Tensor:
        """
        Generate new tokens autoregressively using the trained model.

        This method generates text by:
        1. Taking the most recent block_size tokens as context.
        2. Predicting the next token using the model.
        3. Sampling from the predicted probability distribution.
        4. Appending the sampled token to the context.
        5. Repeating steps 1-4 for max_new_toks iterations.

        Args:
            context: Initial context tensor of shape (B, T) containing token indices,
                     where B is batch size and T is current sequence length.
            max_new_toks: Number of new tokens to generate.

        Returns:
            Extended context tensor of shape (B, T + max_new_toks) containing
            the original context plus the newly generated tokens.
        """
        for _ in range(max_new_toks):
            # model cannot handle sequence len > 8
            trimmed_ctx = context[:, -self.params.block_size :]

            # forward pass in generation mode
            logits, _ = self(trimmed_ctx, None)
            # only last token for each sequence needed
            logits = logits[:, -1, :]  # shape: (B, C)
            probs = F.softmax(logits, dim=-1)
            # sample from each batch's probability distribution for next token
            next_toks = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            context = torch.cat((context, next_toks), dim=1)  # inputs shape: (B, T+1)

        return context


def main():
    """
    Main training and generation pipeline for LiminalGPT.

    This function orchestrates the complete workflow:
    1. Loads and tokenizes the training text corpus.
    2. Splits data into training (90%) and validation (10%) sets.
    3. Initializes the LiminalGPT model with configured hyperparameters.
    4. Trains the model using AdamW optimizer with periodic loss evaluation.
    5. Generates sample text using the trained model.
    6. Saves the generated text to an output file.

    The training loop runs for max_iters iterations, evaluating and reporting
    training and validation losses at regular intervals (eval_interval).

    Returns:
        None. Trains the model and saves generated output to './output.txt'.
    """
    # load training dataset
    with open(TXT_PATH, "r", encoding="utf-8") as bk:
        text = bk.read()

    chars = sorted(set(text))
    vocab_size = len(chars)

    # * 1 token =  1 character

    # define character <-> token mappings
    char2tok = {char: i for i, char in enumerate(chars)}
    tok2char = {i: char for char, i in char2tok.items()}

    # separate dataset into batches
    lim = int(0.9 * len(text))
    toks = TokenStore(_encode(text[:lim], char2tok), _encode(text[lim:], char2tok))

    params = ModelParams(
        vocab_size, embd_dims, n_heads, block_size, ffn_layer_scale, drop_rate, n_layers
    )

    model = LiminalGPT(params)
    # move model parameters to gpu
    model = model.to(device)

    # print model's parameter size
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # gradient optimizer for updating hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop
    for iter_id in range(max_iters):
        # evaluate average model performance over different batch types
        if iter_id % eval_interval == 0:
            losses = estimate_loss(model, toks, device)
            print(
                f"iteration {iter_id}: "
                f"train loss: {losses[BatchType.train]:.4f}, "
                f"val loss: {losses[BatchType.val]:.4f}"
            )

        # sample a batch of data
        binputs, btargets = _get_batch(BatchType.train, toks)
        # move batches to gpu
        binputs, btargets = binputs.to(device), btargets.to(device)

        # evaluate loss
        _, loss = model.forward(binputs, btargets)
        assert loss is not None  # stupid type checker
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # text generation
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context = model.generate(context, 500)
    txt = _decode(context, tok2char)
    _save_generated_text(txt, "./output.txt")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    main()

    # val loss with 5000 iters,32 embds,1 head: 2.3310
    # val loss with 5000 iters,32 embds,4 heads: 2.1903
    # val loss with 5000 iters,32 embds,4 heads, with ffn: 2.1735
    # val loss with 5000 iters,32 embds,4 heads, 4 blocks: 2.3096 (net too deep; need residual connections)
    # val loss with 5000 iters,32 embds,4 heads, 4 blocks, skip conns + 4-ffn_scale: 1.9157
    # val loss with 5000 iters,32 embds,4 heads, 4 blocks, skip conns + 4-ffn_scale, ln: 1.8945
    # val loss with 5000 iters,32 embds,4 heads, 4 blocks, skip conns + 4-ffn_scale, ln, dropouts: 2.000 (prevented overfitting?)
# 2.0811
