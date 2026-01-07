"""
#TODO add module doc
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
    train = "training_dataset"
    val = "validation_dataset"


@dataclass(frozen=True)
class TokenStore:
    train: torch.Tensor
    val: torch.Tensor


@dataclass(frozen=True)
class Params:
    n_embd: int
    n_heads: int
    head_size: int
    block_size: int
    vocab_size: int


# model hyperparameters
batch_size = 32
block_size = 8

max_iters = 5000
learning_rate = 1e-3
eval_interval = 1000

n_embd = 32
n_heads = 4

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


# define token encoder & decoder
def _encode(s: str, char2tok: dict[str, int]) -> torch.Tensor:
    return torch.tensor([char2tok[char] for char in s], dtype=torch.long)


def _decode(sequences: torch.Tensor, tok2char: dict[int, str]) -> str:
    return "".join(tok2char[int(tok.item())] for seq in sequences for tok in seq)


# define mini batch data loader
def _get_batch(
    btype: BatchType, tokens: TokenStore
) -> tuple[torch.Tensor, torch.Tensor]:
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
    model.eval()
    # try-finally ensures model.train() called even if there's an error
    # duing loss estimation so that training loop is not affected.
    try:
        avg_losses: dict[str, torch.Tensor] = {}
        for btype in BatchType:
            losses = torch.zeros(eval_interval)
            for k in range(eval_interval):
                X, Y = _get_batch(btype, tokens)
                _, loss = model(X.to(device), Y.to(device))
                losses[k] = loss.item()
            avg_losses[btype] = losses.mean()
        return avg_losses
    finally:
        model.train()


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, params: Params) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            Head(params.n_embd, params.block_size, params.head_size)
            for _ in range(params.n_heads)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # each head output: (B, T, head_size)
        # concat result: (B,T,n_embd) as head_size = n_embd/n_heads
        return torch.cat([h(x) for h in self.heads], dim=-1)


class Head(nn.Module):
    """
    One head of self-attention.
    """

    def __init__(self, n_embd: int, block_size: int, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size

        self.key = apply_module(nn.Linear(n_embd, head_size, bias=False))
        self.query = apply_module(nn.Linear(n_embd, head_size, bias=False))
        self.value = apply_module(nn.Linear(n_embd, head_size, bias=False))
        self.tril_mask: torch.Tensor
        # register mask as buffer as we need to move the mask between cpu/gpu
        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)  # (B,T,head_size)
        # (B,T,head_size) @ (B,head_size, T) --> (B,T,T)
        w = q @ k.transpose(-2, -1)
        scaled_w: torch.Tensor = w * self.head_size**-0.5
        # slice mask to match actual sequence length T
        masked_w = scaled_w.masked_fill(self.tril_mask[:T, :T] == 0, float("-inf"))
        att_mat = masked_w.softmax(-1)
        # (B,T,T) @ (B,T,head_size) --> (B,T,head_size)
        weighted_att_mat = att_mat @ v
        return weighted_att_mat


class FeedForward(nn.Module):
    """
    A simple feedforward network with a single hidden layer.

    This module applies a linear transformation, followed by RELU non-linearity.
    """

    def __init__(self, n_embd: int, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BigramModel(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.token_embd_table = nn.Embedding(params.vocab_size, params.n_embd)
        self.position_embd_table = nn.Embedding(params.block_size, params.n_embd)

        self.sa_heads = MultiHeadAttention(params)

        self.ffwd = FeedForward(params.n_embd, params.vocab_size)

        # lm_head is the final projection layer that converts model's
        # internal representations back into predictions over vocabulary
        self.lm_head = apply_module(nn.Linear(params.n_embd, params.vocab_size))

        self.block_size = params.block_size

    def forward(
        self, input_toks: torch.Tensor, target_toks: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = input_toks.shape

        tok_embds = self.token_embd_table(input_toks)  # (B, T, n_embd)
        pos_embds = self.position_embd_table(
            torch.arange(T, device=device)
        )  # (T, n_embd)

        input_embds = tok_embds + pos_embds  # (B, T, n_embd)

        concat_weights = self.sa_heads(input_embds)  # (B,T,n_embd)

        # (B,T,n_embd) --> (B,T,n_embd)
        activations = self.ffwd(concat_weights)

        # (B,T,n_embd) @ (n_embd, vocab_size) --> (B,T,vocab_size)
        logits = self.lm_head(activations)

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
        for _ in range(max_new_toks):
            # model cannot handle sequence len > 8
            trimmed_ctx = context[:, -self.block_size :]

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

    head_size = n_embd // n_heads
    params = Params(n_embd, n_heads, head_size, block_size, vocab_size)

    model = BigramModel(params)
    # move model parameters to gpu
    model = model.to(device)

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
    print("Generated text:", _decode(context, tok2char))


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU Device:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")
    main()

    # val loss with 5000 iters,32 embds,1 head: 2.3310
    # val loss with 5000 iters,32 embds,4 heads: 2.1903
    # val loss with 5000 iters,32 embds,4 heads, with ffwd: 2.1735
