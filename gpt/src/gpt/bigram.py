"""
Bigram language model implementation for baseline evaluation of gpt's performance.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED: Final[int] = 3252
TXT_PATH: Final[int] = "./book.txt"


class BatchType(StrEnum):
    train = "training_dataset"
    val = "validation_dataset"


@dataclass(frozen=True)
class TokenStore:
    train: torch.Tensor
    val: torch.Tensor


# model hyperparameters
batch_size = 32
block_size = 8
max_iters = 100_00
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)


# define token encoder & decoder
def _encode(s: str, char2tok: dict[str, int]) -> torch.Tensor:
    return torch.tensor([char2tok[char] for char in s], dtype=torch.long)


def _decode(sequences: torch.Tensor, tok2char: dict[int, str]) -> str:
    return "".join(tok2char[tok.item()] for seq in sequences for tok in seq)


# define mini batch data loader
def _get_batch(btype: BatchType, tokens: TokenStore) -> (torch.Tensor, torch.Tensor):
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


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, input_toks: torch.Tensor, target_toks: torch.Tensor | None
    ) -> (torch.Tensor, torch.Tensor | None):
        logits = self.token_embed_table(input_toks)

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

    def generate(self, input_toks: torch.Tensor, max_new_toks: int) -> torch.Tensor:
        for _ in range(max_new_toks):
            # forward pass in generation mode
            logits, _ = self(input_toks, None)
            # only last token for each sequence needed
            logits = logits[:, -1, :]  # shape: (B, C)
            probs = F.softmax(logits, dim=1)
            # sample from each batch's probability distribution for next token
            next_toks = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            input_toks = torch.cat(
                (input_toks, next_toks), dim=1
            )  # inputs shape: (B, T+1)

        return input_toks


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

    model = BigramModel(vocab_size)

    # gradient optimizer for updating hyperparameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # training loop
    for _ in range(max_iters):
        binputs, btargets = _get_batch(BatchType.train, toks)
        _, loss = model.forward(binputs, btargets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss}")

    # text generation
    input_toks = torch.zeros((1, 1), dtype=torch.long)
    gen_toks = model.generate(input_toks, 500)
    print("Generated text:", _decode(gen_toks, tok2char))


if __name__ == "__main__":
    main()
