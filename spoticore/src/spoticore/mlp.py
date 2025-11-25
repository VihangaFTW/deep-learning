import torch
import torch.nn.functional as F

from reader import read_all_unique_words
from constants import (
    LEARNING_RATE,
    SAMPLE_SEED,
    BLOCK_SIZE,
    EMBEDDING_DIM,
    HIDDEN_LAYER_SIZE,
)


def build_vocab_from_words():
    """
    Build vocabulary mappings from unique words.

    Reads all unique words from the CSV file and creates string-to-index (stoi)
    and index-to-string (itos) mappings for all characters found in the words.
    The period character (.) is assigned index 0.

    Returns:
        tuple: A tuple containing:
            - stoi (dict[str, int]): Mapping from character to index.
            - itos (dict[int, str]): Mapping from index to character.
            - words (list[str]): List of all unique words.
    """
    words = read_all_unique_words()
    chars = sorted(list(set(".".join(words))))
    stoi = {char: i for i, char in enumerate(chars)}
    stoi["."] = 0
    itos = {i: char for i, char in enumerate(stoi)}

    return stoi, itos, words


def build_dataset(
    words: list[str], itos: dict[int, str], block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build training dataset from words using character-level context windows.

    Creates input-output pairs where each input is a context window of characters
    and the output is the next character to predict. Uses a sliding window approach
    with a fixed context length (block_size).

    Args:
        words: List of words to build the dataset from.
        itos: Mapping from index to character for converting indices back to characters.
        block_size: Number of characters in the context window. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - X (torch.Tensor): Input tensor of shape (n_samples, block_size) with context windows.
            - Y (torch.Tensor): Output tensor of shape (n_samples,) with target character indices.
            - block_size (int): The context window size used.
    """
    # input and corresponding output (labels) matrices
    X, Y = [], []

    for word in words:
        print(word)
        prev_char_idxs = [0] * block_size
        for char in word + ".":
            next_char_idx = stoi[char]
            X.append(prev_char_idxs)
            Y.append(next_char_idx)
            print(
                f" {''.join(itos[i] for i in prev_char_idxs)} --> {itos[next_char_idx]}"
            )
            prev_char_idxs = prev_char_idxs[1:] + [next_char_idx]

    return torch.tensor(X), torch.tensor(Y), block_size


def forward_pass(
    X: torch.Tensor, Y: torch.Tensor, vocab_size: int, block_size: int
) -> torch.Tensor:
    """
    Perform forward pass through the MLP network for character-level language modeling.

    The network consists of:
    1. Embedding layer: Converts character indices to dense embeddings.
    2. Hidden layer: Fully connected layer with tanh activation.
    3. Output layer: Produces logits for next character prediction.

    Args:
        X: Input tensor of shape (n_samples, block_size) containing character indices.
        Y: Target tensor of shape (n_samples,) containing the true next character indices.
        vocab_size: Size of the vocabulary (number of unique characters).
        block_size: Number of characters in the context window.

    Returns:
        torch.Tensor: The cross-entropy loss value (scalar tensor).
    """
    g = torch.Generator().manual_seed(SAMPLE_SEED)

    emb_dims = EMBEDDING_DIM
    hidden_layer_size = HIDDEN_LAYER_SIZE
    C = torch.randn((vocab_size, emb_dims), generator=g, requires_grad=True)  # 37 X 10

    # ? hidden layer
    b1 = torch.randn(hidden_layer_size, generator=g)

    emb = C[X]  #  [num_samples, block_size, emb_dims] = [36, 3, 10]
    num_samples = emb.shape[0]

    # Flatten emb tensor to 2d for matrix multiplication with weight matrix
    # One input sample contains 3 chracters and each character is embedded as a vector of size 10.
    # * Each sample becomes a single 30 element vector containing all the context information
    emb = emb.view(num_samples, block_size * emb_dims)

    # Each row of W1 corresponds to one of the 30 input features (after flattening) of an input sample.
    # * Each hidden neuron receives a weighted sum of all 30 features
    W1 = torch.randn((block_size * emb_dims, hidden_layer_size), generator=g)

    h = torch.tanh(emb @ W1 + b1)  # [num_samples, hidden_size]

    # ? output layer
    # Output layer contains a node for each character that comes next; i.e. vocab_size neurons
    b2 = torch.randn(vocab_size, generator=g)
    W2 = torch.randn((hidden_layer_size, vocab_size), generator=g)

    logits = h @ W2 + b2  # [num_samples, vocab_size]

    return F.cross_entropy(logits, Y)


def backward_pass(parameters: list[torch.Tensor], loss: torch.Tensor):
    """
    Perform backward pass and update parameters using gradient descent.

    Args:
        parameters: List of parameters to update (must have requires_grad=True).
        loss: The computed loss tensor.
    """
    for p in parameters:
        p.requires_grad = True

    # ? Compute gradients via backpropagation.
    for p in parameters:
        p.grad = None

    loss.backward()

    # ? Update parameters using gradient descent.
    for p in parameters:
        if p.grad:
            p.data -= LEARNING_RATE * p.grad
        else:
            # This shouldn't happen.
            print(
                f"Warning: Parameter with shape {p.shape} has no gradient (requires_grad={p.requires_grad})"
            )


if __name__ == "__main__":
    stoi, itos, words = build_vocab_from_words()

    vocab_size = len(stoi)

    print("vocab size is ", vocab_size)

    print(f"{stoi=}")

    X, Y, block_size = build_dataset(words[:3], itos)

    forward_pass(X, Y, vocab_size, block_size)
