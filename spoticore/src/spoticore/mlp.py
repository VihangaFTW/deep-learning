import torch
import torch.nn.functional as F
from dataclasses import dataclass

from reader import read_all_unique_words
from constants import (
    LEARNING_RATE,
    SAMPLE_SEED,
    BLOCK_SIZE,
    EMBEDDING_DIM,
    HIDDEN_LAYER_SIZE,
)


@dataclass(frozen=True)
class Parameters:
    embedding_weights: torch.Tensor
    W1: torch.Tensor
    W2: torch.Tensor
    b1: torch.Tensor
    b2: torch.Tensor

    @property
    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.embedding_weights, self.W1, self.W2, self.b1, self.b2)


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
    words: list[str],
    stoi: dict[str, int],
    itos: dict[int, str],
    block_size: int = BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build training dataset from words using character-level context windows.

    Creates input-output pairs where each input is a context window of characters
    and the output is the next character to predict. Uses a sliding window approach
    with a fixed context length (block_size).

    Args:
        words: List of words to build the dataset from.
        stoi: Mapping from character to index for converting characters to tensor inputs.
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
            # print(
            #     f" {''.join(itos[i] for i in prev_char_idxs)} --> {itos[next_char_idx]}"
            # )
            prev_char_idxs = prev_char_idxs[1:] + [next_char_idx]

    return torch.tensor(X), torch.tensor(Y), block_size


def initialize_parameters(
    vocab_size: int, block_size: int, generator: torch.Generator | None = None
) -> Parameters:
    """
    Initialize network parameters with random values.

    Args:
        vocab_size: Size of the vocabulary (number of unique characters).
        block_size: Number of characters in the context window.
        generator: Optional random number generator for reproducible initialization.

    Returns:
        ForwardPassResult: Container with initialized parameters.
    """
    if generator is None:
        generator = torch.Generator().manual_seed(SAMPLE_SEED)

    emb_dims = EMBEDDING_DIM
    hidden_layer_size = HIDDEN_LAYER_SIZE

    # Embedding layer weights.
    C = torch.randn((vocab_size, emb_dims), generator=generator, requires_grad=True)

    # Hidden layer weights and bias.
    W1 = torch.randn(
        (block_size * emb_dims, hidden_layer_size),
        generator=generator,
        requires_grad=True,
    )
    b1 = torch.randn(hidden_layer_size, generator=generator, requires_grad=True)

    # Output layer weights and bias.
    W2 = torch.randn(
        (hidden_layer_size, vocab_size), generator=generator, requires_grad=True
    )
    b2 = torch.randn(vocab_size, generator=generator, requires_grad=True)

    return Parameters(C, W1, W2, b1, b2)


def forward_pass(
    X: torch.Tensor,
    Y: torch.Tensor,
    parameters: Parameters,
    block_size: int,
) -> tuple[Parameters, torch.Tensor]:
    """
    Perform forward pass through the MLP network for character-level language modeling.

    The network consists of:
    1. Embedding layer: Converts character indices to dense embeddings.
    2. Hidden layer: Fully connected layer with tanh activation.
    3. Output layer: Produces logits for next character prediction.

    Args:
        X: Input tensor of shape (n_samples, block_size) containing character indices.
        Y: Target tensor of shape (n_samples,) containing the true next character indices.
        parameters: Container with all network parameters (embedding weights, W1, W2, b1, b2).
        block_size: Number of characters in the context window.

    Returns:
        Tuple containing:
            - Parameters: Container with the parameter tensors (same reference as input).
            - loss: The cross entropy loss tensor.
    """
    emb_dims = EMBEDDING_DIM

    emb = parameters.embedding_weights[X]  # [num_samples, block_size, emb_dims]
    num_samples = emb.shape[0]

    # Flatten emb tensor to 2d for matrix multiplication with weight matrix.
    # One input sample contains 3 characters and each character is embedded as a vector of size 10.
    # * Each sample becomes a single 30 element vector containing all the context information.
    emb = emb.view(num_samples, block_size * emb_dims)

    # Each row of W1 corresponds to one of the 30 input features (after flattening) of an input sample.
    # * Each hidden neuron receives a weighted sum of all 30 features.
    h = torch.tanh(emb @ parameters.W1 + parameters.b1)  # [num_samples, hidden_size]

    # Output layer contains a node for each character that comes next; i.e. vocab_size neurons.
    logits = h @ parameters.W2 + parameters.b2  # [num_samples, vocab_size]
    loss = F.cross_entropy(logits, Y)

    return parameters, loss


def backward_pass(parameters: list[torch.Tensor], loss: torch.Tensor):
    """
    Perform backward pass and update parameters using gradient descent.

    Args:
        parameters: List of parameters to update (must have requires_grad=True).
        loss: The computed loss tensor.
    Returns:
        None.

    Note:
    The parameters are updated in place.
    """

    # just in case
    for p in parameters:
        p.requires_grad = True

    # ? Compute gradients via backpropagation.
    for p in parameters:
        p.grad = None

    loss.backward()

    # ? Update parameters using gradient descent.
    for p in parameters:
        if p.grad is not None:
            p.data -= LEARNING_RATE * p.grad
        else:
            # This shouldn't happen.
            print(
                f"Warning: Parameter with shape {p.shape} has no gradient (requires_grad={p.requires_grad})"
            )


def train(
    num_iterations: int = 1000,
    print_interval: int = 10,
    generator: torch.Generator | None = None,
) -> None:
    """
    Train the MLP network for character-level language modeling.

    Performs the complete training process: data preparation, parameter initialization,
    and training loop with forward and backward passes.

    Args:
        num_iterations: Number of training iterations (default: 1000).
        print_interval: Print loss every N iterations (default: 10).
        generator: Optional random number generator. If None, creates one with SAMPLE_SEED.
    """
    # Initialize random number generator with fixed seed for reproducibility.
    if generator is None:
        generator = torch.Generator().manual_seed(SAMPLE_SEED)

    # Build vocabulary and dataset.
    stoi, itos, words = build_vocab_from_words()
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")

    # Build training dataset from all words.
    X, Y, block_size = build_dataset(words, stoi, itos)
    print(f"Dataset shape: X={X.shape}, Y={Y.shape}")

    # Initialize network parameters.
    params = initialize_parameters(vocab_size, block_size, generator=generator)
    print("Parameters initialized")

    n_params = sum(p.nelement() for p in params.parameters)
    print("Total paramters:", n_params)

    # Training loop.
    for i in range(num_iterations):
        # Forward pass: compute predictions and loss.
        params, loss = forward_pass(X, Y, params, block_size)

        # Print loss at specified intervals.
        if i % print_interval == 0:
            print(f"Iteration {i}: loss = {loss.item():.4f}")

        # ? Backward pass: compute gradients and update parameters.
        backward_pass(list(params.parameters), loss)

    print(f"Training complete. Final loss: {loss.item():.4f}")


if __name__ == "__main__":
    train(100)
