from bigram import lyrics_to_indices, build_vocabulary, SAMPLE_SEED
from reader import process_csv_file
import torch
import torch.nn as nn

EMBEDDING_DIM = 10


def create_bigram_set() -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create input-output pairs for bigram language modeling.

    Processes lyrics, builds vocabulary, converts to indices, and creates
    pairs of consecutive characters for training a bigram model.

    Returns:
        Tuple of (first_chars tensor, second_chars tensor, vocab_size).
        first_chars contains all characters except the last.
        second_chars contains all characters except the first.
        Together they form bigram pairs (first_chars[i], second_chars[i]).
    """
    lyrics = process_csv_file()

    stoi, _, vocab_size = build_vocabulary(lyrics)

    all_indices: torch.Tensor = lyrics_to_indices(lyrics, stoi)

    return all_indices[:-1], all_indices[1:], vocab_size


def encode_integer_inputs(
    indices: torch.Tensor, vocab_size: int, embedding_dim: int
) -> torch.Tensor:
    """
    Encode integer indices using embedding layer instead of one-hot encoding.

    Memory-efficient alternative to F.one_hot() for large datasets. An integer character index is represented via a dense 10-dimensional vector
    instead of a sparse 51-dimensional vector.

    Creates a simple embedding layer to convert indices to dense vectors.

    Args:
        indices: Tensor of integer character indices.
        vocab_size: Size of vocabulary.
        embedding_dim: Dimension of embedding vectors.

    Returns:
        Tensor of embedding vectors of shape (N, embedding_dim).
    """
    # Create a simple embedding layer for encoding.
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    # Convert indices to long dtype and get embeddings.
    return embedding(indices.long())


def forward_pass(x: torch.Tensor, vocab_size: int, num_samples: int = 50_000):
    """
    Perform forward pass through the neural network.

    Computes logits from embeddings and applies softmax to get probabilities
    for predicting the next character.

    Args:
        x: Input embeddings tensor of shape (N, embedding_dim).
        vocab_size: Size of vocabulary.
        num_samples: Number of samples to process (default: 50,000).
    """
    # Initialize random number generator with fixed seed for reproducibility.
    g = torch.Generator().manual_seed(SAMPLE_SEED)

    # Create weight matrix that maps embeddings to logits over vocabulary.
    # Shape: (embedding_dim, vocab_size)
    W = torch.randn((EMBEDDING_DIM, vocab_size), generator=g)
    print(f"{W=}")

    # Compute logits by matrix multiplication: embeddings @ weights.
    # Logits are raw scores (log counts) for each possible next character.
    # Shape: (num_samples, embedding_dim) @ (embedding_dim, vocab_size) = (num_samples, vocab_size)
    logits = x[:num_samples] @ W
    print(f"{logits=}")

    # Apply softmax activation to convert logits to probabilities.
    # Step 1: Exponentiate logits to get unnormalized counts.
    counts = logits.exp()
    print(f"{counts=}")
    # Step 2: Normalize each row so probabilities sum to 1.
    # This gives probability distribution over vocabulary for each input.
    probs = counts / counts.sum(1, keepdim=True)
    print(f"{probs[20]=}")
    print(f"{probs.shape=}")


if __name__ == "__main__":
    first_chars, second_chars, vocab_size = create_bigram_set()

    print(f"Vocabulary size: {vocab_size}")

    x = encode_integer_inputs(first_chars, vocab_size, EMBEDDING_DIM)

    print(f"Embedding shape: {x.shape}")
    print(f"First embedding: {x[1]}")

    forward_pass(x, vocab_size)
