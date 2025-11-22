from bigram import lyrics_to_indices, build_vocabulary, SAMPLE_SEED
from reader import process_csv_file
import torch
import torch.nn as nn

EMBEDDING_DIM = 10
LEARNING_RATE = 4.1
REGULARIZATION_FACTOR = 0.001


def create_bigram_set() -> tuple[
    torch.Tensor, torch.Tensor, int, dict[str, int], dict[int, str]
]:
    """
    Create input-output pairs for bigram language modeling.

    Processes lyrics, builds vocabulary, converts to indices, and creates
    pairs of consecutive characters for training a bigram model.

    Returns:
        Tuple of (first_chars tensor, second_chars tensor, vocab_size, stoi, itos).
        first_chars contains all characters except the last.
        second_chars contains all characters except the first.
        Together they form bigram pairs (first_chars[i], second_chars[i]).
        stoi is string-to-index mapping dictionary.
        itos is index-to-string mapping dictionary.
    """
    lyrics = process_csv_file()

    stoi, itos, vocab_size = build_vocabulary(lyrics)

    all_indices: torch.Tensor = lyrics_to_indices(lyrics, stoi)

    return all_indices[:-1], all_indices[1:], vocab_size, stoi, itos


def encode_integer_inputs(
    indices: torch.Tensor,
    vocab_size: int,
    embedding_dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Encode integer indices using embedding layer instead of one-hot encoding.

    Memory-efficient alternative to F.one_hot() for large datasets.
    An integer character index is represented via a dense vector instead of
    a sparse one-hot vector.

    Creates a simple embedding layer to convert indices to dense vectors.

    Args:
        indices: Tensor of integer character indices.
        vocab_size: Size of vocabulary.
        embedding_dim: Dimension of embedding vectors.
        generator: Optional random number generator for reproducible initialization.

    Returns:
        Tensor of embedding vectors of shape (N, embedding_dim).
    """
    # Create embedding layer (lookup table: vocab_size rows × embedding_dim columns).
    # nn.Embedding initializes weights randomly, so we overwrite with seeded generator.
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    if generator:
        # Use no_grad() for initialization operations (not part of training loop).
        with torch.no_grad():
            embedding.weight.data = torch.randn(
                vocab_size, embedding_dim, generator=generator
            )
    # Detach embeddings since we are not training the embedding layer weights.
    return embedding(indices.long()).detach()


def forward_pass(
    emb_inputs: torch.Tensor,
    second_chars: torch.Tensor,
    W: torch.Tensor,
    num_samples: int = 50_000,
    regularization_factor: float = REGULARIZATION_FACTOR,
) -> torch.Tensor:
    """
    Perform forward pass through the neural network and compute loss.

    Computes logits from embeddings, applies softmax to get probabilities,
    and calculates regularized negative log likelihood loss.

    Args:
        emb_inputs: Input embeddings tensor of shape (N, embedding_dim).
        second_chars: Target character indices for computing loss.
        W: Weight matrix tensor of shape (embedding_dim, vocab_size).
        num_samples: Number of samples to process (default: 50,000).
        regularization_factor: L2 regularization factor (default: REGULARIZATION_FACTOR).

    Returns:
        Loss tensor (scalar) combining NLL loss and L2 regularization.

    Note on L2 Regularization (modified to use mean() instead of sum()):

    During backpropagation, the gradient of the regularization term (2λW),
    where λ represents the regularization factor, pulls weights toward zero.
    This is combined with the NLL loss gradient, creating a "weight decay" effect
    where weights are multiplied by a factor slightly less than 1 on each update.

    Weights closer to zero means that exp()-ing them during softmax results in logits
    closer to 1. After normalization step, we get a "smoother" probability distribution.

    Thus, L2 basically penalizes large weights and encourages the model to
    use smaller, more distributed weights. This prevents the model from
    memorizing training data with extreme weight values (why this is bad is another
    rabbit hole) and promotes generalization to unseen data.

    This approach is conceptually similar to label smoothing used in the raw-count
    based bigram model: both techniques encourage smoother, more distributed
    probability distributions rather than sharp concentration on a few outcomes.
    """

    # Compute logits by matrix multiplication: embeddings @ weights.
    # Logits are raw scores for each possible next character.
    # Shape: (num_samples, embedding_dim) @ (embedding_dim, vocab_size) = (num_samples, vocab_size)
    logits = emb_inputs[:num_samples] @ W

    # Apply softmax activation to convert logits to probabilities.
    # Step 1: Exponentiate logits (maps any real number to positive number).
    counts = logits.exp()
    # Step 2: Normalize each row so probabilities sum to 1.
    probs = counts / counts.sum(1, keepdim=True)

    # Get probabilities for the actual observed bigrams.
    predictions = probs[torch.arange(len(probs)), second_chars[: len(probs)]]

    # Compute regularized negative log likelihood loss.

    # The weights are multiplied by a constant slightly less than 1 on every
    # gradient descent update. It decays each weight toward zero each step.
    reg_loss = regularization_factor * (W**2).mean()

    nll_loss = -predictions.log().mean()

    # Total Loss = NLL loss + L2 regularization term
    return nll_loss + reg_loss


def backward_pass(
    W: torch.Tensor, loss: torch.Tensor, learning_rate: float = LEARNING_RATE
) -> None:
    """
    Perform backward pass to compute gradients and update weights.

    Computes gradients via backpropagation and applies gradient descent update
    with a constant learning rate.

    Args:
        W: Weight tensor to update.
        loss: Scalar loss tensor.
        learning_rate: Learning rate for gradient descent (default: LEARNING_RATE).
    """
    # Clear previous gradients.
    W.grad = None
    # Compute gradients via backpropagation.
    loss.backward()
    # Verify gradients were computed.
    assert W.grad is not None
    # Update weights using gradient descent.
    W.data += -learning_rate * W.grad


def sampling(
    W: torch.Tensor,
    itos: dict[int, str],
    vocab_size: int,
    embedding_dim: int,
    generator: torch.Generator,
    num_samples: int = 20,
) -> None:
    """
    Generate text samples using the trained neural network model.

    Samples characters sequentially based on the model's probability distributions,
    starting from the start token "*" (index 0) and continuing until the end token
    is sampled. Generates multiple sample sequences and prints each one.

    Args:
        W: Trained weight matrix tensor of shape (embedding_dim, vocab_size).
        itos: Index-to-string mapping dictionary.
        vocab_size: Size of vocabulary.
        embedding_dim: Dimension of embedding vectors.
        generator: Random number generator for reproducible sampling.
        num_samples: Number of text samples to generate (default: 20).

    Returns:
        None
    """
    # Create embedding layer with the same initialization as training.
    # This ensures embeddings match those used during training.
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    with torch.no_grad():
        embedding.weight.data = torch.randn(
            vocab_size, embedding_dim, generator=generator
        )

    # Generate text samples.
    for i in range(num_samples):
        outputs = []
        # Start sampling from index 0, which represents the start token "*".
        sample_idx = 0

        while True:
            # Get embedding for the current character index.
            # Shape: (1, embedding_dim)
            char_embedding = embedding(torch.tensor([[sample_idx]], dtype=torch.long))

            # Compute logits using the trained weight matrix.
            # Shape: (1, embedding_dim) @ (embedding_dim, vocab_size) = (1, vocab_size)
            logits = char_embedding @ W

            # Apply softmax to convert logits to probabilities.
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            # Sample the next character index from the probability distribution.
            # Flatten to 1D for multinomial sampling.
            sample_idx = int(
                torch.multinomial(
                    probs[0], num_samples=1, replacement=True, generator=generator
                ).item()
            )

            # Convert the sampled index to its corresponding character and append to outputs.
            outputs.append(itos[sample_idx])

            # Stop sampling when the end token "*" (index 0) is chosen.
            if not sample_idx:
                break

        print(f"\nSample {i + 1}:\n", "".join(outputs))


def train(
    num_iterations: int = 1000,
    embedding_dim: int = EMBEDDING_DIM,
    learning_rate: float = LEARNING_RATE,
    regularization_factor: float = REGULARIZATION_FACTOR,
    print_interval: int = 10,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, int], dict[int, str], int, torch.Generator]:
    """
    Train the neural network bigram language model.

    Encapsulates the complete training process: data preparation, model initialization,
    and training loop with forward and backward passes.

    Args:
        num_iterations: Number of training iterations (default: 1000).
        embedding_dim: Dimension of embedding vectors (default: EMBEDDING_DIM).
        learning_rate: Learning rate for gradient descent (default: LEARNING_RATE).
        regularization_factor: L2 regularization factor (default: REGULARIZATION_FACTOR).
        print_interval: Print loss every N iterations (default: 10).
        generator: Optional random number generator. If None, creates one with SAMPLE_SEED.

    Returns:
        Tuple of (trained weight matrix W, stoi dictionary, itos dictionary,
        vocab_size, generator used for training).
    """
    # Initialize random number generator with fixed seed for reproducibility.
    if generator is None:
        generator = torch.Generator().manual_seed(SAMPLE_SEED)

    # Create bigram dataset and vocabulary mappings.
    first_chars, second_chars, vocab_size, stoi, itos = create_bigram_set()

    print(f"Vocabulary size: {vocab_size}")

    # Use the same generator for embedding initialization to ensure reproducibility.
    emb_inputs = encode_integer_inputs(
        first_chars, vocab_size, embedding_dim, generator=generator
    )

    print(f"Embedding shape: {emb_inputs.shape}")
    print(f"First embedding: {emb_inputs[1]}")

    # Initialize weight matrix with seeded generator for reproducibility.
    W = torch.randn(
        (embedding_dim, vocab_size), generator=generator, requires_grad=True
    )

    # Training loop.
    for i in range(num_iterations):
        # Forward pass: compute predictions and loss.
        loss = forward_pass(
            emb_inputs, second_chars, W, regularization_factor=regularization_factor
        )

        # Print loss at specified intervals.
        if i % print_interval == 0:
            print(f"Iteration {i}: loss = {loss.item()}")

        # Backward pass: compute gradients and update weights.
        backward_pass(W, loss, learning_rate)

    return W, stoi, itos, vocab_size, generator


if __name__ == "__main__":
    # Train the model.
    W, stoi, itos, vocab_size, g = train()

    # Generate samples using the trained model.
    print("\n" + "=" * 50)
    print("Generating text samples:")
    print("=" * 50)
    sampling(W, itos, vocab_size, EMBEDDING_DIM, g)
