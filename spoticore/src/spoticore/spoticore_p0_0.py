from reader import read_all_unique_words
import matplotlib.pyplot as plt
import torch

from constants import SAMPLE_SEED


def build_vocab_from_words(
    words: list[str],
) -> tuple[dict[str, int], dict[int, str], int]:
    """
    Build vocabulary from words and create string-to-index and index-to-string mappings.

    Uses "." as the special start/end token at index 0 for consistency with MLP model.

    Args:
        words: List of unique words.

    Returns:
        Tuple of (stoi dictionary, itos dictionary, vocab_size).
    """
    # Get all unique characters from the words.
    chars = sorted(list(set(".".join(words))))

    # Create mappings with special token "." at index 0.
    stoi: dict[str, int] = {char: i for i, char in enumerate(chars)}
    stoi["."] = 0

    itos: dict[int, str] = {i: char for char, i in stoi.items()}
    vocab_size = len(stoi)

    return stoi, itos, vocab_size


def words_to_indices(
    words: list[str], stoi: dict[str, int], progress_interval: int = 10000
) -> torch.Tensor:
    """
    Convert all words to character indices with start/end tokens.

    Args:
        words: List of words.
        stoi: String-to-index mapping dictionary.
        progress_interval: Print progress every N words.

    Returns:
        Tensor of character indices.
    """
    print(f"Processing {len(words)} words...")
    all_indices = []

    for i, word in enumerate(words):
        if (i + 1) % progress_interval == 0:
            print(f"Processed {i + 1}/{len(words)} words...")

        # Add start token, character indices, and end token.
        word_indices = (
            [stoi["."]] + [stoi.get(c, stoi["."]) for c in word] + [stoi["."]]
        )
        all_indices.extend(word_indices)

    return torch.tensor(all_indices, dtype=torch.int32)


def count_bigrams(indices_tensor: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Count bigram frequencies using vectorized operations.

    Args:
        indices_tensor: Tensor of character indices.
        vocab_size: Size of vocabulary.

    Returns:
        2D tensor of bigram frequencies (vocab_size x vocab_size).
    """
    print("Converting to tensor and counting bigrams...")

    # Create bigram pairs using slicing.
    first_chars = indices_tensor[:-1]
    second_chars = indices_tensor[1:]

    # Flatten 2D indices to 1D for scatter_add.
    # PyTorch requires long dtype for indexing operations.
    bigram_flat_indices = (first_chars * vocab_size + second_chars).long()
    N_flat = torch.zeros(vocab_size * vocab_size, dtype=torch.int32)
    N_flat.scatter_add_(
        0, bigram_flat_indices, torch.ones_like(bigram_flat_indices, dtype=torch.int32)
    )

    return N_flat.view(vocab_size, vocab_size)


def plot_bigram_heatmap(
    N: torch.Tensor,
    itos: dict[int, str],
    vocab_size: int,
    figsize: tuple[int, int] = (24, 24),
    cmap: str = "Blues",
) -> None:
    """
    Create and display a bigram frequency heatmap.

    Args:
        N: 2D tensor of bigram frequencies.
        itos: Index-to-string mapping dictionary.
        vocab_size: Size of vocabulary.
        figsize: Figure size tuple.
        cmap: Colormap name.
    """

    # Use log scale for better visualization.
    N_log = torch.log1p(N).numpy()

    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(N_log, cmap=cmap, aspect="auto")

    # Add colorbar.
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log Frequency", rotation=270, labelpad=20)

    # Set axis labels to show characters.
    ax.set_xticks(range(vocab_size))
    ax.set_yticks(range(vocab_size))
    ax.set_xticklabels([itos[i] for i in range(vocab_size)], fontsize=8)
    ax.set_yticklabels([itos[i] for i in range(vocab_size)], fontsize=8)
    ax.set_xlabel("Second Character", fontsize=12)
    ax.set_ylabel("First Character", fontsize=12)

    for row in range(vocab_size):
        for col in range(vocab_size):
            count = N[row, col].item()
            bigram: str = itos[row] + itos[col]

            # Use white text on dark blue, black on light blue.
            text_color = "white" if N_log[row, col] > N_log.max() * 0.5 else "black"

            # Smaller font size to fit all bigrams.
            fontsize = 7 if count > 1000 else 6

            # Show bigram.
            ax.text(
                col,
                row,
                bigram,
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                fontweight="bold" if count > 10000 else "normal",
            )

    plt.title("Bigram Frequency Heatmap (Log Scale)", fontsize=14, pad=20)
    plt.show()


def sampling(N: torch.Tensor, itos: dict[int, str]) -> None:
    """
    Generate text samples using bigram probability distributions.

    Samples characters sequentially based on bigram probabilities, starting from
    the start token and continuing until the end token is sampled. Generates
    20 sample sequences and prints each one.

    Args:
        N: 2D tensor of bigram frequency counts (vocab_size x vocab_size).
        itos: Index-to-string mapping dictionary.

    Returns:
        None
    """
    # Initialize random number generator with fixed seed for reproducibility.
    generator = torch.Generator().manual_seed(SAMPLE_SEED)

    # Convert bigram counts to float for probability calculations.
    P = N.float()
    # Normalize each row to convert counts into probabilities.
    # Each row sums to 1.0, representing the probability distribution of the next character given the current character.
    P /= P.sum(1, keepdim=True)

    print(P[0].sum())

    # Generate 20 samples of text sequences.
    for i in range(20):
        outputs = []
        # Start sampling from index 0, which represents the start token ".".
        sample_idx = 0
        while True:
            # Get the probability distribution for the next character given the current character.
            row_probs = P[sample_idx]

            # Sample the next character index from the probability distribution.
            # multinomial samples according to the probabilities in row_probs.
            sample_idx = int(
                torch.multinomial(
                    row_probs, num_samples=1, replacement=True, generator=generator
                ).item()
            )

            # Convert the sampled index to its corresponding character and append to outputs.
            outputs.append(itos[sample_idx])

            # Stop sampling when the end token "." (index 0) is chosen.
            if not sample_idx:
                # index 0 means end character . chosen.
                break

        print(f"\nIteration {i}:\n", "".join(outputs))


def calculate_nll(N: torch.Tensor, words: list[str], stoi: dict[str, int]) -> float:
    """
    Calculate the negative log likelihood of words under the bigram model.

    Computes the average negative log likelihood per character using smoothed
    bigram probabilities. Applies add-one smoothing to avoid zero probabilities.

    Args:
        N: 2D tensor of bigram frequency counts (vocab_size x vocab_size).
        words: List of words to evaluate.
        stoi: String-to-index mapping dictionary.

    Returns:
        Average negative log likelihood per character.
    """
    # Apply smoothing to avoid zero probabilities that would cause log(0) = -inf.
    P = (N + 1).float()
    # Normalize each row to convert counts into probabilities.
    P /= P.sum(1, keepdim=True)

    # Convert to log probabilities for numerical stability.
    log_P = torch.log(P)

    # Convert all words to indices using the existing function.
    indices_tensor = words_to_indices(
        words, stoi, progress_interval=len(words) + 1
    ).long()

    # Create bigram pairs using slicing.
    first_chars = indices_tensor[:-1]
    second_chars = indices_tensor[1:]

    # Get all log probabilities at once via vector based indexing.
    # * If first_chars = [0, 5, 12] and second_chars = [5, 12, 3]
    # * Then log_probs = [log_P[0, 5], log_P[5, 12], log_P[12, 3]]
    log_probs = log_P[first_chars, second_chars]

    # Sum all log probabilities and compute average negative log likelihood.
    log_likelihood = log_probs.sum().item()
    char_count = len(first_chars)

    # Return negative log likelihood averaged over all characters.
    return -log_likelihood / char_count if char_count > 0 else float("inf")


def main() -> None:
    """Main function to process words and create bigram heatmap."""
    # Load unique words from lyrics.
    words = read_all_unique_words()

    # Build vocabulary.
    stoi, itos, vocab_size = build_vocab_from_words(words)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {stoi}")

    # Convert words to indices.
    indices_tensor = words_to_indices(words, stoi)

    # Count bigrams.
    N = count_bigrams(indices_tensor, vocab_size)

    print("Done processing.")

    # Visualize.
    # plot_bigram_heatmap(N, itos, vocab_size)

    sampling(N, itos)

    # Calculate how well the model predicts a given word.
    nll = calculate_nll(N, words, stoi)
    print(f"{nll=}")


if __name__ == "__main__":
    main()
