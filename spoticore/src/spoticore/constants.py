"""
Constants used across the spoticore package.

This module centralizes all configuration constants to make them easy to find,
modify, and maintain. Constants are grouped by category.
"""

from typing import Final

# ============================================================================
# Random Seed
# ============================================================================
SAMPLE_SEED: Final[int] = 534150593

# ============================================================================
# Data Processing
# ============================================================================
LYRICS_COLUMN: Final[str] = "text"
DEFAULT_CSV_PATH: Final[str] = "spotify_lyrics.csv"

# ============================================================================
# Model Architecture - MLP
# ============================================================================
BLOCK_SIZE: Final[int] = 3  # Context window size (number of characters)
EMBEDDING_DIM: Final[int] = 10  # Dimension of character embeddings
HIDDEN_LAYER_SIZE: Final[int] = 100  # Number of neurons in the hidden layer

# ============================================================================
# Training Hyperparameters
# ============================================================================
LEARNING_RATE: Final[float] = 0.1
REGULARIZATION_FACTOR: Final[float] = 0.001
