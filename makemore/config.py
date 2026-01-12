"""Configuration definitions for the Makemore project."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the language model architecture.

    Attributes:
        vocab_size: Number of unique characters in the vocabulary.
        block_size: Context length (sequence length) for predictions.
        n_layer: Number of transformer blocks or RNN layers.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.
        n_embd2: Secondary embedding dimension (used in specific architectures like BoW).
        dropout: Dropout rate (added for standard practice).
    """
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    dropout: float = 0.1

@dataclass
class TrainConfig:
    """Configuration for the training process.

    Attributes:
        work_dir: Directory to save checkpoints and logs.
        max_steps: Maximum optimization steps (-1 for infinite).
        batch_size: Number of samples per batch.
        learning_rate: Optimizer learning rate.
        weight_decay: Optimizer weight decay.
        device: Compute device ('cpu', 'cuda', etc.).
        num_workers: Number of DataLoader workers.
        seed: Random seed for reproducibility.
    """
    work_dir: str = 'out'
    max_steps: int = -1
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    device: str = 'cpu'
    num_workers: int = 4
    seed: int = 3407