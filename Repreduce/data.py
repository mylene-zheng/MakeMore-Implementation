"""Data loading and preprocessing utilities."""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Set

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler

logger = logging.getLogger(__name__)

class CharDataset(Dataset):
    """Character-level dataset for language modeling.

    Attributes:
        words: List of words/tokens in the dataset.
        chars: List of unique characters.
        stoi: Mapping from character to integer.
        itos: Mapping from integer to character.
        max_word_length: The length of the longest word in the dataset.
    """

    def __init__(self, words: List[str], chars: List[str], max_word_length: int):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi: Dict[str, int] = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: s for s, i in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.words)

    def get_vocab_size(self) -> int:
        return len(self.chars) + 1  # +1 for special 0 token

    def get_output_length(self) -> int:
        return self.max_word_length + 1

    def encode(self, word: str) -> torch.Tensor:
        return torch.tensor([self.stoi[w] for w in word], dtype=torch.long)

    def decode(self, ix: List[int]) -> str:
        return ''.join(self.itos[i] for i in ix if i in self.itos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        
        # Format: <START> word
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1  # Mask loss
        return x, y

def create_datasets(input_file: str) -> Tuple[CharDataset, CharDataset]:
    """Reads input file and creates train/test split."""
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    words = [w.strip() for w in data.splitlines() if w.strip()]
    chars = sorted(list(set(''.join(words))))
    max_len = max(len(w) for w in words)

    logger.info(f"Dataset stats: {len(words)} examples, Max len: {max_len}")
    logger.info(f"Vocab size: {len(chars)}")

    # 90/10 Split
    test_set_size = min(1000, int(len(words) * 0.1))
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]

    return (
        CharDataset(train_words, chars, max_len),
        CharDataset(test_words, chars, max_len)
    )

class InfiniteDataLoader:
    """Wraps a standard DataLoader to provide an infinite stream of batches."""
    
    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, device: str):
        sampler = RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.loader = DataLoader(
            dataset, sampler=sampler, batch_size=batch_size, 
            num_workers=num_workers, pin_memory=True
        )
        self.iterator = iter(self.loader)
        self.device = device

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return tuple(t.to(self.device) for t in batch)