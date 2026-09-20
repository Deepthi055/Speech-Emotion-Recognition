"""
Stratified Emotion Batch Sampler for Speech Emotion Recognition.

Guarantees that every emotion class appears at least once per batch,
which stabilizes SupCon loss gradients, especially for minority classes
like `neutral` (lowest recall at 45.35%) and `sad`.

Strategy:
    For a batch of size B with C classes:
    1. Sample exactly 1 index from each class (C forced anchor slots).
    2. Fill the remaining B - C slots uniformly at random from all indices.
    Sampling within each class uses replacement if a class has fewer
    samples than required.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class StratifiedEmotionBatchSampler(Sampler):
    """Yields batches where every emotion class appears at least once.

    Args:
        labels: Array of integer emotion labels aligned with the dataset.
        batch_size: Total number of samples per batch. Must be >= num_classes.
        num_classes: Number of emotion classes (default 6).
        shuffle: Whether to shuffle the per-class index pools each epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 64,
        num_classes: int = 6,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if batch_size < num_classes:
            raise ValueError(
                f"batch_size ({batch_size}) must be >= num_classes ({num_classes}) "
                "to guarantee one sample per class per batch."
            )
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Build per-class index lists
        self.class_indices: list[np.ndarray] = [
            np.where(self.labels == c)[0] for c in range(num_classes)
        ]

        # Total samples (one full pass over the dataset, rounded to complete batches)
        self.n_samples = len(self.labels)
        self.n_batches = self.n_samples // batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        # Optionally shuffle per-class pools at the start of each epoch
        if self.shuffle:
            pools = [self.rng.permutation(idx) for idx in self.class_indices]
        else:
            pools = [idx.copy() for idx in self.class_indices]

        # Cursor into each class pool (for cycling)
        cursors = [0] * self.num_classes

        # Global pool for the "fill" slots (remaining B - C indices per batch)
        all_indices = np.arange(self.n_samples)
        if self.shuffle:
            all_indices = self.rng.permutation(all_indices)
        fill_cursor = 0

        for _ in range(self.n_batches):
            batch: list[int] = []

            # --- Forced anchor slot: one per class ---
            for c in range(self.num_classes):
                pool = pools[c]
                if len(pool) == 0:
                    # Class has no samples; skip (shouldn't happen in practice)
                    continue
                idx = cursors[c] % len(pool)
                batch.append(int(pool[idx]))
                cursors[c] += 1

            # --- Fill remaining slots from global pool ---
            fill_needed = self.batch_size - len(batch)
            if fill_needed > 0:
                remaining = self.n_samples - fill_cursor
                if remaining < fill_needed:
                    # Wrap global pool
                    if self.shuffle:
                        all_indices = self.rng.permutation(np.arange(self.n_samples))
                    fill_cursor = 0
                fill_slice = all_indices[fill_cursor: fill_cursor + fill_needed].tolist()
                batch.extend([int(i) for i in fill_slice])
                fill_cursor += fill_needed

            yield batch
