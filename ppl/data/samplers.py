"""Series-balanced batch sampler: every batch contains each congeneric series."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import BatchSampler


class SeriesBalancedBatchSampler(BatchSampler):
    """Create train batches that contain every series at least once.

    Every dataset index is emitted at least once per epoch. Rare series are
    repeated as needed so every batch still contains every congeneric series.
    """

    def __init__(
        self,
        series_labels: Sequence[str],
        batch_size: int,
        *,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.series_labels = [str(label) for label in series_labels]
        self.batch_size = int(batch_size)
        self.generator = torch.Generator()
        self.generator.manual_seed(int(seed))

        self.series_to_indices: dict[str, list[int]] = {}
        for idx, label in enumerate(self.series_labels):
            self.series_to_indices.setdefault(label, []).append(idx)

        self.series = sorted(self.series_to_indices)
        if not self.series:
            raise ValueError("SeriesBalancedBatchSampler requires at least one series")
        if self.batch_size < len(self.series):
            raise ValueError(
                "batch_size must be at least the number of series when "
                "balance_train_batches_by_series=True: "
                f"batch_size={self.batch_size}, series={len(self.series)}"
            )

        counts = np.asarray(
            [len(self.series_to_indices[label]) for label in self.series],
            dtype=np.float64,
        )
        self.num_batches = self._compute_num_batches(counts.astype(np.int64))
        self.series_probs = torch.as_tensor(counts / counts.sum(), dtype=torch.double)

    def __len__(self) -> int:
        return self.num_batches

    def _compute_num_batches(self, counts: np.ndarray) -> int:
        """Find enough batches to cover all bags plus rare-series repeats."""
        num_batches = max(1, int(np.ceil(len(self.series_labels) / self.batch_size)))
        total_unique = int(counts.sum())

        while True:
            rare_repeats = int(np.maximum(0, num_batches - counts).sum())
            required_draws = total_unique + rare_repeats
            if num_batches * self.batch_size >= required_draws:
                return num_batches
            num_batches += 1

    def _shuffle_indices(self, indices: list[int]) -> list[int]:
        order = torch.randperm(len(indices), generator=self.generator).tolist()
        return [indices[i] for i in order]

    def __iter__(self):
        shuffled_by_series = {
            label: self._shuffle_indices(indices)
            for label, indices in self.series_to_indices.items()
        }
        batches: list[list[int]] = [[] for _ in range(self.num_batches)]

        # First reserve one slot per series in every batch. If a series has
        # fewer molecules than batches, cycle through it with replacement.
        for label in self.series:
            indices = shuffled_by_series[label]
            for batch_idx in range(self.num_batches):
                idx = indices[batch_idx % len(indices)]
                batches[batch_idx].append(idx)

        # Then place the not-yet-used molecules from larger series. This is the
        # coverage guarantee: every train bag appears at least once per epoch.
        remaining: list[int] = []
        for label in self.series:
            indices = shuffled_by_series[label]
            if len(indices) > self.num_batches:
                remaining.extend(indices[self.num_batches :])

        if remaining:
            order = torch.randperm(len(remaining), generator=self.generator).tolist()
            remaining = [remaining[i] for i in order]
            batch_order = torch.randperm(
                self.num_batches,
                generator=self.generator,
            ).tolist()
            cursor = 0
            for idx in remaining:
                placed = False
                for _ in range(self.num_batches):
                    batch_idx = batch_order[cursor % self.num_batches]
                    cursor += 1
                    if len(batches[batch_idx]) < self.batch_size:
                        batches[batch_idx].append(idx)
                        placed = True
                        break
                if not placed:
                    raise RuntimeError(
                        "SeriesBalancedBatchSampler could not place every "
                        "training bag while preserving per-batch series coverage"
                    )

        covered = {idx for batch in batches for idx in batch}
        missing = sorted(set(range(len(self.series_labels))) - covered)
        if missing:
            raise RuntimeError(
                "SeriesBalancedBatchSampler failed to cover all train bags; "
                f"first missing indices: {missing[:10]}"
            )

        pools = {
            label: list(indices)
            for label, indices in shuffled_by_series.items()
        }
        positions = {label: 0 for label in self.series}

        def next_index(label: str) -> int:
            if positions[label] >= len(pools[label]):
                pools[label] = self._shuffle_indices(self.series_to_indices[label])
                positions[label] = 0
            idx = pools[label][positions[label]]
            positions[label] += 1
            return idx

        for batch in batches:
            while len(batch) < self.batch_size:
                sampled = torch.multinomial(
                    self.series_probs,
                    num_samples=1,
                    replacement=True,
                    generator=self.generator,
                ).item()
                batch.append(next_index(self.series[int(sampled)]))

            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[i] for i in order]
