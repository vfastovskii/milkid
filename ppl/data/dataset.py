"""In-memory PyTorch Dataset for multiple-instance learning (MIL)."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class MILDataset(Dataset):
    """In-memory torch Dataset for Multiple Instance Learning.

    Parameters
    ----------
    bags : Sequence[np.ndarray]
        A sequence of per-bag instance arrays.
    labels : np.ndarray
        Label for each bag.
    bag_ids : Sequence[str]
        Unique identifier for each bag.
    cluster_ids : Optional[Sequence[np.ndarray]]
        Optional per-bag conformer cluster assignments.
    series_labels : Optional[Sequence[str]]
        Optional per-bag series label used for series-balanced batching.
    dtype : torch.dtype
        Data type for the bag tensors.
    """

    def __init__(
        self,
        bags: Sequence[np.ndarray],
        labels: np.ndarray,
        bag_ids: Sequence[str],
        *,
        cluster_ids: Optional[Sequence[np.ndarray]] = None,
        series_labels: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.dtype = dtype
        self._bag_ids = list(bag_ids)
        self._labels = torch.as_tensor(labels, dtype=torch.float32)
        self._has_cluster_ids = cluster_ids is not None
        if series_labels is not None:
            if len(series_labels) != len(self._bag_ids):
                raise ValueError(
                    "series_labels length must match number of bags: "
                    f"{len(series_labels)} != {len(self._bag_ids)}"
                )
            self._series_labels = [str(label) for label in series_labels]
        else:
            self._series_labels = None

        if isinstance(bags, (dict, str)):
            raise TypeError(
                "MILDataset is in-memory only; pass a sequence of bag arrays "
                "(disk caching was removed)."
            )

        # In-memory mode: convert all bags to tensors.
        self._bags = [torch.as_tensor(b, dtype=dtype) for b in bags]
        if cluster_ids is not None:
            if len(cluster_ids) != len(self._bags):
                raise ValueError(
                    "cluster_ids length must match number of bags: "
                    f"{len(cluster_ids)} != {len(self._bags)}"
                )
            self._cluster_ids = [
                torch.as_tensor(c, dtype=torch.long) for c in cluster_ids
            ]
            for bag_id, bag, clusters in zip(
                self._bag_ids, self._bags, self._cluster_ids
            ):
                if clusters.dim() != 1 or clusters.size(0) != bag.size(0):
                    raise ValueError(
                        f"Cluster IDs for bag {bag_id} must have shape "
                        f"[{bag.size(0)}], got {tuple(clusters.shape)}"
                    )
        else:
            self._cluster_ids = None

    # torch API
    def __len__(self) -> int:
        return len(self._bag_ids)

    def apply_instance_selection(
        self,
        selection_by_bag_id: Dict[str, Sequence[int]],
        *,
        require_complete: bool = True,
        min_selected_per_bag: Optional[int] = None,
        max_selected_per_bag: Optional[int] = None,
    ) -> Dict[str, float]:
        """Materialize a filtered in-memory dataset from per-bag instance indices."""
        dataset_keys = [str(bag_id) for bag_id in self._bag_ids]
        dataset_key_set = set(dataset_keys)
        selection_key_set = {str(bag_id) for bag_id in selection_by_bag_id}
        missing = [
            bag_id
            for bag_id in dataset_keys
            if bag_id not in selection_key_set
        ]
        extra = sorted(selection_key_set - dataset_key_set)

        if require_complete and (missing or extra):
            details = []
            if missing:
                details.append(
                    f"missing {len(missing)} train bags; first={missing[:10]}"
                )
            if extra:
                details.append(
                    f"unexpected {len(extra)} non-train bags; first={extra[:10]}"
                )
            raise ValueError(
                "Instance selection must cover the train dataset exactly: "
                + "; ".join(details)
            )
        if missing:
            LOGGER.warning(
                "[DATASET] Instance selection missing %d bags; keeping those bags unfiltered",
                len(missing),
            )
        if extra:
            LOGGER.warning(
                "[DATASET] Ignoring instance selections for %d non-train bags",
                len(extra),
            )

        new_bags = []
        new_cluster_ids = [] if self._has_cluster_ids else None
        selected_counts = []
        original_counts = []
        single_conformer_bags = 0

        for idx, bag_id in enumerate(self._bag_ids):
            item = self[idx]
            bag = item[0]
            cluster_ids = None
            if len(item) >= 4 and isinstance(item[3], torch.Tensor):
                cluster_ids = item[3]

            bag = torch.as_tensor(bag, dtype=self.dtype)
            original_count = int(bag.size(0))
            if original_count <= 0:
                raise ValueError(f"Bag {bag_id} has no instances")
            if original_count == 1:
                single_conformer_bags += 1

            raw_indices = selection_by_bag_id.get(str(bag_id))
            if raw_indices is None:
                indices = torch.arange(original_count, dtype=torch.long)
            else:
                valid_indices = sorted(
                    {
                        int(i)
                        for i in raw_indices
                        if 0 <= int(i) < original_count
                    }
                )
                if not valid_indices:
                    valid_indices = [0]
                indices = torch.as_tensor(valid_indices, dtype=torch.long)

            selected_count = int(indices.numel())
            if min_selected_per_bag is not None:
                expected_min = min(int(min_selected_per_bag), original_count)
                if selected_count < expected_min:
                    raise ValueError(
                        f"Bag {bag_id} selected {selected_count} instances, "
                        f"below expected minimum {expected_min}"
                    )
            if max_selected_per_bag is not None:
                expected_max = min(int(max_selected_per_bag), original_count)
                if selected_count > expected_max:
                    raise ValueError(
                        f"Bag {bag_id} selected {selected_count} instances, "
                        f"above expected maximum {expected_max}"
                    )

            new_bags.append(bag.index_select(0, indices).clone())
            if new_cluster_ids is not None:
                if cluster_ids is None:
                    new_cluster_ids.append(
                        torch.zeros(indices.numel(), dtype=torch.long)
                    )
                else:
                    cluster_ids = torch.as_tensor(cluster_ids, dtype=torch.long)
                    new_cluster_ids.append(
                        cluster_ids.index_select(0, indices).clone()
                    )
            selected_counts.append(int(indices.numel()))
            original_counts.append(original_count)

        self._bags = new_bags
        self._cluster_ids = new_cluster_ids
        self._has_cluster_ids = new_cluster_ids is not None

        total_original = int(sum(original_counts))
        total_selected = int(sum(selected_counts))
        kept_fraction = (
            float(total_selected / total_original)
            if total_original > 0
            else 0.0
        )
        LOGGER.info(
            "[DATASET] Applied instance selection: bags=%d instances=%d/%d "
            "kept=%.3f single_conformer_bags=%d",
            len(self._bag_ids),
            total_selected,
            total_original,
            kept_fraction,
            single_conformer_bags,
        )
        return {
            "bags": float(len(self._bag_ids)),
            "instances_original": float(total_original),
            "instances_selected": float(total_selected),
            "kept_fraction": kept_fraction,
            "mean_selected_per_bag": float(np.mean(selected_counts))
            if selected_counts
            else 0.0,
            "single_conformer_bags": float(single_conformer_bags),
        }

    def _series_label(self, idx: int) -> Optional[str]:
        if self._series_labels is None:
            return None
        return self._series_labels[idx]

    def __getitem__(self, idx: int):
        series_label = self._series_label(idx)
        bag = self._bags[idx]
        if self._cluster_ids is not None:
            if series_label is not None:
                return (
                    bag,
                    self._labels[idx],
                    self._bag_ids[idx],
                    self._cluster_ids[idx],
                    series_label,
                )
            return bag, self._labels[idx], self._bag_ids[idx], self._cluster_ids[idx]
        if series_label is not None:
            return bag, self._labels[idx], self._bag_ids[idx], None, series_label
        return bag, self._labels[idx], self._bag_ids[idx]
