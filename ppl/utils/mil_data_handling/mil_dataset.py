"""PyTorch Dataset for multiple-instance learning (MIL).

This module provides the MILDataset class for handling multiple-instance learning data.
It supports both in-memory and on-demand loading modes to optimize memory usage.
"""
from __future__ import annotations

import logging
import pickle
import time
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ppl.utils.mil_data_handling.data_loader_impl import manage_cache

LOGGER = logging.getLogger(__name__)


class MILDataset(Dataset):
    """torch Dataset for Multiple Instance Learning.

    This dataset supports both in-memory and on-demand loading modes.
    In on-demand mode, bags are loaded from the disk when needed, reducing memory usage.

    Parameters
    ----------
    bags : Sequence[np.ndarray] or Dict[str, str] or str
        Either a sequence of numpy arrays (in-memory mode), a dictionary mapping
        bag IDs to file paths (on-demand mode), or a single file path containing all bags
    labels : np.ndarray
        Labels for each bag
    bag_ids : Sequence[str]
        Unique identifiers for each bag
    dtype : torch.dtype
        Data type for tensors
    cache_dir : Optional[str]
        Directory to cache processed bags (only used in on-demand mode)
    memory_limit : Optional[float]
        Maximum memory usage in GB (only used in on-demand mode)
    """

    def __init__(
        self,
        bags: Union[Sequence[np.ndarray], Dict[str, str], str],
        labels: np.ndarray,
        bag_ids: Sequence[str],
        *,
        cluster_ids: Optional[Sequence[np.ndarray]] = None,
        series_labels: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        memory_limit: Optional[float] = None,
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

        # Determine if we're in in-memory or on-demand mode
        self._on_demand = isinstance(bags, dict) or isinstance(bags, str)
        self._single_file_mode = isinstance(bags, str)

        if self._on_demand:
            # On-demand mode: store file paths
            self._bag_paths = bags
            self._cache = {}  # Cache for frequently accessed bags
            self._cache_dir = cache_dir
            self._memory_limit = memory_limit or 4.0  # Default 4GB limit
            self._last_access = {}  # Track when bags were last accessed

            if self._single_file_mode:
                LOGGER.info(f"MILDataset initialized in single-file mode with {len(bag_ids)} bags")
                # Load the data dictionary once to get the bag_ids mapping
                try:
                    with open(self._bag_paths, 'rb') as f:
                        data = pickle.load(f)
                    self._file_bag_ids = data["bag_ids"]
                    self._file_cluster_ids = data.get("cluster_ids")
                    file_series_labels = data.get("series_labels")
                    self._has_cluster_ids = self._file_cluster_ids is not None
                    if self._series_labels is None and file_series_labels is not None:
                        self._series_labels = [
                            str(label) for label in file_series_labels
                        ]
                    # Create a mapping from bag_id to index in the file
                    self._bag_id_to_idx = {bag_id: i for i, bag_id in enumerate(self._file_bag_ids)}
                    LOGGER.info(f"Loaded bag_ids mapping from {self._bag_paths}")
                except Exception as e:
                    LOGGER.error(f"Error loading bag_ids mapping from {self._bag_paths}: {e}")
                    self._file_bag_ids = []
                    self._bag_id_to_idx = {}
            else:
                LOGGER.info(f"MILDataset initialized in on-demand mode with {len(bag_ids)} bags")
        else:
            # In-memory mode: convert all bags to tensors
            start_time = time.time()
            LOGGER.info(f"Converting {len(bags)} bags to tensors...")
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
                    self._bag_ids,
                    self._bags,
                    self._cluster_ids,
                ):
                    if clusters.dim() != 1 or clusters.size(0) != bag.size(0):
                        raise ValueError(
                            f"Cluster IDs for bag {bag_id} must have shape "
                            f"[{bag.size(0)}], got {tuple(clusters.shape)}"
                        )
            else:
                self._cluster_ids = None
            elapsed = time.time() - start_time
            LOGGER.info(f"Tensor conversion completed in {elapsed:.2f} seconds")

            # Log memory usage
            mem_usage = sum(bag.element_size() * bag.nelement() for bag in self._bags) / (1024**3)
            LOGGER.info(f"MILDataset initialized in in-memory mode, using {mem_usage:.2f} GB of memory")

    def _load_bag(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Load a bag from a disk or cache."""
        bag_id = self._bag_ids[idx]

        # Check if a bag is in the cache
        if bag_id in self._cache:
            self._last_access[bag_id] = time.time()
            return self._cache[bag_id]

        try:
            if self._single_file_mode:
                # Load all bags from a single file
                with open(self._bag_paths, 'rb') as f:
                    data = pickle.load(f)

                # Get the bag index from the bag_id
                if bag_id in self._bag_id_to_idx:
                    bag_idx = self._bag_id_to_idx[bag_id]
                    bag_data = data["bags"][bag_idx]
                    cluster_data = (
                        data.get("cluster_ids", [None] * len(data["bags"]))[bag_idx]
                        if self._has_cluster_ids
                        else None
                    )
                else:
                    LOGGER.error(f"Bag ID {bag_id} not found in file {self._bag_paths}")
                    return torch.zeros((1, 1), dtype=self.dtype)
            else:
                # Load bag from individual file
                file_path = self._bag_paths[bag_id]
                with open(file_path, 'rb') as f:
                    bag_data = pickle.load(f)
                cluster_data = None

            bag_tensor = torch.as_tensor(bag_data, dtype=self.dtype)
            if self._has_cluster_ids and cluster_data is not None:
                cluster_tensor = torch.as_tensor(cluster_data, dtype=torch.long)
                cached_item = (bag_tensor, cluster_tensor)
            else:
                cached_item = bag_tensor

            # Add to the cache if we have space
            self._manage_cache()
            self._cache[bag_id] = cached_item
            self._last_access[bag_id] = time.time()

            return cached_item
        except Exception as e:
            file_path = self._bag_paths if self._single_file_mode else self._bag_paths[bag_id]
            LOGGER.error(f"Error loading bag {bag_id} from {file_path}: {e}")
            # Return an empty tensor as fallback
            return torch.zeros((1, 1), dtype=self.dtype)

    def _manage_cache(self) -> None:
        """Manage the cache to stay within memory limits."""
        manage_cache(self._cache, self._last_access, self._memory_limit)

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
        self._on_demand = False
        self._single_file_mode = False
        self._cache = {}
        self._last_access = {}

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
        if self._on_demand:
            loaded = self._load_bag(idx)
            if isinstance(loaded, tuple):
                bag, clusters = loaded
                if series_label is not None:
                    return (
                        bag,
                        self._labels[idx],
                        self._bag_ids[idx],
                        clusters,
                        series_label,
                    )
                return bag, self._labels[idx], self._bag_ids[idx], clusters
            bag = loaded
            if series_label is not None:
                return bag, self._labels[idx], self._bag_ids[idx], None, series_label
        else:
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

    def get_memory_usage(self) -> float:
        """Get the current memory usage of the dataset in GB."""
        if not self._on_demand:
            return sum(bag.element_size() * bag.nelement() for bag in self._bags) / (1024**3)
        else:
            total = 0
            for item in self._cache.values():
                bag = item[0] if isinstance(item, tuple) else item
                total += bag.element_size() * bag.nelement()
            return total / (1024**3)
