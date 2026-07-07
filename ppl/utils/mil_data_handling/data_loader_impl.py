"""Implementation details for the MIL data loading functionality.

This module contains helper functions and implementation details for the
MILDataModule and MILDataset classes defined in data_loader.py.
"""
import gc
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from ppl.utils.modelling_configs.data_loader_config import DataLoaderConfig

LOGGER = logging.getLogger(__name__)


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


def _make_agglomerative_clustering(**kwargs) -> AgglomerativeClustering:
    """Create AgglomerativeClustering across sklearn metric/affinity APIs."""
    try:
        return AgglomerativeClustering(**kwargs)
    except TypeError as exc:
        if "metric" not in kwargs:
            raise
        legacy_kwargs = dict(kwargs)
        legacy_kwargs["affinity"] = legacy_kwargs.pop("metric")
        try:
            return AgglomerativeClustering(**legacy_kwargs)
        except TypeError:
            raise exc


def _normalize_cluster_labels(labels: np.ndarray) -> np.ndarray:
    """Remap arbitrary cluster labels to contiguous int64 IDs starting at 0."""
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64, copy=False)


def _fit_agglomerative_labels(
    bag: np.ndarray,
    *,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: str,
    metric: str,
) -> np.ndarray:
    """Fit agglomerative clustering for one molecule bag."""
    kwargs = {
        "linkage": linkage,
        "metric": metric,
    }
    if distance_threshold is not None:
        kwargs["n_clusters"] = None
        kwargs["distance_threshold"] = float(distance_threshold)
    else:
        kwargs["n_clusters"] = int(n_clusters)

    model = _make_agglomerative_clustering(**kwargs)
    return _normalize_cluster_labels(
        model.fit_predict(bag.astype(np.float32, copy=False))
    )


def _choose_silhouette_cluster_count(
    bag: np.ndarray,
    *,
    min_clusters: int,
    max_clusters: int,
    min_silhouette: float,
    linkage: str,
    metric: str,
) -> int:
    """Choose a per-bag cluster count using silhouette in descriptor space."""
    n = int(bag.shape[0])
    min_k = max(1, min(int(min_clusters), n))
    max_k = max(1, min(int(max_clusters), n))

    if n <= 1 or max_k <= 1:
        return 1

    # Silhouette is only defined for 2 <= n_labels <= n_samples - 1.
    candidate_max = min(max_k, n - 1)
    candidate_min = max(2, min_k)
    if candidate_max < candidate_min:
        return min_k

    # If all instances are effectively identical, forcing multiple clusters
    # only creates artificial KID structure.
    if np.allclose(np.var(bag, axis=0).sum(), 0.0):
        return min_k

    best_k = min_k
    best_score = -np.inf

    for k in range(candidate_min, candidate_max + 1):
        labels = _fit_agglomerative_labels(
            bag,
            n_clusters=k,
            linkage=linkage,
            metric=metric,
        )
        n_labels = int(np.unique(labels).size)
        if n_labels < 2 or n_labels >= n:
            continue
        score = float(silhouette_score(bag, labels, metric=metric))
        if score > best_score:
            best_score = score
            best_k = k

    if min_k <= 1 and best_score < float(min_silhouette):
        return 1

    return int(best_k)


def cluster_config_signature(cfg: DataLoaderConfig) -> Dict[str, object]:
    """Return the clustering settings that affect cached cluster labels."""
    return {
        "cluster_instances": bool(getattr(cfg, "cluster_instances", False)),
        "cluster_selection_method": str(
            getattr(cfg, "cluster_selection_method", "silhouette")
        ).lower(),
        "cluster_max_clusters": int(getattr(cfg, "cluster_max_clusters", 6)),
        "cluster_min_clusters": int(getattr(cfg, "cluster_min_clusters", 1)),
        "cluster_min_silhouette": float(
            getattr(cfg, "cluster_min_silhouette", 0.05)
        ),
        "cluster_distance_threshold": getattr(
            cfg,
            "cluster_distance_threshold",
            None,
        ),
        "cluster_linkage": str(getattr(cfg, "cluster_linkage", "average")),
        "cluster_metric": str(getattr(cfg, "cluster_metric", "euclidean")),
        "cluster_feature_space": "per_fingerprint_scaled_no_sqrt",
    }


def _seed_worker(worker_id: int) -> None:
    """Seed NumPy/Python RNGs inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_mil(batch):
    """Collate function for MIL batches - returns padded tensor format (padding before embedding)."""
    if not batch:
        raise ValueError("collate_mil received an empty batch")

    item_len = len(batch[0])
    series_labels = None
    if item_len == 3:
        bags, labels, bag_ids = zip(*batch)
        cluster_ids = None
    elif item_len == 4:
        bags, labels, bag_ids, cluster_ids = zip(*batch)
    elif item_len == 5:
        bags, labels, bag_ids, cluster_ids, series_labels = zip(*batch)
        series_labels = [str(label) for label in series_labels]
    else:
        raise ValueError(
            "collate_mil expects dataset items of length 3, 4, or 5, "
            f"got {item_len}"
        )
    if cluster_ids is not None and all(cluster is None for cluster in cluster_ids):
        cluster_ids = None
    
    # Log batch formation details
    batch_size = len(bags)
    LOGGER.debug(f"[BATCH_FORMATION] Processing batch with {batch_size} bags")
    
    # Convert bags to tensors if they aren't already
    bag_tensors = []
    conversion_stats = {'already_tensors': 0, 'converted': 0}
    
    for i, bag in enumerate(bags):
        if isinstance(bag, torch.Tensor):
            bag_tensor = bag.float()
            conversion_stats['already_tensors'] += 1
        else:
            bag_tensor = torch.tensor(bag, dtype=torch.float32)
            conversion_stats['converted'] += 1

        if bag_tensor.dim() != 2:
            raise ValueError(
                f"Bag {bag_ids[i]} must have shape [num_instances, num_features], "
                f"got {tuple(bag_tensor.shape)}"
            )
        if bag_tensor.size(0) == 0:
            raise ValueError(f"Bag {bag_ids[i]} contains no instances")
        bag_tensors.append(bag_tensor)
        
        # Log individual bag details
        LOGGER.debug(f"[BATCH_FORMATION] Bag {i} (ID: {bag_ids[i]}): shape={bag_tensor.shape}, "
                    f"dtype={bag_tensor.dtype}, device={bag_tensor.device}")
    
    # Calculate batch statistics
    bag_lengths = [bag.shape[0] for bag in bag_tensors]
    max_instances = max(bag_lengths)
    min_instances = min(bag_lengths)
    total_instances = sum(bag_lengths)
    avg_instances = total_instances / batch_size
    feature_dim = bag_tensors[0].shape[1] if bag_tensors else 0
    for bag_id, bag_tensor in zip(bag_ids, bag_tensors):
        if bag_tensor.shape[1] != feature_dim:
            raise ValueError(
                f"Bag {bag_id} has feature_dim={bag_tensor.shape[1]}, "
                f"expected {feature_dim}"
            )
    
    # Calculate padding overhead
    total_padded_positions = batch_size * max_instances
    padding_overhead = total_padded_positions - total_instances
    padding_efficiency = (total_instances / total_padded_positions) * 100 if total_padded_positions > 0 else 100
    
    # Log comprehensive batch statistics
    LOGGER.debug(f"[BATCH_FORMATION] Batch statistics:")
    LOGGER.debug(f"  - Batch size: {batch_size}")
    LOGGER.debug(f"  - Feature dimension: {feature_dim}")
    LOGGER.debug(f"  - Bag lengths: min={min_instances}, max={max_instances}, avg={avg_instances:.1f}")
    LOGGER.debug(f"  - Total instances: {total_instances}")
    LOGGER.debug(f"  - Padding overhead: {padding_overhead} positions ({(padding_overhead/total_padded_positions*100):.1f}% padding)")
    LOGGER.debug(f"  - Tensor conversions: {conversion_stats['already_tensors']} already tensors, "
                f"{conversion_stats['converted']} converted")
    
    # Create padded tensor and mask (padding before embedding)
    padded_bags = torch.zeros(batch_size, max_instances, feature_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_instances, dtype=torch.bool)  # True = padding
    padded_cluster_ids = None
    if cluster_ids is not None:
        padded_cluster_ids = torch.full(
            (batch_size, max_instances),
            -1,
            dtype=torch.long,
        )
    
    # Fill padded tensor and create mask
    for i, bag in enumerate(bag_tensors):
        bag_len = bag.shape[0]
        padded_bags[i, :bag_len, :] = bag
        padding_mask[i, :bag_len] = False  # False = real data, True = padding
        if padded_cluster_ids is not None:
            clusters = torch.as_tensor(cluster_ids[i], dtype=torch.long).flatten()
            if clusters.numel() != bag_len:
                raise ValueError(
                    f"Cluster IDs for bag {bag_ids[i]} must have length {bag_len}, "
                    f"got {clusters.numel()}"
                )
            padded_cluster_ids[i, :bag_len] = clusters
    
    # Log memory implications
    if batch_size > 0:
        padded_memory_mb = (padded_bags.numel() * 4) / (1024 * 1024)  # 4 bytes per float32
        mask_memory_mb = (padding_mask.numel() * 1) / (1024 * 1024)  # 1 byte per bool
        total_memory_mb = padded_memory_mb + mask_memory_mb
        LOGGER.debug(f"  - Memory usage: {padded_memory_mb:.2f} MB padded data + {mask_memory_mb:.2f} MB mask = {total_memory_mb:.2f} MB total")
        LOGGER.debug(f"  - Padding approach: before embedding (simpler, all instances processed through embedder)")
    
    label_tensor = torch.stack(
        [torch.as_tensor(label, dtype=torch.float32).reshape(()) for label in labels]
    )

    if padded_cluster_ids is not None:
        if series_labels is not None:
            return (
                padded_bags,
                label_tensor,
                list(bag_ids),
                padding_mask,
                padded_cluster_ids,
                series_labels,
            )
        return padded_bags, label_tensor, list(bag_ids), padding_mask, padded_cluster_ids

    if series_labels is not None:
        return padded_bags, label_tensor, list(bag_ids), padding_mask, None, series_labels

    return padded_bags, label_tensor, list(bag_ids), padding_mask




def validate_dataframe(df: pd.DataFrame, cfg: DataLoaderConfig) -> pd.DataFrame:
    """Checks for NaNs and non-numeric descriptors."""
    required_cols = {cfg.bag_id_col, cfg.inst_id_col, cfg.endpoint_value_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # Drop rows with NaN labels or bag/instance IDs
    n_before = len(df)
    df = df.dropna(subset=[cfg.bag_id_col, cfg.inst_id_col, cfg.endpoint_value_col])
    n_dropped = n_before - len(df)
    if n_dropped:
        LOGGER.warning("Dropped %d rows with NaN in required columns", n_dropped)

    # Ensure descriptor columns are numeric and NaN-free
    if cfg.descriptor_cols is not None:
        non_num = [c for c in cfg.descriptor_cols
                   if not pd.api.types.is_numeric_dtype(df[c])]
        if non_num:
            raise TypeError(f"Descriptor columns must be numeric: {non_num}")

        if df[cfg.descriptor_cols].isna().any().any():
            bad = df[cfg.descriptor_cols].isna().sum().sum()
            raise ValueError(f"{bad} NaNs found in descriptor columns")

    return df


def build_bags(df: pd.DataFrame, cfg: DataLoaderConfig, split_name: str = "") -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """Build bags from instances in the dataframe.

    Implements experimental pose handling:
    - Train: remove instances whose instance IDs contain "_experimental_pose".
    - Val/Test: duplicate each original bag into two derived bags:
        • the original bag_id containing ALL instances (including experimental poses)
        • a duplicate without experimental poses (suffix "__noexp")
      Special case: if a bag contains exactly one instance and it is experimental, still create
      the "__noexp" duplicate and include the same (experimental) instance in that copy.
      Validation loss should be computed only on "__noexp" bags (handled downstream),
      while logging will include all val bags.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing instances
    cfg : DataLoaderConfig
        Configuration for data loading
    split_name : str
        Name of the split for progress reporting

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray, List[str]]
        Bags, labels, and bag IDs
    """
    if df.empty:
        return [], np.array([], dtype=np.float32), []

    bags, labels, ids = [], [], []

    # Get unique bag IDs for progress reporting
    unique_bag_ids = df[cfg.bag_id_col].unique()
    LOGGER.info(f"Building {len(unique_bag_ids)} bags for {split_name} split")

    # Precompute experimental pose mask column name
    inst_col = cfg.inst_id_col
    if inst_col not in df.columns:
        raise KeyError(f"Instance ID column '{inst_col}' not found in dataframe")

    # Group by bag ID with progress reporting
    groups = df.groupby(cfg.bag_id_col, sort=False)
    for mol_id, sub in tqdm.tqdm(groups, total=len(unique_bag_ids), desc=f"Building {split_name} bags", disable=len(unique_bag_ids) < 10):
        try:
            # Determine experimental vs non-experimental instances within this bag
            inst_series = sub[inst_col].astype(str)
            exp_mask = inst_series.str.contains("_experimental_pose", na=False)
            nonexp_mask = ~exp_mask

            # Extract label for this original bag
            tgt_raw = sub[cfg.endpoint_value_col].iat[0]
            label_val = float(int(tgt_raw)) if cfg.task == "classification" else float(tgt_raw)

            if split_name.lower() == "train":
                # TRAIN: prefer non-experimental instances; if none exist, fall back to using all instances
                sub_kept = sub[nonexp_mask]
                if sub_kept.empty:
                    # Fallback: retain the bag using all instances to avoid silently dropping bags
                    LOGGER.warning(f"[build_bags] Train bag {mol_id} has only experimental instances; using all instances for training")
                    bag_data = sub[cfg.descriptor_cols].to_numpy(dtype=np.float32)
                else:
                    bag_data = sub_kept[cfg.descriptor_cols].to_numpy(dtype=np.float32)
                if not np.isfinite(bag_data).all():
                    LOGGER.warning(f"Non-finite values found in bag {mol_id}, replacing with zeros")
                    bag_data = np.nan_to_num(bag_data, nan=0.0, posinf=0.0, neginf=0.0)
                bags.append(bag_data)
                labels.append(label_val)
                ids.append(str(mol_id))
            else:
                # VAL/TEST: create two bags per original where applicable
                # 1) Full bag with ALL original instances (keeps experimental poses)
                bag_full = sub[cfg.descriptor_cols].to_numpy(dtype=np.float32)
                if not np.isfinite(bag_full).all():
                    LOGGER.warning(f"Non-finite values found in full bag {mol_id}, replacing with zeros")
                    bag_full = np.nan_to_num(bag_full, nan=0.0, posinf=0.0, neginf=0.0)
                bags.append(bag_full)
                labels.append(label_val)
                ids.append(f"{mol_id}")

                # 2) Always create a '__noexp' duplicate for validation/test
                if nonexp_mask.any():
                    sub_nonexp = sub[nonexp_mask]
                    bag_nonexp = sub_nonexp[cfg.descriptor_cols].to_numpy(dtype=np.float32)
                    if not np.isfinite(bag_nonexp).all():
                        LOGGER.warning(f"Non-finite values found in bag {mol_id}__noexp, replacing with zeros")
                        bag_nonexp = np.nan_to_num(bag_nonexp, nan=0.0, posinf=0.0, neginf=0.0)
                    bags.append(bag_nonexp)
                    labels.append(label_val)
                    ids.append(f"{mol_id}__noexp")
                else:
                    # No non-experimental instances available; duplicate full bag under '__noexp'
                    bags.append(bag_full.copy())
                    labels.append(label_val)
                    ids.append(f"{mol_id}__noexp")
        except Exception as e:
            LOGGER.error(f"Error processing bag {mol_id}: {e}")
            # Skip this bag
            continue

    if not bags:
        LOGGER.warning(f"No valid bags found for {split_name} split")
        return [], np.array([], dtype=np.float32), []

    return bags, np.asarray(labels, dtype=np.float32), ids


def build_bag_series_labels(
    df: pd.DataFrame,
    cfg: DataLoaderConfig,
    bag_ids: Sequence[str],
    split_name: str = "",
) -> Optional[List[str]]:
    """Resolve one series label per bag ID, preserving dataset order."""
    series_col = getattr(cfg, "series_col", None)
    if not series_col:
        return None
    if series_col not in df.columns:
        LOGGER.warning(
            "[SERIES_BATCH] series_col=%s not found in %s split; disabling series labels",
            series_col,
            split_name,
        )
        return None

    def base_bag_id(bag_id) -> str:
        bag_id = str(bag_id)
        return bag_id[: -len("__noexp")] if bag_id.endswith("__noexp") else bag_id

    series_by_bag = (
        df.drop_duplicates(cfg.bag_id_col)
        .assign(_bag_id_key=lambda d: d[cfg.bag_id_col].astype(str))
        .set_index("_bag_id_key")[series_col]
    )

    labels: List[str] = []
    missing: list[str] = []
    for bag_id in bag_ids:
        key = base_bag_id(bag_id)
        if key in series_by_bag.index:
            value = series_by_bag.loc[key]
            labels.append("UNKNOWN" if pd.isna(value) else str(value))
        else:
            missing.append(key)
            labels.append("UNKNOWN")

    if missing:
        LOGGER.warning(
            "[SERIES_BATCH] Missing series labels for %d %s bags; first missing IDs: %s",
            len(missing),
            split_name,
            missing[:5],
        )

    if labels:
        values, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
        summary = ", ".join(
            f"{value}:{count}" for value, count in zip(values.tolist(), counts.tolist())
        )
        LOGGER.info(
            "[SERIES_BATCH] %s split series distribution: %s",
            split_name,
            summary,
        )

    return labels


def log_dataset_stats(tag: str,
                       splits: dict[str, Tuple[List[np.ndarray], np.ndarray]],
                       *,
                       task: str) -> None:
    """
    Log bag-level and feature-level diagnostics for each split.

    Parameters
    ----------
    tag
        A label like `"raw"` or `"scaled"`.
    splits
        Dict  {"train": (bags, y), "val": (bags, y), "test": (bags, y)}
        Bags  : list of np.ndarray  [N, D]
        y     : np.ndarray  [n_bags]
    task
        `"classification"` or `"regression"` – controls label histogram.

    Notes
    -----
    • μ/σ are averaged over descriptor dimensions to keep the log short.
    • For classification report class counts; for regression mean±std.
    • For predefined_split=True, train bags go to CV (train/val), test bags become heldout test set.
    """
    # Log summary of all splits for easier comparison
    LOGGER.info(f"[{tag}] Dataset split summary:")
    for split_name, (bags, _) in splits.items():
        if not len(bags):  # empty val set (final model / no val_partition)
            continue
        LOGGER.info(f"[{tag}] - {split_name}: {len(bags)} bags")

    # If we have both validation and test sets, this indicates we're using CV with predefined split
    for split, (bags, y) in splits.items():
        if not len(bags):               # empty val set (final model / no val_partition)
            continue

        n_bags = len(bags)
        lens   = np.fromiter((len(b) for b in bags), dtype=int)
        inst_tot = lens.sum()

        mat = np.vstack(bags)
        mu_feat = mat.mean(axis=0).mean()      # mean over dims THEN average
        sd_feat = mat.std(axis=0).mean()

        if task == "classification":
            unique, counts = np.unique(y.astype(int), return_counts=True)
            cls_hist = "/".join(f"{u}:{c}" for u, c in zip(unique, counts))
            lbl_info = f"class hist [{cls_hist}]"
        else:  # regression
            lbl_info = f"labels μ={y.mean():.3f} σ={y.std():.3f}"

        LOGGER.info(
            "[%s] %-5s  bags=%4d  inst=%6d  "
            "bag_len μ=%.1f σ=%.1f  "
            "descriptors μ=%.5f σ=%.5f  %s",
            tag, split, n_bags, inst_tot,
            lens.mean(), lens.std(),
            mu_feat, sd_feat, lbl_info
        )


def cluster_bags(
    bags: Sequence[np.ndarray],
    cfg: DataLoaderConfig,
    split_name: str = "",
) -> List[np.ndarray]:
    """Cluster conformers inside each bag in the scaled descriptor space."""
    selection_method = str(
        getattr(cfg, "cluster_selection_method", "silhouette")
    ).lower()
    max_clusters = int(getattr(cfg, "cluster_max_clusters", 6))
    min_clusters = int(getattr(cfg, "cluster_min_clusters", 1))
    min_silhouette = float(getattr(cfg, "cluster_min_silhouette", 0.05))
    distance_threshold = getattr(cfg, "cluster_distance_threshold", None)
    linkage = str(getattr(cfg, "cluster_linkage", "average"))
    metric = str(getattr(cfg, "cluster_metric", "euclidean"))

    if max_clusters <= 0 and distance_threshold is None:
        raise ValueError(
            "cluster_max_clusters must be positive unless cluster_distance_threshold is set"
        )
    if min_clusters <= 0:
        raise ValueError(f"cluster_min_clusters must be positive, got {min_clusters}")
    if max_clusters > 0 and min_clusters > max_clusters:
        raise ValueError(
            "cluster_min_clusters cannot exceed cluster_max_clusters: "
            f"{min_clusters} > {max_clusters}"
        )

    if distance_threshold is not None:
        selection_method = "distance"
    if selection_method not in {"silhouette", "distance", "fixed"}:
        raise ValueError(
            "cluster_selection_method must be one of silhouette, distance, fixed; "
            f"got {selection_method!r}"
        )
    if selection_method == "distance" and distance_threshold is None:
        raise ValueError(
            "cluster_distance_threshold must be set when "
            "cluster_selection_method='distance'"
        )

    cluster_ids: List[np.ndarray] = []
    cluster_counts: List[int] = []

    LOGGER.info(
        "[CLUSTER] %s split: method=%s min_clusters=%d max_clusters=%d "
        "distance_threshold=%s linkage=%s metric=%s",
        split_name,
        selection_method,
        min_clusters,
        max_clusters,
        distance_threshold,
        linkage,
        metric,
    )

    for bag in tqdm.tqdm(
        bags,
        desc=f"Clustering {split_name} bags",
        disable=len(bags) < 10,
    ):
        n = int(bag.shape[0])
        if n <= 1:
            labels = np.zeros(n, dtype=np.int64)
        else:
            try:
                max_k = min(max(1, max_clusters), n)
                min_k = min(max(1, min_clusters), max_k)

                if selection_method == "distance":
                    labels = _fit_agglomerative_labels(
                        bag,
                        distance_threshold=float(distance_threshold),
                        linkage=linkage,
                        metric=metric,
                    )
                    found_k = int(labels.max() + 1) if labels.size else 0
                    if max_clusters > 0 and found_k > max_k:
                        labels = _fit_agglomerative_labels(
                            bag,
                            n_clusters=max_k,
                            linkage=linkage,
                            metric=metric,
                        )
                else:
                    if selection_method == "silhouette":
                        k = _choose_silhouette_cluster_count(
                            bag,
                            min_clusters=min_k,
                            max_clusters=max_k,
                            min_silhouette=min_silhouette,
                            linkage=linkage,
                            metric=metric,
                        )
                    else:
                        k = max_k

                    if k <= 1:
                        labels = np.zeros(n, dtype=np.int64)
                    else:
                        labels = _fit_agglomerative_labels(
                            bag,
                            n_clusters=k,
                            linkage=linkage,
                            metric=metric,
                        )
            except Exception as exc:
                LOGGER.warning(
                    "[CLUSTER] Failed to cluster bag with %d instances in %s split: %s. "
                    "Falling back to one cluster.",
                    n,
                    split_name,
                    exc,
                )
                labels = np.zeros(n, dtype=np.int64)

        cluster_ids.append(labels)
        cluster_counts.append(int(labels.max() + 1) if labels.size else 0)

    if cluster_counts:
        unique_counts, unique_freq = np.unique(cluster_counts, return_counts=True)
        count_hist = ", ".join(
            f"{int(k)}:{int(v)}" for k, v in zip(unique_counts, unique_freq)
        )
        LOGGER.info(
            "[CLUSTER] %s split: clustered %d bags; clusters/bag min=%d max=%d "
            "mean=%.2f hist={%s}",
            split_name,
            len(cluster_counts),
            int(np.min(cluster_counts)),
            int(np.max(cluster_counts)),
            float(np.mean(cluster_counts)),
            count_hist,
        )

    return cluster_ids


def make_dataloader(ds, cfg: DataLoaderConfig, *, shuffle: bool) -> DataLoader:
    """Create a DataLoader with optimized settings for the current hardware.

    Parameters
    ----------
    ds : Dataset
        Dataset to load
    cfg : DataLoaderConfig
        Configuration for data loading
    shuffle : bool
        Whether to shuffle the data

    Returns
    -------
    DataLoader
        Configured DataLoader
    """
    # Detect available hardware
    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )

    # Optimize settings based on hardware
    pin_memory = cfg.pin_memory and device.type == "cuda"

    # Validate and adjust num_workers
    cpu_count = os.cpu_count() or 4
    num_workers = min(cfg.num_workers, cpu_count)
    if cfg.num_workers > cpu_count:
        LOGGER.warning(
            f"Reducing num_workers from {cfg.num_workers} to {num_workers} "
            f"to match available CPU cores"
        )

    # Validate batch size
    if cfg.batch_size <= 0:
        LOGGER.warning(f"Invalid batch size {cfg.batch_size}, using default of 1")
        batch_size = 1
    else:
        batch_size = cfg.batch_size

    # Persistent workers and timeout are only valid when worker processes exist.
    use_persistent = num_workers > 0 and (len(ds) > 100 or num_workers > 1)
    loader_timeout = 120 if num_workers > 0 else 0

    generator_seed = int(getattr(cfg, "seed", 42))
    generator = torch.Generator()
    generator.manual_seed(generator_seed)

    if shuffle:
        LOGGER.info(
            "Using seeded DataLoader generator for shuffled training batches: seed=%d",
            generator_seed,
        )

    # Create and return the DataLoader
    try:
        series_labels = getattr(ds, "_series_labels", None)
        use_series_balanced_batches = (
            shuffle
            and bool(getattr(cfg, "balance_train_batches_by_series", False))
            and series_labels is not None
        )
        if use_series_balanced_batches:
            batch_sampler = SeriesBalancedBatchSampler(
                series_labels,
                batch_size=batch_size,
                seed=generator_seed,
            )
            LOGGER.info(
                "[SERIES_BATCH] Using balanced train batch sampler: "
                "batch_size=%d, series=%d, batches_per_epoch=%d, "
                "coverage=all_train_bags",
                batch_size,
                len(batch_sampler.series),
                len(batch_sampler),
            )
            loader = DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_mil,
                worker_init_fn=_seed_worker,
                persistent_workers=use_persistent,
                prefetch_factor=2 if num_workers > 0 else None,
                timeout=loader_timeout,
            )
            LOGGER.info(
                f"Created series-balanced DataLoader with batch_size={batch_size}, "
                f"num_workers={num_workers}, pin_memory={pin_memory}, "
                f"persistent_workers={use_persistent}, device={device.type}"
            )
            return loader

        if shuffle and bool(getattr(cfg, "balance_train_batches_by_series", False)):
            LOGGER.warning(
                "[SERIES_BATCH] balance_train_batches_by_series=True but dataset has no series labels; "
                "falling back to standard shuffled DataLoader"
            )

        # Explicitly use a SequentialSampler for eval loaders to prevent any shuffling
        sampler = None if shuffle else SequentialSampler(ds)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_mil,
            generator=generator,
            worker_init_fn=_seed_worker,
            persistent_workers=use_persistent,
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
            timeout=loader_timeout,  # 2-minute timeout for worker initialization
        )
        LOGGER.info(
            f"Created DataLoader with batch_size={batch_size}, num_workers={num_workers}, "
            f"pin_memory={pin_memory}, persistent_workers={use_persistent}, "
            f"device={device.type}"
        )
        return loader
    except Exception as e:
        LOGGER.error(f"Failed to create DataLoader: {e}")
        if shuffle and bool(getattr(cfg, "balance_train_batches_by_series", False)):
            raise
        # Fallback to a simple DataLoader with minimal settings
        LOGGER.warning("Using fallback DataLoader with minimal settings")
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=shuffle,
            num_workers=0,
            generator=generator,
            collate_fn=collate_mil,
        )


def save_bags_to_disk(
    bags: List[np.ndarray],
    bag_ids: List[str],
    split: str,
    cache_dir: Path,
    cluster_ids: Optional[List[np.ndarray]] = None,
    cluster_config: Optional[Dict[str, object]] = None,
    series_labels: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Save bags to disk for on-demand loading.

    Parameters
    ----------
    bags : List[np.ndarray]
        List of bags to save
    bag_ids : List[str]
        List of bag IDs
    split : str
        Dataset split name (train, val, test)
    cache_dir : Path
        Directory to save bags to

    Returns
    -------
    Dict[str, str]
        Mapping from bag IDs to file paths
    """
    if not cache_dir:
        raise ValueError("Cache directory not specified")

    LOGGER.info(f"Saving bags to disk: split={split}")

    split_dir = cache_dir / split
    LOGGER.info(f"Creating directory for {split} data: {split_dir}")

    split_dir.mkdir(parents=True, exist_ok=True)

    # Save all bags to a single file
    file_path = split_dir / f"{split}.pkl"
    LOGGER.info(f"Saving {len(bags)} bags to {file_path}")

    # Create a dictionary with all the data
    data = {
        "bags": bags,
        "bag_ids": bag_ids
    }
    if cluster_ids is not None:
        if len(cluster_ids) != len(bags):
            raise ValueError(
                "cluster_ids length must match bags length when saving: "
                f"{len(cluster_ids)} != {len(bags)}"
            )
        data["cluster_ids"] = cluster_ids
    if cluster_config is not None:
        data["cluster_config"] = dict(cluster_config)
    if series_labels is not None:
        if len(series_labels) != len(bags):
            raise ValueError(
                "series_labels length must match bags length when saving: "
                f"{len(series_labels)} != {len(bags)}"
            )
        data["series_labels"] = [str(label) for label in series_labels]

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    # Return a dictionary mapping bag IDs to the single file path
    # This is for compatibility with the existing code
    return {bag_id: str(file_path) for bag_id in bag_ids}


def manage_cache(cache, last_access, memory_limit):
    """Manage the cache to stay within memory limits."""
    if not memory_limit:
        return

    # Check current memory usage
    process = psutil.Process(os.getpid())
    current_memory_gb = process.memory_info().rss / (1024**3)

    # If we're over the limit, remove least recently used items
    if current_memory_gb > memory_limit:
        LOGGER.debug(f"Memory usage ({current_memory_gb:.2f} GB) exceeds limit ({memory_limit:.2f} GB), cleaning cache")

        # Sort cache items by last access time
        sorted_items = sorted(last_access.items(), key=lambda x: x[1])

        # Remove items until we're under the limit or the cache is empty
        for bag_id, _ in sorted_items:
            if bag_id in cache:
                del cache[bag_id]
                del last_access[bag_id]

                # Check if we're under the limit
                current_memory_gb = process.memory_info().rss / (1024**3)
                if current_memory_gb < memory_limit * 0.8:  # 20% buffer
                    break

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
