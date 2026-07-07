"""Implementation details for the MILDataModule class.

This module contains the implementation details for the MILDataModule class,
including data loading, preprocessing, and dataset creation.
"""
import logging
import math
import os
import pickle
import re
import time
from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from ppl.utils.mil_data_handling.data_loader_impl import cluster_config_signature
from ppl.utils.mil_data_handling.mil_dataset import MILDataset
from ppl.utils.modelling_configs.data_loader_config import DataLoaderConfig

LOGGER = logging.getLogger(__name__)


def _fingerprint_block_name(feature_name: str) -> str:
    """Infer a fingerprint block name from descriptor column basename.

    Expected columns look like ``GETAWAYFingerprint_42``. If a column does not
    end with a numeric suffix, the full name is treated as its own block.
    """
    match = re.match(r"^(?P<base>.+)_\d+$", str(feature_name))
    return match.group("base") if match else str(feature_name)


def _feature_blocks(feature_names: list[str]) -> "OrderedDict[str, list[int]]":
    blocks: "OrderedDict[str, list[int]]" = OrderedDict()
    for idx, name in enumerate(feature_names):
        blocks.setdefault(_fingerprint_block_name(name), []).append(idx)
    return blocks


def _format_block_sizes(feature_names: list[str]) -> str:
    blocks = _feature_blocks(feature_names)
    if not blocks:
        return "none"
    return ", ".join(f"{name}:{len(indices)}" for name, indices in blocks.items())


def _log_descriptor_blocks(stage: str, feature_names: list[str]) -> None:
    LOGGER.info(
        "[FEATURE_BLOCKS] %s total_features=%d blocks={%s}",
        stage,
        len(feature_names),
        _format_block_sizes(feature_names),
    )


def _transform_bags_by_fingerprint_blocks(
    *,
    train_bags: list[np.ndarray],
    val_bags: list[np.ndarray],
    test_bags: list[np.ndarray],
    feature_names: list[str],
) -> dict:
    """Fit one StandardScaler per fingerprint block on train instances only.

    This stage intentionally does not apply block-size normalization. Clustering
    uses this scaled-only descriptor space.
    """
    if not train_bags:
        raise ValueError("Cannot fit feature block scalers without training bags")

    blocks = _feature_blocks(feature_names)
    train_fit_data = np.vstack(train_bags).astype(np.float32, copy=False)
    scalers: dict[str, StandardScaler] = {}
    block_sizes = {name: len(indices) for name, indices in blocks.items()}

    LOGGER.info(
        "[FEATURE_SCALE] fit=train_only scaler=StandardScaler "
        "normalization=none stage=scale_for_clustering train_bags=%d "
        "train_instances=%d blocks=%d features=%d",
        len(train_bags),
        train_fit_data.shape[0],
        len(blocks),
        train_fit_data.shape[1],
    )
    LOGGER.info(
        "[FEATURE_SCALE] blocks_after_zero_var={%s}",
        ", ".join(
            f"{name}:n={size},scale=standardized"
            for name, size in block_sizes.items()
        ),
    )

    for block_name, indices in blocks.items():
        idx = np.asarray(indices, dtype=np.int64)
        scaler = StandardScaler().fit(train_fit_data[:, idx])
        scalers[block_name] = scaler
        LOGGER.info(
            "[FEATURE_SCALE] fitted block=%s features=%d train_mean_abs=%.6g "
            "train_scale_mean=%.6g",
            block_name,
            len(indices),
            float(np.abs(scaler.mean_).mean()),
            float(np.asarray(scaler.scale_).mean()),
        )

    def transform_split(split_name: str, bags: list[np.ndarray]) -> None:
        if not bags:
            LOGGER.info("[FEATURE_SCALE] transform split=%s skipped_empty", split_name)
            return
        n_instances = int(sum(len(bag) for bag in bags))
        LOGGER.info(
            "[FEATURE_SCALE] transform split=%s bags=%d instances=%d",
            split_name,
            len(bags),
            n_instances,
        )
        iterator = tqdm.tqdm(
            range(len(bags)),
            desc=f"Scaling {split_name} bags by fingerprint block",
            disable=len(bags) < 10,
        )
        for bag_idx in iterator:
            bag = bags[bag_idx].astype(np.float32, copy=False)
            out = np.empty_like(bag, dtype=np.float32)
            for block_name, indices in blocks.items():
                idx = np.asarray(indices, dtype=np.int64)
                block = scalers[block_name].transform(bag[:, idx])
                out[:, idx] = block.astype(np.float32)
            bags[bag_idx] = out
        transformed = np.vstack(bags).astype(np.float32, copy=False)
        summary = []
        for block_name, indices in blocks.items():
            idx = np.asarray(indices, dtype=np.int64)
            block_values = transformed[:, idx]
            summary.append(
                f"{block_name}:mean_abs={float(np.abs(block_values.mean(axis=0)).mean()):.4g},"
                f"var_sum={float(block_values.var(axis=0).sum()):.4g}"
            )
        LOGGER.info(
            "[FEATURE_SCALE] transformed split=%s stats={%s}",
            split_name,
            "; ".join(summary),
        )

    transform_split("train", train_bags)
    transform_split("validation", val_bags)
    transform_split("test", test_bags)

    return {
        "mode": "per_fingerprint_standard_scaler",
        "scalers": scalers,
        "block_sizes": block_sizes,
        "feature_blocks": {name: list(indices) for name, indices in blocks.items()},
    }


def _normalize_scaled_bags_by_fingerprint_blocks(
    *,
    train_bags: list[np.ndarray],
    val_bags: list[np.ndarray],
    test_bags: list[np.ndarray],
    feature_blocks: dict[str, list[int]],
) -> None:
    """Apply sqrt block-size normalization after clustering, before embedding."""
    LOGGER.info(
        "[FEATURE_NORM] normalization=divide_by_sqrt_block_size "
        "stage=embedder_input blocks=%d",
        len(feature_blocks),
    )
    LOGGER.info(
        "[FEATURE_NORM] blocks={%s}",
        ", ".join(
            f"{name}:n={len(indices)},scale=1/sqrt({len(indices)})="
            f"{1.0 / math.sqrt(max(1, len(indices))):.6f}"
            for name, indices in feature_blocks.items()
        ),
    )

    def normalize_split(split_name: str, bags: list[np.ndarray]) -> None:
        if not bags:
            LOGGER.info("[FEATURE_NORM] transform split=%s skipped_empty", split_name)
            return
        for bag_idx, bag in enumerate(bags):
            out = bag.astype(np.float32, copy=True)
            for indices in feature_blocks.values():
                idx = np.asarray(indices, dtype=np.int64)
                out[:, idx] = out[:, idx] / math.sqrt(max(1, len(indices)))
            bags[bag_idx] = out
        transformed = np.vstack(bags).astype(np.float32, copy=False)
        summary = []
        for block_name, indices in feature_blocks.items():
            idx = np.asarray(indices, dtype=np.int64)
            block_values = transformed[:, idx]
            summary.append(
                f"{block_name}:mean_abs={float(np.abs(block_values.mean(axis=0)).mean()):.4g},"
                f"var_sum={float(block_values.var(axis=0).sum()):.4g}"
            )
        LOGGER.info(
            "[FEATURE_NORM] transformed split=%s stats={%s}",
            split_name,
            "; ".join(summary),
        )

    normalize_split("train", train_bags)
    normalize_split("validation", val_bags)
    normalize_split("test", test_bags)


def setup_data_module(self, stage: str | None = None, is_final_model: bool = False):
    """Set up the data module by loading and preprocessing data.

    This method handles:
    1. Loading and validating the CSV data
    2. Splitting data into train/val/test sets
    3. Building bags from instances
    4. Scaling features
    5. Creating datasets

    Parameters
    ----------
    self : MILDataModule
        The data module instance
    stage : str | None
        The stage of the pipeline (fit, validate, test, predict)
    is_final_model : bool, optional
        Whether this is for the final model, by default False
    """
    LOGGER.info("[DM] DataModule.setup(stage=%s, is_final_model=%s) called", stage, is_final_model)
    if self._train is not None:
        return

    start_time = time.time()

    # Monitor initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)
    LOGGER.info(f"[DM] Initial memory usage: {initial_memory:.2f} GB")

    # Check if we can load pre-processed datasets from cache
    if self._cache_dir and self._cache_dir.exists():
        # Final model caches under "final/"; standard runs cache at the root.
        if is_final_model:
            final_dir = self._cache_dir / "final"
            train_dir = final_dir / "train"
            val_dir = final_dir / "val"
            LOGGER.info(f"[DM] Looking for final model datasets in {final_dir}")
        else:
            train_dir = self._cache_dir / "train"
            val_dir = self._cache_dir / "val"

        # Test directory
        test_dir = (final_dir / "test") if is_final_model else (self._cache_dir / "test")

        # Check if the required directories and files exist
        if train_dir.exists():
            train_file = train_dir / "train.pkl"
            LOGGER.info(f"[DM] Train directory exists: {train_dir}, checking for train.pkl")
            if train_file.exists():
                LOGGER.info(f"[DM] Found train.pkl: {train_file}")
            else:
                LOGGER.info(f"[DM] train.pkl not found in {train_dir}")
                train_file = None
        else:
            LOGGER.info(f"[DM] Train directory does not exist: {train_dir}")
            train_file = None

        if test_dir.exists():
            test_file = test_dir / "test.pkl"
            LOGGER.info(f"[DM] Test directory exists: {test_dir}, checking for test.pkl")
            if test_file.exists():
                LOGGER.info(f"[DM] Found test.pkl: {test_file}")
            else:
                LOGGER.info(f"[DM] test.pkl not found in {test_dir}")
                test_file = None
        else:
            LOGGER.info(f"[DM] Test directory does not exist: {test_dir}")
            test_file = None

        if val_dir.exists():
            val_file = val_dir / "val.pkl"
            LOGGER.info(f"[DM] Val directory exists: {val_dir}, checking for val.pkl")
            if val_file.exists():
                LOGGER.info(f"[DM] Found val.pkl: {val_file}")
            else:
                LOGGER.info(f"[DM] val.pkl not found in {val_dir}")
                val_file = None
        else:
            LOGGER.info(f"[DM] Val directory does not exist: {val_dir}")
            val_file = None

        if train_file and test_file:
            LOGGER.info(f"[DM] Found existing train/test splits in {self._cache_dir}")

            # Try to load datasets from cache
            try:
                # Load labels from CSV
                df = pd.read_csv(self._csv)
                df = self._validate_dataframe(df, self.cfg)

                # Get bag-level labels. Cached val/test IDs may include the
                # "__noexp" suffix, while the source CSV contains only the
                # original molecule IDs, so labels are resolved through a
                # string-normalized base-ID map.
                bag_to_lbl = (
                    df.drop_duplicates(self.cfg.bag_id_col)
                    .assign(
                        _bag_id_key=lambda d: d[self.cfg.bag_id_col].astype(str)
                    )
                    .set_index("_bag_id_key")[self.cfg.endpoint_value_col]
                    .sort_index()
                )

                def _base_bag_id(bag_id) -> str:
                    bag_id = str(bag_id)
                    return (
                        bag_id[: -len("__noexp")]
                        if bag_id.endswith("__noexp")
                        else bag_id
                    )

                def _labels_for_cached_ids(bag_ids):
                    base_ids = [_base_bag_id(bag_id) for bag_id in bag_ids]
                    missing = sorted(set(base_ids) - set(bag_to_lbl.index))
                    if missing:
                        preview = missing[:5]
                        raise KeyError(
                            "Cached bag IDs are missing from the source CSV "
                            f"label map. First missing IDs: {preview}"
                        )
                    return bag_to_lbl.loc[base_ids].to_numpy(dtype=np.float32)

                def _cached_cluster_ids(cache_data, split_name: str):
                    if not getattr(self.cfg, "cluster_instances", False):
                        return None

                    cluster_ids = cache_data.get("cluster_ids")
                    if cluster_ids is None:
                        raise ValueError(
                            f"Cached {split_name} data has no cluster_ids, "
                            "but cluster_instances=True"
                        )

                    expected_config = cluster_config_signature(self.cfg)
                    cached_config = cache_data.get("cluster_config")
                    if cached_config != expected_config:
                        raise ValueError(
                            f"Cached {split_name} cluster config does not match "
                            "current clustering settings; rebuilding cache"
                        )

                    return cluster_ids

                # Load train dataset
                with open(train_file, 'rb') as f:
                    train_data = pickle.load(f)
                train_bag_ids = train_data["bag_ids"]
                train_cluster_ids = _cached_cluster_ids(train_data, "train")
                train_series_labels = train_data.get("series_labels")
                if train_series_labels is None:
                    train_series_labels = self._build_bag_series_labels(
                        df, self.cfg, train_bag_ids, "train"
                    )
                train_y = _labels_for_cached_ids(train_bag_ids)

                # Load val dataset if it exists
                val_bag_ids = []
                val_y = []
                if val_file and val_file.exists():
                    with open(val_file, 'rb') as f:
                        val_data = pickle.load(f)
                    val_bag_ids = val_data["bag_ids"]
                    val_cluster_ids = _cached_cluster_ids(val_data, "validation")
                    val_series_labels = val_data.get("series_labels")
                    if val_series_labels is None:
                        val_series_labels = self._build_bag_series_labels(
                            df, self.cfg, val_bag_ids, "validation"
                        )
                    val_y = _labels_for_cached_ids(val_bag_ids)
                else:
                    val_cluster_ids = None
                    val_series_labels = None

                # Load test dataset
                with open(test_file, 'rb') as f:
                    test_data = pickle.load(f)
                test_bag_ids = test_data["bag_ids"]
                test_cluster_ids = _cached_cluster_ids(test_data, "test")
                test_series_labels = test_data.get("series_labels")
                if test_series_labels is None:
                    test_series_labels = self._build_bag_series_labels(
                        df, self.cfg, test_bag_ids, "test"
                    )
                test_y = _labels_for_cached_ids(test_bag_ids)

                # Create datasets
                self._train = MILDataset(
                    str(train_file),
                    train_y,
                    train_bag_ids,
                    cluster_ids=train_cluster_ids,
                    series_labels=train_series_labels,
                    cache_dir=str(self._cache_dir),
                    memory_limit=self._memory_limit,
                )
                self._val = (
                    MILDataset(
                        str(val_file),
                        val_y,
                        val_bag_ids,
                        cluster_ids=val_cluster_ids,
                        series_labels=val_series_labels,
                        cache_dir=str(self._cache_dir),
                        memory_limit=self._memory_limit,
                    )
                    if val_file and val_file.exists()
                    else None
                )
                self._test = MILDataset(
                    str(test_file),
                    test_y,
                    test_bag_ids,
                    cluster_ids=test_cluster_ids,
                    series_labels=test_series_labels,
                    cache_dir=str(self._cache_dir),
                    memory_limit=self._memory_limit,
                )

                # Detect descriptor columns
                if self.cfg.descriptor_cols is None:
                    # Only exclude columns that actually exist in the dataframe
                    excluded = {col for col in {self.cfg.bag_id_col, self.cfg.inst_id_col, self.cfg.endpoint_value_col,
                                self.cfg.energy_col, self.cfg.smiles_col, self.cfg.pdb_id_col, self.cfg.split_col,
                                getattr(self.cfg, "series_col", None), "conf_id"}
                                if col in df.columns}
                    # Always exclude the Energy column even if it's not in the excluded set
                    excluded.add("Energy")
                    numeric_cols = df.select_dtypes("number").columns.tolist()
                    LOGGER.info(f"[DM] Total numeric columns: {len(numeric_cols)}")
                    LOGGER.info(f"[DM] Excluding columns from descriptors: {sorted(excluded)}")
                    LOGGER.info(f"[DM] Numeric columns being excluded: {[c for c in numeric_cols if c in excluded]}")
                    self.cfg.descriptor_cols = [c for c in numeric_cols if c not in excluded]
                self.feature_names = self.cfg.descriptor_cols
                LOGGER.info(f"[DM] Detected {len(self.cfg.descriptor_cols)} descriptor columns")
                _log_descriptor_blocks("cache_descriptor_detection", list(self.feature_names))

                LOGGER.info(f"[DM] Successfully loaded datasets from cache: {len(train_bag_ids)} train, {len(val_bag_ids)} val, {len(test_bag_ids)} test bags")
                return

            except Exception as e:
                LOGGER.warning(f"[DM] Failed to load datasets from cache: {e}. Will recreate datasets.")

    # If we get here, we need to create the datasets from scratch
    LOGGER.info(f"[DM] Loading data from {self._csv}")
    try:
        df = pd.read_csv(self._csv)
        LOGGER.info(f"[DM] Loaded {len(df)} rows from {self._csv}")
    except Exception as e:
        LOGGER.error(f"[DM] Failed to load CSV file: {e}")
        raise

    cfg = self.cfg

    # Validate and clean the raw df
    try:
        df = self._validate_dataframe(df, cfg)
        LOGGER.info("[DM] Dataframe validation completed - no structural issues found")
    except Exception as e:
        LOGGER.error(f"[DM] Dataframe validation failed: {e}")
        raise

    # Detect descriptor columns
    if cfg.descriptor_cols is None:
        # Only exclude columns that actually exist in the dataframe
        excluded = {col for col in {cfg.bag_id_col, cfg.inst_id_col, cfg.endpoint_value_col,
                    cfg.energy_col, cfg.smiles_col, cfg.pdb_id_col, cfg.split_col,
                    getattr(cfg, "series_col", None), "conf_id"}
                    if col in df.columns}
        # Always exclude the Energy column even if it's not in the excluded set
        excluded.add("Energy")
        numeric_cols = df.select_dtypes("number").columns.tolist()
        LOGGER.info(f"[DM] Total numeric columns: {len(numeric_cols)}")
        LOGGER.info(f"[DM] Excluding columns from descriptors: {sorted(excluded)}")
        LOGGER.info(f"[DM] Numeric columns being excluded: {[c for c in numeric_cols if c in excluded]}")
        cfg.descriptor_cols = [c for c in numeric_cols if c not in excluded]
    self.feature_names = cfg.descriptor_cols
    LOGGER.info(f"[DM] Detected {len(cfg.descriptor_cols)} descriptor columns")
    _log_descriptor_blocks("raw_descriptor_detection", list(self.feature_names))

    # Process data and create datasets
    train_dataset, val_dataset, test_dataset = process_data(
        self, df, cfg, initial_memory, start_time, is_final_model=is_final_model
    )

    # Store the datasets
    self._train = train_dataset
    self._val = val_dataset
    self._test = test_dataset


def process_data(self, df: pd.DataFrame, cfg: DataLoaderConfig, 
                initial_memory: float, start_time: float, is_final_model: bool = False):
    """Process the data and create datasets.

    Parameters
    ----------
    self : MILDataModule
        The data module instance
    df : pd.DataFrame
        The input dataframe
    cfg : DataLoaderConfig
        Configuration for data loading
    initial_memory : float
        Initial memory usage in GB
    start_time : float
        Start time of the setup process
    is_final_model : bool, optional
        Whether this is for the final model, by default False

    Returns
    -------
    Tuple[MILDataset, Optional[MILDataset], MILDataset]
        Train, validation, and test datasets
    """
    # Bag‑level endpoint vector
    LOGGER.info("[DM] Creating bag-level endpoint vector")
    bag_to_lbl = (
        df.drop_duplicates(cfg.bag_id_col)
        .set_index(cfg.bag_id_col)[cfg.endpoint_value_col]
        .sort_index()
    )
    bag_ids_all = bag_to_lbl.index.to_numpy()
    y_all = bag_to_lbl.to_numpy()

    # Discretise continuous labels for stratification
    if cfg.task == "regression":
        y_strat = pd.qcut(y_all, q=cfg.n_strat_bins, labels=False, duplicates="drop")
    else:
        y_strat = y_all  # already categorical

    # Train/validation/test split
    LOGGER.info("[DM] Performing data split")
    has_validation = True
    if cfg.predefined_split and cfg.split_col in df.columns:
        # Predefined split: 0=train, 1=val, 2=test.
        df_train = df[df[cfg.split_col] == 0].copy()
        df_val = df[df[cfg.split_col] == 1].copy()
        df_test = df[df[cfg.split_col] == 2].copy()

        if is_final_model:
            # Final model: fit the scaler and train on train+val, keep test held out.
            df_train = pd.concat([df_train, df_val], ignore_index=True)
            df_val = df.iloc[0:0].copy()
            has_validation = False
            LOGGER.info(
                "[DM] Predefined split (final): %d train+val bags, %d test bags",
                df_train[cfg.bag_id_col].nunique(),
                df_test[cfg.bag_id_col].nunique(),
            )
        else:
            LOGGER.info(
                "[DM] Predefined split: %d train / %d val / %d test bags",
                df_train[cfg.bag_id_col].nunique(),
                df_val[cfg.bag_id_col].nunique(),
                df_test[cfg.bag_id_col].nunique(),
            )
    else:
        # No predefined split: carve a stratified test set, then optionally a val set.
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.seed
        )
        train_idx, test_idx = next(sss.split(bag_ids_all, y_strat))
        train_bags = set(bag_ids_all[train_idx])
        test_bags = set(bag_ids_all[test_idx])
        df_train = df[df[cfg.bag_id_col].isin(train_bags)].copy()
        df_test = df[df[cfg.bag_id_col].isin(test_bags)].copy()

        if is_final_model or not cfg.val_partition:
            df_val = df.iloc[0:0].copy()
            has_validation = False
            LOGGER.info(
                "[DM] Stratified split (no val): %d train / %d test bags",
                len(train_bags),
                len(test_bags),
            )
        else:
            bag_ids_tr = df_train[cfg.bag_id_col].unique()
            y_tr = bag_to_lbl.loc[bag_ids_tr].to_numpy()
            y_tr_strat = (
                pd.qcut(y_tr, q=cfg.n_strat_bins, labels=False, duplicates="drop")
                if cfg.task == "regression"
                else y_tr
            )
            sss_val = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=cfg.seed
            )
            tr_idx, va_idx = next(sss_val.split(bag_ids_tr, y_tr_strat))
            df_val = df_train[df_train[cfg.bag_id_col].isin(set(bag_ids_tr[va_idx]))]
            df_train = df_train[df_train[cfg.bag_id_col].isin(set(bag_ids_tr[tr_idx]))]
            LOGGER.info(
                "[DM] Stratified split: %d train / %d val / %d test bags",
                df_train[cfg.bag_id_col].nunique(),
                df_val[cfg.bag_id_col].nunique(),
                len(test_bags),
            )

    # Build datasets with progress reporting
    LOGGER.info("[DM] Building bags from instances")
    tr_bags, tr_y, tr_ids = self._build_bags(df_train, cfg, "train")
    tr_series_labels = self._build_bag_series_labels(
        df_train, cfg, tr_ids, "train"
    )

    if has_validation:
        va_bags, va_y, va_ids = self._build_bags(df_val, cfg, "validation")
        va_series_labels = self._build_bag_series_labels(
            df_val, cfg, va_ids, "validation"
        )
    else:
        LOGGER.info("[DM] No validation set for this run")
        va_bags, va_y, va_ids = [], np.array([], dtype=np.float32), []
        va_series_labels = None
    
    te_bags, te_y, te_ids = self._build_bags(df_test, cfg, "test")
    te_series_labels = self._build_bag_series_labels(
        df_test, cfg, te_ids, "test"
    )

    # Log raw statistics before scaling
    self._log_dataset_stats(
        "RAW",
        {"train": (tr_bags, tr_y),
         "val": (va_bags, va_y),
         "test": (te_bags, te_y)},
        task=cfg.task,
    )

    # Feature variance removal - identify zero-variance features on all training
    # instances only, then apply the same mask to train/validation/test.
    pre_zero_var_feature_names = list(self.feature_names)
    _log_descriptor_blocks("before_zero_variance_filter", pre_zero_var_feature_names)
    LOGGER.info(
        "[FEATURE_ZERO_VAR] fit=train_only threshold=1e-8 train_bags=%d "
        "train_instances=%d",
        len(tr_bags),
        int(sum(len(bag) for bag in tr_bags)),
    )
    train_data = np.vstack(tr_bags)
    feature_variances = np.var(train_data, axis=0)
    
    # Identify features with zero or near-zero variance (threshold: 1e-8)
    zero_var_mask = feature_variances > 1e-8
    n_removed_features = np.sum(~zero_var_mask)
    removed_by_block: "OrderedDict[str, int]" = OrderedDict()
    kept_by_block: "OrderedDict[str, int]" = OrderedDict()
    removed_feature_names = []
    for keep, name in zip(zero_var_mask.tolist(), pre_zero_var_feature_names):
        block = _fingerprint_block_name(name)
        removed_by_block.setdefault(block, 0)
        kept_by_block.setdefault(block, 0)
        if keep:
            kept_by_block[block] += 1
        else:
            removed_by_block[block] += 1
            removed_feature_names.append(name)
    
    if n_removed_features > 0:
        LOGGER.info(
            "[FEATURE_ZERO_VAR] removed=%d kept=%d original=%d removed_by_block={%s}",
            int(n_removed_features),
            int(np.sum(zero_var_mask)),
            len(feature_variances),
            ", ".join(
                f"{block}:{count}"
                for block, count in removed_by_block.items()
                if count > 0
            )
            or "none",
        )
        LOGGER.info(
            "[FEATURE_ZERO_VAR] kept_by_block={%s}",
            ", ".join(f"{block}:{count}" for block, count in kept_by_block.items()),
        )
        if removed_feature_names:
            LOGGER.info(
                "[FEATURE_ZERO_VAR] removed_features_first20=%s",
                removed_feature_names[:20],
            )
        
        # Apply feature removal to all splits
        for split_name, bags in zip(["train", "validation", "test"], [tr_bags, va_bags, te_bags]):
            if not bags:
                LOGGER.info(
                    "[FEATURE_ZERO_VAR] apply split=%s skipped_empty",
                    split_name,
                )
                continue
            LOGGER.info(
                "[FEATURE_ZERO_VAR] apply split=%s bags=%d features=%d->%d",
                split_name,
                len(bags),
                len(feature_variances),
                int(np.sum(zero_var_mask)),
            )
            for i in range(len(bags)):
                bags[i] = bags[i][:, zero_var_mask]
    else:
        LOGGER.info(
            "[FEATURE_ZERO_VAR] removed=0 kept=%d original=%d",
            len(feature_variances),
            len(feature_variances),
        )
    
    # Store the feature mask for later use
    self.feature_mask = zero_var_mask
    
    # Update feature_names to reflect the remaining features after zero-variance removal
    if hasattr(self, 'feature_names') and self.feature_names is not None:
        if len(self.feature_names) == len(zero_var_mask):
            # Filter feature_names using the mask
            original_feature_names = self.feature_names.copy()
            self.feature_names = [name for i, name in enumerate(original_feature_names) if zero_var_mask[i]]
            LOGGER.info(f"[DM] Updated feature_names from {len(original_feature_names)} to {len(self.feature_names)} features")
            _log_descriptor_blocks("after_zero_variance_filter", list(self.feature_names))
        else:
            LOGGER.warning(f"[DM] Feature names count ({len(self.feature_names)}) doesn't match feature mask size ({len(zero_var_mask)}), skipping feature_names update")
    
    # Fit all preprocessing statistics on train instances only, then reuse the
    # frozen transform for train/validation/test to avoid leakage. Clustering
    # uses per-fingerprint standardized features before block-size
    # normalization. The embedder then receives the same scaled features after
    # sqrt block-size normalization.
    if getattr(self, "_scaled", False):
        LOGGER.warning(
            "[DM] Scaling has already been applied in this session; skipping "
            "to avoid double scaling."
        )
    else:
        self.scaler = _transform_bags_by_fingerprint_blocks(
            train_bags=tr_bags,
            val_bags=va_bags,
            test_bags=te_bags,
            feature_names=list(self.feature_names),
        )
        self.feature_blocks = self.scaler["feature_blocks"]
        self._scaled = True

    # Log statistics after scaling, before block normalization. This is the
    # feature space used for per-molecule conformer clustering.
    self._log_dataset_stats(
        "SCALED_FOR_CLUSTERING",
        {"train": (tr_bags, tr_y),
         "val": (va_bags, va_y),
         "test": (te_bags, te_y)},
        task=cfg.task,
    )

    tr_cluster_ids = va_cluster_ids = te_cluster_ids = None
    if getattr(cfg, "cluster_instances", False):
        LOGGER.info(
            "[CLUSTER] Clustering conformers per bag in scaled-only descriptor space"
        )
        tr_cluster_ids = self._cluster_bags(tr_bags, "train") if tr_bags else []
        va_cluster_ids = self._cluster_bags(va_bags, "validation") if va_bags else []
        te_cluster_ids = self._cluster_bags(te_bags, "test") if te_bags else []

    if getattr(self, "_block_normalized", False):
        LOGGER.warning(
            "[DM] Block-size normalization has already been applied in this "
            "session; skipping to avoid double normalization."
        )
    else:
        _normalize_scaled_bags_by_fingerprint_blocks(
            train_bags=tr_bags,
            val_bags=va_bags,
            test_bags=te_bags,
            feature_blocks=self.feature_blocks,
        )
        self._block_normalized = True

    self._log_dataset_stats(
        "EMBEDDER_INPUT",
        {"train": (tr_bags, tr_y),
         "val": (va_bags, va_y),
         "test": (te_bags, te_y)},
        task=cfg.task,
    )

    # Always save bags to disk if cache_dir is specified
    if self._cache_dir:
        LOGGER.info(f"[DM] Saving datasets to {self._cache_dir}")

        # Save bags to disk and create path mappings
        LOGGER.info(f"[DM] Saving train bags to disk: {len(tr_bags)} bags, {len(tr_ids)} IDs")
        tr_file = self._save_bags_to_disk(
            tr_bags,
            tr_ids,
            "train",
            is_final_model=is_final_model,
            cluster_ids=tr_cluster_ids,
            series_labels=tr_series_labels,
        )

        if va_bags:
            LOGGER.info(f"[DM] Saving val bags to disk: {len(va_bags)} bags, {len(va_ids)} IDs")
            va_file = self._save_bags_to_disk(
                va_bags,
                va_ids,
                "val",
                is_final_model=is_final_model,
                cluster_ids=va_cluster_ids,
                series_labels=va_series_labels,
            )
        else:
            LOGGER.info("[DM] No val bags to save")
            va_file = None

        LOGGER.info(f"[DM] Saving test bags to disk: {len(te_bags)} bags, {len(te_ids)} IDs")
        te_file = self._save_bags_to_disk(
            te_bags,
            te_ids,
            "test",
            is_final_model=is_final_model,
            cluster_ids=te_cluster_ids,
            series_labels=te_series_labels,
        )

        # Get the file paths
        if is_final_model:
            # For final model, use the "final" directory
            train_file = self._cache_dir / "final" / "train" / "train.pkl"
            val_file = self._cache_dir / "final" / "val" / "val.pkl" if va_bags else None
            test_file = self._cache_dir / "final" / "test" / "test.pkl"
            LOGGER.info(f"[DM] Using final model directories for data files: train={train_file}, val={val_file}, test={test_file}")

            # Ensure the directories exist
            train_dir = train_file.parent
            train_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[DM] Ensured train directory exists: {train_dir}")

            if va_bags:
                val_dir = val_file.parent
                val_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"[DM] Ensured val directory exists: {val_dir}")

            test_dir = test_file.parent
            test_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[DM] Ensured test directory exists: {test_dir}")
        else:
            train_file = self._cache_dir / "train" / "train.pkl"
            val_file = self._cache_dir / "val" / "val.pkl" if va_bags else None
            test_file = self._cache_dir / "test" / "test.pkl"

        # Create datasets based on loading mode
        if self._on_demand_loading:
            LOGGER.info("[DM] Creating datasets in single-file mode")
            # Create datasets with single-file loading
            train_dataset = MILDataset(
                str(train_file),
                tr_y,
                tr_ids,
                cluster_ids=tr_cluster_ids,
                series_labels=tr_series_labels,
                cache_dir=str(self._cache_dir),
                memory_limit=self._memory_limit,
            )
            # Validation dataset only when validation bags exist
            val_dataset = (
                MILDataset(
                    str(val_file),
                    va_y,
                    va_ids,
                    cluster_ids=va_cluster_ids,
                    series_labels=va_series_labels,
                    cache_dir=str(self._cache_dir),
                    memory_limit=self._memory_limit,
                )
                if val_file and len(va_bags)
                else None
            )
            test_dataset = MILDataset(
                str(test_file),
                te_y,
                te_ids,
                cluster_ids=te_cluster_ids,
                series_labels=te_series_labels,
                cache_dir=str(self._cache_dir),
                memory_limit=self._memory_limit,
            )
        else:
            LOGGER.info("[DM] Creating datasets in in-memory mode")
            # Create datasets with in-memory data
            train_dataset = MILDataset(
                tr_bags,
                tr_y,
                tr_ids,
                cluster_ids=tr_cluster_ids,
                series_labels=tr_series_labels,
            )
            # Validation dataset only when validation bags exist
            val_dataset = (
                MILDataset(
                    va_bags,
                    va_y,
                    va_ids,
                    cluster_ids=va_cluster_ids,
                    series_labels=va_series_labels,
                )
                if len(va_bags)
                else None
            )
            test_dataset = MILDataset(
                te_bags,
                te_y,
                te_ids,
                cluster_ids=te_cluster_ids,
                series_labels=te_series_labels,
            )
    else:
        LOGGER.info("[DM] Creating datasets in in-memory mode (no cache directory specified)")
        # Create datasets with in-memory data
        train_dataset = MILDataset(
            tr_bags,
            tr_y,
            tr_ids,
            cluster_ids=tr_cluster_ids,
            series_labels=tr_series_labels,
        )
        # Validation dataset only when validation bags exist
        val_dataset = (
            MILDataset(
                va_bags,
                va_y,
                va_ids,
                cluster_ids=va_cluster_ids,
                series_labels=va_series_labels,
            )
            if len(va_bags)
            else None
        )
        test_dataset = MILDataset(
            te_bags,
            te_y,
            te_ids,
            cluster_ids=te_cluster_ids,
            series_labels=te_series_labels,
        )

    # Report memory usage after setup
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / (1024**3)
    LOGGER.info(f"[DM] Final memory usage: {current_memory:.2f} GB (delta: {current_memory - initial_memory:.2f} GB)")

    # Report total setup time
    elapsed_time = time.time() - start_time
    LOGGER.info(f"[DM] Setup completed in {elapsed_time:.2f} seconds")

    return train_dataset, val_dataset, test_dataset
