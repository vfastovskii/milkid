"""Implementation details for the MILDataModule class.

This module contains the implementation details for the MILDataModule class,
including data loading, preprocessing, and dataset creation.
"""
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedShuffleSplit

from ppl.data.dataset import MILDataset
from ppl.config.data_loader_config import DataLoaderConfig

LOGGER = logging.getLogger(__name__)


from ppl.data.feature_scaling import (
    _fingerprint_block_name,
    _log_descriptor_blocks,
    _normalize_scaled_bags_by_fingerprint_blocks,
    _transform_bags_by_fingerprint_blocks,
)


def setup_data_module(self, stage: str | None = None):
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
    """
    LOGGER.info("[DM] DataModule.setup(stage=%s) called", stage)
    if self._train is not None:
        return

    start_time = time.time()

    # Monitor initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024**3)
    LOGGER.info(f"[DM] Initial memory usage: {initial_memory:.2f} GB")

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
        self, df, cfg, initial_memory, start_time
    )

    # Store the datasets
    self._train = train_dataset
    self._val = val_dataset
    self._test = test_dataset


def process_data(self, df: pd.DataFrame, cfg: DataLoaderConfig,
                initial_memory: float, start_time: float):
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

        if not cfg.val_partition:
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
    tr_bags, tr_y, tr_ids, tr_conf_ids = self._build_bags(df_train, cfg, "train")
    tr_series_labels = self._build_bag_series_labels(
        df_train, cfg, tr_ids, "train"
    )

    if has_validation:
        va_bags, va_y, va_ids, va_conf_ids = self._build_bags(df_val, cfg, "validation")
        va_series_labels = self._build_bag_series_labels(
            df_val, cfg, va_ids, "validation"
        )
    else:
        LOGGER.info("[DM] No validation set for this run")
        va_bags, va_y, va_ids, va_conf_ids = [], np.array([], dtype=np.float32), [], []
        va_series_labels = None

    te_bags, te_y, te_ids, te_conf_ids = self._build_bags(df_test, cfg, "test")
    te_series_labels = self._build_bag_series_labels(
        df_test, cfg, te_ids, "test"
    )

    # Describe what bag construction produced (checked against the built bags):
    # train drops experimental poses (1 bag/molecule); val/test keep each molecule
    # twice — a full bag WITH experimental poses and a "__noexp" bag WITHOUT.
    run_log = logging.getLogger("milk")
    n_train_exp = sum(
        1 for cids in tr_conf_ids if any("_experimental_pose" in str(c) for c in cids)
    )
    run_log.info(
        "Train bags: %d molecules → %d bags, experimental poses removed, 1 bag/molecule%s.",
        len(tr_ids), len(tr_bags),
        f" ({n_train_exp} all-experimental molecule(s) kept as-is)" if n_train_exp else "",
    )
    for split_name, ids in (("Validation", va_ids), ("Test", te_ids)):
        if not ids:
            continue
        n_molecules = sum(1 for i in ids if str(i).endswith("__noexp"))
        run_log.info(
            "%s bags: %d molecules → %d bags — each molecule kept twice: a full bag "
            "(with experimental poses) + a '__noexp' bag (without). Only '__noexp' "
            "counts toward eval metrics and plots.",
            split_name, n_molecules, len(ids),
        )

    # Capture the exact per-bag conformer IDs (aligned with each bag's instance
    # rows, including "__noexp" bags) so attention weights are always labeled from
    # the same rows they were computed on — never re-derived by re-reading the CSV.
    self.bag_conf_ids = {
        str(bid): [str(c) for c in cids]
        for ids_list, conf_list in (
            (tr_ids, tr_conf_ids), (va_ids, va_conf_ids), (te_ids, te_conf_ids)
        )
        for bid, cids in zip(ids_list, conf_list)
    }

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
        n_bags = len(tr_bags) + len(va_bags) + len(te_bags)
        logging.getLogger("milk").info(
            "Clustering conformers per molecule across %d bags — this can take a few minutes…",
            n_bags,
        )
        LOGGER.info(
            "[CLUSTER] Clustering conformers per bag in scaled-only descriptor space"
        )
        tr_cluster_ids = self._cluster_bags(tr_bags, "train") if tr_bags else []
        va_cluster_ids = self._cluster_bags(va_bags, "validation") if va_bags else []
        te_cluster_ids = self._cluster_bags(te_bags, "test") if te_bags else []

        _all_cids = [
            c for c in list(tr_cluster_ids or []) + list(va_cluster_ids or []) + list(te_cluster_ids or [])
            if c is not None
        ]
        if _all_cids:
            _n_clusters = [int(np.unique(c).size) for c in _all_cids]
            logging.getLogger("milk").info(
                "Clustered %d molecules: mean %.1f clusters/molecule (min %d, max %d).",
                len(_n_clusters), float(np.mean(_n_clusters)), min(_n_clusters), max(_n_clusters),
            )

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

    # Create in-memory datasets (caching removed for reproducibility)
    train_dataset = MILDataset(
        tr_bags, tr_y, tr_ids,
        cluster_ids=tr_cluster_ids, series_labels=tr_series_labels,
    )
    val_dataset = (
        MILDataset(
            va_bags, va_y, va_ids,
            cluster_ids=va_cluster_ids, series_labels=va_series_labels,
        )
        if len(va_bags)
        else None
    )
    test_dataset = MILDataset(
        te_bags, te_y, te_ids,
        cluster_ids=te_cluster_ids, series_labels=te_series_labels,
    )
    # Report memory usage after setup
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / (1024**3)
    LOGGER.info(f"[DM] Final memory usage: {current_memory:.2f} GB (delta: {current_memory - initial_memory:.2f} GB)")

    # Report total setup time
    elapsed_time = time.time() - start_time
    LOGGER.info(f"[DM] Setup completed in {elapsed_time:.2f} seconds")

    return train_dataset, val_dataset, test_dataset
