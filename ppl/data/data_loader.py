"""PyTorch‑Lightning DataModule for multiple‑instance learning (MIL).

This module provides a high-level interface for loading and preprocessing data for multiple-instance learning.
Implementation details are in separate modules:
- dataset.py: Contains the MILDataset class
- data_loader_impl.py: Contains helper functions for data loading
- data_module_impl.py: Contains implementation details for the MILDataModule class
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, List, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ppl.config.data_loader_config import DataLoaderConfig
from ppl.data.data_loader_impl import (
    make_dataloader,
    build_bags,
    build_bag_series_labels,
    cluster_bags,
    validate_dataframe,
    log_dataset_stats,
)
from ppl.data.data_module_impl import setup_data_module

LOGGER = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the project root directory.

    Returns
    -------
    Path
        Path to the project root directory
    """
    # Navigate up from ppl/data/data_loader.py to the project root
    return Path(__file__).parent.parent.parent

def resolve_path(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Resolve a path relative to a base directory or the project root.

    This function handles both absolute and relative paths:
    - Absolute paths are returned as-is
    - Relative paths are resolved relative to base_dir if provided, or the project root

    Parameters
    ----------
    path : Union[str, Path]
        The path to resolve
    base_dir : Optional[Path], optional
        Base directory to resolve relative paths against, by default None (uses project root)

    Returns
    -------
    Path
        Resolved absolute path
    """
    path_obj = Path(path).expanduser()

    # If it's an absolute path, return it directly
    if path_obj.is_absolute():
        return path_obj

    # If base_dir is not provided, use the project root
    if base_dir is None:
        base_dir = get_project_root()

    # Resolve the path relative to the base directory
    return (base_dir / path_obj).resolve()


class MILDataModule(pl.LightningDataModule):
    """MIL DataModule with memory-efficient data loading and progress reporting.

    This class provides a high-level interface for loading and preprocessing data
    for multiple-instance learning. Implementation details are in data_module_impl.py.
    """

    def __init__(self, cfg: DataLoaderConfig):
        """Initialize the MILDataModule with configuration.

        Parameters
        ----------
        cfg : DataLoaderConfig
            Configuration for data loading
        """
        super().__init__()
        self.cfg = cfg
        LOGGER.info("[DM] Initialising MILDataModule with DataLoaderConfig.")

        # Resolve CSV path relative to project root to ensure it works from any entry point
        self._csv = resolve_path(cfg.csv_path)
        if not self._csv.exists():
            raise FileNotFoundError(f"CSV file not found: {self._csv}. Current working directory: {Path.cwd()}")
        if cfg.task not in {"regression", "classification"}:
            raise ValueError("task must be 'classification' or 'regression'")

        # Initialize with default values (datasets are always built in-memory)
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        self._train = self._val = self._test = None  # filled in setup()
        # {bag_id: [conformer instance IDs]} captured during bag construction,
        # aligned with each bag's rows (incl. "__noexp" bags) for attention labels.
        self.bag_conf_ids: dict[str, list[str]] = {}
        # {series: importance weight} for balanced-batch training; None when off.
        self.series_importance_weights: Optional[dict[str, float]] = None

    # Lightning API
    def setup(self, stage: str | None = None):
        """Set up the data module by loading and preprocessing data.

        This method handles:
        1. Loading and validating the CSV data
        2. Splitting data into train/val/test sets
        3. Building bags from instances
        4. Scaling features
        5. Creating datasets

        Parameters
        ----------
        stage : str | None
            The stage of the pipeline (fit, validate, test, predict)
        """
        # Implementation details are in data_module_impl.py
        setup_data_module(self, stage)
        self.series_importance_weights = self._compute_series_importance_weights()

    def _compute_series_importance_weights(self):
        """Per-series importance weight w(s) = |s|·n_series/N over the TRAIN bags.

        Corrects the group-balanced sampler's oversampling of rare series so the
        weighted train loss is an unbiased estimate of the true-distribution loss.
        Returns None unless balanced batching AND reweighting are both enabled and
        the train split carries series labels.
        """
        if not (
            bool(getattr(self.cfg, "balance_train_batches_by_series", False))
            and bool(getattr(self.cfg, "series_balance_importance_weight", True))
        ):
            return None
        labels = list(getattr(self._train, "_series_labels", None) or [])
        if not labels:
            return None
        from collections import Counter

        sizes = Counter(str(s) for s in labels)
        n = float(sum(sizes.values()))
        n_series = float(len(sizes))
        return {s: (float(sz) * n_series / n) for s, sz in sizes.items()}

    def _cluster_bags(self, bags, split_name=""):
        """Cluster conformers per bag in scaled descriptor space."""
        return cluster_bags(bags, self.cfg, split_name)

    @staticmethod
    def _build_bag_series_labels(df, cfg, bag_ids, split_name=""):
        """Build one series label per bag ID."""
        return build_bag_series_labels(df, cfg, bag_ids, split_name)

    # Dataloaders
    def train_dataloader(self): 
        return self._make_loader(self._train, shuffle=True)

    def train_full_dataloader(self):
        """One-pass sequential train loader for audits/ablation, never balanced."""
        if self._train is None:
            raise RuntimeError("DataModule.setup must run before creating train loader")
        return self._make_loader(self._train, shuffle=False)

    def val_bag_ids(self):
        """Return validation bag IDs in dataset order."""
        if self._val is None:
            return []
        return list(getattr(self._val, "_bag_ids", []))

    def val_dataloader(self): 
        return [] if self._val is None else self._make_loader(self._val, shuffle=False)

    def test_dataloader(self): 
        return self._make_loader(self._test, shuffle=False)

    def apply_train_instance_selection(
        self,
        selection_by_bag_id,
        *,
        require_complete=True,
        min_selected_per_bag=None,
        max_selected_per_bag=None,
    ):
        """Filter train bags to selected instance indices."""
        if self._train is None:
            raise RuntimeError("DataModule.setup must run before applying selection")
        if not hasattr(self._train, "apply_instance_selection"):
            raise TypeError("Current train dataset does not support instance selection")
        return self._train.apply_instance_selection(
            selection_by_bag_id,
            require_complete=require_complete,
            min_selected_per_bag=min_selected_per_bag,
            max_selected_per_bag=max_selected_per_bag,
        )

    # Helpers
    def _make_loader(self, ds, *, shuffle: bool) -> DataLoader:
        """Create a DataLoader with optimized settings."""
        return make_dataloader(ds, self.cfg, shuffle=shuffle)

    @staticmethod
    def _build_bags(df, cfg, split_name=""):
        """Build bags from instances in the dataframe."""
        return build_bags(df, cfg, split_name)

    @staticmethod
    def _validate_dataframe(df, cfg):
        """Checks for NaNs and non-numeric descriptors."""
        return validate_dataframe(df, cfg)

    @staticmethod
    def _log_dataset_stats(tag, splits, *, task):
        """Log bag-level and feature-level diagnostics for each split."""
        log_dataset_stats(tag, splits, task=task)
