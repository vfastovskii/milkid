"""Cross-validation utilities for the Multi-Instance Learning Kit (MILK).

This module provides a class for performing cross-validation in the MILK project.
It handles creating train/validation splits, training models on each fold,
and summarizing the results.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from ppl.utils.mil_data_handling.data_loader import DataLoaderConfig, MILDataModule
from ppl.utils.pipeline.mlflow_utils import SafeMLFlowLogger, create_mlflow_logger
from ppl.utils.model_trainer.model_trainer import ModelTrainer
from ppl.utils.pipeline.visualization import log_split_distributions
from ppl.utils.pipeline.results_directory import create_results_directory

LOGGER = logging.getLogger(__name__)


class CrossValidator:
    """Class for performing cross-validation.

    This class is responsible for:
    - Creating train/validation splits
    - Training models on each fold
    - Collecting and summarizing validation metrics

    Parameters
    ----------
    data_cfg : DataLoaderConfig
        Configuration for data loading
    model_trainer : ModelTrainer
        Model trainer instance
    log_save_dir : Path
        Directory to save logs
    experiment_name : str
        Name of the experiment
    run_name : str
        Name of the run
    tracking_uri : str | None
        URI for MLFlow tracking server
    """

    def __init__(
        self,
        data_cfg: DataLoaderConfig,
        model_trainer: ModelTrainer,
        log_save_dir: Path,
        experiment_name: str,
        run_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_trainer = model_trainer
        self.log_save_dir = log_save_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.num_folds = data_cfg.num_folds
        self.cv_seed = data_cfg.cv_seed
        self._fold_results: Dict[str, List[float]] = {}
        self._mlflow_logger = create_mlflow_logger(
            self.log_save_dir,
            experiment_name=self.experiment_name,
            run_name=self.run_name,
            tracking_uri=self.tracking_uri,
        )

    def _new_logger(self, run_suffix: str) -> SafeMLFlowLogger:
        """Create a new MLFlow logger with a run suffix.

        Parameters
        ----------
        run_suffix : str
            Suffix to append to the run name

        Returns
        -------
        SafeMLFlowLogger
            Configured MLFlow logger
        """
        return create_mlflow_logger(
            save_dir=self.log_save_dir,
            experiment_name=self.experiment_name,
            run_name=f"{self.run_name}-{run_suffix}",
            tracking_uri=self.tracking_uri,
        )

    def run_cross_validation(self) -> None:
        """Run k-fold cross-validation on the training data.

        This method:
        1. Creates k different train/validation splits
        2. For each fold:
           - Sets up the data module
           - Logs the label distributions
           - Fits the model on the training data
           - Validates the model on the validation data
           - Collects the validation metrics
           - Generates attention weights and true vs. predicted plots
        3. Summarizes the cross-validation results
        """
        LOGGER.info("[CV] Running %d‑fold CV on the train split", self.num_folds)

        # Create results directory
        results_dir = create_results_directory(self.experiment_name)
        validation_dir = results_dir / "validation"

        for fold_idx in range(self.num_folds):
            LOGGER.info("[CV] ──────── Fold %d / %d ────────", fold_idx + 1, self.num_folds)

            fold_cfg = replace(
                self.data_cfg,
                num_folds=self.num_folds,
                fold_idx=fold_idx,
                cv_seed=self.cv_seed,
            )

            dm = MILDataModule(fold_cfg)
            dm.setup(f"fold_{fold_idx+1}")

            log_split_distributions(dm, stage=f"fold{fold_idx}")
            logger = self._new_logger(f"fold{fold_idx + 1}")
            metrics = self.model_trainer.fit_validate(dm, logger, fold_idx)

            # Aggregate metrics over folds
            for k, v in metrics.items():
                self._fold_results.setdefault(k, []).append(v)

            # Generate plots for validation and training data
            try:
                # Create fold-specific directories
                fold_val_dir = validation_dir / f"fold_{fold_idx}"
                fold_val_dir.mkdir(parents=True, exist_ok=True)
                train_root_dir = results_dir / "train"
                fold_train_dir = train_root_dir / f"fold_{fold_idx}"
                fold_train_dir.mkdir(parents=True, exist_ok=True)

                # Get the best model from the trainer
                model = self.model_trainer.get_best_model()

                save_attention_artifacts = bool(
                    getattr(
                        self.model_trainer.trainer_cfg,
                        "save_attention_artifacts",
                        True,
                    )
                )

                if model is not None and save_attention_artifacts:
                    LOGGER.info(f"[CV] Generating plots for fold {fold_idx + 1}")
                    from ppl.utils.plotting.plot_attention_weights import (
                        plot_attention_weights_from_model,
                    )
                    from ppl.utils.plotting.plot_true_vs_pred import (
                        plot_true_vs_pred_from_model,
                    )

                    # Prepare conformer IDs mapping from the CSV file (shared for train and val)
                    attention_val_dir = fold_val_dir / "attention_weights"
                    attention_train_dir = fold_train_dir / "attention_weights"

                    # Get conformer IDs from the CSV file
                    import pandas as pd
                    from pathlib import Path
                    from ppl.utils.mil_data_handling.data_loader import resolve_path
                    csv_path = resolve_path(self.data_cfg.csv_path)
                    df = pd.read_csv(csv_path)

                    # Create a dictionary mapping bag IDs to lists of conformer IDs
                    # aligned with experimental-pose handling used in bag building
                    conf_ids = {}
                    bag_col = self.data_cfg.bag_id_col
                    inst_col = self.data_cfg.inst_id_col
                    split_col = getattr(self.data_cfg, 'split_col', None)
                    predefined = getattr(self.data_cfg, 'predefined_split', False)

                    if predefined and split_col in df.columns:
                        tr_df = df[df[split_col] == 0]
                        va_df = df[df[split_col] == 1]
                        te_df = df[df[split_col] == 2]

                        # TRAIN: non-experimental only (to avoid experimental-only leakage in interpretation)
                        for bag_id, group in tr_df.groupby(bag_col):
                            inst_list = list(map(str, group[inst_col].astype(str)))
                            inst_nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                            if len(inst_nonexp) == 0:
                                continue
                            conf_ids[str(bag_id)] = inst_nonexp

                        # VAL/TEST: create full and __noexp entries
                        def add_split(split_df):
                            for bag_id, group in split_df.groupby(bag_col):
                                inst_list = list(map(str, group[inst_col].astype(str)))
                                nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                                # Full bag mapping (ALL instances)
                                conf_ids[f"{bag_id}"] = inst_list
                                # __noexp only if fewer than full
                                if len(nonexp) > 0 and len(nonexp) < len(inst_list):
                                    conf_ids[f"{bag_id}__noexp"] = nonexp
                                else:
                                    # Special case: single-instance bag that is experimental-only → create '__noexp' with the same instance
                                    if len(inst_list) == 1 and ("_experimental_pose" in inst_list[0]):
                                        conf_ids[f"{bag_id}__noexp"] = inst_list
                        add_split(va_df)
                        add_split(te_df)
                    else:
                        # Fallback: create full and __noexp entries per bag
                        for bag_id, group in df.groupby(bag_col):
                            inst_list = list(map(str, group[inst_col].astype(str)))
                            nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                            # Full bag
                            conf_ids[f"{bag_id}"] = inst_list
                            # __noexp only if fewer than full
                            if len(nonexp) > 0 and len(nonexp) < len(inst_list):
                                conf_ids[f"{bag_id}__noexp"] = nonexp

                    # --- Validation attention weights ---
                    plot_attention_weights_from_model(
                        model=model,
                        dataloader=dm.val_dataloader(),
                        save_dir=str(attention_val_dir),
                        max_bags=1000,
                        conf_ids=conf_ids
                    )

                    # --- Training attention weights (mirror validation) ---
                    train_dl = dm.train_dataloader()
                    try:
                        has_train = train_dl is not None and len(train_dl) > 0
                    except Exception:
                        has_train = train_dl is not None
                    if has_train:
                        plot_attention_weights_from_model(
                            model=model,
                            dataloader=train_dl,
                            save_dir=str(attention_train_dir),
                            max_bags=1000,
                            conf_ids=conf_ids
                        )

                    # --- Validation true vs predicted ---
                    true_vs_pred_val_path = fold_val_dir / "true_vs_pred.png"
                    plot_true_vs_pred_from_model(
                        model=model,
                        dataloader=dm.val_dataloader(),
                        save_path=str(true_vs_pred_val_path),
                        title="Experimental vs. Predicted Endpoint Value — Validation",
                        task=self.model_trainer.task
                    )

                    # --- Training true vs predicted (mirror validation) ---
                    if has_train:
                        true_vs_pred_train_path = fold_train_dir / "true_vs_pred.png"
                        plot_true_vs_pred_from_model(
                            model=model,
                            dataloader=train_dl,
                            save_path=str(true_vs_pred_train_path),
                            title="Experimental vs. Predicted Endpoint Value — Train",
                            task=self.model_trainer.task
                        )
                elif model is not None:
                    LOGGER.info(
                        "[CV] Plot/artifact export disabled for fold %d "
                        "(trainer.save_attention_artifacts=false)",
                        fold_idx + 1,
                    )
                else:
                    LOGGER.warning(f"[CV] Could not get best model for fold {fold_idx + 1}, skipping plot generation")
            except Exception as e:
                LOGGER.error(f"[CV] Error generating plots for fold {fold_idx + 1}: {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())

        self._summarize_cv()

    def _summarize_cv(self) -> None:
        """Summarize cross-validation results.

        This method:
        1. Calculates mean and standard deviation of each metric across folds
        2. Logs the summary to console
        3. Logs the summary to MLFlow
        """
        LOGGER.info("──────── CV Summary ────────")
        for metric_name, values in self._fold_results.items():
            values_np = np.asarray(values, dtype=np.float32)
            mean, std = values_np.mean(), values_np.std()
            LOGGER.info("CV %-15s: %.4f ± %.4f (n=%d)", metric_name, mean, std, len(values))
            self._mlflow_logger.log_metrics(
                {f"cv_{metric_name}_mean": float(mean), f"cv_{metric_name}_std": float(std)}
            )
