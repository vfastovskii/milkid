"""Model evaluation utilities for the Multi-Instance Learning Kit (MILK).

This module provides a class for evaluating models in the MILK project.
It handles training models on the full training data and evaluating them on test data.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from ppl.utils.mil_data_handling.data_loader import DataLoaderConfig, MILDataModule
from ppl.utils.pipeline.mlflow_utils import SafeMLFlowLogger, create_mlflow_logger
from ppl.utils.model_trainer.model_trainer import ModelTrainer
from ppl.utils.pipeline.visualization import log_split_distributions
from ppl.utils.pipeline.results_directory import create_results_directory

LOGGER = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating models.

    This class is responsible for:
    - Training models on the full training data
    - Evaluating models on test data
    - Logging metrics and artifacts

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
        self.cv_seed = data_cfg.cv_seed

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

    def final_fit_test(self) -> None:
        """Fit the model on the full training data and evaluate on the test data.

        This method:
        1. Creates a data module with the full training data
        2. Logs the label distributions
        3. Fits the model on the full training data
        4. Evaluates the model on the test data
        5. Generates attention weights and true vs. predicted plots
        """
        LOGGER.info("[FINAL] Stage 2: fit on combined train+val; evaluate on held‑out test (scaler fit on train+val only, no leak)")

        # If num_folds is already 1, don't override it
        if self.data_cfg.num_folds == 1:
            full_cfg = replace(
                self.data_cfg,
                fold_idx=0,
                cv_seed=self.cv_seed,
            )
        else:
            full_cfg = replace(
                self.data_cfg,
                num_folds=5,  # Still use 5 folds for consistency, but only use fold 0
                fold_idx=0,
                cv_seed=self.cv_seed,
            )

        dm = MILDataModule(full_cfg)
        dm.setup(is_final_model=True)
        LOGGER.info("[FINAL] Fit/test – Using DataLoaderConfig:\n%s", full_cfg)
        LOGGER.info("[FINAL] Creating directory for final model data")

        log_split_distributions(dm, stage="final")
        logger = self._new_logger("test")
        self.model_trainer.fit_test(dm, logger)

        # Generate plots for validation and test data
        try:
            # Create results directory
            results_dir = create_results_directory(self.experiment_name)
            validation_dir = results_dir / "validation"
            test_dir = results_dir / "test"

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
                LOGGER.info("[FINAL] Generating plots for validation and test data")
                from ppl.utils.plotting.export_bag_data_to_csv import (
                    export_bag_data_to_csv,
                )
                from ppl.utils.plotting.plot_attention_weights import (
                    plot_attention_weights_from_model,
                )
                from ppl.utils.plotting.plot_true_vs_pred import (
                    plot_true_vs_pred_from_model,
                )

                # Generate validation plots
                LOGGER.info("[FINAL] Generating validation plots")

                # Generate attention weights plot for validation data
                val_attention_dir = validation_dir / "final"

                # Get conformer IDs from the CSV file
                import pandas as pd
                from pathlib import Path
                from ppl.utils.mil_data_handling.data_loader import resolve_path
                csv_path = resolve_path(self.data_cfg.csv_path)
                df = pd.read_csv(csv_path)

                # Create a dictionary mapping bag IDs to lists of conformer IDs
                conf_ids = {}
                for bag_id, group in df.groupby(self.data_cfg.bag_id_col):
                    conf_ids[str(bag_id)] = group[self.data_cfg.inst_id_col].tolist()

                plot_attention_weights_from_model(
                    model=model,
                    dataloader=dm.val_dataloader(),
                    save_dir=str(val_attention_dir),
                    max_bags=1000,
                    conf_ids=conf_ids
                )

                # Export bag data to CSV for validation data
                val_csv_dir = validation_dir / "final" / "bag_data_csv"
                export_bag_data_to_csv(
                    model=model,
                    dataloader=dm.val_dataloader(),
                    save_dir=str(val_csv_dir),
                    max_bags=1000,
                    conf_ids=conf_ids,
                    task=self.model_trainer.task
                )

                # Generate true vs. predicted plot for validation data
                val_true_vs_pred_path = validation_dir / "final" / "true_vs_pred.png"
                plot_true_vs_pred_from_model(
                    model=model,
                    dataloader=dm.val_dataloader(),
                    save_path=str(val_true_vs_pred_path),
                    title="Experimental vs. Predicted Endpoint Value — Validation",
                    task=self.model_trainer.task
                )
                
                # Generate training plots
                LOGGER.info("[FINAL] Generating training plots")
                train_dir = results_dir / "train"
                
                # Generate attention weights plot for training data
                train_attention_dir = train_dir / "final"
                plot_attention_weights_from_model(
                    model=model,
                    dataloader=dm.train_dataloader(),
                    save_dir=str(train_attention_dir),
                    max_bags=1000,
                    conf_ids=conf_ids
                )
                
                # Generate true vs. predicted plot for training data
                train_true_vs_pred_path = train_dir / "final" / "true_vs_pred.png"
                plot_true_vs_pred_from_model(
                    model=model,
                    dataloader=dm.train_dataloader(),
                    save_path=str(train_true_vs_pred_path),
                    title="Experimental vs. Predicted Endpoint Value — Train",
                    task=self.model_trainer.task
                )
                
                # Export bag data to CSV for training data
                train_csv_dir = train_dir / "final" / "bag_data_csv"
                export_bag_data_to_csv(
                    model=model,
                    dataloader=dm.train_dataloader(),
                    save_dir=str(train_csv_dir),
                    max_bags=1000,
                    conf_ids=conf_ids,
                    task=self.model_trainer.task
                )

                # Generate test plots if test data is available
                if hasattr(dm, 'test_dataloader') and dm.test_dataloader() is not None:
                    LOGGER.info("[FINAL] Generating test plots")

                    # Generate attention weights plot for test data
                    test_attention_dir = test_dir / "final"

                    # We can reuse the conf_ids dictionary created earlier
                    plot_attention_weights_from_model(
                        model=model,
                        dataloader=dm.test_dataloader(),
                        save_dir=str(test_attention_dir),
                        max_bags=1000,
                        conf_ids=conf_ids
                    )
                    
                    # Export bag data to CSV for test data
                    test_csv_dir = test_dir / "final" / "bag_data_csv"
                    export_bag_data_to_csv(
                        model=model,
                        dataloader=dm.test_dataloader(),
                        save_dir=str(test_csv_dir),
                        max_bags=1000,
                        conf_ids=conf_ids,
                        task=self.model_trainer.task
                    )

                    # Generate true vs. predicted plot for test data
                    test_true_vs_pred_path = test_dir / "final" / "true_vs_pred.png"
                    plot_true_vs_pred_from_model(
                        model=model,
                        dataloader=dm.test_dataloader(),
                        save_path=str(test_true_vs_pred_path),
                        title="Experimental vs. Predicted Endpoint Value — Test",
                        task=self.model_trainer.task
                    )
                else:
                    LOGGER.warning("[FINAL] No test dataloader available, skipping test plots")
            elif model is not None:
                LOGGER.info(
                    "[FINAL] Plot/artifact export disabled "
                    "(trainer.save_attention_artifacts=false)"
                )
            else:
                LOGGER.warning("[FINAL] Could not get best model, skipping plot generation")
        except Exception as e:
            LOGGER.error(f"[FINAL] Error generating plots: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
