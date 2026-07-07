"""Single train/validation/test run for the Multi-Instance Learning Kit (MILK).

The dataset uses a predefined split (0=train, 1=val, 2=test): the model trains
on the train split, is selected on the validation split, and the best checkpoint
is evaluated on the test split.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ppl.utils.mil_data_handling.data_loader import (
    DataLoaderConfig,
    MILDataModule,
    resolve_path,
)
from ppl.utils.pipeline.mlflow_utils import create_mlflow_logger
from ppl.utils.pipeline.results_directory import create_results_directory
from ppl.utils.pipeline.visualization import log_split_distributions
from ppl.utils.model_trainer.model_trainer import ModelTrainer

LOGGER = logging.getLogger(__name__)


class SplitTrainer:
    """Train on the predefined split and validate/test the best checkpoint.

    Parameters
    ----------
    data_cfg : DataLoaderConfig
        Configuration for data loading.
    model_trainer : ModelTrainer
        Model trainer instance.
    log_save_dir : Path
        Directory to save logs.
    experiment_name : str
        Name of the experiment.
    run_name : str
        Name of the run.
    tracking_uri : str | None
        URI for the MLflow tracking server.
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

    def run(self) -> dict[str, float]:
        """Fit on the train split, validate on val, and test the best model.

        Returns
        -------
        dict[str, float]
            Validation metrics for the best checkpoint.
        """
        LOGGER.info("[TRAIN] Train on split=0, validate on split=1, test on split=2")

        dm = MILDataModule(self.data_cfg)
        dm.setup("fit")

        log_split_distributions(dm, stage="train")
        logger = create_mlflow_logger(
            save_dir=self.log_save_dir,
            experiment_name=self.experiment_name,
            run_name=f"{self.run_name}-train" if self.run_name else "train",
            tracking_uri=self.tracking_uri,
        )
        metrics = self.model_trainer.fit_validate(dm, logger)

        self._export_plots(dm)
        return metrics

    def _export_plots(self, dm: MILDataModule) -> None:
        """Export attention-weight and true-vs-predicted plots for train/val."""
        save_attention_artifacts = bool(
            getattr(self.model_trainer.trainer_cfg, "save_attention_artifacts", True)
        )
        if not save_attention_artifacts:
            LOGGER.info("[TRAIN] Plot export disabled (save_attention_artifacts=false)")
            return

        model = self.model_trainer.get_best_model()
        if model is None:
            LOGGER.warning("[TRAIN] No best model available, skipping plot generation")
            return

        try:
            from ppl.utils.plotting.plot_attention_weights import (
                plot_attention_weights_from_model,
            )
            from ppl.utils.plotting.plot_true_vs_pred import (
                plot_true_vs_pred_from_model,
            )

            results_dir = create_results_directory(self.experiment_name)
            val_dir = results_dir / "validation"
            train_dir = results_dir / "train"
            val_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)

            conf_ids = self._build_conf_ids()
            task = self.model_trainer.task

            # Validation artifacts
            val_loader = dm.val_dataloader()
            if val_loader is not None and (not hasattr(val_loader, "__len__") or len(val_loader) > 0):
                plot_attention_weights_from_model(
                    model=model,
                    dataloader=val_loader,
                    save_dir=str(val_dir / "attention_weights"),
                    max_bags=1000,
                    conf_ids=conf_ids,
                )
                plot_true_vs_pred_from_model(
                    model=model,
                    dataloader=val_loader,
                    save_path=str(val_dir / "true_vs_pred.png"),
                    title="Experimental vs. Predicted Endpoint Value — Validation",
                    task=task,
                )

            # Training artifacts
            train_loader = dm.train_dataloader()
            if train_loader is not None:
                plot_attention_weights_from_model(
                    model=model,
                    dataloader=train_loader,
                    save_dir=str(train_dir / "attention_weights"),
                    max_bags=1000,
                    conf_ids=conf_ids,
                )
                plot_true_vs_pred_from_model(
                    model=model,
                    dataloader=train_loader,
                    save_path=str(train_dir / "true_vs_pred.png"),
                    title="Experimental vs. Predicted Endpoint Value — Train",
                    task=task,
                )
        except Exception as e:
            LOGGER.error(f"[TRAIN] Error generating plots: {e}")
            import traceback

            LOGGER.debug(traceback.format_exc())

    def _build_conf_ids(self) -> dict[str, list[str]]:
        """Map each bag ID to its conformer instance IDs from the source CSV."""
        df = pd.read_csv(resolve_path(self.data_cfg.csv_path))
        bag_col = self.data_cfg.bag_id_col
        inst_col = self.data_cfg.inst_id_col
        conf_ids: dict[str, list[str]] = {}
        for bag_id, group in df.groupby(bag_col):
            conf_ids[str(bag_id)] = list(map(str, group[inst_col].astype(str)))
        return conf_ids
