"""Single train/validation/test run for the Multi-Instance Learning Kit (MILK).

The dataset uses a predefined split (0=train, 1=val, 2=test): the model trains
on the train split, is selected on the validation split, and the best checkpoint
is evaluated on the test split.
"""
from __future__ import annotations

import logging
from pathlib import Path

from ppl.data.data_loader import (
    DataLoaderConfig,
    MILDataModule,
)
from ppl.data.feature_scaling import _feature_blocks
from ppl.pipeline.mlflow_utils import create_mlflow_logger
from ppl.pipeline.results_directory import create_results_directory
from ppl.pipeline.visualization import log_split_distributions
from ppl.training.model_trainer import ModelTrainer

LOGGER = logging.getLogger(__name__)


class SplitTrainer:
    """Train on the predefined split and validate/test the best checkpoint.

    Parameters
    ----------
    data_cfg : DataLoaderConfig
        Configuration for data loading.
    model_trainer : ModelTrainer
        Model trainer instance.
    results_dir : Path
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
        results_dir: Path,
        experiment_name: str,
        run_name: str,
        tracking_uri: str | None = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_trainer = model_trainer
        self.results_dir = results_dir
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
        run_log = logging.getLogger("milk")
        if self.experiment_name:
            run_log.info("Experiment: %s", self.experiment_name)
        run_log.info("Preparing data…")

        dm = MILDataModule(self.data_cfg)
        dm.setup("fit")

        def _n(ds):
            return len(ds) if ds is not None else 0

        # Val/test keep each molecule twice — a full bag (with experimental poses)
        # and a "__noexp" bag (without); train is 1 bag/molecule.
        def _bags_desc(ds):
            n = _n(ds)
            ids = list(getattr(ds, "_bag_ids", [])) if ds is not None else []
            n_noexp = sum(1 for i in ids if str(i).endswith("__noexp"))
            if 0 < n_noexp < n:
                return f"{n} ({n - n_noexp} with exp pose + {n_noexp} __noexp)"
            return str(n)

        # Per-fingerprint-block feature breakdown (several fingerprints are concatenated).
        blocks = ", ".join(
            f"{name}:{len(idx)}" for name, idx in _feature_blocks(dm.feature_names).items()
        )
        run_log.info(
            "Data ready: %s train / %s val / %s test bags · %d features [%s]",
            _bags_desc(dm._train), _bags_desc(dm._val), _bags_desc(dm._test),
            len(dm.feature_names), blocks,
        )
        if self.experiment_name:
            self._save_preprocessing(dm)

        log_split_distributions(dm, stage="train")
        logger = create_mlflow_logger(
            save_dir=self.results_dir,
            experiment_name=self.experiment_name,
            run_name=f"{self.run_name}-train" if self.run_name else "train",
            tracking_uri=self.tracking_uri,
        )
        metrics = self.model_trainer.fit_validate(dm, logger)

        self._export_plots(dm)
        if self.experiment_name:
            run_log.info(
                "Results saved to %s", create_results_directory(self.experiment_name)
            )
        return metrics

    def _save_preprocessing(self, dm) -> None:
        """Persist fitted scalers + feature layout so preprocessing is reproducible.

        The saved ``preprocessing.joblib`` (per-fingerprint-block StandardScalers,
        the retained-feature mask, and descriptor column order) lets you apply the
        exact same transform to new conformers without re-fitting.
        """
        try:
            import joblib

            results_dir = create_results_directory(self.experiment_name)
            payload = {
                "scaler": getattr(dm, "scaler", None),
                "feature_names": list(getattr(dm, "feature_names", [])),
                "feature_mask": getattr(dm, "feature_mask", None),
                "descriptor_cols": list(getattr(dm.cfg, "descriptor_cols", []) or []),
            }
            path = results_dir / "preprocessing.joblib"
            joblib.dump(payload, path)
            logging.getLogger("milk").info("Saved preprocessing (scalers) to %s", path)
        except Exception as e:
            LOGGER.warning("Failed to save preprocessing artifacts: %s", e)

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

        logging.getLogger("milk").info(
            "Exporting attention weights and true-vs-predicted plots for train/validation "
            "molecules — this can take a few minutes…"
        )

        try:
            from ppl.plotting.plot_attention_weights import (
                plot_attention_weights_from_model,
            )
            from ppl.plotting.plot_true_vs_pred import (
                plot_true_vs_pred_from_model,
            )

            results_dir = create_results_directory(self.experiment_name)
            val_dir = results_dir / "validation"
            train_dir = results_dir / "train"
            val_dir.mkdir(parents=True, exist_ok=True)
            train_dir.mkdir(parents=True, exist_ok=True)

            # Conformer IDs captured during bag construction (guaranteed aligned
            # with each bag's rows, including "__noexp" bags).
            conf_ids = dict(getattr(dm, "bag_conf_ids", {}) or {})
            task = self.model_trainer.task

            def _plot_split(loader, out_dir, label):
                plot_attention_weights_from_model(
                    model=model, dataloader=loader,
                    save_dir=str(out_dir / "attention_weights"),
                    max_bags=1000, conf_ids=conf_ids,
                )
                plot_true_vs_pred_from_model(
                    model=model, dataloader=loader,
                    save_path=str(out_dir / "true_vs_pred.png"),
                    title=f"Experimental vs. Predicted Endpoint Value — {label}",
                    task=task,
                )

            val_loader = dm.val_dataloader()
            if val_loader is not None and (not hasattr(val_loader, "__len__") or len(val_loader) > 0):
                _plot_split(val_loader, val_dir, "Validation")

            train_loader = dm.train_dataloader()
            if train_loader is not None:
                _plot_split(train_loader, train_dir, "Train")
        except Exception as e:
            LOGGER.error(f"[TRAIN] Error generating plots: {e}")
            import traceback

            LOGGER.debug(traceback.format_exc())

