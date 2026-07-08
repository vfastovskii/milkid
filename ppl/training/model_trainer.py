"""Model trainer for the Multi-Instance Learning Kit (MILK).

This module provides a class for building and training models for the MILK project.
It handles model building, trainer configuration, and model training.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)

from ppl.data.data_loader import MILDataModule
from ppl.models.model_builder import ModelBuilder
from ppl.config.model_builder_config import ModelBuilderConfig
from ppl.config.trainer_config import TrainerConfig, TrainerBuilder
from ppl.pipeline.mlflow_utils import SafeMLFlowLogger, log_metrics
from ppl.training.artifacts import export_fit_artifacts, evaluate_on_test
from ppl.training.callbacks import build_callbacks, extract_conformer_ids
from ppl.training.runtime_config import attach_trainer_runtime_config

LOGGER = logging.getLogger(__name__)


class ModelTrainer:
    """Class for building and training models.

    This class is responsible for:
    - Building models with the correct input dimension
    - Configuring PyTorch Lightning trainers
    - Training models on data
    - Logging hyperparameters and metrics

    Parameters
    ----------
    model_cfg : ModelBuilderConfig
        Configuration for model building
    trainer_cfg : ModelTrainerConfig
        Configuration for model training
    log_save_dir : Path
        Directory to save logs
    seed : int
        Global RNG seed
    task : str
        Task type (regression or classification)
    """

    def __init__(
        self,
        model_cfg: ModelBuilderConfig,
        trainer_cfg: TrainerConfig,
        log_save_dir: Path,
        seed: int,
        task: str,
        experiment_name: str = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.log_save_dir = log_save_dir
        self.seed = seed
        self.task = task
        self.max_epochs = trainer_cfg.max_epochs
        self.device = trainer_cfg.device
        self.experiment_name = experiment_name
        self.data_module = None  # Will store the MILDataModule instance

    def build_model(self, input_dim: int) -> pl.LightningModule:
        """Build a model with the given input dimension.

        Parameters
        ----------
        input_dim : int
            Input dimension for the model (number of features)

        Returns
        -------
        pl.LightningModule
            Built model
        """
        model_cfg = replace(self.model_cfg, input_dim=input_dim, task=self.task)
        LOGGER.info("[MODEL] Injected input_dim=%d into ModelBuilderConfig", input_dim)

        model = ModelBuilder(model_cfg).build()
        self._attach_trainer_runtime_config(model)
        LOGGER.info("[MODEL] Model successfully built – parameters verified.")

        n_params = sum(p.numel() for p in model.parameters())
        logging.getLogger("milk").info(
            "Model: %s → %s → %s (%.1fM params)",
            self.model_cfg.embedder_type,
            self.model_cfg.aggregator_type,
            self.model_cfg.predictor_type,
            n_params / 1e6,
        )
        return model

    def _attach_trainer_runtime_config(self, model: pl.LightningModule) -> None:
        """Reattach TrainerConfig runtime controls to a (re)loaded model."""
        attach_trainer_runtime_config(
            model, self.trainer_cfg, self._checkpoint_monitor_metric()
        )

    def _validation_monitor_metric(self) -> str:
        """Metric used for checkpointing/early stopping on validation data."""
        configured_monitor = getattr(self.trainer_cfg, "checkpoint_monitor", None)
        if configured_monitor:
            monitor = str(configured_monitor).strip()
            if monitor:
                return monitor if monitor.startswith("val_") else f"val_{monitor}"
        if str(self.task).lower() == "regression":
            return "val_mae"
        return "val_loss"

    def _checkpoint_monitor_metric(self) -> str:
        """Metric used for selecting checkpoints and early stopping."""
        return self._validation_monitor_metric()

    def _extract_conformer_ids(self):
        """Conformer-ID mapping for attention/embedding artifacts."""
        return extract_conformer_ids(self.data_module)

    def callbacks(self) -> Sequence[pl.callbacks.Callback]:
        """Build the Lightning callback list for a training run."""
        return build_callbacks(self)

    def create_trainer(self, logger: SafeMLFlowLogger) -> pl.Trainer:
        """Create a PyTorch Lightning Trainer using TrainerBuilder.

        Parameters
        ----------
        logger : SafeMLFlowLogger
            MLFlow logger to use

        Returns
        -------
        pl.Trainer
            Configured PyTorch Lightning Trainer
        """
        # Create a trainer using TrainerBuilder
        return TrainerBuilder.build(
            config=self.trainer_cfg,
            logger=logger,
            callbacks=self.callbacks()
        )

    def log_hyperparams(self, logger: SafeMLFlowLogger, model: pl.LightningModule) -> None:
        """Log hyperparameters to MLFlow.

        Parameters
        ----------
        logger : SafeMLFlowLogger
            MLFlow logger to use
        model : pl.LightningModule
            Model to extract hyperparameters from
        """
        hparams = dict(
            task=self.task,
            embedder=self.model_cfg.embedder_type,
            aggregator=self.model_cfg.aggregator_type,
            predictor=self.model_cfg.predictor_type,
            max_epochs=self.max_epochs,
            optimiser=model.hparams.get("optimizer", "adamw"),
        )
        logger.log_hyperparams(hparams)


    def get_best_model(self) -> pl.LightningModule:
        """Get the best model from the last training run.

        Returns
        -------
        pl.LightningModule
            Best model from the last training run, or None if no model is available
        """
        # Check if we have a trainer with a checkpoint callback
        if not hasattr(self, "_last_trainer") or not self._last_trainer:
            LOGGER.warning("No trainer available to get best model from")
            return None

        # Get the checkpoint callback
        ckpt_callback = None
        for callback in self._last_trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                ckpt_callback = callback
                break

        if not ckpt_callback:
            LOGGER.warning("No checkpoint callback found in trainer")
            return None

        # Get the best model path
        best_model_path = ckpt_callback.best_model_path
        if not best_model_path:
            LOGGER.warning("No best model path found in checkpoint callback")
            return None

        # Load the best model
        try:
            LOGGER.info(f"Loading best model from {best_model_path}")
            model_class = self._last_model.__class__

            # Extract the required arguments from the original model
            core = self._last_model.core
            optim_cfg = self._last_model.optim_cfg
            task = self._last_model.task

            # Load the model with the required arguments
            model = model_class.load_from_checkpoint(
                best_model_path,
                core=core,
                task=task,
                optim_cfg=optim_cfg
            )
            self._attach_trainer_runtime_config(model)
            best_epoch = self._get_best_epoch_from_trainer(self._last_trainer)
            if best_epoch is not None:
                setattr(model, "_evaluation_epoch_override", best_epoch)
                setattr(model, "_instance_importance_epoch", best_epoch)
                if hasattr(model, "_apply_attention_refinement_phase"):
                    model._apply_attention_refinement_phase(best_epoch, log=False)
            return model
        except Exception as e:
            LOGGER.error(f"Error loading best model: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
            return None

    def _get_best_epoch_from_trainer(self, trainer: pl.Trainer) -> Optional[int]:
        """Extract the epoch index of the best checkpoint from a trainer's ModelCheckpoint.
        Returns None if unavailable.
        """
        try:
            ckpt_cb = None
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    ckpt_cb = cb
                    break
            if ckpt_cb is None or not getattr(ckpt_cb, "best_model_path", None):
                return None
            import os, re
            fname = os.path.basename(ckpt_cb.best_model_path)
            m = re.search(r"(?:ep|epoch=)(\d+)", fname)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    def _get_best_checkpoint_score_from_trainer(
        self,
        trainer: pl.Trainer,
    ) -> Optional[float]:
        """Extract the best monitored checkpoint score, if available."""
        try:
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint):
                    score = getattr(cb, "best_model_score", None)
                    if score is None:
                        return None
                    return float(score.detach().cpu().item())
        except Exception:
            pass
        return None

    def _maybe_export_validation_instance_importance(
        self,
        *,
        dm: MILDataModule,
        source_model: pl.LightningModule,
        source_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Optionally export validation-set instance importance.

        This is an interpretation pass only. It never filters the train set and
        never launches a second training/refit iteration.
        """
        if not getattr(
            self.trainer_cfg,
            "validation_instance_importance_enabled",
            False,
        ):
            return source_metrics

        from ppl.training.post_training_ablation import (
            run_instance_importance_selection,
        )
        from ppl.pipeline.results_directory import create_results_directory

        if self.experiment_name:
            save_dir = create_results_directory(self.experiment_name) / "validation"
        else:
            save_dir = self.log_save_dir / "validation"

        conf_ids = self._extract_conformer_ids()
        expected_val_bag_ids = (
            dm.val_bag_ids()
            if hasattr(dm, "val_bag_ids")
            else list(getattr(getattr(dm, "_val", None), "_bag_ids", []))
        )
        if not expected_val_bag_ids:
            LOGGER.info("[VAL_IMPORTANCE] No validation set; skipping export")
            return source_metrics

        min_keep = getattr(
            self.trainer_cfg,
            "validation_instance_importance_min_keep",
            3,
        )
        max_keep = getattr(
            self.trainer_cfg,
            "validation_instance_importance_max_keep",
            12,
        )
        LOGGER.info(
            "[VAL_IMPORTANCE] Exporting validation instance importance for %d bags",
            len(expected_val_bag_ids),
        )
        result = run_instance_importance_selection(
            model=source_model,
            dataloader=dm.val_dataloader(),
            save_dir=save_dir,
            min_keep=min_keep,
            max_keep=max_keep,
            attention_weight=getattr(
                self.trainer_cfg,
                "validation_instance_importance_attention_weight",
                0.5,
            ),
            impact_weight=getattr(
                self.trainer_cfg,
                "validation_instance_importance_impact_weight",
                0.5,
            ),
            conf_ids=conf_ids,
            expected_bag_ids=expected_val_bag_ids,
            split_name="validation",
        )
        log_metrics(
            result.summary,
            stage="validation_instance_importance",
            prefix="val",
        )

        metrics = dict(source_metrics)
        metrics.update(
            {
                f"validation_importance_{k}": v
                for k, v in result.summary.items()
            }
        )
        return metrics

    def fit_validate(self, dm: MILDataModule, logger: SafeMLFlowLogger) -> Dict[str, float]:
        """Fit the model on training data and validate on validation data.

        Parameters
        ----------
        dm : MILDataModule
            Data module containing the data loaders
        logger : SafeMLFlowLogger
            MLFlow logger to use

        Returns
        -------
        Dict[str, float]
            Validation metrics
        """
        # Store the data module for later use
        self.data_module = dm
        
        model = self.build_model(len(dm.feature_names))
        trainer = self.create_trainer(logger)
        self.log_hyperparams(logger, model)

        # Store the trainer and model for later use
        self._last_trainer = trainer
        self._last_model = model

        logging.getLogger("milk").info(
            "Training up to %d epochs on %s…", self.max_epochs, self.device
        )
        trainer.fit(model, datamodule=dm)

        # Get the best model from checkpoint instead of using the model in memory (last epoch)
        best_epoch = self._get_best_epoch_from_trainer(trainer)
        best_model = self.get_best_model()
        if best_model is not None:
            LOGGER.info("[MODEL] Using best model checkpoint for final validation")
            if best_epoch is not None:
                setattr(best_model, "_evaluation_epoch_override", best_epoch)
                setattr(best_model, "_instance_importance_epoch", best_epoch)
            # Disable progress bar and verbosity for validation to avoid multiple progress bars
            # This ensures only one progress bar is displayed at a time
            val_metrics = trainer.validate(best_model, datamodule=dm, verbose=False)[0]
        else:
            if bool(
                getattr(
                    self.trainer_cfg,
                    "checkpoint_after_attention_refinement",
                    False,
                )
            ):
                raise RuntimeError(
                    "No eligible best checkpoint was produced after the "
                    "attention-refinement gate. Refusing to register last-epoch "
                    "metrics for this trial."
                )
            LOGGER.warning("[MODEL] Could not load best model checkpoint, using model in memory")
            if best_epoch is not None:
                setattr(model, "_evaluation_epoch_override", best_epoch)
                setattr(model, "_instance_importance_epoch", best_epoch)
            # Fallback to model in memory if best model loading fails
            val_metrics = trainer.validate(model, datamodule=dm, verbose=False)[0]
        log_metrics(val_metrics, stage="val", prefix="val")

        plot_model_candidate = best_model if best_model is not None else model
        if best_epoch is not None:
            setattr(plot_model_candidate, "_instance_importance_epoch", best_epoch)
        val_metrics = self._maybe_export_validation_instance_importance(
            dm=dm,
            source_model=plot_model_candidate,
            source_metrics=dict(val_metrics),
        )
        best_model = plot_model_candidate

        if self.experiment_name:
            export_fit_artifacts(self, dm, model, best_model, val_metrics)
        evaluate_on_test(self, dm, model)

        return val_metrics

