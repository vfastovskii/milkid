"""Base MIL model lightning wrapper."""

from __future__ import annotations

import dataclasses as dc
import logging
import traceback

import pytorch_lightning as pl
import torchmetrics as tm

from ppl.models.loss import Loss
from ppl.config.trainer_optim_config import TrainerOptimConfig
from ppl.models.core import MILCore

from ppl.models.lightning.training import TrainingMethods
from ppl.models.lightning.optimization import OptimizationMethods
from ppl.models.lightning._helpers import log_memory_usage, log_model_size

LOGGER = logging.getLogger(__name__)

# Lightning‑native MIL model wrapper
class MILModelLightningWrapper(
    TrainingMethods,
    OptimizationMethods,
    pl.LightningModule
):
    """LightningModule wrapping the core MIL network with memory management and mixed precision."""

    def __init__(self, core: MILCore, task: str, optim_cfg: TrainerOptimConfig) -> None:
        """Initialize the MIL model wrapper.

        Parameters
        ----------
        core : MILCore
            Core MIL network
        task : str
            Task type (classification or regression)
        optim_cfg : TrainerOptimConfig
            Optimizer configuration
        """
        super().__init__()
        self.core = core
        self.task = task.lower()
        self._last_train_metrics = {}
        self._last_train_loss = None
        self._train_epoch_metrics_consumed_epoch = None
        self._attention_refinement_bad_epochs = 0
        self._attention_refinement_best_val = None
        self._attention_refinement_trigger_epoch = None
        self._attention_refinement_lr_reduced = False
        self._attention_refinement_stop_logged = False
        # Phase-B (aggregator-focus) plateau detector for the loss-linked stop.
        self._focus_best_val = None
        self._focus_bad_epochs = 0
        self._epoch_loss_sums = {"train": 0.0, "val": 0.0, "test": 0.0}
        self._epoch_loss_counts = {"train": 0, "val": 0, "test": 0}
        self.optim_cfg = optim_cfg

        log_memory_usage("Initialization")

        # Hyper‑params snapshot for Lightning loggers / checkpoints
        if dc.is_dataclass(optim_cfg):
            hparams = dc.asdict(optim_cfg)
        else:
            hparams = dict(optim_cfg)  # fallback if already a dict

        # Add mixed precision and distributed training settings to hparams
        hparams.update({
            "task": self.task,
            "use_mixed_precision": getattr(optim_cfg, "use_mixed_precision", True),
            "gradient_accumulation_steps": getattr(optim_cfg, "gradient_accumulation_steps", 1),
        })

        self.save_hyperparameters(hparams)

        # Loss: supervised objective (+ optional CCC anti-shrinkage term); attention
        # diagnostics stay out of loss. CCC only affects the TRAIN loss — eval uses
        # criterion.base (plain MSE) so val/test metrics stay comparable.
        try:
            self.criterion = Loss(
                task=self.task,
                ccc_weight=float(getattr(optim_cfg, "ccc_weight", 0.0) or 0.0),
            )
        except Exception as e:
            LOGGER.error(f"Failed to initialize loss function: {e}")
            LOGGER.error(traceback.format_exc())
            raise ValueError(f"Loss function initialization failed: {e}") from e

        # Metrics (shared collections for train/val/test)
        try:
            if self.task == "classification":
                base_metrics = tm.MetricCollection({
                    "auroc": tm.classification.BinaryAUROC(),
                    "accuracy": tm.classification.BinaryAccuracy(),
                    "f1": tm.classification.BinaryF1Score(),
                    "precision": tm.classification.BinaryPrecision(),
                    "recall": tm.classification.BinaryRecall(),
                })
            else:
                base_metrics = tm.MetricCollection({
                    "mae": tm.regression.MeanAbsoluteError(),
                    "rmse": tm.regression.MeanSquaredError(squared=False),
                    "r2": tm.regression.R2Score(),
                    "pearson": tm.PearsonCorrCoef(),
                    "spearman": tm.SpearmanCorrCoef(),
                })

            self.train_metrics = base_metrics.clone()
            self.val_metrics = base_metrics.clone()
            self.test_metrics = base_metrics.clone()
        except Exception as e:
            LOGGER.error(f"Failed to initialize metrics: {e}")
            LOGGER.error(traceback.format_exc())
            raise ValueError(f"Metrics initialization failed: {e}") from e

        # Log model size
        log_model_size(self)
