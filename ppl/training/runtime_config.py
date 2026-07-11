"""Attach non-hparam trainer runtime controls to a (re)loaded LightningModule.

Checkpoint loading reconstructs the LightningModule from its hparams and explicit
constructor args. Runtime controls owned by TrainerConfig (e.g. attention-refinement
scheduling) are not constructor args, so they must be reattached before
validation/test/export.
"""
from __future__ import annotations

import logging

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)

_RUNTIME_ATTRS = (
    "overfit_gap_stop_enabled",
    "overfit_gap_metric",
    "overfit_gap_patience",
    "overfit_gap_min_delta",
    "overfit_gap_abs_threshold",
    "overfit_gap_rel_threshold",
    "overfit_gap_require_val_worse_than_best",
    "attention_refinement_enabled",
    "attention_refinement_metric",
    "attention_refinement_patience",
    "attention_refinement_min_delta",
    "attention_refinement_lr_factor",
    "attention_refinement_query_ramp_epochs",
    "attention_refinement_query_max_weight",
)


def attach_trainer_runtime_config(
    model: pl.LightningModule, trainer_cfg, monitor_metric: str
) -> None:
    """Copy TrainerConfig runtime controls onto ``model`` as private attributes."""
    checkpoint_min_epoch = int(getattr(trainer_cfg, "checkpoint_min_epoch", 0) or 0)
    setattr(model, "_checkpoint_min_epoch", checkpoint_min_epoch)
    for attr in _RUNTIME_ATTRS:
        setattr(model, f"_{attr}", getattr(trainer_cfg, attr, None))

    if checkpoint_min_epoch > 0:
        LOGGER.info(
            "[MODEL] Best-checkpoint selection uses %s only for epochs >= %d",
            monitor_metric, checkpoint_min_epoch,
        )
    if bool(getattr(trainer_cfg, "overfit_gap_stop_enabled", False)):
        LOGGER.info(
            "[MODEL] Overfit-gap early stop enabled: metric=%s "
            "patience=%d abs_gap>=%.4f rel_gap>=%.4f",
            getattr(trainer_cfg, "overfit_gap_metric", "rmse"),
            int(getattr(trainer_cfg, "overfit_gap_patience", 2) or 2),
            float(getattr(trainer_cfg, "overfit_gap_abs_threshold", 0.10)),
            float(getattr(trainer_cfg, "overfit_gap_rel_threshold", 0.15)),
        )
    if bool(getattr(trainer_cfg, "attention_refinement_enabled", False)):
        LOGGER.info(
            "[MODEL] Aggregator-focus curriculum enabled: metric=%s plateau_patience=%d "
            "min_delta=%.4f embedder/predictor_lr_factor=%.4f "
            "query ramps to weight=%.2f over %d epochs; stops when val plateaus again",
            getattr(trainer_cfg, "attention_refinement_metric", "rmse"),
            int(getattr(trainer_cfg, "attention_refinement_patience", 3) or 3),
            float(getattr(trainer_cfg, "attention_refinement_min_delta", 0.005)),
            float(getattr(trainer_cfg, "attention_refinement_lr_factor", 0.1)),
            float(getattr(trainer_cfg, "attention_refinement_query_max_weight", 0.8)),
            int(getattr(trainer_cfg, "attention_refinement_query_ramp_epochs", 2) or 2),
        )
    if bool(getattr(trainer_cfg, "checkpoint_after_attention_refinement", False)):
        LOGGER.info(
            "[MODEL] Best-checkpoint selection is gated to attention "
            "refinement: min_query_epochs=%d",
            int(getattr(trainer_cfg, "checkpoint_min_query_epochs", 1) or 1),
        )
