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
    for attr in _RUNTIME_ATTRS:
        setattr(model, f"_{attr}", getattr(trainer_cfg, attr, None))

    if bool(getattr(trainer_cfg, "attention_refinement_enabled", False)):
        LOGGER.info(
            "[MODEL] Aggregator-focus curriculum enabled: metric=%s plateau_patience=%d "
            "min_delta=%.4f embedder/predictor_lr_factor=%.4f "
            "query ramps to weight=%.2f over %d epochs; stops when val plateaus again",
            getattr(trainer_cfg, "attention_refinement_metric", "loss"),
            int(getattr(trainer_cfg, "attention_refinement_patience", 3) or 3),
            float(getattr(trainer_cfg, "attention_refinement_min_delta", 0.005)),
            float(getattr(trainer_cfg, "attention_refinement_lr_factor", 0.1)),
            float(getattr(trainer_cfg, "attention_refinement_query_max_weight", 0.8)),
            int(getattr(trainer_cfg, "attention_refinement_query_ramp_epochs", 2) or 2),
        )
