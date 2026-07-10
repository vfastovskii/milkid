"""Lightning callback construction for ModelTrainer.

Functions take the ``ModelTrainer`` instance (``mt``) to reuse its config and
metric helpers; kept here to keep ModelTrainer focused on build/fit.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from ppl.pipeline.results_directory import create_results_directory
from ppl.training.checkpoints import MinEpochModelCheckpoint

LOGGER = logging.getLogger(__name__)


def make_model_checkpoint(mt, **kwargs) -> ModelCheckpoint:
    """ModelCheckpoint, gated by MinEpochModelCheckpoint when configured."""
    min_epoch = int(getattr(mt.trainer_cfg, "checkpoint_min_epoch", 0) or 0)
    monitor = kwargs.get("monitor")
    require_refinement = bool(
        getattr(mt.trainer_cfg, "checkpoint_after_attention_refinement", False)
    )
    min_query_epochs = int(getattr(mt.trainer_cfg, "checkpoint_min_query_epochs", 1) or 1)
    if (min_epoch > 0 or require_refinement) and monitor == mt._checkpoint_monitor_metric():
        return MinEpochModelCheckpoint(
            min_epoch=min_epoch,
            require_attention_refinement=require_refinement,
            min_query_epochs=min_query_epochs,
            **kwargs,
        )
    return ModelCheckpoint(**kwargs)


def make_early_stopping_callback(mt, validation_monitor: str):
    """EarlyStopping, or None when attention-refinement owns the post-overfit phase."""
    if bool(getattr(mt.trainer_cfg, "attention_refinement_enabled", False)):
        LOGGER.info(
            "[MODEL] Lightning EarlyStopping disabled because "
            "attention-refinement controls the post-overfit training phase."
        )
        return None
    return EarlyStopping(
        monitor=validation_monitor,
        mode="min",
        patience=mt.model_cfg.optim.lr_patience,
        verbose=True,
        check_on_train_epoch_end=False,
    )


def _checkpoint_dirs(mt):
    """Return (checkpoint_dirpath, exp_dir, attention_dir) for the run."""
    if mt.experiment_name:
        exp_dir = create_results_directory(mt.experiment_name)
        dirpath = exp_dir / "models"
        dirpath.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"[MODEL] Saving best model to {dirpath}")
        attention_dir = None
        if mt.trainer_cfg.log_per_epoch:
            attention_dir = exp_dir / "attention_weights_per_epoch"
            attention_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[MODEL] Saving per-epoch attention weights to {attention_dir}")
        return dirpath, exp_dir, attention_dir

    dirpath = mt.results_dir / "checkpoints"
    attention_dir = (
        mt.results_dir / "attention_weights_per_epoch"
        if mt.trainer_cfg.log_per_epoch
        else None
    )
    if attention_dir is not None:
        attention_dir.mkdir(parents=True, exist_ok=True)
    return dirpath, None, attention_dir


def build_callbacks(mt) -> Sequence[pl.callbacks.Callback]:
    """Build the Lightning callback list for a training run."""
    run_suffix = f"seed{mt.seed}"
    iteration_label = getattr(mt, "_training_iteration_label", "")
    if iteration_label:
        run_suffix = f"{run_suffix}_{iteration_label}"

    dirpath, exp_dir, attention_dir = _checkpoint_dirs(mt)

    validation_monitor = mt._checkpoint_monitor_metric()
    ckpt_cb = make_model_checkpoint(
        mt,
        dirpath=dirpath,
        filename=f"{run_suffix}_ep{{epoch:03d}}",
        monitor=validation_monitor,
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    # Conformer IDs captured during bag construction (guaranteed aligned with each
    # bag's rows, including "__noexp" bags); None -> plots fall back to indices.
    conf_ids = getattr(mt.data_module, "bag_conf_ids", None) or None
    if conf_ids is None:
        LOGGER.info("[MODEL] Using indices as instance IDs for attention weight plots")
    else:
        LOGGER.info(f"[MODEL] Using conformer IDs for attention weight plots ({len(conf_ids)} bags)")

    callbacks_list = [ckpt_cb, LearningRateMonitor(logging_interval="step")]
    early_stop_cb = make_early_stopping_callback(mt, validation_monitor)
    if early_stop_cb is not None:
        callbacks_list.insert(1, early_stop_cb)
    if mt.trainer_cfg.enable_progress_bar:
        callbacks_list.append(TQDMProgressBar(refresh_rate=20, leave=False))
    if mt.trainer_cfg.enable_model_summary:
        callbacks_list.append(ModelSummary(max_depth=2))

    if mt.trainer_cfg.log_per_epoch:
        from ppl.training.epoch_attention_weight_logger import EpochAttentionWeightLogger
        from ppl.training.epoch_embedding_logger import EpochEmbeddingLogger

        base = exp_dir if exp_dir is not None else mt.results_dir
        embeddings_dir = base / "embeddings_per_epoch"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"[MODEL] Saving per-epoch embeddings to {embeddings_dir}")
        if attention_dir is None:
            attention_dir = base / "attention_weights_per_epoch"
            attention_dir.mkdir(parents=True, exist_ok=True)

        callbacks_list += [
            EpochAttentionWeightLogger(save_dir=str(attention_dir), max_bags=1000, conf_ids=conf_ids, max_epochs=mt.max_epochs),
            EpochEmbeddingLogger(save_dir=str(embeddings_dir), max_bags=1000, conf_ids=conf_ids, max_epochs=mt.max_epochs),
        ]

    return callbacks_list
