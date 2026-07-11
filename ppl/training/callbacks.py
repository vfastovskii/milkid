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

LOGGER = logging.getLogger(__name__)


class _FocusPhaseModelCheckpoint(ModelCheckpoint):
    """Best-val checkpoint restricted to the aggregator-focus phase.

    Skips saving until the active-prototype query is actually engaged (focus phase
    entered AND query weight > 0 this epoch), so the selected best model is one the
    prototype injection has corrected the aggregator with — not an earlier
    joint-training epoch whose evaluation (use_on_eval=True) would run with the
    query off. The gate is purely state-driven (curriculum trigger + query weight),
    so it stays automatic and reproducible. If the focus phase never triggers, no
    checkpoint is saved and the caller falls back to the in-memory model.
    """

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        if super()._should_skip_saving_checkpoint(trainer):
            return True
        module = trainer.lightning_module
        if getattr(module, "_attention_refinement_trigger_epoch", None) is None:
            return True  # focus phase not entered yet
        weight_fn = getattr(module, "_attention_refinement_query_weight", None)
        if callable(weight_fn) and float(weight_fn(int(trainer.current_epoch))) <= 0.0:
            return True  # active-prototype query not injected this epoch
        return False


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

    # Best-val checkpoint. When the aggregator-focus curriculum is on, restrict
    # selection to focus-phase epochs (active-prototype query engaged) so the chosen
    # model is one the prototype injection corrected the aggregator with; otherwise
    # plain best-val over all epochs.
    validation_monitor = mt._checkpoint_monitor_metric()
    ckpt_cls = (
        _FocusPhaseModelCheckpoint
        if bool(getattr(mt.trainer_cfg, "attention_refinement_enabled", False))
        else ModelCheckpoint
    )
    ckpt_cb = ckpt_cls(
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
