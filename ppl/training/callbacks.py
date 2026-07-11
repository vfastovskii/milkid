"""Lightning callback construction for ModelTrainer.

Functions take the ``ModelTrainer`` instance (``mt``) to reuse its config and
metric helpers; kept here to keep ModelTrainer focused on build/fit.
"""
from __future__ import annotations

import logging
from pathlib import Path
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


class _GatedModelCheckpoint(ModelCheckpoint):
    """Best-val checkpoint with two optional eligibility gates:

    - ``min_epoch``: only epochs >= min_epoch are considered (a late-training floor;
      training is made to reach it via TrainerConfig's min_epochs coupling).
    - ``require_focus``: only focus-phase epochs where the active-prototype query is
      engaged (curriculum triggered AND query weight > 0) are considered, so the
      chosen model is one the prototype injection has corrected the aggregator with
      — not an earlier epoch whose evaluation (use_on_eval=True) runs with the query
      off.

    Both gates are state-driven, so selection stays automatic and reproducible. With
    both off it behaves as a plain best-val ModelCheckpoint. If the gates admit no
    epoch, nothing is saved and the caller falls back to the in-memory model.
    """

    def __init__(self, *args, min_epoch: int = 0, require_focus: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epoch = int(min_epoch)
        self.require_focus = bool(require_focus)

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        if super()._should_skip_saving_checkpoint(trainer):
            return True
        if int(trainer.current_epoch) < self.min_epoch:
            return True  # late-training floor
        if self.require_focus:
            module = trainer.lightning_module
            if getattr(module, "_attention_refinement_trigger_epoch", None) is None:
                return True  # focus phase not entered yet
            weight_fn = getattr(module, "_attention_refinement_query_weight", None)
            if callable(weight_fn) and float(weight_fn(int(trainer.current_epoch))) <= 0.0:
                return True  # active-prototype query not injected this epoch
        return False


class MetricsFileLogger(pl.Callback):
    """Append every per-epoch metric to ``<results_dir>/log.txt`` (one line/epoch).

    Runs at on_train_epoch_end, which fires after validation + all other callbacks
    (KID, LR), so trainer.callback_metrics holds the full epoch record — train/val
    loss+rmse+mae, KID top-k, learning rates, everything logged that epoch. Each
    line is self-describing ``epoch=N key=value …`` (sorted), so it stays parseable
    even as the metric set changes.
    """

    def __init__(self, log_path) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._written_epoch = None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        epoch = int(trainer.current_epoch)
        if epoch == self._written_epoch:
            return
        metrics = {}
        for k, v in (trainer.callback_metrics or {}).items():
            try:
                metrics[str(k)] = float(v.item() if hasattr(v, "item") else v)
            except (TypeError, ValueError):
                continue
        if not metrics:
            return
        self._written_epoch = epoch
        line = " ".join(
            [f"epoch={epoch}"] + [f"{k}={metrics[k]:.6g}" for k in sorted(metrics)]
        )
        try:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            LOGGER.warning("[MODEL] Could not write per-epoch metrics to %s: %s",
                           self.log_path, e)


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

    # Best-val checkpoint, gated so the chosen model is late and prototype-corrected:
    # only epochs >= checkpoint_min_epoch (a late-training floor) and, when the
    # aggregator-focus curriculum is on, only focus-phase epochs (active-prototype
    # query engaged) are eligible.
    validation_monitor = mt._checkpoint_monitor_metric()
    ckpt_cb = _GatedModelCheckpoint(
        dirpath=dirpath,
        filename=f"{run_suffix}_ep{{epoch:03d}}",
        monitor=validation_monitor,
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        min_epoch=int(getattr(mt.trainer_cfg, "checkpoint_min_epoch", 0) or 0),
        require_focus=bool(getattr(mt.trainer_cfg, "attention_refinement_enabled", False)),
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

    # Per-run plain-text metrics log (all metrics, one line per epoch).
    log_path = (exp_dir if exp_dir is not None else mt.results_dir) / "log.txt"
    callbacks_list.append(MetricsFileLogger(log_path))
    LOGGER.info("[MODEL] Per-epoch metrics log: %s", log_path)

    if bool(getattr(mt.trainer_cfg, "kid_metric_enabled", False)):
        sdf = getattr(mt.trainer_cfg, "kid_sdf_path", None)
        if not sdf:
            LOGGER.warning("[MODEL] kid_metric_enabled but kid_sdf_path is not set; skipping KID")
        else:
            try:
                from ppl.training.kid_calculator import KidCalculator, KidMetricCallback

                calc = KidCalculator(
                    sdf,
                    top_k=getattr(mt.trainer_cfg, "kid_top_k", [1, 3, 5]),
                    rmsd_threshold=getattr(mt.trainer_cfg, "kid_rmsd_threshold", 2.0),
                    o3a_threshold=getattr(mt.trainer_cfg, "kid_o3a_threshold", 0.8),
                    active_threshold=getattr(mt.trainer_cfg, "kid_active_threshold", 7.0),
                    pred_tol=getattr(mt.trainer_cfg, "kid_pred_tol", 1.0),
                    pdb_only=bool(getattr(mt.trainer_cfg, "kid_pdb_only", True)),
                )
                callbacks_list.append(KidMetricCallback(calc))
                LOGGER.info("[MODEL] KID pose-selection metric enabled (per-epoch train + val)")
            except Exception as e:
                LOGGER.warning("[MODEL] KID metric disabled (SDF load failed: %s)", e)
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
