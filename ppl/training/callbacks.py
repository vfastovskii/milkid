"""Lightning callback construction and conformer-id extraction for ModelTrainer.

Functions take the ``ModelTrainer`` instance (``mt``) to reuse its config and
metric helpers; kept here to keep ModelTrainer focused on build/fit.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from ppl.training.checkpoints import MinEpochModelCheckpoint

LOGGER = logging.getLogger(__name__)


def extract_conformer_ids(data_module) -> Dict[str, List[str]] | None:
    """Extract conformer IDs consistent with bag construction rules.

    - Train: exclude instances containing "_experimental_pose".
    - Val/Test: create a full-bag entry plus a ``<bag_id>__noexp`` entry when
      experimental poses would be removed.
    """
    if data_module is None:
        LOGGER.warning("[MODEL] No data module available, cannot extract conformer IDs")
        return None
    try:
        csv_path = data_module._csv
        cfg = data_module.cfg
        bag_col = cfg.bag_id_col
        inst_col = cfg.inst_id_col
        split_col = getattr(cfg, "split_col", None)
        predefined = getattr(cfg, "predefined_split", False)

        LOGGER.info(f"[MODEL] Extracting conformer IDs from {csv_path}")
        df = pd.read_csv(csv_path)

        conf_ids: Dict[str, List[str]] = {}

        def add_split(split_df):
            for bag_id, group in split_df.groupby(bag_col):
                inst_list = list(map(str, group[inst_col].astype(str)))
                nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                conf_ids[f"{bag_id}"] = inst_list
                if len(nonexp) > 0 and len(nonexp) < len(inst_list):
                    conf_ids[f"{bag_id}__noexp"] = nonexp
                elif len(inst_list) == 1 and ("_experimental_pose" in inst_list[0]):
                    conf_ids[f"{bag_id}__noexp"] = inst_list

        if predefined and split_col in df.columns:
            # TRAIN (split==0): non-experimental only
            for bag_id, group in df[df[split_col] == 0].groupby(bag_col):
                inst_list = list(map(str, group[inst_col].astype(str)))
                inst_nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                if inst_nonexp:
                    conf_ids[str(bag_id)] = inst_nonexp
            add_split(df[df[split_col] == 1])  # VAL
            add_split(df[df[split_col] == 2])  # TEST
        else:
            add_split(df)

        LOGGER.info(
            f"[MODEL] Prepared conformer IDs for {len(conf_ids)} bag keys "
            "(with suffixes where applicable)"
        )
        return conf_ids
    except Exception as e:
        LOGGER.error(f"[MODEL] Error extracting conformer IDs: {e}")
        import traceback

        LOGGER.debug(traceback.format_exc())
        return None


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
        exp_dir = mt.log_save_dir.parent / mt.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        dirpath = exp_dir / "models" / "cv"
        dirpath.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"[MODEL] Saving best CV model to {dirpath}")
        attention_dir = None
        if mt.trainer_cfg.log_per_epoch:
            attention_dir = exp_dir / "attention_weights_per_epoch"
            attention_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[MODEL] Saving per-epoch attention weights to {attention_dir}")
        return dirpath, exp_dir, attention_dir

    dirpath = mt.log_save_dir / "checkpoints"
    attention_dir = (
        mt.log_save_dir / "attention_weights_per_epoch"
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

    conf_ids = extract_conformer_ids(mt.data_module)
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

        base = exp_dir if exp_dir is not None else mt.log_save_dir
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
