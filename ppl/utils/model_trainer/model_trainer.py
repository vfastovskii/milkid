"""Model trainer for the Multi-Instance Learning Kit (MILK).

This module provides a class for building and training models for the MILK project.
It handles model building, trainer configuration, and model training.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)

from ppl.utils.mil_data_handling.data_loader import MILDataModule
from ppl.utils.model_builder.model_builder import ModelBuilder
from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.modelling_configs.trainer_config import TrainerConfig, TrainerBuilder
from ppl.utils.pipeline.mlflow_utils import SafeMLFlowLogger, log_metrics

LOGGER = logging.getLogger(__name__)


class MinEpochModelCheckpoint(ModelCheckpoint):
    """Select checkpoints only after configured epoch/attention-refinement gates."""

    def __init__(
        self,
        *args,
        min_epoch: int = 0,
        require_attention_refinement: bool = False,
        min_query_epochs: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_epoch = int(min_epoch)
        self.require_attention_refinement = bool(require_attention_refinement)
        self.min_query_epochs = max(1, int(min_query_epochs))

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        if trainer.current_epoch < self.min_epoch:
            return True
        if self.require_attention_refinement:
            module = getattr(trainer, "lightning_module", None)
            trigger_epoch = getattr(
                module,
                "_attention_refinement_trigger_epoch",
                None,
            )
            if trigger_epoch is None:
                return True
            query_epochs_seen = trainer.current_epoch - int(trigger_epoch)
            if query_epochs_seen < self.min_query_epochs:
                return True
        return super()._should_skip_saving_checkpoint(trainer)


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

        return model

    def _attach_trainer_runtime_config(self, model: pl.LightningModule) -> None:
        """Attach non-hparam trainer controls to a freshly loaded model.

        Checkpoint loading reconstructs the LightningModule from checkpoint
        hparams and explicit constructor args. Runtime controls owned by
        TrainerConfig, such as attention-refinement scheduling, are not
        constructor args, so reattach them before validation/test/export.
        """
        checkpoint_min_epoch = int(
            getattr(self.trainer_cfg, "checkpoint_min_epoch", 0) or 0
        )
        setattr(model, "_checkpoint_min_epoch", checkpoint_min_epoch)
        for attr in (
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
            "attention_refinement_gap_threshold",
            "attention_refinement_rel_gap_threshold",
            "attention_refinement_require_val_worse_than_best",
            "attention_refinement_lr_factor",
            "attention_refinement_query_epochs",
            "attention_refinement_query_weight_start",
            "attention_refinement_query_weight_end",
            "attention_refinement_stop_after_query_epochs",
        ):
            setattr(model, f"_{attr}", getattr(self.trainer_cfg, attr, None))
        if checkpoint_min_epoch > 0:
            LOGGER.info(
                "[MODEL] Best-checkpoint selection uses %s only for epochs >= %d",
                self._checkpoint_monitor_metric(),
                checkpoint_min_epoch,
            )
        if bool(getattr(self.trainer_cfg, "overfit_gap_stop_enabled", False)):
            LOGGER.info(
                "[MODEL] Overfit-gap early stop enabled: metric=%s "
                "patience=%d abs_gap>=%.4f rel_gap>=%.4f",
                getattr(self.trainer_cfg, "overfit_gap_metric", "rmse"),
                int(getattr(self.trainer_cfg, "overfit_gap_patience", 2) or 2),
                float(getattr(self.trainer_cfg, "overfit_gap_abs_threshold", 0.10)),
                float(getattr(self.trainer_cfg, "overfit_gap_rel_threshold", 0.15)),
            )
        if bool(getattr(self.trainer_cfg, "attention_refinement_enabled", False)):
            LOGGER.info(
                "[MODEL] Attention-refinement schedule enabled: metric=%s "
                "patience=%d gap>=%.4f rel_gap>=%.4f lr_factor=%.4f "
                "query_epochs=%d query_weight=%.3f->%.3f",
                getattr(self.trainer_cfg, "attention_refinement_metric", "rmse"),
                int(getattr(self.trainer_cfg, "attention_refinement_patience", 1) or 1),
                float(
                    getattr(
                        self.trainer_cfg,
                        "attention_refinement_gap_threshold",
                        0.08,
                    )
                ),
                float(
                    getattr(
                        self.trainer_cfg,
                        "attention_refinement_rel_gap_threshold",
                        0.15,
                    )
                ),
                float(getattr(self.trainer_cfg, "attention_refinement_lr_factor", 0.1)),
                int(getattr(self.trainer_cfg, "attention_refinement_query_epochs", 25) or 25),
                float(
                    getattr(
                        self.trainer_cfg,
                        "attention_refinement_query_weight_start",
                        0.0,
                    )
                ),
                float(
                    getattr(
                        self.trainer_cfg,
                        "attention_refinement_query_weight_end",
                        1.0,
                    )
                ),
            )
        if bool(
            getattr(self.trainer_cfg, "checkpoint_after_attention_refinement", False)
        ):
            LOGGER.info(
                "[MODEL] Best-checkpoint selection is gated to attention "
                "refinement: min_query_epochs=%d",
                int(getattr(self.trainer_cfg, "checkpoint_min_query_epochs", 1) or 1),
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

    def _make_model_checkpoint(self, **kwargs) -> ModelCheckpoint:
        min_epoch = int(getattr(self.trainer_cfg, "checkpoint_min_epoch", 0) or 0)
        monitor = kwargs.get("monitor")
        require_refinement = bool(
            getattr(self.trainer_cfg, "checkpoint_after_attention_refinement", False)
        )
        min_query_epochs = int(
            getattr(self.trainer_cfg, "checkpoint_min_query_epochs", 1) or 1
        )
        if (
            (min_epoch > 0 or require_refinement)
            and monitor == self._checkpoint_monitor_metric()
        ):
            return MinEpochModelCheckpoint(
                min_epoch=min_epoch,
                require_attention_refinement=require_refinement,
                min_query_epochs=min_query_epochs,
                **kwargs,
            )
        return ModelCheckpoint(**kwargs)

    def _make_early_stopping_callback(self, validation_monitor: str):
        if bool(getattr(self.trainer_cfg, "attention_refinement_enabled", False)):
            LOGGER.info(
                "[MODEL] Lightning EarlyStopping disabled because "
                "attention-refinement controls the post-overfit training phase."
            )
            return None
        return EarlyStopping(
            monitor=validation_monitor,
            mode="min",
            patience=self.model_cfg.optim.lr_patience,
            verbose=True,
            check_on_train_epoch_end=False,
        )
        
    def _extract_conformer_ids(self) -> Dict[str, List[str]]:
        """Extract conformer IDs consistent with bag construction rules.
        
        - Train: exclude instances containing "_experimental_pose".
        - Val/Test: create two entries per original bag where applicable:
          <bag_id>__noexp with non-experimental instances, and <bag_id>__exp with experimental instances.
        """
        if self.data_module is None:
            LOGGER.warning("[MODEL] No data module available, cannot extract conformer IDs")
            return None
        try:
            import pandas as pd
            
            # Get CSV and columns
            csv_path = self.data_module._csv
            cfg = self.data_module.cfg
            bag_col = cfg.bag_id_col
            inst_col = cfg.inst_id_col
            split_col = getattr(cfg, 'split_col', None)
            predefined = getattr(cfg, 'predefined_split', False)
            
            LOGGER.info(f"[MODEL] Extracting conformer IDs from {csv_path}")
            df = pd.read_csv(csv_path)
            
            conf_ids: Dict[str, List[str]] = {}
            
            if predefined and split_col in df.columns:
                # Split masks: 0=train, 1=val, 2=test
                tr_df = df[df[split_col] == 0]
                va_df = df[df[split_col] == 1]
                te_df = df[df[split_col] == 2]
                
                # TRAIN: non-experimental only
                for bag_id, group in tr_df.groupby(bag_col):
                    inst_list = list(map(str, group[inst_col].astype(str)))
                    inst_nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                    if len(inst_nonexp) == 0:
                        continue
                    conf_ids[str(bag_id)] = inst_nonexp
                
                # VAL/TEST: create full and __noexp entries
                def add_split(split_df):
                    for bag_id, group in split_df.groupby(bag_col):
                        inst_list = list(map(str, group[inst_col].astype(str)))
                        nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                        # Full bag with ALL instances
                        conf_ids[f"{bag_id}"] = inst_list
                        # __noexp duplicate only if anything would be removed
                        if len(nonexp) > 0 and len(nonexp) < len(inst_list):
                            conf_ids[f"{bag_id}__noexp"] = nonexp
                        else:
                            # Special case: single-instance bag that is experimental-only → create '__noexp' with the same instance
                            if len(inst_list) == 1 and ("_experimental_pose" in inst_list[0]):
                                conf_ids[f"{bag_id}__noexp"] = inst_list
                add_split(va_df)
                add_split(te_df)
            else:
                # Fallback when predefined split is not available:
                # For all bags, provide full and __noexp entries.
                for bag_id, group in df.groupby(bag_col):
                    inst_list = list(map(str, group[inst_col].astype(str)))
                    nonexp = [iid for iid in inst_list if "_experimental_pose" not in iid]
                    # Full bag key
                    conf_ids[f"{bag_id}"] = inst_list
                    # __noexp only if fewer than full
                    if len(nonexp) > 0 and len(nonexp) < len(inst_list):
                        conf_ids[f"{bag_id}__noexp"] = nonexp
                    else:
                        # Special case: single-instance bag that is experimental-only → create '__noexp' with the same instance
                        if len(inst_list) == 1 and ("_experimental_pose" in inst_list[0]):
                            conf_ids[f"{bag_id}__noexp"] = inst_list
            
            LOGGER.info(f"[MODEL] Prepared conformer IDs for {len(conf_ids)} bag keys (with suffixes where applicable)")
            return conf_ids
        except Exception as e:
            LOGGER.error(f"[MODEL] Error extracting conformer IDs: {e}")
            import traceback
            LOGGER.debug(traceback.format_exc())
            return None

    def callbacks(self) -> Sequence[pl.callbacks.Callback]:
        """Create a list of callbacks for the PyTorch Lightning Trainer.

        Returns
        -------
        Sequence[pl.callbacks.Callback]
            List of callbacks:
            - ModelCheckpoint: Save the best model based on validation loss
            - EarlyStopping: Stop training when validation loss stops improving
            - LearningRateMonitor: Log learning rate during training
            - RichProgressBar: Display a rich progress bar during training
            - ModelSummary: Display a summary of the model architecture
            - EpochAttentionWeightLogger: Log attention weights per epoch for training and validation bags
        """
        run_suffix = f"seed{self.seed}"
        iteration_label = getattr(self, "_training_iteration_label", "")
        if iteration_label:
            run_suffix = f"{run_suffix}_{iteration_label}"

        # If experiment_name is provided, save models to the experiment directory
        if self.experiment_name:
            # Create experiment directory if it doesn't exist
            exp_dir = self.log_save_dir.parent / self.experiment_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Save models to the experiment directory
            dirpath = exp_dir / "models" / "cv"
            dirpath.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[MODEL] Saving best CV model to {dirpath}")
            
            # Create directory for per-epoch attention weights (only if enabled)
            if self.trainer_cfg.log_per_epoch:
                attention_dir = exp_dir / "attention_weights_per_epoch"
                attention_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"[MODEL] Saving per-epoch attention weights to {attention_dir}")
            else:
                attention_dir = None
        else:
            dirpath = self.log_save_dir / "checkpoints"
            attention_dir = self.log_save_dir / "attention_weights_per_epoch" if self.trainer_cfg.log_per_epoch else None
            if attention_dir is not None:
                attention_dir.mkdir(parents=True, exist_ok=True)

        validation_monitor = self._checkpoint_monitor_metric()
        ckpt_cb = self._make_model_checkpoint(
            dirpath=dirpath,
            filename=f"{run_suffix}_ep{{epoch:03d}}",
            monitor=validation_monitor,
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
        )

        # Extract conformer IDs from the CSV file if we have a data module
        conf_ids = self._extract_conformer_ids()
        if conf_ids is None:
            LOGGER.info("[MODEL] Using indices as instance IDs for attention weight plots")
        else:
            LOGGER.info(f"[MODEL] Using conformer IDs for attention weight plots ({len(conf_ids)} bags)")

        # Prepare embedding logger directory (only if per-epoch logging is enabled)
        callbacks_list = [
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
        ]
        early_stop_cb = self._make_early_stopping_callback(validation_monitor)
        if early_stop_cb is not None:
            callbacks_list.insert(1, early_stop_cb)
        if self.trainer_cfg.enable_progress_bar:
            # Higher refresh rate means less frequent updates, which helps
            # prevent progress bar jumping.
            callbacks_list.append(TQDMProgressBar(refresh_rate=20, leave=False))
        if self.trainer_cfg.enable_model_summary:
            callbacks_list.append(ModelSummary(max_depth=2))

        if self.trainer_cfg.log_per_epoch:
            from ppl.utils.epoch_attention_weight_logger import EpochAttentionWeightLogger
            from ppl.utils.epoch_embedding_logger import EpochEmbeddingLogger

            if self.experiment_name:
                embeddings_dir = exp_dir / "embeddings_per_epoch"
            else:
                embeddings_dir = self.log_save_dir / "embeddings_per_epoch"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"[MODEL] Saving per-epoch embeddings to {embeddings_dir}")

            if attention_dir is None:
                # In case experiment_name was not set and attention_dir was disabled earlier
                if self.experiment_name:
                    attention_dir = exp_dir / "attention_weights_per_epoch"
                else:
                    attention_dir = self.log_save_dir / "attention_weights_per_epoch"
                attention_dir.mkdir(parents=True, exist_ok=True)

            callbacks_list += [
                EpochAttentionWeightLogger(save_dir=str(attention_dir), max_bags=1000, conf_ids=conf_ids, max_epochs=self.max_epochs),
                EpochEmbeddingLogger(save_dir=str(embeddings_dir), max_bags=1000, conf_ids=conf_ids, max_epochs=self.max_epochs),
            ]

        return callbacks_list

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

        from ppl.utils.model_trainer.post_training_ablation import (
            run_instance_importance_selection,
        )
        from ppl.utils.pipeline.results_directory import create_results_directory

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

    @staticmethod
    def _unpack_batch(batch):
        """Return (x, y, bag_ids, key_padding_mask, cluster_ids, series_labels)."""
        if isinstance(batch, dict):
            bags = batch.get("bags", batch.get("x"))
            y = batch.get("labels", batch.get("y"))
            bag_ids = batch.get("bag_ids", batch.get("ids"))
            key_padding_mask = batch.get("padding_mask")
            cluster_ids = batch.get("cluster_ids")
            series_labels = batch.get("series_labels")
        else:
            bags, y, bag_ids = batch[0], batch[1], batch[2]
            key_padding_mask = batch[3] if len(batch) > 3 else None
            cluster_ids = batch[4] if len(batch) > 4 else None
            series_labels = batch[5] if len(batch) > 5 else None
        x = bags[0] if isinstance(bags, list) else bags
        return x, y, bag_ids, key_padding_mask, cluster_ids, series_labels

    def _predict_rows(
        self,
        model: pl.LightningModule,
        dataloader,
        *,
        stage: Optional[str] = None,
        eval_epoch: Optional[int] = None,
        use_series_labels: bool = False,
    ) -> List[tuple]:
        """Run inference over a dataloader and collect (bag_id, true, pred) rows."""
        import numpy as np

        rows: List[tuple] = []
        if dataloader is None:
            return rows

        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in dataloader:
                x, y, bag_ids, key_padding_mask, cluster_ids, series_labels = (
                    self._unpack_batch(batch)
                )
                x = x.to(device)
                if key_padding_mask is not None:
                    try:
                        key_padding_mask = key_padding_mask.to(x.device)
                    except Exception:
                        key_padding_mask = None
                if cluster_ids is not None:
                    try:
                        cluster_ids = cluster_ids.to(x.device)
                    except Exception:
                        cluster_ids = None

                core_kwargs = {}
                if key_padding_mask is not None:
                    core_kwargs["key_padding_mask"] = key_padding_mask
                if cluster_ids is not None:
                    core_kwargs["cluster_ids"] = cluster_ids
                if use_series_labels and series_labels is not None:
                    core_kwargs["series_labels"] = series_labels
                if stage is not None:
                    core_kwargs["stage"] = stage
                if eval_epoch is not None:
                    core_kwargs["current_epoch"] = int(eval_epoch)

                try:
                    logit, _ = model.core(x, **core_kwargs)
                except TypeError:
                    logit, _ = model.core(x)

                y_pred = (
                    torch.sigmoid(logit)
                    if self.task.lower() == "classification"
                    else logit
                )
                y_np = np.asarray(y.cpu().numpy()).reshape(-1)
                y_pred_np = np.asarray(y_pred.detach().cpu().numpy()).reshape(-1)
                for i, mol_id in enumerate(bag_ids):
                    rows.append((str(mol_id), float(y_np[i]), float(y_pred_np[i])))
        return rows

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

        # Generate training/validation artifacts if experiment_name is set
        try:
            if self.experiment_name:
                from ppl.utils.pipeline.results_directory import create_results_directory
                import torch
                import numpy as np
                import pandas as pd

                results_dir = create_results_directory(self.experiment_name)

                # Choose model for predictions/plots
                plot_model = best_model if best_model is not None else model

                # Safe dataloaders
                if hasattr(dm, "train_full_dataloader"):
                    train_dl = dm.train_full_dataloader()
                else:
                    train_dl = dm.train_dataloader()
                val_dl = dm.val_dataloader()

                eval_epoch = getattr(
                    plot_model,
                    "_evaluation_epoch_override",
                    getattr(plot_model, "current_epoch", None),
                )

                def _compute_regression_metrics_from_rows(rows, prefix: str):
                    if not rows or self.task.lower() != "regression":
                        return None
                    y_true = np.asarray([row[1] for row in rows], dtype=np.float64)
                    y_pred = np.asarray([row[2] for row in rows], dtype=np.float64)
                    finite = np.isfinite(y_true) & np.isfinite(y_pred)
                    y_true = y_true[finite]
                    y_pred = y_pred[finite]
                    if y_true.size == 0:
                        return None

                    err = y_pred - y_true
                    mse = float(np.mean(err * err))
                    mae = float(np.mean(np.abs(err)))
                    rmse = float(np.sqrt(mse))
                    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
                    r2 = (
                        float("nan")
                        if denom <= 0.0
                        else float(1.0 - np.sum(err * err) / denom)
                    )

                    def _corr(a, b):
                        if a.size < 2:
                            return float("nan")
                        if float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
                            return float("nan")
                        return float(np.corrcoef(a, b)[0, 1])

                    true_rank = pd.Series(y_true).rank(method="average").to_numpy(
                        dtype=np.float64,
                    )
                    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(
                        dtype=np.float64,
                    )

                    return {
                        f"{prefix}_loss": mse,
                        f"{prefix}_mae": mae,
                        f"{prefix}_rmse": rmse,
                        f"{prefix}_r2": r2,
                        f"{prefix}_pearson": _corr(y_true, y_pred),
                        f"{prefix}_spearman": _corr(true_rank, pred_rank),
                    }

                # Write train_fit.csv
                train_rows = []
                try:
                    train_rows = self._predict_rows(
                        plot_model, train_dl, stage="val",
                        eval_epoch=eval_epoch, use_series_labels=True,
                    )
                    if train_rows:
                        train_df = pd.DataFrame(train_rows, columns=["mol_id", "true", "predicted"])
                        # Compute absolute error and round all numeric values to 2 decimals
                        train_df["true"] = train_df["true"].round(2)
                        train_df["predicted"] = train_df["predicted"].round(2)
                        train_df["abs_error"] = (train_df["true"] - train_df["predicted"]).abs().round(2)
                        train_df.to_csv(results_dir / "train_fit.csv", index=False)
                        LOGGER.info(f"[MODEL] Saved train predictions to {results_dir / 'train_fit.csv'}")
                    else:
                        LOGGER.info("[MODEL] No train rows collected for train_fit.csv in fit_validate")
                except Exception as e:
                    LOGGER.warning(f"[MODEL] Failed to write train_fit.csv in fit_validate: {e}")

                # Write val.csv
                try:
                    val_rows = self._predict_rows(
                        plot_model, val_dl, stage="val",
                        eval_epoch=eval_epoch, use_series_labels=True,
                    )
                    if val_rows:
                        val_df = pd.DataFrame(val_rows, columns=["mol_id", "true", "predicted"])
                        # Round all numeric values to 2 decimals
                        val_df["true"] = val_df["true"].round(2)
                        val_df["predicted"] = val_df["predicted"].round(2)
                        val_df["abs_error"] = (val_df["true"] - val_df["predicted"]).abs().round(2)
                        val_df.to_csv(results_dir / "val.csv", index=False)
                        LOGGER.info(f"[MODEL] Saved validation predictions to {results_dir / 'val.csv'}")
                    else:
                        LOGGER.info("[MODEL] No validation rows collected for val.csv in fit_validate")
                except Exception as e:
                    LOGGER.warning(f"[MODEL] Failed to write val.csv in fit_validate: {e}")

                # Collect train metrics from the same best-checkpoint train
                # predictions. This avoids a fragile second Lightning
                # validation pass over the train loader in HPO jobs.
                train_metrics = _compute_regression_metrics_from_rows(
                    train_rows,
                    prefix="train",
                )
                if train_metrics is None:
                    LOGGER.warning(
                        "[MODEL] Could not compute train metrics for res.txt; "
                        "train_rows=%d task=%s",
                        len(train_rows),
                        self.task,
                    )

                # Early stopping epoch
                es_epoch = None
                try:
                    from pytorch_lightning.callbacks.early_stopping import EarlyStopping as _ES
                    for cb in trainer.callbacks:
                        if isinstance(cb, _ES):
                            es_epoch = getattr(cb, "stopped_epoch", None)
                            if es_epoch in (None, 0):
                                es_epoch = getattr(cb, "early_stopped_epoch", None)
                            break
                except Exception:
                    pass

                # Write res.txt
                try:
                    lines = []
                    lines.append("Run summary:\n")
                    lines.append("Validation metrics (final/best):\n")
                    for k, v in val_metrics.items():
                        try:
                            val_str = f"{float(v):.6f}"
                        except Exception:
                            val_str = str(v)
                        lines.append(f"  {k}: {val_str}\n")
                    if train_metrics is not None:
                        lines.append("\nTraining metrics:\n")
                        for k, v in train_metrics.items():
                            metric_key = str(k)
                            if metric_key.startswith("val_"):
                                metric_key = f"train_{metric_key[4:]}"
                            elif not metric_key.startswith("train_"):
                                metric_key = f"train_{metric_key}"
                            try:
                                tr_str = f"{float(v):.6f}"
                            except Exception:
                                tr_str = str(v)
                            lines.append(f"  {metric_key}: {tr_str}\n")
                    trained_epochs = int(getattr(trainer, "current_epoch", 0) or 0)
                    lines.append("\nTraining schedule:\n")
                    lines.append(f"  trained_epochs: {trained_epochs}\n")
                    lines.append(f"  max_epochs: {int(self.max_epochs)}\n")
                    lines.append(f"  early_stopping_patience: {int(self.model_cfg.optim.lr_patience)}\n")
                    validation_monitor = self._checkpoint_monitor_metric()
                    best_epoch = self._get_best_epoch_from_trainer(trainer)
                    best_score = self._get_best_checkpoint_score_from_trainer(trainer)
                    lines.append("\nMetric source:\n")
                    lines.append("  trial_metrics_source: best_checkpoint\n")
                    lines.append(f"  checkpoint_monitor: {validation_monitor}\n")
                    if best_epoch is not None:
                        lines.append(f"  best_epoch: {best_epoch}\n")
                    else:
                        lines.append("  best_epoch: N/A\n")
                    if best_score is not None:
                        lines.append(f"  best_checkpoint_score: {best_score:.6f}\n")
                    else:
                        lines.append("  best_checkpoint_score: N/A\n")
                    lines.append("\nEarly stopping:\n")
                    # Prefer reporting the best checkpoint epoch (user-facing "stopped" epoch)
                    if best_epoch is not None:
                        lines.append(f"  stopped_epoch: {best_epoch}\n")
                        if es_epoch is not None and es_epoch != 0 and es_epoch != best_epoch:
                            lines.append(f"  early_stopped_epoch_internal: {es_epoch}\n")
                    else:
                        if es_epoch is not None and es_epoch != 0:
                            lines.append(f"  stopped_epoch: {es_epoch}\n")
                        else:
                            lines.append("  stopped_epoch: N/A (not triggered or unavailable)\n")
                    with open(results_dir / "res.txt", "w") as f:
                        f.writelines(lines)
                    LOGGER.info(f"[MODEL] Wrote run summary to {results_dir / 'res.txt'}")
                except Exception as e:
                    LOGGER.warning(f"[MODEL] Failed to write res.txt in fit_validate: {e}")
        except Exception as e:
            LOGGER.debug(f"[MODEL] Artifact generation in fit_validate failed: {e}")

        # Evaluate the best model on the test split (if one exists).
        try:
            # Determine if test data exists
            test_dl = None
            try:
                test_dl = dm.test_dataloader()
            except Exception:
                test_dl = None
            has_test = test_dl is not None and len(test_dl) > 0 if test_dl is not None else False

            if has_test:
                LOGGER.info("[MODEL][TEST] Evaluating best model on the test split")
                # Use best model if available; otherwise, fallback to in-memory model
                best_test_model = self.get_best_model()
                test_model = best_test_model if best_test_model is not None else model
                test_best_epoch = self._get_best_epoch_from_trainer(trainer)
                if test_best_epoch is not None:
                    setattr(test_model, "_evaluation_epoch_override", test_best_epoch)

                # Use a quiet temporary trainer to run test to avoid extra progress bars/logging
                from pytorch_lightning import Trainer as _PLTrainer
                temp_trainer = _PLTrainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
                test_metrics = temp_trainer.test(test_model, dataloaders=test_dl, verbose=False)[0]
                log_metrics(test_metrics, stage="test", prefix="test")

                # Optionally save test.csv with predictions if experiment_name is set
                if self.experiment_name:
                    try:
                        from ppl.utils.pipeline.results_directory import create_results_directory
                        import torch
                        import numpy as np
                        import pandas as pd
                        results_dir = create_results_directory(self.experiment_name)
                        test_rows = self._predict_rows(test_model, test_dl)
                        if test_rows:
                            test_df = pd.DataFrame(test_rows, columns=["mol_id", "true", "predicted"])
                            # Round predictions and absolute error to 2 decimals for user-facing CSV
                            test_df["predicted"] = test_df["predicted"].round(2)
                            test_df["abs_error"] = (test_df["true"] - test_df["predicted"]).abs().round(2)
                            test_df.to_csv(results_dir / "test.csv", index=False)
                            LOGGER.info(f"[MODEL][TEST] Saved test predictions to {results_dir / 'test.csv'}")

                        # Additionally, generate attention weights for the test set
                        # only when interpretation artifacts are requested.
                        if bool(
                            getattr(
                                self.trainer_cfg,
                                "save_attention_artifacts",
                                True,
                            )
                        ):
                            try:
                                from ppl.utils.plotting.plot_attention_weights import plot_attention_weights_from_model
                                # Prepare directories: results/<experiment>/test/attention_weights
                                test_dir = results_dir / "test"
                                attention_dir = test_dir / "attention_weights"
                                attention_dir.mkdir(parents=True, exist_ok=True)

                                # Extract conformer IDs consistent with bag construction
                                conf_ids = self._extract_conformer_ids()
                                if conf_ids is None:
                                    LOGGER.info("[MODEL][TEST] Using indices as instance IDs for test attention weight plots")
                                else:
                                    LOGGER.info(f"[MODEL][TEST] Using conformer IDs for test attention weight plots ({len(conf_ids)} bags)")

                                # Generate attention weights for the test dataloader
                                plot_attention_weights_from_model(
                                    model=test_model,
                                    dataloader=test_dl,
                                    save_dir=str(attention_dir),
                                    max_bags=1000,
                                    conf_ids=conf_ids
                                )
                                LOGGER.info(f"[MODEL][TEST] Saved test attention weights to {attention_dir}")
                            except Exception as e_inner:
                                LOGGER.warning(f"[MODEL][TEST] Failed to generate test attention weights: {e_inner}")
                    except Exception as e:
                        LOGGER.warning(f"[MODEL][TEST] Failed to save test.csv: {e}")
            else:
                LOGGER.info("[MODEL][TEST] No test split available – skipping test evaluation")
        except Exception as e:
            LOGGER.debug(f"[MODEL][TEST] Test evaluation step failed: {e}")

        return val_metrics

