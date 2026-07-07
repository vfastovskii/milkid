from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


@dataclass(slots=True, frozen=False)
class TrainerConfig:
    """Holds all configuration parameters for the PyTorch Lightning Trainer.

    This class consolidates all trainer-related configurations in one place,
    making it easier to configure and customize the trainer.

    Parameters
    ----------
    max_epochs : int
        Maximum number of epochs to train for
    device : str
        Device to use for training ("cuda", "cpu", "mps")
    precision : str
        Precision to use for training ("32", "16-mixed")
    accelerator : Optional[str]
        Accelerator to use ("gpu", "cpu", "mps")
    devices : Union[int, List[int]]
        Number of devices to use or list of device indices
    strategy : Optional[str]
        Strategy to use for distributed training
    deterministic : bool
        Whether to use deterministic algorithms
    log_every_n_steps : int
        How often to log metrics
    enable_checkpointing : bool
        Whether to enable checkpointing
    enable_progress_bar : bool
        Whether to enable progress bar
    enable_model_summary : bool
        Whether to enable model summary
    num_sanity_val_steps : int
        Number of validation steps to run before training
    log_save_dir : str
        Directory to save logs
    experiment_name : str
        Name of the experiment
    run_name : Optional[str]
        Name of the run
    tracking_uri : Optional[str]
        URI for tracking
    log_per_epoch : bool
        If True, log attention weights and embeddings at the end of every epoch. If False, only log once for the best model selected by early stopping.
    save_attention_artifacts : bool
        If True, export post-training attention-weight artifacts. Disable during
        HPO sweeps so only the final selected model writes attention files.
    """
    # Basic training parameters
    max_epochs: int = 150
    min_epochs: int = 0
    device: str = "cuda"

    # Advanced training parameters
    precision: str = "16-mixed"
    accelerator: Optional[str] = None
    devices: Union[int, List[int]] = 1
    strategy: Optional[str] = None
    deterministic: bool = True
    log_every_n_steps: int = 50
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    num_sanity_val_steps: int = 0

    # Logging parameters
    log_save_dir: str = "exp_log"
    experiment_name: str = "mil_exp"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    log_per_epoch: bool = True
    save_attention_artifacts: bool = True
    checkpoint_monitor: Optional[str] = None
    checkpoint_min_epoch: int = 0
    checkpoint_after_attention_refinement: bool = False
    checkpoint_min_query_epochs: int = 1

    # Optional early stop on generalization gap. This is useful when the model
    # keeps improving on train bags while validation error starts degrading.
    overfit_gap_stop_enabled: bool = False
    overfit_gap_metric: str = "rmse"
    overfit_gap_patience: int = 2
    overfit_gap_min_delta: float = 0.0
    overfit_gap_abs_threshold: float = 0.10
    overfit_gap_rel_threshold: float = 0.15
    overfit_gap_require_val_worse_than_best: bool = True

    # Optional attention-refinement schedule. When the train/validation RMSE
    # gap exceeds a configured threshold, the model keeps training instead of
    # early-stopping: embedder/predictor LRs are reduced and the active-query
    # path is ramped in for a fixed number of epochs.
    attention_refinement_enabled: bool = False
    attention_refinement_metric: str = "rmse"
    attention_refinement_patience: int = 1
    attention_refinement_min_delta: float = 0.005
    attention_refinement_gap_threshold: float = 0.08
    attention_refinement_rel_gap_threshold: float = 0.15
    attention_refinement_require_val_worse_than_best: bool = True
    attention_refinement_lr_factor: float = 0.1
    attention_refinement_query_epochs: int = 25
    attention_refinement_query_weight_start: float = 0.0
    attention_refinement_query_weight_end: float = 1.0
    attention_refinement_stop_after_query_epochs: bool = True

    # Optional validation-set KID interpretation.
    validation_instance_importance_enabled: bool = False
    validation_instance_importance_min_keep: int = 3
    validation_instance_importance_max_keep: int = 12
    validation_instance_importance_attention_weight: float = 0.5
    validation_instance_importance_impact_weight: float = 0.5


class TrainerBuilder:
    """Utility class for building a PyTorch Lightning Trainer from a TrainerConfig.

    This class provides a clean interface for creating a trainer with the
    appropriate configuration, including handling device-specific settings.
    """

    @staticmethod
    def build(config: TrainerConfig, logger: Optional[Any] = None, 
              callbacks: Optional[List[Callback]] = None) -> pl.Trainer:
        """Build a PyTorch Lightning Trainer from a TrainerConfig.

        Parameters
        ----------
        config : TrainerConfig
            Configuration for the trainer
        logger : Optional[Any]
            Logger to use with the trainer
        callbacks : Optional[List[Callback]]
            Callbacks to use with the trainer

        Returns
        -------
        pl.Trainer
            Configured PyTorch Lightning Trainer
        """
        import torch

        # Determine accelerator if not specified
        accelerator = config.accelerator
        if accelerator is None:
            if config.device.startswith("cuda"):
                accelerator = "gpu"
            elif config.device == "mps":
                accelerator = "mps"
            else:
                accelerator = "cpu"

        # Only use mixed precision with CUDA. MPS/CPU are more reliable in true fp32
        # for this MIL pipeline, especially around attention, normalization, and metrics.
        precision = config.precision
        if isinstance(precision, str):
            precision = precision.strip()
            if precision == "mixed":
                precision = "16-mixed" if accelerator == "gpu" else "32-true"
        if precision in {"16-mixed", "bf16-mixed"} and accelerator != "gpu":
            precision = "32-true"

        # Create trainer kwargs dict to allow conditional inclusion of parameters
        trainer_kwargs = {
            "max_epochs": config.max_epochs,
            "accelerator": accelerator,
            "devices": config.devices,
            "deterministic": config.deterministic,
            "precision": precision,
            "logger": logger,
            "callbacks": callbacks,
            "log_every_n_steps": config.log_every_n_steps,
            "enable_checkpointing": config.enable_checkpointing,
            "enable_progress_bar": config.enable_progress_bar,
            "enable_model_summary": config.enable_model_summary,
            "num_sanity_val_steps": config.num_sanity_val_steps,
        }
        min_epochs = int(config.min_epochs or 0)
        checkpoint_min_epoch = int(getattr(config, "checkpoint_min_epoch", 0) or 0)
        if checkpoint_min_epoch > 0:
            # Lightning displays epochs as zero-based indices. If epoch 30 is the
            # first eligible checkpoint, training must complete at least 31 epochs.
            min_epochs = max(min_epochs, checkpoint_min_epoch + 1)
        if min_epochs > 0:
            trainer_kwargs["min_epochs"] = min_epochs
        # Only include strategy if it's not None
        if config.strategy is not None:
            trainer_kwargs["strategy"] = config.strategy

        return pl.Trainer(**trainer_kwargs)
