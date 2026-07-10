from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


@dataclass(slots=True, frozen=False)
class TrainerConfig:
    """Configuration for the PyTorch Lightning Trainer plus this pipeline's
    training-schedule extras: checkpointing, the optional overfit-gap early stop,
    and the attention-refinement curriculum.

    Each field is documented by its inline comment below rather than an
    exhaustive Parameters block (which went stale as fields were added).
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

        # Determine accelerator if not specified. Resolve "auto" (and prefer CUDA)
        # so a config that says auto/mps still uses the GPU on a CUDA machine.
        accelerator = config.accelerator
        if accelerator is None:
            dev = (config.device or "auto").lower()
            if dev == "auto":
                dev = (
                    "cuda" if torch.cuda.is_available()
                    else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                    else "cpu"
                )
            if dev.startswith("cuda") or dev == "gpu":
                accelerator = "gpu"
            elif dev == "mps":
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
