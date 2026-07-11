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

    # Loss-linked aggregator-focus curriculum (single LR + query authority; the
    # plain LR scheduler should be "none" when this is enabled). When the
    # validation metric plateaus (no improvement > min_delta for `patience`
    # epochs) AND the active-prototype bank is ready (num_active >= min_active),
    # the model enters an "aggregator-focus" phase: embedder/predictor LRs are
    # reduced by `lr_factor` (the aggregator keeps its LR) and the active-prototype
    # query is ramped in to `query_max_weight` over `query_ramp_epochs`. Training
    # then continues until the validation metric plateaus again, at which point
    # it stops. No hardcoded epoch counts — every transition is loss-linked.
    attention_refinement_enabled: bool = False
    attention_refinement_metric: str = "loss"
    attention_refinement_patience: int = 3
    attention_refinement_min_delta: float = 0.005
    attention_refinement_lr_factor: float = 0.1
    attention_refinement_query_ramp_epochs: int = 2
    attention_refinement_query_max_weight: float = 0.8

    # Optional validation-set KID interpretation.
    validation_instance_importance_enabled: bool = False
    validation_instance_importance_min_keep: int = 3
    validation_instance_importance_max_keep: int = 12
    validation_instance_importance_attention_weight: float = 0.5
    validation_instance_importance_impact_weight: float = 0.5


def build_trainer(
    config: TrainerConfig,
    logger: Optional[Any] = None,
    callbacks: Optional[List[Callback]] = None,
) -> pl.Trainer:
    """Build a PyTorch Lightning Trainer from a TrainerConfig.

    Resolves the accelerator (``auto`` -> CUDA/MPS/CPU), forces true fp32 off GPU,
    and enforces ``min_epochs`` so the first eligible checkpoint epoch is reached.
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
    # Stopping is loss-controlled: the aggregator-focus curriculum stops on the
    # validation plateau (or Lightning EarlyStopping when the curriculum is off).
    # min_epochs is only an optional user floor — nothing forces it upward.
    if int(config.min_epochs or 0) > 0:
        trainer_kwargs["min_epochs"] = int(config.min_epochs)
    # Only include strategy if it's not None
    if config.strategy is not None:
        trainer_kwargs["strategy"] = config.strategy

    return pl.Trainer(**trainer_kwargs)
