from __future__ import annotations

from dataclasses import dataclass, field
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
    # Best-model selection floor: only epochs >= checkpoint_min_epoch are eligible
    # for the best-val checkpoint (0 = no floor). Training is guaranteed to reach it
    # (min_epochs is raised to match), so the floor always has eligible epochs; it
    # does NOT prevent the curriculum from stopping once past the floor.
    checkpoint_min_epoch: int = 0

    # Per-epoch KID pose-selection metric: top-k attention success against the
    # experimental pose (best_rmsd_to_exp <= threshold / norm o3a_to_exp >= threshold)
    # for active, correctly-predicted molecules. Needs the SDF carrying per-conformer
    # best_rmsd_to_exp / o3a_to_exp (keyed by conformer _Name).
    kid_metric_enabled: bool = False
    kid_sdf_path: Optional[str] = None
    kid_top_k: List[int] = field(default_factory=lambda: [1, 3, 5])
    kid_rmsd_threshold: float = 2.0
    kid_o3a_threshold: float = 0.8
    kid_active_threshold: float = 7.0
    kid_pred_tol: float = 1.0

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
    # Minimum number of focus-phase epochs before the loss-plateau stop may fire,
    # so the aggregator has time to adapt to the injected active prototypes (the
    # ramp + LR cut make val loss wobble up briefly, which would otherwise stop it).
    attention_refinement_min_focus_epochs: int = 20

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
    # min_epochs is an optional user floor, also raised so training reaches the
    # best-checkpoint floor (the best model can only come from epochs that run).
    min_epochs = int(config.min_epochs or 0)
    checkpoint_min_epoch = int(getattr(config, "checkpoint_min_epoch", 0) or 0)
    if checkpoint_min_epoch > 0:
        # Epoch indices are 0-based, so reaching epoch N requires N+1 epochs.
        min_epochs = max(min_epochs, checkpoint_min_epoch + 1)
    if min_epochs > 0:
        trainer_kwargs["min_epochs"] = min_epochs
    # Only include strategy if it's not None
    if config.strategy is not None:
        trainer_kwargs["strategy"] = config.strategy

    return pl.Trainer(**trainer_kwargs)
