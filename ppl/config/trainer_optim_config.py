from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, frozen=False)
class TrainerOptimConfig:
    lr: float = 1e-4
    embedder_lr: Optional[float] = None
    self_attention_lr: Optional[float] = None
    aggregator_lr: Optional[float] = None
    predictor_lr: Optional[float] = None
    active_query_lr: Optional[float] = None
    weight_decay: float = 1e-4
    scheduler: str = "cosine"   # {"plateau", "cosine", "none"}
    lr_patience: int = 20        # ReduceLROnPlateau
    factor: float = 0.01         # ReduceLROnPlateau decay factor
    lr_t_max: int = 200           # CosineAnnealingLR
    eta_min: float = 1e-5
    embedder_lr_factor: float = 0.1  # Explicit LR factor for embedder
    self_attention_lr_factor: float = 0.1  # LR factor for contextualizer/self-attention
    aggregator_lr_factor: float = 1.0  # Explicit LR factor for aggregator
    predictor_lr_factor: float = 1.0  # Explicit LR factor for predictor
    active_query_lr_factor: float = 1.0  # LR factor for bioactive query builder
    enable_gradient_tracking: bool = True  # Control whether to track gradients during training
