from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, frozen=False)
class TrainerOptimConfig:
    """Optimiser and LR-scheduler settings.

    ``lr`` is the base learning rate; the per-component ``*_lr`` fields override
    it for the embedder / self-attention / aggregator / predictor / active-query
    parameter groups (with ``*_lr_factor`` as a relative fallback). The remaining
    fields configure weight decay and the selected scheduler.
    """
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
    # Concordance-correlation (CCC) auxiliary loss weight: loss += ccc_weight·(1 − CCC(ŷ,y)).
    # CCC rewards slope→1 and variance-match, directly countering MSE regression-to-mean
    # shrinkage. 0 = pure MSE (default). Batch-estimated, so use a larger effective batch
    # (trainer.accumulate_grad_batches) when enabling on small batch sizes.
    ccc_weight: float = 0.0
