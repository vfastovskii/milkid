from __future__ import annotations

from dataclasses import dataclass, field, replace

from ppl.config.trainer_optim_config import TrainerOptimConfig


@dataclass(slots=True, frozen=False)
class ModelBuilderConfig:
    # Model Components
    embedder_type: str = "contextualized_mlp_embedder_v1"
    aggregator_type: str = "cluster_hier_mha_v1"
    predictor_type: str = "mlp_predictor_v3"
    # Injected from the processed data feature count before model construction.
    input_dim: int | None = None
    task: str = "regression"

    embedder_kwargs: dict = field(default_factory=dict)
    self_attention_kwargs: dict = field(default_factory=dict)
    aggregator_kwargs: dict = field(default_factory=dict)
    predictor_kwargs: dict = field(default_factory=dict)
    active_prototype_kwargs: dict = field(default_factory=dict)

    # Optimiser
    optim: TrainerOptimConfig = field(default_factory=lambda: TrainerOptimConfig(
        lr=1e-3,
        weight_decay=1e-4,
        scheduler="plateau",
        lr_patience=20,
        lr_t_max=50,
    ))

    def update(self, **kwargs) -> "ModelBuilderConfig":
        """
        Return a new BuilderConfig with the provided overrides.

        Any keyword that matches an attribute of the nested TrainerOptimConfig
        is routed there automatically.

        # >>> cfg2 = cfg.update(aggregator_type="gated_att", lr=5e-4)
        """

        # separate optimiser-level overrides
        optim_keys   = {k for k in kwargs if hasattr(self.optim, k)}
        optim_kwargs = {k: kwargs.pop(k) for k in optim_keys}

        new_optim = replace(self.optim, **optim_kwargs) if optim_kwargs else self.optim
        return replace(self, optim=new_optim, **kwargs)
