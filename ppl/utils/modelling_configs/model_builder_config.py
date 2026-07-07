from __future__ import annotations

from dataclasses import dataclass, field, replace

from ppl.utils.modelling_configs.trainer_optim_config import TrainerOptimConfig
from ppl.utils.model_builder.template_registry import Template


@dataclass(slots=True, frozen=False)
class ModelBuilderConfig:
    # Model Template
    template: Template = Template.BAG_ATTENTION

    # Model Components
    embedder_type: str = "mlp_embedder"
    aggregator_type: str = "mha_att"
    predictor_type: str = "mlp_predictor"
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
