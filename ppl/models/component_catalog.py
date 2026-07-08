"""Explicit catalog of MIL model components.

The model is a fixed embedder -> aggregator -> predictor pipeline. Components are
selected by name from the plain dictionaries below (no plug-in registry, no
template indirection). To add a component, import its class and add one entry.
"""
from __future__ import annotations

import logging
from typing import Any, Tuple

from ppl.models.components.embedders import ContextualizedMLPEmbedder
from ppl.models.components.aggregators import ClusterHierarchicalAttentionAggregator
from ppl.models.components.predictors import MLPPredictor

LOGGER = logging.getLogger(__name__)

# name -> class. Names are matched case-insensitively.
EMBEDDERS = {
    "contextualized_mlp_embedder": ContextualizedMLPEmbedder,
}

AGGREGATORS = {
    "cluster_hier_mha": ClusterHierarchicalAttentionAggregator,
}

PREDICTORS = {
    "mlp_predictor": MLPPredictor,
}

COMPONENT_ORDER: Tuple[str, str, str] = ("embedder", "aggregator", "predictor")


def _lookup(catalog: dict, kind: str, name: str):
    cls = catalog.get(str(name).lower())
    if cls is None:
        raise ValueError(
            f"Unknown {kind} '{name}'. Available: {sorted(catalog)}"
        )
    return cls


def _merge_self_attention_kwargs(cfg: Any, embedder_kwargs: dict) -> dict:
    """Fold optional self-attention kwargs into the embedder kwargs."""
    self_attention_kwargs = getattr(cfg, "self_attention_kwargs", {}) or {}
    if not self_attention_kwargs:
        return embedder_kwargs

    overlap = set(embedder_kwargs).intersection(self_attention_kwargs)
    if overlap:
        raise ValueError(
            "Duplicate keys in embedder_kwargs and self_attention_kwargs: "
            f"{sorted(overlap)}. Keep each hyperparameter in only one group."
        )

    merged = embedder_kwargs.copy()
    merged.update(self_attention_kwargs)
    LOGGER.info(
        "[MODEL] Applying self_attention_kwargs to embedder: %s",
        sorted(self_attention_kwargs),
    )
    return merged


def build_components(cfg: Any, input_dim: int) -> Tuple[Any, Any, Any]:
    """Build the embedder, aggregator, and predictor with dimension chaining.

    Parameters
    ----------
    cfg : ModelBuilderConfig
        Model configuration.
    input_dim : int
        Descriptor dimension fed to the embedder.

    Returns
    -------
    tuple
        (embedder, aggregator, predictor).
    """
    embedder_cls = _lookup(EMBEDDERS, "embedder_type", cfg.embedder_type)
    aggregator_cls = _lookup(AGGREGATORS, "aggregator_type", cfg.aggregator_type)
    predictor_cls = _lookup(PREDICTORS, "predictor_type", cfg.predictor_type)

    embedder_kwargs = dict(getattr(cfg, "embedder_kwargs", {}) or {})
    embedder_kwargs = _merge_self_attention_kwargs(cfg, embedder_kwargs)
    embedder = embedder_cls(input_dim=input_dim, **embedder_kwargs)
    LOGGER.info("[MODEL] Embedder: %s", cfg.embedder_type)

    aggregator_kwargs = dict(getattr(cfg, "aggregator_kwargs", {}) or {})
    aggregator = aggregator_cls(input_dim=embedder.output_dim, **aggregator_kwargs)
    LOGGER.info("[MODEL] Aggregator: %s", cfg.aggregator_type)

    predictor_kwargs = dict(getattr(cfg, "predictor_kwargs", {}) or {})
    if cfg.task == "regression" and "output_dim" not in predictor_kwargs:
        predictor_kwargs["output_dim"] = 1
    predictor = predictor_cls(input_dim=aggregator.output_dim, **predictor_kwargs)
    LOGGER.info("[MODEL] Predictor: %s", cfg.predictor_type)

    return embedder, aggregator, predictor
