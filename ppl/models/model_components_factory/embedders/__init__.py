"""Embedder components for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.models.model_components_factory.embedders.base_embedder import EmbedderBase
from ppl.models.model_components_factory.embedders.contextualized_mlp_embedder import (
    BagSelfAttentionBlock,
    ContextualizedMLPEmbedder,
)
from ppl.models.model_components_factory.embedders.mlp_embedder_v3 import MLPEmbedderV3
