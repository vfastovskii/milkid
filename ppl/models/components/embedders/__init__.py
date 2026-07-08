"""Embedder components for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.models.components.embedders.base_embedder import EmbedderBase
from ppl.models.components.embedders.mlp_embedder import MLPEmbedder
from ppl.models.components.embedders.contextualized_mlp_embedder import (
    BagSelfAttentionBlock,
    ContextualizedMLPEmbedder,
)
