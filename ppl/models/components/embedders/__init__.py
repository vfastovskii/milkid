"""Embedder components for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.models.components.embedders.mlp_embedder import MLPEmbedder
from ppl.models.components.embedders.contextualized_mlp_embedder import (
    BagSelfAttentionBlock,
    ContextualizedMLPEmbedder,
)
from ppl.models.components.embedders.simple_swiglu_embedder import (
    SimpleSwiGLUEmbedder,
)
from ppl.models.components.embedders.simple_swiglu_attn_embedder import (
    SimpleSwiGLUAttnEmbedder,
)
