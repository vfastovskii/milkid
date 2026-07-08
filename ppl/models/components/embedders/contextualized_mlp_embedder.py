"""Contextualized MLP embedder for MIL conformer bags.

This module keeps the local descriptor embedder separate from bag-level
self-attention. Conformer bags are unordered, so no positional embeddings are
used by default; the self-attention block remains permutation-equivariant.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppl.models.components.embedders.base_embedder import (
    EmbedderBase,
)
from ppl.models.components.embedders.mlp_embedder import (
    DropPath,
    MLPEmbedder,
)


class BagSelfAttentionBlock(nn.Module):
    """Transformer-style self-attention block over instances inside each bag.

    Parameters
    ----------
    dim
        Instance embedding dimension.
    num_heads
        Number of attention heads.
    mlp_ratio
        Expansion ratio for the SwiGLU FFN branch.
    attn_dropout
        Dropout inside multi-head attention.
    dropout
        Dropout inside the FFN branch.
    drop_path
        Stochastic depth probability for each residual branch.
    residual_scale_init
        Initial value for per-channel LayerScale gamma parameters.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
        drop_path: float = 0.05,
        residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.attn_dropout = float(attn_dropout)
        self.dropout_rate = float(dropout)
        self.drop_path_rate = float(drop_path)
        self.residual_scale_init = float(residual_scale_init)
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            batch_first=True,
        )
        self.drop_path1 = DropPath(self.drop_path_rate)
        self.gamma_attn = nn.Parameter(
            self.residual_scale_init * torch.ones(self.dim)
        )

        self.norm2 = nn.LayerNorm(self.dim)
        inner_dim = int(math.ceil((self.dim * self.mlp_ratio) / 64.0) * 64)
        self.inner_dim = inner_dim
        self.ff1 = nn.Linear(self.dim, 2 * inner_dim)
        self.ff2 = nn.Linear(inner_dim, self.dim)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.drop_path2 = DropPath(self.drop_path_rate)
        self.gamma_ffn = nn.Parameter(
            self.residual_scale_init * torch.ones(self.dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Contextualize instance embeddings inside each bag.

        Parameters
        ----------
        x
            Tensor with shape [B, N, D].
        mask
            Optional boolean tensor with shape [B, N], where True marks valid
            conformers and False marks padding.
        return_attn
            If True, also return attention maps with shape [B, H, N, N].
        """
        if x.dim() != 3:
            raise ValueError(
                f"BagSelfAttentionBlock expects [B, N, D], got shape {x.shape}"
            )

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()

        h = self.norm1(x)
        attn_out, attn_weights = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        gamma_attn = self.gamma_attn.to(device=attn_out.device, dtype=attn_out.dtype)
        x = x + self.drop_path1(gamma_attn * attn_out)

        h = self.norm2(x)
        u, v = torch.chunk(self.ff1(h), 2, dim=-1)
        h = F.silu(u) * v
        h = self.dropout(h)
        h = self.ff2(h)

        gamma_ffn = self.gamma_ffn.to(device=h.device, dtype=h.dtype)
        x = x + self.drop_path2(gamma_ffn * h)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(device=x.device, dtype=x.dtype)

        if return_attn:
            return x, attn_weights
        return x

    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "dim": self.dim,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "inner_dim": self.inner_dim,
            "attn_dropout": self.attn_dropout,
            "dropout": self.dropout_rate,
            "drop_path": self.drop_path_rate,
            "residual_scale_init": self.residual_scale_init,
            "positional_embeddings": False,
        }


class ContextualizedMLPEmbedder(EmbedderBase):
    """Local MLP embedder followed by bag-level self-attention.

    Input shape is [B, N, F] and output shape is [B, N, D]. The optional mask
    uses True for valid conformers and False for padding.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_mlp_layers: int = 2,
        num_attn_layers: int = 1,
        num_heads: int = 4,
        mlp_expansion: float = 2.0,
        attn_mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        stochastic_depth: float = 0.05,
        attn_ffn_dropout: Optional[float] = None,
        attn_stochastic_depth: Optional[float] = None,
        attn_residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()

        self._input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.num_mlp_layers = int(num_mlp_layers)
        self.num_attn_layers = int(num_attn_layers)
        self.num_heads = int(num_heads)
        self.mlp_expansion = float(mlp_expansion)
        self.attn_mlp_ratio = float(attn_mlp_ratio)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.stochastic_depth = float(stochastic_depth)
        self.attn_ffn_dropout = (
            self.dropout if attn_ffn_dropout is None else float(attn_ffn_dropout)
        )
        self.attn_stochastic_depth = (
            self.stochastic_depth
            if attn_stochastic_depth is None
            else float(attn_stochastic_depth)
        )
        self.attn_residual_scale_init = float(attn_residual_scale_init)
        if self.num_attn_layers < 0:
            raise ValueError(
                f"num_attn_layers must be non-negative, got {num_attn_layers}"
            )
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}"
            )

        self.local_embedder = MLPEmbedder(
            input_dim=self._input_dim,
            hidden_dim=self.embed_dim,
            num_layers=self.num_mlp_layers,
            expansion=self.mlp_expansion,
            dropout=self.dropout,
            output_dim=self.embed_dim,
            final_norm="none",
            block_norm="pre",
            gated=True,
            stochastic_depth=self.stochastic_depth,
        )

        self.context_blocks = nn.ModuleList()
        for layer_idx in range(self.num_attn_layers):
            if self.num_attn_layers > 1:
                dp_rate = self.attn_stochastic_depth * layer_idx / max(
                    1, self.num_attn_layers - 1
                )
            else:
                dp_rate = self.attn_stochastic_depth

            self.context_blocks.append(
                BagSelfAttentionBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.attn_mlp_ratio,
                    attn_dropout=self.attn_dropout,
                    dropout=self.attn_ffn_dropout,
                    drop_path=dp_rate,
                    residual_scale_init=self.attn_residual_scale_init,
                )
            )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    @property
    def out_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Embed and contextualize a padded bag batch.

        Parameters
        ----------
        x
            Tensor with shape [B, N, F].
        mask
            Optional boolean tensor with shape [B, N], where True marks valid
            conformers and False marks padding.
        return_attn
            If True, return the contextual embeddings plus all attention maps.
        """
        if x.dim() != 3:
            raise ValueError(
                f"Self-attention requires bag-shaped input [B, N, F], got {x.shape}"
            )

        z = self.local_embedder(x)
        if mask is not None:
            z = z * mask.unsqueeze(-1).to(device=z.device, dtype=z.dtype)

        attn_maps = []
        for block in self.context_blocks:
            if return_attn:
                z, attn = block(z, mask=mask, return_attn=True)
                attn_maps.append(attn)
            else:
                z = block(z, mask=mask, return_attn=False)

        if return_attn:
            return z, attn_maps
        return z

    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "output_dim": self.embed_dim,
            "num_mlp_layers": self.num_mlp_layers,
            "num_attn_layers": self.num_attn_layers,
            "num_heads": self.num_heads,
            "mlp_expansion": self.mlp_expansion,
            "attn_mlp_ratio": self.attn_mlp_ratio,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "stochastic_depth": self.stochastic_depth,
            "attn_ffn_dropout": self.attn_ffn_dropout,
            "attn_stochastic_depth": self.attn_stochastic_depth,
            "attn_residual_scale_init": self.attn_residual_scale_init,
            "positional_embeddings": False,
            "local_embedder": self.local_embedder.describe()
            if hasattr(self.local_embedder, "describe")
            else self.local_embedder.__class__.__name__,
            "context_blocks": [
                block.describe() if hasattr(block, "describe") else block.__class__.__name__
                for block in self.context_blocks
            ],
        }
