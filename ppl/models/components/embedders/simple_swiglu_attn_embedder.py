"""Minimal SwiGLU MLP embedder + within-bag self-attention.

Same stripped-down style as :class:`SimpleSwiGLUEmbedder` (no DropPath /
LayerScale / exotic init), but after the per-instance SwiGLU blocks it adds
one or more pre-LN self-attention blocks so conformers can attend to each
other inside a bag. A lighter alternative to :class:`ContextualizedMLPEmbedder`.

Input ``[B, N, F]`` -> output ``[B, N, D]``. The optional mask uses True for
valid conformers and False for padding.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ppl.models.components.embedders.simple_swiglu_embedder import _SwiGLUBlock


class _SelfAttnBlock(nn.Module):
    """Pre-LN self-attention residual + a pre-LN SwiGLU FFN residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion: float,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ffn = _SwiGLUBlock(dim, expansion, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        h = self.norm(x)
        attn_out, attn_w = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        x = x + attn_out
        x = self.ffn(x)  # _SwiGLUBlock is itself a pre-LN residual FFN
        if return_attn:
            return x, attn_w
        return x


class SimpleSwiGLUAttnEmbedder(nn.Module):
    """Per-instance SwiGLU MLP followed by within-bag self-attention."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_mlp_layers: int = 1,
        num_attn_layers: int = 1,
        num_heads: int = 4,
        expansion: float = 2.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_mlp_layers < 1:
            raise ValueError(f"num_mlp_layers must be >= 1, got {num_mlp_layers}")
        if num_attn_layers < 1:
            raise ValueError(f"num_attn_layers must be >= 1, got {num_attn_layers}")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self._input_dim = int(input_dim)
        self._embed_dim = int(embed_dim)
        self.num_mlp_layers = int(num_mlp_layers)
        self.num_attn_layers = int(num_attn_layers)
        self.num_heads = int(num_heads)
        self.expansion = float(expansion)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)

        self.in_proj = nn.Linear(self._input_dim, self._embed_dim)
        self.mlp_blocks = nn.ModuleList(
            _SwiGLUBlock(self._embed_dim, self.expansion, self.dropout)
            for _ in range(self.num_mlp_layers)
        )
        self.attn_blocks = nn.ModuleList(
            _SelfAttnBlock(
                self._embed_dim, self.num_heads, self.expansion,
                self.dropout, self.attn_dropout,
            )
            for _ in range(self.num_attn_layers)
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    @property
    def out_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """Embed each instance, then contextualize within each bag.

        Parameters
        ----------
        x
            Tensor with shape ``[B, N, F]``.
        mask
            Optional boolean tensor ``[B, N]``, True marks valid conformers.
        return_attn
            If True, also return the per-block attention maps ``[B, H, N, N]``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"SimpleSwiGLUAttnEmbedder expects [B, N, F], got {tuple(x.shape)}"
            )
        key_padding_mask = ~mask.bool() if mask is not None else None

        z = self.in_proj(x)
        for block in self.mlp_blocks:
            z = block(z)

        attn_maps = []
        for block in self.attn_blocks:
            if return_attn:
                z, attn_w = block(z, key_padding_mask=key_padding_mask, return_attn=True)
                attn_maps.append(attn_w)
            else:
                z = block(z, key_padding_mask=key_padding_mask)

        if mask is not None:
            z = z * mask.unsqueeze(-1).to(device=z.device, dtype=z.dtype)
        if return_attn:
            return z, attn_maps
        return z

    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "output_dim": self._embed_dim,
            "num_mlp_layers": self.num_mlp_layers,
            "num_attn_layers": self.num_attn_layers,
            "num_heads": self.num_heads,
            "expansion": self.expansion,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "self_attention": True,
            "positional_embeddings": False,
        }
