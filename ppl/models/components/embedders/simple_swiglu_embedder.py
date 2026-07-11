"""Minimal per-instance SwiGLU MLP embedder for MIL conformer bags.

A stripped-down alternative to :class:`ContextualizedMLPEmbedder`: no within-bag
self-attention and none of the exotic residual machinery (DropPath, LayerScale
warmup, zero-init/depth-scale, l2/layernorm output modes). Each conformer is
embedded independently; the MIL aggregator handles all cross-conformer mixing.

Input ``[B, N, F]`` -> output ``[B, N, D]`` (also accepts ``[N, F]``/``[F]``).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLUBlock(nn.Module):
    """Pre-LN residual SwiGLU FFN block: ``x + ff2(silu(u) * v)``, ``u,v = ff1(LN(x))``."""

    def __init__(self, dim: int, expansion: float, dropout: float) -> None:
        super().__init__()
        inner = int(math.ceil(max(4.0, dim * expansion) / 64.0) * 64)
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, 2 * inner)
        self.ff2 = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(self.ff1(self.norm(x)), 2, dim=-1)
        return x + self.ff2(self.drop(F.silu(u) * v))


class SimpleSwiGLUEmbedder(nn.Module):
    """Per-instance SwiGLU MLP embedder. Does not mix information across instances."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        num_layers: int = 1,
        expansion: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self._input_dim = int(input_dim)
        self._embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.expansion = float(expansion)
        self.dropout = float(dropout)

        self.in_proj = nn.Linear(self._input_dim, self._embed_dim)
        self.blocks = nn.ModuleList(
            _SwiGLUBlock(self._embed_dim, self.expansion, self.dropout)
            for _ in range(self.num_layers)
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
        """Embed each instance independently.

        Parameters
        ----------
        x
            Tensor with shape ``[B, N, F]`` (or ``[N, F]`` / ``[F]``).
        mask
            Optional boolean tensor ``[B, N]``, True marks valid conformers.
            Padded positions are zeroed on output (cosmetic — the aggregator
            masks them regardless).
        return_attn
            Interface parity with attention embedders; this one has no attention,
            so the attention-map list is always empty.
        """
        z = self.in_proj(x)
        for block in self.blocks:
            z = block(z)
        if mask is not None:
            z = z * mask.unsqueeze(-1).to(device=z.device, dtype=z.dtype)
        if return_attn:
            return z, []
        return z

    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "output_dim": self._embed_dim,
            "num_layers": self.num_layers,
            "expansion": self.expansion,
            "dropout": self.dropout,
            "self_attention": False,
            "positional_embeddings": False,
        }
