"""Minimal SwiGLU MLP predictor for MIL bag-level outputs.

A stripped-down alternative to :class:`MLPPredictor`: pre-LN residual SwiGLU FFN
block(s) plus a linear head, and nothing else (no DropPath, no tanh-bounded
residual scale, no split-init tricks, no dynamic input_dim setter).

The head is small-gain-initialised and its bias seeded with ``output_bias`` so a
regression model starts by predicting the training mean.

Input ``bag_repr`` ``[B, D]`` (or ``[D]``) -> ``[B]`` (or scalar) when
``output_dim == 1``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SwiGLUBlock(nn.Module):
    """Pre-LN residual SwiGLU FFN block, with a linear shortcut when dims differ."""

    def __init__(self, in_dim: int, out_dim: int, expansion: float, dropout: float) -> None:
        super().__init__()
        inner = int(math.ceil(max(4.0, out_dim * expansion) / 64.0) * 64)
        self.norm = nn.LayerNorm(in_dim)
        self.ff1 = nn.Linear(in_dim, 2 * inner)
        self.ff2 = nn.Linear(inner, out_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = torch.chunk(self.ff1(self.norm(x)), 2, dim=-1)
        return self.proj(x) + self.ff2(self.drop(F.silu(u) * v))


class SimpleMLPPredictor(nn.Module):
    """SwiGLU MLP head mapping a bag representation to an output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        expansion: float = 2.0,
        dropout: float = 0.1,
        output_dim: int = 1,
        output_bias: Optional[float] = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self._input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.expansion = float(expansion)
        self.dropout = float(dropout)
        self._output_dim = int(output_dim)
        self.output_bias = output_bias

        self.input_norm = nn.LayerNorm(self._input_dim)
        blocks = []
        d_in = self._input_dim
        for _ in range(self.num_layers):
            blocks.append(_SwiGLUBlock(d_in, self.hidden_dim, self.expansion, self.dropout))
            d_in = self.hidden_dim
        self.blocks = nn.ModuleList(blocks)

        self.fc_out = nn.Linear(self.hidden_dim, self._output_dim)
        # Small-gain head + seeded bias: the model starts by predicting the mean.
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        if self.fc_out.bias is not None:
            if output_bias is not None and self._output_dim == 1:
                nn.init.constant_(self.fc_out.bias, float(output_bias))
            else:
                nn.init.zeros_(self.fc_out.bias)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, bag_repr: torch.Tensor) -> torch.Tensor:
        single = bag_repr.dim() == 1
        if single:
            bag_repr = bag_repr.unsqueeze(0)
        x = self.input_norm(bag_repr)
        for block in self.blocks:
            x = block(x)
        y = self.fc_out(x)
        if single:
            y = y.squeeze(0)
        return y.squeeze(-1) if self._output_dim == 1 else y

    def describe(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "input_dim": self._input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "expansion": self.expansion,
            "dropout": self.dropout,
            "output_dim": self._output_dim,
            "output_bias": self.output_bias,
        }
