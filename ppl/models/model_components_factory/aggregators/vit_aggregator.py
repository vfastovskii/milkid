from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint



# -------------------- utils --------------------
class DropPath(nn.Module):
    """Stochastic depth (per-sample)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor()
        return x.div(keep) * mask


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02):
    # Approximate truncated normal (±2σ)
    with torch.no_grad():
        tensor.normal_(0, std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor


class _EncoderBlock(nn.Module):
    """
    Pre/Post-LN Transformer encoder block (ViT-style, no positional encodings).
    ViT-like init: truncated normal (std=0.02) for linear weights, bias=0.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        pre_layer_norm: bool,
        drop_path: float,
        gated_mlp: bool = False,  # ViT default: GELU MLP (no gating)
        activation: Literal["gelu","silu","relu"] = "gelu",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.pre_ln = bool(pre_layer_norm)
        self.gated = bool(gated_mlp)

        # norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # attention
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=attn_dropout, bias=True, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.dp1 = DropPath(drop_path)

        # MLP
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden * (2 if self.gated else 1))
        self.act = nn.GELU() if activation == "gelu" else (nn.SiLU() if activation == "silu" else nn.ReLU())
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)
        self.dp2 = DropPath(drop_path)

        self._init_weights()

    def _init_weights(self):
        _trunc_normal_(self.fc1.weight, 0.02); nn.init.zeros_(self.fc1.bias)
        _trunc_normal_(self.fc2.weight, 0.02); nn.init.zeros_(self.fc2.bias)
        if hasattr(self.mha, "in_proj_weight") and self.mha.in_proj_weight is not None:
            _trunc_normal_(self.mha.in_proj_weight, 0.02)
        if hasattr(self.mha, "in_proj_bias") and self.mha.in_proj_bias is not None:
            nn.init.zeros_(self.mha.in_proj_bias)
        if hasattr(self.mha, "out_proj") and hasattr(self.mha.out_proj, "weight"):
            _trunc_normal_(self.mha.out_proj.weight, 0.02)
            if self.mha.out_proj.bias is not None:
                nn.init.zeros_(self.mha.out_proj.bias)

    def _forward_impl(self, x, key_padding_mask=None, need_attn: bool = False):
        # Pre-LN
        h = self.ln1(x) if self.pre_ln else x
        attn_out, attn_w = self.mha(
            h, h, h,
            need_weights=need_attn,
            average_attn_weights=False,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dp1(self.drop1(attn_out))
        x = self.ln1(x) if not self.pre_ln else x

        h = self.ln2(x) if self.pre_ln else x
        h = self.fc1(h)
        if self.gated:
            v, g = h.chunk(2, dim=-1)
            h = F.silu(g) * v
        else:
            h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        x = x + self.dp2(self.drop2(h))
        x = self.ln2(x) if not self.pre_ln else x
        return x, attn_w

    def forward(self, x, key_padding_mask=None, need_attn: bool = False):
        if self.use_checkpoint and self.training:
            x_in = x
            def core(z, mask):
                z, _ = self._forward_impl(z, key_padding_mask=mask, need_attn=False)
                return z
            x = checkpoint.checkpoint(core, x_in, key_padding_mask, use_reentrant=False)
            attn_w = None
            if need_attn:
                with torch.no_grad():
                    _, attn_w = self._forward_impl(x_in, key_padding_mask=key_padding_mask, need_attn=True)
            return x, attn_w
        else:
            return self._forward_impl(x, key_padding_mask=key_padding_mask, need_attn=need_attn)


# -------------------- ViT aggregator (CLS readout only) --------------------
class VITAggregator(nn.Module):
    r"""
    ViT-style MIL aggregator (no positional encodings).
    - Always uses **CLS token output** as bag representation.
    - Returns `alpha` for interpretability from **last CLS→instances** attention.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_heads: int = 4,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.05,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        pre_layer_norm: bool = True,
        use_checkpoint: bool = True,
        eps: float = 1e-8,
        use_cls_token: bool = True,    # True for pure ViT; kept for API symmetry
        final_norm: bool = True,       # make this True OR predictor.input_layernorm=True, not both
    ):
        super().__init__()

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self._input_dim = self.out_dim = input_dim
        self.num_heads = int(num_heads)
        self.head_dim = input_dim // num_heads
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.pre_layer_norm = bool(pre_layer_norm)
        self.use_checkpoint = bool(use_checkpoint)
        self.eps = float(eps)
        self.use_cls_token = bool(use_cls_token)
        self.use_final_norm = bool(final_norm)

        # store for describe() and rebuilds
        self._depth = int(depth)
        self._mlp_ratio = float(mlp_ratio)
        self.drop_path_rate = float(drop_path_rate)

        dpr = [self.drop_path_rate * i / max(1, self._depth - 1) for i in range(self._depth)]
        self.blocks = nn.ModuleList([
            _EncoderBlock(
                dim=input_dim,
                num_heads=num_heads,
                mlp_ratio=self._mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                pre_layer_norm=self.pre_layer_norm,
                drop_path=dpr[i],
                gated_mlp=False,
                activation="gelu",
                use_checkpoint=self.use_checkpoint,
            )
            for i in range(self._depth)
        ])

        # CLS + final norm on CLS (optional)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, 1, input_dim))
            _trunc_normal_(self.cls_token, 0.02)
        else:
            self.register_parameter("cls_token", None)
        self.cls_final_norm = nn.LayerNorm(input_dim) if self.use_final_norm else nn.Identity()

        # state for viz
        self.last_attn = None        # [B,H,1,N]
        self.last_attn_full = None   # [B,H,L,L]

    @property
    def output_dim(self):
        return self.out_dim

    @property
    def input_dim(self):
        return self._input_dim

    @input_dim.setter
    def input_dim(self, value: int):
        if self._input_dim == value:
            return
        assert value > 0
        self._input_dim = value
        self.out_dim = value
        if value % self.num_heads != 0:
            for i in range(self.num_heads, 0, -1):
                if value % i == 0:
                    self.num_heads = i
                    break
        self.head_dim = value // self.num_heads
        # rebuild blocks with preserved depth/mlp_ratio/drop_path_rate
        dpr = [self.drop_path_rate * i / max(1, self._depth - 1) for i in range(self._depth)]
        self.blocks = nn.ModuleList([
            _EncoderBlock(
                dim=value,
                num_heads=self.num_heads,
                mlp_ratio=self._mlp_ratio,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                pre_layer_norm=self.pre_layer_norm,
                drop_path=dpr[i],
                gated_mlp=False,
                activation="gelu",
                use_checkpoint=self.use_checkpoint,
            )
            for i in range(self._depth)
        ])
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.empty(1, 1, value)); _trunc_normal_(self.cls_token, 0.02)
        else:
            self.register_parameter("cls_token", None)
        self.cls_final_norm = nn.LayerNorm(value) if self.use_final_norm else nn.Identity()

    # --- forward --------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_entropy: bool = True,
        return_attn: bool = False,
    ):
        # AMP-friendly: cast inputs to module dtype
        p0 = next(self.parameters(), None)
        if p0 is not None:
            h = h.to(p0.dtype)

        # Ensure batch dim
        if h.dim() == 2:
            h_seq = h.unsqueeze(0)           # [1, N, D]
        else:
            h_seq = h                        # [B, N, D]
        B, N, D = h_seq.shape

        # Build sequence (+CLS) and mask
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)      # [B,1,D]
            x = torch.cat([cls, h_seq], dim=1)          # [B,1+N,D]
            mask = None
            if key_padding_mask is not None:
                pad = torch.zeros(B, 1, dtype=torch.bool, device=h_seq.device)
                mask = torch.cat([pad, key_padding_mask], dim=1)  # [B,1+N]
        else:
            x = h_seq
            mask = key_padding_mask

        # Blocks (only request attn from last block)
        attn_full = None
        for i, blk in enumerate(self.blocks):
            need_attn = (return_attn or return_entropy) and (i == self._depth - 1)
            x, attn_w = blk(x, key_padding_mask=mask, need_attn=need_attn)
            if attn_w is not None:
                attn_full = attn_w
        self.last_attn_full = attn_full

        # CLS readout (no pooling of original embeddings)
        if self.use_cls_token:
            cls_out = x[:, 0, :]                      # [B,D]
            bag_repr = self.cls_final_norm(cls_out)   # [B,D]
        else:
            # fallback if CLS disabled: mean of transformed tokens (mask-aware)
            tokens = x
            if mask is not None:
                valid = (~mask).float()
                sums = (tokens * valid.unsqueeze(-1)).sum(dim=1)
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                bag_repr = sums / denom
            else:
                bag_repr = tokens.mean(dim=1)
            bag_repr = self.cls_final_norm(bag_repr)

        # α for interpretability from last block CLS→instances
        if self.use_cls_token and attn_full is not None:
            cls_to_inst = attn_full[:, :, 0:1, 1:]               # [B,H,1,N]
            alpha_raw = cls_to_inst.mean(dim=1).squeeze(1)       # [B,N]
            if key_padding_mask is not None:
                alpha_raw = alpha_raw.masked_fill(key_padding_mask, 0.0)
            alpha = alpha_raw / (alpha_raw.sum(dim=-1, keepdim=True).clamp_min(self.eps))
            alpha_std = torch.zeros_like(alpha)
            self.last_attn = cls_to_inst
        else:
            alpha = torch.full((B, N), 1.0 / max(1, N), device=h_seq.device, dtype=h_seq.dtype)
            alpha_std = torch.zeros_like(alpha)
            self.last_attn = None

        # Squeeze back for single-bag
        if h.dim() == 2:
            bag_repr = bag_repr.squeeze(0)   # [D]
            alpha = alpha.squeeze(0)         # [N]
            alpha_std = alpha_std.squeeze(0) # [N]

        # Extras
        extras = {"alpha": alpha, "alpha_std": alpha_std}
        if return_entropy:
            a = alpha.clamp_min(self.eps)
            extras["entropy"] = (-(a * a.log()).sum() if a.dim() == 1 else -(a * a.log()).sum(dim=1))
        if return_attn and self.last_attn is not None:
            extras["attn"] = self.last_attn   # [B,H,1,N]

        return bag_repr, extras

    # --- describe -------------------------------------------------------------
    def describe(self) -> dict:
        """Human-readable config summary (parity with previous aggregators)."""
        # Recreate current drop-path schedule for reporting
        dpr = [self.drop_path_rate * i / max(1, self._depth - 1) for i in range(self._depth)]
        return {
            "class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.out_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "pre_layer_norm": self.pre_layer_norm,
            "use_checkpoint": self.use_checkpoint,
            "use_cls_token": self.use_cls_token,
            "final_norm": self.use_final_norm,
            "depth": self._depth,
            "mlp_ratio": self._mlp_ratio,
            "drop_path_rate": self.drop_path_rate,
            "architecture": {
                "encoder_blocks": self._depth,
                "drop_path_schedule": [round(x, 6) for x in dpr],
                "attention_type": "self-attention (no positional encodings)",
                "alpha_source": "CLS→instances (last block)" if self.use_cls_token else "uniform (no CLS)",
                "cls_token_init": "trunc_normal(std=0.02)" if self.use_cls_token else None,
                "init": "trunc_normal(std=0.02) for linear weights, bias=0",
            },
        }
