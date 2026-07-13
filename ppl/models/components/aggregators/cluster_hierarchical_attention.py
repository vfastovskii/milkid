"""Cluster-aware hierarchical attention aggregator for MIL."""

from __future__ import annotations

import math
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _det_index_add_(target: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """In-place index_add_ that stays deterministic on MPS.

    ``torch.use_deterministic_algorithms(True)`` covers index_add_ on CUDA/CPU
    but not on the Metal (MPS) backend, where the scatter-reduction order is
    unspecified. On MPS we run the reduction on CPU (serial, deterministic) and
    copy the result back; other backends use the native op.
    """
    if target.device.type == "mps":
        acc = target.cpu()
        acc.index_add_(dim, index.cpu(), source.cpu())
        target.copy_(acc.to(target.device))
    else:
        target.index_add_(dim, index, source)
    return target


def _det_scatter_add_(target: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """In-place scatter_add_ that stays deterministic on MPS (see _det_index_add_)."""
    if target.device.type == "mps":
        acc = target.cpu()
        acc.scatter_add_(dim, index.cpu(), source.cpu())
        target.copy_(acc.to(target.device))
    else:
        target.scatter_add_(dim, index, source)
    return target


class SwiGLUFeedForward(nn.Module):
    """Transformer FFN with SwiGLU gating and rounded hidden width."""

    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")

        inner_dim = int(math.ceil((dim * mlp_ratio) / 64.0) * 64)
        self.fc1 = nn.Linear(dim, 2 * inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(u) * v
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class CLSCrossAttentionBlock(nn.Module):
    """Pre-norm query cross-attention block with residual FFN refinement."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mlp_ratio: float = 2.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if residual_scale_init < 0.0:
            raise ValueError(
                "residual_scale_init must be non-negative, got "
                f"{residual_scale_init}"
            )

        norm = nn.LayerNorm if use_layer_norm else nn.Identity
        self.norm_q = norm(dim)
        self.norm_kv = norm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=True,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.gamma_attn = nn.Parameter(residual_scale_init * torch.ones(dim))

        self.norm_ffn = norm(dim)
        self.ffn = SwiGLUFeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.gamma_ffn = nn.Parameter(residual_scale_init * torch.ones(dim))

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_temperature: Optional[torch.Tensor] = None,
        return_attn: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_q(query)
        kv = self.norm_kv(memory)
        if query_temperature is not None:
            q = q / query_temperature.to(device=q.device, dtype=q.dtype)

        attn_out, attn_weights = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        gamma_attn = self.gamma_attn.to(device=query.device, dtype=query.dtype)
        query = query + gamma_attn * self.attn_dropout(attn_out)

        ffn_out = self.ffn(self.norm_ffn(query))
        gamma_ffn = self.gamma_ffn.to(device=query.device, dtype=query.dtype)
        query = query + gamma_ffn * ffn_out

        return query, attn_weights


class ClusterHierarchicalAttentionAggregator(nn.Module):
    """Hierarchical query-attention MIL aggregator.

    The aggregator performs cluster-level selection first, then refines attention
    inside the selected clusters. Both stages use a canonical pre-norm
    transformer-style query cross-attention block:

    MIL query -> cross-attention over memory -> residual update -> SwiGLU FFN.

    No positional embeddings are used; conformer bags remain unordered.

    Inputs
    ------
    h
        Instance embeddings with shape [B, N, D] or [N, D].
    cluster_ids
        Integer cluster labels with shape [B, N]. Padding should be -1.
    key_padding_mask
        Boolean mask with shape [B, N], where True marks padding.

    Outputs
    -------
    bag_repr
        Bag-level representation with shape [B, D].
    extras
        Includes final instance attention ``alpha`` / ``alpha_final``,
        query-stage cluster attention ``cluster_alpha``, final cluster mass
        induced by ``alpha_final`` as ``cluster_alpha_final``, uniform
        cluster-stage attention ``alpha_cluster_uniform``, refined instance
        attention ``alpha_instance_refined``, stage CLS representations, and
        optional ``reg_loss``.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_layer_norm: bool = True,
        num_query_tokens: int = 1,
        pool_from: Literal["inputs", "normed_inputs"] = "inputs",
        pool_v_proj: Union[bool, Literal[False, "linear", "tie_mha_v"]] = False,
        pool_v_bias: bool = False,
        use_temperature: bool = True,
        temperature_init: float = 0.2,
        attn_mlp_ratio: float = 2.0,
        residual_scale_init: float = 1e-2,
        bag_repr_source: Literal[
            "weighted_sum",
            "cls",
            "cls_plus_weighted",
        ] = "weighted_sum",
        refine_top_k_clusters: int = 1,
        cluster_refinement_mode: Literal["topk", "soft"] = "topk",
        refine_mix: float = 0.5,
        refine_mix_learnable: bool = False,
        external_query_gate_init: float = 0.5,
        cluster_loss_coeff: float = 0.005,
        cluster_compactness_weight: float = 1.0,
        cluster_separation_weight: float = 0.1,
        cluster_separation_max_sim: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_query_tokens <= 0:
            raise ValueError(
                f"num_query_tokens must be positive, got {num_query_tokens}"
            )
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
            )
        if temperature_init <= 0.0:
            raise ValueError(
                f"temperature_init must be positive, got {temperature_init}"
            )
        if not 0.0 <= refine_mix <= 1.0:
            raise ValueError(f"refine_mix must be in [0, 1], got {refine_mix}")
        if attn_mlp_ratio <= 0.0:
            raise ValueError(f"attn_mlp_ratio must be positive, got {attn_mlp_ratio}")
        if residual_scale_init < 0.0:
            raise ValueError(
                "residual_scale_init must be non-negative, got "
                f"{residual_scale_init}"
            )
        if bag_repr_source not in {"weighted_sum", "cls", "cls_plus_weighted"}:
            raise ValueError(
                "bag_repr_source must be one of "
                "{'weighted_sum', 'cls', 'cls_plus_weighted'}, got "
                f"{bag_repr_source!r}"
            )
        if pool_from not in {"inputs", "normed_inputs"}:
            raise ValueError(
                "pool_from must be either 'inputs' or 'normed_inputs', got "
                f"{pool_from!r}"
            )
        if refine_top_k_clusters < 0:
            raise ValueError(
                "refine_top_k_clusters must be non-negative, got "
                f"{refine_top_k_clusters}"
            )
        if cluster_refinement_mode not in {"topk", "soft"}:
            raise ValueError(
                "cluster_refinement_mode must be either 'topk' or 'soft', "
                f"got {cluster_refinement_mode!r}"
            )
        if not 0.0 <= external_query_gate_init <= 1.0:
            raise ValueError(
                "external_query_gate_init must be in [0, 1], got "
                f"{external_query_gate_init}"
            )

        self._input_dim = self.out_dim = int(input_dim)
        self.use_cls_token = True
        self.num_query_tokens = int(num_query_tokens)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.use_layer_norm = bool(use_layer_norm)
        self.pool_from = pool_from
        self.bag_repr_source = bag_repr_source
        self.eps = float(eps)
        self.use_temperature = bool(use_temperature)
        self.refine_top_k_clusters = int(refine_top_k_clusters)
        self.cluster_refinement_mode = cluster_refinement_mode
        self.cluster_loss_coeff = float(cluster_loss_coeff)
        self.cluster_compactness_weight = float(cluster_compactness_weight)
        self.cluster_separation_weight = float(cluster_separation_weight)
        self.cluster_separation_max_sim = float(cluster_separation_max_sim)

        if isinstance(pool_v_proj, bool):
            self.pool_v_mode = "linear" if pool_v_proj else False
        else:
            self.pool_v_mode = pool_v_proj
        if self.pool_v_mode not in {False, "linear", "tie_mha_v"}:
            raise ValueError(
                "pool_v_proj must be False, True, 'linear', or 'tie_mha_v', "
                f"got {pool_v_proj!r}"
            )

        self.pre_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.cluster_cls = nn.Parameter(
            torch.randn(1, self.num_query_tokens, input_dim) * 0.02
        )
        self.log_tau = nn.Parameter(
            torch.tensor(math.log(float(temperature_init))),
            requires_grad=True,
        )

        self.cluster_block = CLSCrossAttentionBlock(
            dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            residual_scale_init=residual_scale_init,
        )
        self.instance_block = CLSCrossAttentionBlock(
            dim=input_dim,
            num_heads=num_heads,
            mlp_ratio=attn_mlp_ratio,
            attn_dropout=attn_dropout,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
            residual_scale_init=residual_scale_init,
        )
        self.refine_query_proj = nn.Sequential(
            nn.LayerNorm(2 * input_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(2 * input_dim, input_dim),
        )
        self.external_query_norm = (
            nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        )
        self.external_query_proj = nn.Linear(input_dim, input_dim)
        # external_query_gate_init now sets the INITIAL magnitude of the external-query
        # conditioning (applied to the proj weights in _reset_parameters), not a separate
        # learned global sigmoid gate. Per-bag gating is supplied by the caller each
        # forward via `external_query_gate` (the query builder's match_gate), so a single
        # learnable projection + a per-bag data gate replace the old proj × global-gate
        # stack. Kept as a constructor arg for config back-compat.
        self._external_query_init_scale = min(max(float(external_query_gate_init), 1e-4), 1.0)
        self.output_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()

        self._reset_parameters()

        if self.pool_v_mode == "linear":
            self.pool_v = nn.Linear(input_dim, input_dim, bias=pool_v_bias)
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()

        if refine_mix_learnable:
            mix = min(max(float(refine_mix), 1e-4), 1.0 - 1e-4)
            self.refine_mix_logit = nn.Parameter(
                torch.tensor(math.log(mix / (1.0 - mix)))
            )
        else:
            self.register_buffer("refine_mix_value", torch.tensor(float(refine_mix)))
            self.register_parameter("refine_mix_logit", None)

        self.out_dropout = nn.Dropout(dropout)
        self.last_cluster_attn: Optional[torch.Tensor] = None
        self.last_attn: Optional[torch.Tensor] = None

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cluster_cls, mean=0.0, std=0.02)
        linear = self.refine_query_proj[-1]
        if isinstance(linear, nn.Linear):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.external_query_proj.weight)
        # Sets the INITIAL magnitude of the prototype conditioning. Note the learned CLS
        # query inits tiny (std 0.02, ‖·‖≈0.32), so scale=0.15 already makes ‖condition‖≈2.4
        # — the prototype term DOMINATES the learned query ~7.5× at init (not a mild
        # refinement). That is deliberate: it guarantees the active path differs from the
        # base path so blending it in (curriculum weight w) actually steers. The proj then
        # trains, so this is only the starting scale; w ramps 0->max and match_gate (per-bag)
        # keep the early influence controlled.
        self.external_query_proj.weight.data.mul_(self._external_query_init_scale)
        nn.init.zeros_(self.external_query_proj.bias)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.out_dim

    def _tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.1, 10.0)

    def _refine_mix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.refine_mix_logit is not None:
            return torch.sigmoid(self.refine_mix_logit).to(device=device, dtype=dtype)
        return self.refine_mix_value.to(device=device, dtype=dtype)

    def _as_bag_gate(
        self,
        external_query_gate: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Per-bag gate on the external-query conditioning, broadcast to [B, 1, 1].

        This replaces the old global learned sigmoid gate: the caller passes the query
        builder's match_gate (max conformer-prototype cosine, per bag), so a bag that
        resembles no active prototype gets ~zero conditioning -> its active path equals
        its base path -> the downstream blend leaves it untouched. Defaults to 1.0 when
        no gate is supplied (external queries with no relevance signal).
        """
        if external_query_gate is None:
            return torch.ones((), device=reference.device, dtype=reference.dtype)
        gate = external_query_gate.to(
            device=reference.device,
            dtype=reference.dtype,
        ).clamp(0.0, 1.0)
        if gate.dim() == 0:
            return gate
        if gate.dim() == 1:  # [B] -> [B, 1, 1]
            return gate.view(-1, 1, 1)
        if gate.dim() == 2:  # [B, 1] -> [B, 1, 1]
            return gate.unsqueeze(-1)
        raise ValueError(
            f"external_query_gate must be scalar, [B] or [B, 1], got {tuple(gate.shape)}"
        )

    def _as_query_weight(
        self,
        external_query_weight: Optional[Union[float, torch.Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if external_query_weight is None:
            return torch.ones((), device=device, dtype=dtype)
        if isinstance(external_query_weight, torch.Tensor):
            weight = external_query_weight.to(device=device, dtype=dtype)
        else:
            weight = torch.tensor(float(external_query_weight), device=device, dtype=dtype)
        return weight.clamp(0.0, 1.0)

    def _match_query_count(self, queries: torch.Tensor) -> torch.Tensor:
        """Match external query count to the configured MIL query count."""
        if queries.size(1) == self.num_query_tokens:
            return queries
        if queries.size(1) == 1:
            return queries.expand(-1, self.num_query_tokens, -1)
        if self.num_query_tokens == 1:
            return queries.mean(dim=1, keepdim=True)
        raise ValueError(
            "external_queries query-token count must equal num_query_tokens "
            "or be 1 for broadcasting, got "
            f"{queries.size(1)} vs {self.num_query_tokens}"
        )

    def _condition_queries(
        self,
        learned_queries: torch.Tensor,
        external_queries: Optional[torch.Tensor],
        external_query_weight: Optional[Union[float, torch.Tensor]],
        external_query_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, bool]:
        weight = self._as_query_weight(
            external_query_weight,
            device=learned_queries.device,
            dtype=learned_queries.dtype,
        )
        if external_queries is None:
            return learned_queries, None, weight.new_zeros(()), False

        queries = external_queries.to(
            device=learned_queries.device,
            dtype=learned_queries.dtype,
        )
        if queries.dim() != 3:
            raise ValueError(
                "external_queries must have shape [B, K, D], got "
                f"{tuple(queries.shape)}"
            )
        if queries.size(0) != learned_queries.size(0) or queries.size(-1) != self._input_dim:
            raise ValueError(
                "external_queries must be [B, K, D] with "
                f"B={learned_queries.size(0)}, D={self._input_dim}, got "
                f"{tuple(queries.shape)}"
            )

        queries = self._match_query_count(queries)
        condition = self.external_query_proj(self.external_query_norm(queries))
        gate = self._as_bag_gate(external_query_gate, learned_queries)
        conditioned = learned_queries + gate * condition
        return learned_queries, conditioned, weight, True

    def _project_for_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_v_mode == "linear":
            return self.pool_v(x)
        if self.pool_v_mode == "tie_mha_v":
            D = self._input_dim
            instance_attn = self.instance_block.attn
            W = instance_attn.in_proj_weight[2 * D : 3 * D]
            b = (
                None
                if instance_attn.in_proj_bias is None
                else instance_attn.in_proj_bias[2 * D : 3 * D]
            )
            return F.linear(x, W, b)
        return x

    def _default_cluster_ids(
        self,
        B: int,
        N: int,
        key_padding_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        cluster_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        if key_padding_mask is not None:
            cluster_ids = cluster_ids.masked_fill(key_padding_mask.bool(), -1)
        return cluster_ids

    def _make_cluster_tokens(
        self,
        h: torch.Tensor,
        cluster_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        valid = cluster_ids >= 0
        if key_padding_mask is not None:
            valid = valid & (~key_padding_mask.bool())

        if valid.any():
            C = int(cluster_ids[valid].max().item()) + 1
        else:
            C = 1

        tokens = h.new_zeros((B, C, D))
        counts = h.new_zeros((B, C))

        for b in range(B):
            valid_b = valid[b]
            if not valid_b.any():
                continue
            ids_b = cluster_ids[b, valid_b].clamp(0, C - 1)
            h_b = h[b, valid_b]
            _det_index_add_(tokens[b], 0, ids_b, h_b)
            _det_index_add_(counts[b], 0, ids_b, h.new_ones(ids_b.shape, dtype=h.dtype))

        cluster_valid = counts > 0
        tokens = tokens / counts.clamp_min(self.eps).unsqueeze(-1)
        return tokens, cluster_valid, counts

    def _cluster_uniform_alpha(
        self,
        beta: torch.Tensor,
        counts: torch.Tensor,
        cluster_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        C = beta.size(1)
        safe_ids = cluster_ids.clamp(0, C - 1)
        beta_i = beta.gather(1, safe_ids)
        count_i = counts.gather(1, safe_ids).clamp_min(1.0)
        alpha = beta_i / count_i
        alpha = alpha.masked_fill(~valid, 0.0)
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _cluster_gate_per_instance(
        self,
        beta: torch.Tensor,
        cluster_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        C = beta.size(1)
        safe_ids = cluster_ids.clamp(0, C - 1)
        gate = beta.gather(1, safe_ids)
        return gate.masked_fill(~valid, 0.0)

    def _normalize_instance_alpha(
        self,
        alpha: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = alpha.masked_fill(~valid, 0.0).clamp_min(0.0)
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _cluster_mass_from_instance_alpha(
        self,
        alpha: torch.Tensor,
        cluster_ids: torch.Tensor,
        cluster_valid: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate instance attention into cluster mass actually used downstream."""
        B, _ = alpha.shape
        C = cluster_valid.size(1)
        safe_ids = cluster_ids.clamp(0, C - 1)
        alpha_valid = alpha.masked_fill(~valid, 0.0)
        mass = alpha.new_zeros((B, C))
        _det_scatter_add_(mass, 1, safe_ids, alpha_valid)
        mass = mass.masked_fill(~cluster_valid, 0.0)
        return mass / mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _selected_instance_mask(
        self,
        beta: torch.Tensor,
        cluster_valid: torch.Tensor,
        cluster_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.refine_top_k_clusters <= 0:
            return valid, torch.full(
                (beta.size(0), 0),
                -1,
                dtype=torch.long,
                device=beta.device,
            )

        B, C = beta.shape
        k = min(self.refine_top_k_clusters, C)
        logits = beta.masked_fill(~cluster_valid, -1.0)
        top_idx = torch.topk(logits, k=k, dim=-1).indices

        selected = torch.zeros_like(valid)
        for b in range(B):
            selected_b = valid[b] & (
                cluster_ids[b].unsqueeze(-1) == top_idx[b]
            ).any(dim=-1)
            if not selected_b.any():
                selected_b = valid[b]
            selected[b] = selected_b

        return selected, top_idx

    def _cluster_preservation_loss(
        self,
        h: torch.Tensor,
        cluster_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.cluster_loss_coeff <= 0.0:
            return h.new_zeros(())

        compact_terms = []
        separation_terms = []
        z = F.normalize(h, p=2, dim=-1)

        for b in range(z.size(0)):
            ids = cluster_ids[b, valid[b]]
            z_b = z[b, valid[b]]
            if z_b.numel() == 0:
                continue
            unique_ids = torch.unique(ids)
            centroids = []
            for cid in unique_ids:
                members = z_b[ids == cid]
                centroid = members.mean(dim=0)
                centroids.append(centroid)
                if members.size(0) > 1:
                    compact_terms.append(
                        (members - centroid).pow(2).sum(dim=-1).mean()
                    )

            if len(centroids) > 1:
                C = F.normalize(torch.stack(centroids, dim=0), p=2, dim=-1)
                sim = C @ C.T
                eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
                off_diag = sim.masked_select(~eye)
                separation_terms.append(
                    F.relu(off_diag - self.cluster_separation_max_sim).pow(2).mean()
                )

        loss = h.new_zeros(())
        if compact_terms:
            loss = (
                loss
                + self.cluster_compactness_weight
                * torch.stack(compact_terms).mean()
            )
        if separation_terms:
            loss = (
                loss
                + self.cluster_separation_weight
                * torch.stack(separation_terms).mean()
            )
        return self.cluster_loss_coeff * loss

    def _compose_bag_repr(
        self,
        *,
        cls_repr: torch.Tensor,
        alpha: torch.Tensor,
        pool_source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_repr = torch.bmm(alpha.unsqueeze(1), pool_source).squeeze(1)
        if self.bag_repr_source == "cls":
            bag_repr = self.output_ln(cls_repr)
        elif self.bag_repr_source == "cls_plus_weighted":
            bag_repr = self.output_ln(cls_repr + weighted_repr)
        else:
            bag_repr = weighted_repr
        return self.out_dropout(bag_repr), weighted_repr

    def _run_attention_path(
        self,
        *,
        h_seq: torch.Tensor,
        cluster_tokens: torch.Tensor,
        cluster_valid: torch.Tensor,
        cluster_counts: torch.Tensor,
        cluster_ids: torch.Tensor,
        valid: torch.Tensor,
        queries: torch.Tensor,
        pool_source: torch.Tensor,
        tau: Optional[torch.Tensor],
    ) -> dict:
        cluster_queries, cluster_attn = self.cluster_block(
            queries,
            cluster_tokens,
            key_padding_mask=~cluster_valid,
            query_temperature=tau,
            return_attn=True,
        )

        beta_per_query = cluster_attn.mean(dim=1)
        beta_per_query = torch.nan_to_num(
            beta_per_query,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        beta_per_query = beta_per_query.masked_fill(~cluster_valid.unsqueeze(1), 0.0)
        beta_per_query = beta_per_query / beta_per_query.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(self.eps)

        beta = beta_per_query.mean(dim=1)
        beta = beta.masked_fill(~cluster_valid, 0.0)
        beta = beta / beta.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        alpha_cluster = self._cluster_uniform_alpha(
            beta,
            cluster_counts,
            cluster_ids,
            valid,
        )
        selected_instance_mask, selected_clusters = self._selected_instance_mask(
            beta,
            cluster_valid,
            cluster_ids,
            valid,
        )
        if self.cluster_refinement_mode == "soft":
            refinement_mask = valid
        else:
            refinement_mask = selected_instance_mask

        cluster_context = torch.einsum(
            "bqc,bcd->bqd",
            beta_per_query,
            cluster_tokens,
        )
        refine_query = self.refine_query_proj(
            torch.cat([cluster_queries, cluster_context], dim=-1)
        )

        instance_query, inst_attn = self.instance_block(
            refine_query,
            h_seq,
            key_padding_mask=~refinement_mask,
            query_temperature=tau,
            return_attn=True,
        )

        B, H, Q, N = inst_attn.shape
        alpha_units = inst_attn.reshape(B, H * Q, N)
        alpha_raw = self._normalize_instance_alpha(
            alpha_units.mean(dim=1),
            refinement_mask,
        )
        if self.cluster_refinement_mode == "soft":
            cluster_gate = self._cluster_gate_per_instance(beta, cluster_ids, valid)
            alpha_refined = self._normalize_instance_alpha(
                alpha_raw * cluster_gate,
                valid,
            )
        else:
            cluster_gate = selected_instance_mask.to(dtype=h_seq.dtype)
            alpha_refined = alpha_raw

        alpha_std = (
            alpha_units.std(dim=1, correction=0)
            if alpha_units.size(1) > 1
            else torch.zeros_like(alpha_refined)
        )
        alpha_std = alpha_std.masked_fill(~valid, 0.0)

        mix = self._refine_mix(device=h_seq.device, dtype=h_seq.dtype)
        alpha = self._normalize_instance_alpha(
            (1.0 - mix) * alpha_cluster + mix * alpha_refined,
            valid,
        )

        cls_repr = instance_query.mean(dim=1)
        bag_repr, weighted_repr = self._compose_bag_repr(
            cls_repr=cls_repr,
            alpha=alpha,
            pool_source=pool_source,
        )

        return {
            "bag_repr": bag_repr,
            "weighted_repr": weighted_repr,
            "alpha": alpha,
            "alpha_std": alpha_std,
            "cluster_alpha": beta,
            "cluster_alpha_per_query": beta_per_query,
            "cluster_valid": cluster_valid,
            "cluster_counts": cluster_counts,
            "cluster_gate": cluster_gate,
            "alpha_cluster_uniform": alpha_cluster,
            "alpha_instance_raw": alpha_raw,
            "alpha_instance_refined": alpha_refined,
            "selected_clusters": selected_clusters,
            "selected_instance_mask": selected_instance_mask,
            "cluster_cls_repr": cluster_queries.mean(dim=1),
            "instance_cls_repr": cls_repr,
            "cluster_attn": cluster_attn,
            "instance_attn": inst_attn,
        }

    def forward(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        external_queries: Optional[torch.Tensor] = None,
        external_query_weight: Optional[Union[float, torch.Tensor]] = None,
        external_query_gate: Optional[torch.Tensor] = None,
        return_entropy: bool = True,
        return_attn: bool = False,
    ):
        squeeze_out = h.dim() == 2
        h_seq = h.unsqueeze(0) if squeeze_out else h
        if h_seq.dim() != 3:
            raise ValueError(
                f"Expected h with shape [B, N, D], got {tuple(h.shape)}"
            )

        B, N, D = h_seq.shape
        if D != self._input_dim:
            raise ValueError(
                f"Expected input dim {self._input_dim}, got {D}"
            )

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=torch.bool, device=h_seq.device)
        else:
            key_padding_mask = key_padding_mask.to(device=h_seq.device).bool()

        if cluster_ids is None:
            cluster_ids = self._default_cluster_ids(B, N, key_padding_mask, h_seq.device)
        else:
            cluster_ids = cluster_ids.to(device=h_seq.device, dtype=torch.long)

        valid = (~key_padding_mask) & (cluster_ids >= 0)
        if not valid.any(dim=1).all():
            raise ValueError(
                "ClusterHierarchicalAttentionAggregator received at least one "
                "bag with no valid instances"
            )

        normed_h = self.pre_ln(h_seq)
        cluster_tokens, cluster_valid, cluster_counts = self._make_cluster_tokens(
            h_seq,
            cluster_ids,
            key_padding_mask,
        )
        pool_input = h_seq if self.pool_from == "inputs" else normed_h
        pool_source = self._project_for_pool(pool_input)

        learned_queries = self.cluster_cls.expand(B, -1, -1).to(
            device=h_seq.device,
            dtype=h_seq.dtype,
        )
        base_queries, conditioned_queries, query_weight, external_queries_used = (
            self._condition_queries(
                learned_queries,
                external_queries,
                external_query_weight,
                external_query_gate,
            )
        )

        tau = self._tau() if self.use_temperature else None
        base_path = self._run_attention_path(
            h_seq=h_seq,
            cluster_tokens=cluster_tokens,
            cluster_valid=cluster_valid,
            cluster_counts=cluster_counts,
            cluster_ids=cluster_ids,
            valid=valid,
            queries=base_queries,
            pool_source=pool_source,
            tau=tau,
        )

        active_path = None
        if conditioned_queries is not None and bool((query_weight > 0.0).any().item()):
            active_path = self._run_attention_path(
                h_seq=h_seq,
                cluster_tokens=cluster_tokens,
                cluster_valid=cluster_valid,
                cluster_counts=cluster_counts,
                cluster_ids=cluster_ids,
                valid=valid,
                queries=conditioned_queries,
                pool_source=pool_source,
                tau=tau,
            )
            alpha = self._normalize_instance_alpha(
                (1.0 - query_weight) * base_path["alpha"]
                + query_weight * active_path["alpha"],
                valid,
            )
            cls_repr = (
                (1.0 - query_weight) * base_path["instance_cls_repr"]
                + query_weight * active_path["instance_cls_repr"]
            )
            bag_repr, weighted_repr = self._compose_bag_repr(
                cls_repr=cls_repr,
                alpha=alpha,
                pool_source=pool_source,
            )
            alpha_std = (
                (1.0 - query_weight) * base_path["alpha_std"]
                + query_weight * active_path["alpha_std"]
            )
            cluster_alpha = (
                (1.0 - query_weight) * base_path["cluster_alpha"]
                + query_weight * active_path["cluster_alpha"]
            )
            alpha_cluster = self._normalize_instance_alpha(
                (1.0 - query_weight) * base_path["alpha_cluster_uniform"]
                + query_weight * active_path["alpha_cluster_uniform"],
                valid,
            )
            alpha_refined = self._normalize_instance_alpha(
                (1.0 - query_weight) * base_path["alpha_instance_refined"]
                + query_weight * active_path["alpha_instance_refined"],
                valid,
            )
            alpha_raw = self._normalize_instance_alpha(
                (1.0 - query_weight) * base_path["alpha_instance_raw"]
                + query_weight * active_path["alpha_instance_raw"],
                valid,
            )
            selected_clusters = active_path["selected_clusters"]
            cluster_gate = (
                (1.0 - query_weight) * base_path["cluster_gate"]
                + query_weight * active_path["cluster_gate"]
            )
            cluster_cls_repr = (
                (1.0 - query_weight) * base_path["cluster_cls_repr"]
                + query_weight * active_path["cluster_cls_repr"]
            )
        else:
            alpha = base_path["alpha"]
            alpha_std = base_path["alpha_std"]
            cluster_alpha = base_path["cluster_alpha"]
            alpha_cluster = base_path["alpha_cluster_uniform"]
            alpha_refined = base_path["alpha_instance_refined"]
            alpha_raw = base_path["alpha_instance_raw"]
            selected_clusters = base_path["selected_clusters"]
            cluster_gate = base_path["cluster_gate"]
            cluster_cls_repr = base_path["cluster_cls_repr"]
            cls_repr = base_path["instance_cls_repr"]
            weighted_repr = base_path["weighted_repr"]
            bag_repr = base_path["bag_repr"]

        cluster_alpha_final = self._cluster_mass_from_instance_alpha(
            alpha,
            cluster_ids,
            cluster_valid,
            valid,
        )
        cluster_loss = self._cluster_preservation_loss(normed_h, cluster_ids, valid)
        self.last_cluster_attn = (
            active_path["cluster_attn"].detach()
            if active_path is not None
            else base_path["cluster_attn"].detach()
        )
        self.last_attn = (
            active_path["instance_attn"].detach()
            if active_path is not None
            else base_path["instance_attn"].detach()
        )

        extras = {
            "alpha": alpha,
            "alpha_final": alpha,
            "alpha_std": alpha_std,
            "cluster_alpha": cluster_alpha,
            "cluster_alpha_final": cluster_alpha_final,
            "cluster_valid": cluster_valid,
            "cluster_counts": cluster_counts,
            "cluster_gate": cluster_gate,
            "alpha_cluster_uniform": alpha_cluster,
            "alpha_instance_raw": alpha_raw,
            "alpha_instance_refined": alpha_refined,
            "selected_clusters": selected_clusters,
            "cluster_refinement_mode": self.cluster_refinement_mode,
            "cluster_cls_repr": cluster_cls_repr,
            "instance_cls_repr": cls_repr,
            "weighted_repr": weighted_repr,
            "cluster_preservation_loss": cluster_loss,
            "reg_loss": cluster_loss,
            "external_queries_used": external_queries_used,
            "external_query_weight": query_weight.detach(),
            "external_query_gate": (
                external_query_gate.detach()
                if isinstance(external_query_gate, torch.Tensor)
                else torch.ones((), device=h_seq.device, dtype=h_seq.dtype)
            ),
        }
        if active_path is not None:
            extras.update(
                {
                    "alpha_base": base_path["alpha"],
                    "alpha_active": active_path["alpha"],
                    "cluster_alpha_base": base_path["cluster_alpha"],
                    "cluster_alpha_active": active_path["cluster_alpha"],
                    "alpha_instance_refined_base": base_path[
                        "alpha_instance_refined"
                    ],
                    "alpha_instance_refined_active": active_path[
                        "alpha_instance_refined"
                    ],
                }
            )

        if return_entropy:
            alpha_safe = alpha.clamp_min(self.eps)
            extras["entropy"] = -(alpha_safe * alpha_safe.log()).sum(dim=-1)

        if return_attn:
            attn = base_path["instance_attn"].mean(dim=2, keepdim=True)
            cluster_attn = base_path["cluster_attn"]
            if active_path is not None:
                attn = (
                    (1.0 - query_weight) * attn
                    + query_weight * active_path["instance_attn"].mean(
                        dim=2,
                        keepdim=True,
                    )
                )
                cluster_attn = (
                    (1.0 - query_weight) * cluster_attn
                    + query_weight * active_path["cluster_attn"]
                )
                extras["attn_base"] = base_path["instance_attn"].mean(
                    dim=2,
                    keepdim=True,
                )
                extras["attn_active"] = active_path["instance_attn"].mean(
                    dim=2,
                    keepdim=True,
                )
            extras["attn"] = attn
            extras["cluster_attn"] = cluster_attn

        if squeeze_out:
            bag_repr = bag_repr.squeeze(0)
            for key in (
                "alpha",
                "alpha_final",
                "alpha_base",
                "alpha_active",
                "alpha_std",
                "cluster_alpha",
                "cluster_alpha_final",
                "cluster_alpha_base",
                "cluster_alpha_active",
                "cluster_valid",
                "cluster_counts",
                "cluster_gate",
                "alpha_cluster_uniform",
                "alpha_instance_raw",
                "alpha_instance_refined",
                "alpha_instance_refined_base",
                "alpha_instance_refined_active",
                "selected_clusters",
                "cluster_cls_repr",
                "instance_cls_repr",
                "weighted_repr",
                "entropy",
            ):
                value = extras.get(key)
                if isinstance(value, torch.Tensor) and value.dim() > 0:
                    extras[key] = value.squeeze(0)

        return bag_repr, extras
