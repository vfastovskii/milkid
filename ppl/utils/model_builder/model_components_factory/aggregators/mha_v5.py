import math
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint



class MultiHeadAttentionAggregatorV5(nn.Module):
    r"""
    V5 upgrades over V4 (while preserving the same forward I/O contract):
      1) Head-diversity penalty (returned in extras as "head_div_penalty" and "reg_loss")
      2) Instance scoring branch + fusion with attention to produce final alpha
      3) 2-step CLS attention:
           Step1 CLS -> alpha1 (+ preliminary summary)
           Step2 refined CLS (and optionally reweighted tokens) -> alpha2 used for pooling

    Inputs
    ------
    h : [N, D] or [B, N, D]
    key_padding_mask : Optional[BoolTensor] with shape [B, N] where True marks PAD

    Outputs
    -------
    bag_repr : [D] or [B, D]
    extras : dict with at least:
        "alpha":      [N] or [B, N]
        "alpha_std":  [N] or [B, N]
        "entropy":    scalar or [B] (if requested)
        "attn":       attention tensor (if requested)
        "alpha_topk": selected weights (if topk enabled)
        "alpha_topk_idx": indices (if topk enabled)

      plus V5 additions (safe to ignore in your pipeline):
        "alpha_step1", "alpha_attn_step2", "alpha_score", "head_div_penalty", "reg_loss"
    """

    def __init__(
        self,
        input_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.00,
        attn_dropout: float = 0.0,
        use_layer_norm: bool = True,
        std_correction: float = 0.0,
        use_checkpoint: bool = True,
        eps: float = 1e-8,
        prune_below: float = 0.0,
        pre_layer_norm: bool = True,
        use_cls_token: bool = True,
        num_cls_tokens: int = 2,
        use_weighted_sum: bool = True,

        # --- pooling config ---
        pool_from: Literal["attn_out", "inputs", "normed_inputs"] = "normed_inputs",
        pool_v_proj: Union[bool, Literal[False, "linear", "tie_mha_v"]] = "tie_mha_v",
        pool_v_bias: bool = False,
        residual_pooling: bool = False,
        residual_pool_from: Literal["same", "inputs", "normed_inputs"] = "same",
        residual_mix_learnable: bool = True,

        # --- top-k pooling knobs ---
        topk_n: int = 0,
        topk_strategy: Literal["renorm", "mean", "sum", "argmax"] = "renorm",

        # --- temperature on Q / CLS Q ---
        use_temperature: bool = True,
        temperature_init: float = 0.2,

        # --- out-proj init ---
        out_proj_init: Literal["zero", "tiny"] = "tiny",

        # --- multi-CLS mixer ---
        multi_cls_mixer: bool = True,

        # =========================
        # V5 NEW KNOBS
        # =========================
        two_step: bool = True,
        token_reweight_step2: bool = True,
        token_reweight_init: float = 0.0,  # gamma (learnable scalar)
        refine_query_mode: Literal["add_summary", "replace_with_summary"] = "replace_with_summary",

        use_instance_scorer: bool = True,
        instance_scorer_hidden_dim: int = 256,
        score_dropout: float = 0.0,
        score_scale_init: float = 0.1,              # lambda on scorer logits in fusion
        score_scale_learnable: bool = True,
        score_scale_nonnegative: bool = True,
        fuse_mode: Literal["logit_add", "convex"] = "convex",
        fuse_convex_init: float = 0.5,              # used if fuse_mode="convex"

        head_diversity_coeff: float = 3e-2,          # set >0 to activate regularization
    ):
        super().__init__()

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if input_dim % num_heads != 0:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by num_heads={num_heads}"
            )
        if temperature_init <= 0:
            raise ValueError(
                f"temperature_init must be positive, got {temperature_init}"
            )
        if topk_n < 0:
            raise ValueError(f"topk_n must be non-negative, got {topk_n}")
        if fuse_mode == "convex" and not 0.0 < fuse_convex_init < 1.0:
            raise ValueError(
                f"fuse_convex_init must be in (0, 1), got {fuse_convex_init}"
            )
        self._input_dim = self.out_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.head_dim = input_dim // num_heads
        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.use_layernorm = bool(use_layer_norm)
        self.std_correction = float(std_correction)
        self.use_checkpoint = bool(use_checkpoint)
        self.eps = float(eps)
        self.pre_layer_norm = bool(pre_layer_norm)
        self.prune_below = float(prune_below)

        # ---- CLS config ----
        if num_cls_tokens > 0:
            self.num_cls_tokens = int(num_cls_tokens)
            self.use_cls_token = True
        else:
            self.use_cls_token = bool(use_cls_token)
            self.num_cls_tokens = 1 if self.use_cls_token else 0

        self.use_weighted_sum = bool(use_weighted_sum)

        # --- pooling knobs
        self.pool_from = pool_from
        if isinstance(pool_v_proj, bool):
            self.pool_v_mode = "linear" if pool_v_proj else False
        else:
            self.pool_v_mode = pool_v_proj

        self.residual_pooling = bool(residual_pooling)
        self.residual_pool_from = residual_pool_from
        self.residual_mix_learnable = bool(residual_mix_learnable)
        if self.residual_mix_learnable:
            self.residual_mix = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5
        else:
            self.register_parameter("residual_mix", None)

        # --- top-k
        self.topk_n = int(topk_n)
        self.topk_strategy = topk_strategy

        # --- MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=self.attn_dropout,
            bias=True,
            batch_first=True,
        )

        # --- norms / dropout
        self.pre_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        self.post_ln = nn.LayerNorm(input_dim) if (use_layer_norm and not pre_layer_norm) else nn.Identity()
        self.post_dropout = nn.Dropout(dropout)

        self.output_projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout * 0.5),
        )

        # --- CLS tokens
        if self.use_cls_token:
            self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, input_dim) * 0.02)
            self.cls_ln = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        else:
            self.register_parameter("cls_tokens", None)
            self.cls_ln = nn.Identity()

        # --- multi-CLS convex mixer over K
        self.multi_cls_mixer = bool(multi_cls_mixer)
        if self.use_cls_token and self.num_cls_tokens > 1 and self.multi_cls_mixer:
            self.cls_mix_logits = nn.Parameter(torch.zeros(self.num_cls_tokens))
        else:
            self.register_parameter("cls_mix_logits", None)

        # --- pool value projection
        if self.pool_v_mode == "linear":
            self.pool_v = nn.Linear(input_dim, input_dim, bias=pool_v_bias)
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()  # for False or "tie_mha_v"

        # --- temperature
        self.use_temperature = bool(use_temperature)
        self.log_tau = nn.Parameter(torch.tensor(math.log(float(temperature_init))), requires_grad=True)

        # --- init mode
        self.out_proj_init = out_proj_init
        self._init_weights()

        # =========================
        # V5: 2-step attention
        # =========================
        self.two_step = bool(two_step)
        self.token_reweight_step2 = bool(token_reweight_step2)
        self.refine_query_mode = refine_query_mode

        if self.two_step:
            self.refine_q = nn.Linear(input_dim, input_dim, bias=True)
            nn.init.xavier_uniform_(self.refine_q.weight)
            nn.init.zeros_(self.refine_q.bias)

            if self.token_reweight_step2:
                self.logit_gamma = nn.Parameter(torch.tensor(float(token_reweight_init)))
            else:
                self.register_parameter("logit_gamma", None)
        else:
            self.refine_q = nn.Identity()
            self.register_parameter("logit_gamma", None)

        # =========================
        # V5: instance scoring branch
        # =========================
        self.use_instance_scorer = bool(use_instance_scorer)
        self.score_scale_nonnegative = bool(score_scale_nonnegative)
        self.score_scale_is_raw = False
        if self.use_instance_scorer:
            self.instance_scorer = nn.Sequential(
                nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity(),
                nn.Linear(input_dim, int(instance_scorer_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(score_dropout)),
                nn.Linear(int(instance_scorer_hidden_dim), 1),
            )
            if score_scale_learnable:
                scale_init = float(score_scale_init)
                if self.score_scale_nonnegative:
                    scale_init = self._inverse_softplus(scale_init)
                    self.score_scale_is_raw = True
                self.score_scale = nn.Parameter(torch.tensor(scale_init))
            else:
                self.register_buffer("score_scale", torch.tensor(float(score_scale_init)))
            self.fuse_mode = fuse_mode

            if self.fuse_mode == "convex":
                # sigmoid -> [0,1] mixing weight
                self.fuse_mix_logit = nn.Parameter(torch.tensor(math.log(fuse_convex_init / (1 - fuse_convex_init))))
            else:
                self.register_parameter("fuse_mix_logit", None)
        else:
            self.instance_scorer = None
            self.register_parameter("score_scale", None)
            self.fuse_mode = "logit_add"
            self.register_parameter("fuse_mix_logit", None)

        # =========================
        # V5: head diversity penalty
        # =========================
        self.head_diversity_coeff = float(head_diversity_coeff)

        # Debug/vis
        self.last_attn = None  # Step2 attn in CLS mode, or self-attn mode
        self.last_attn_step1 = None
        self.last_reg_loss = None

    # --- helpers ---------------------------------------------------------------

    @staticmethod
    def _inverse_softplus(value: float) -> float:
        value = max(float(value), 1e-8)
        if value > 20.0:
            return value
        return math.log(math.expm1(value))

    def _tau(self):
        return self.log_tau.exp().clamp(0.1, 10.0)

    def _score_scale_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.score_scale is None:
            return torch.zeros((), device=device, dtype=dtype)
        scale = self.score_scale.to(device=device, dtype=dtype)
        if self.score_scale_nonnegative:
            if self.score_scale_is_raw:
                return F.softplus(scale)
            return scale.clamp_min(0.0)
        return scale

    def _resolve_cls_queries(
        self,
        B: int,
        D: int,
        h_seq: torch.Tensor,
        external_queries: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, bool]:
        """Return CLS queries, optionally replacing learned CLS tokens."""
        if external_queries is None:
            queries = self.cls_ln(self.cls_tokens).expand(B, -1, -1)
            external_queries_used = False
        else:
            if external_queries.dim() != 3:
                raise ValueError(
                    "external_queries must have shape [B, K, D], got "
                    f"{tuple(external_queries.shape)}"
                )
            if external_queries.size(0) != B or external_queries.size(-1) != D:
                raise ValueError(
                    "external_queries must have shape [B, K, D] with "
                    f"B={B}, D={D}, got {tuple(external_queries.shape)}"
                )
            queries = external_queries.to(device=h_seq.device, dtype=h_seq.dtype)
            queries = self.cls_ln(queries)
            external_queries_used = True

        if self.use_temperature:
            queries = queries / self._tau()

        return queries, external_queries_used

    def _mix_query_attention(
        self,
        alpha_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse per-query attention maps into one bag attention vector."""
        query_count = alpha_q.size(1)
        if query_count > 1:
            if (
                query_count == self.num_cls_tokens
                and self.multi_cls_mixer
                and self.cls_mix_logits is not None
            ):
                w = torch.softmax(self.cls_mix_logits, dim=0)
                alpha = (alpha_q * w.view(1, -1, 1)).sum(dim=1)
            else:
                alpha = alpha_q.mean(dim=1)
            alpha_std = alpha_q.std(dim=1, correction=self.std_correction)
        else:
            alpha = alpha_q.squeeze(1)
            alpha_std = torch.zeros_like(alpha)

        return alpha, alpha_std

    def _project_for_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_v_mode == "linear":
            return self.pool_v(x)
        elif self.pool_v_mode == "tie_mha_v":
            D = self._input_dim
            W = self.mha.in_proj_weight[2 * D : 3 * D]
            b = None if self.mha.in_proj_bias is None else self.mha.in_proj_bias[2 * D : 3 * D]
            return F.linear(x, W, b)
        else:
            return x

    def _init_weights(self):
        if hasattr(self.mha, "out_proj") and hasattr(self.mha.out_proj, "weight"):
            if self.out_proj_init == "zero":
                nn.init.zeros_(self.mha.out_proj.weight)
            else:
                nn.init.xavier_uniform_(self.mha.out_proj.weight, gain=1e-3)
            if self.mha.out_proj.bias is not None:
                nn.init.zeros_(self.mha.out_proj.bias)

        if hasattr(self.output_projection[0], "weight"):
            nn.init.xavier_uniform_(self.output_projection[0].weight, gain=0.02)
            if self.output_projection[0].bias is not None:
                nn.init.constant_(self.output_projection[0].bias, 0)

    def _topk_pool(self, alpha: torch.Tensor, pool_source: torch.Tensor) -> torch.Tensor:
        squeeze_out = False
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
            pool_source = pool_source.unsqueeze(0)
            squeeze_out = True

        B, N = alpha.shape
        D = pool_source.shape[-1]

        # hard prune + renorm
        if getattr(self, "prune_below", 0.0) > 0.0:
            a = alpha.clone()
            a = a.masked_fill(a < self.prune_below, 0.0)
            s = a.sum(dim=-1, keepdim=True)
            need_fallback = (s <= self.eps).squeeze(-1)
            if need_fallback.any():
                onehot = torch.zeros_like(a)
                argmax_idx = alpha.argmax(dim=-1, keepdim=True)
                onehot.scatter_(1, argmax_idx, 1.0)
                a = torch.where(need_fallback.unsqueeze(-1), onehot, a)
                s = a.sum(dim=-1, keepdim=True)
            alpha = a / s.clamp_min(self.eps)

        if self.topk_n <= 0 or self.topk_n >= N:
            out = torch.bmm(alpha.unsqueeze(1), pool_source).squeeze(1)
            return out.squeeze(0) if squeeze_out else out

        k = max(1, min(self.topk_n, N))
        topv, topi = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)
        idx = topi.unsqueeze(-1).expand(-1, -1, D)
        selected = torch.gather(pool_source, dim=1, index=idx)

        if k == 1:
            out = selected.squeeze(1)
            return out.squeeze(0) if squeeze_out else out

        if self.topk_strategy == "mean":
            out = selected.mean(dim=1)
        elif self.topk_strategy == "sum":
            out = selected.sum(dim=1)
        elif self.topk_strategy == "argmax":
            out = selected[:, 0, :]
        else:  # renorm
            w = topv / topv.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            out = (w.unsqueeze(-1) * selected).sum(dim=1)

        return out.squeeze(0) if squeeze_out else out

    def _compute_head_div_penalty(self, attn_w: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        attn_w: [B, H, K, N] (CLS mode) or [B, H, N, N] (self-attn mode)
        Returns scalar penalty (mean head cosine similarity; lower is better).
        """
        if attn_w is None or attn_w.dim() < 4:
            return torch.tensor(0.0, device=attn_w.device if attn_w is not None else "cpu")

        if attn_w.shape[-2] == attn_w.shape[-1]:
            # self-attn: average over queries -> head distributions over keys: [B,H,N]
            p = attn_w.mean(dim=2)  # [B,H,N]
        else:
            # CLS: average over K queries -> head distributions over instances: [B,H,N]
            p = attn_w.mean(dim=2)  # [B,H,N]

        if key_padding_mask is not None:
            p = p.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        # cosine similarity between heads per batch
        p_norm = p / (p.norm(dim=-1, keepdim=True).clamp_min(self.eps))  # [B,H,N]
        sim = torch.einsum("bhn,bkn->bhk", p_norm, p_norm)  # [B,H,H]
        H = sim.size(-1)
        off = sim - torch.eye(H, device=sim.device, dtype=sim.dtype).unsqueeze(0)
        denom = float(H * (H - 1)) if H > 1 else 1.0
        return off.sum(dim=(1, 2)).mean() / denom

    def _fuse_alpha_with_scores(
        self,
        alpha_attn: torch.Tensor,          # [B,N]
        pool_source_for_scores: torch.Tensor,  # [B,N,D]
        key_padding_mask: Optional[torch.Tensor],
    ):
        """
        Returns:
          alpha_final [B,N],
          alpha_score [B,N] or None
        """
        if not self.use_instance_scorer or self.instance_scorer is None:
            return alpha_attn, None

        # scorer logits s: [B,N]
        s = self.instance_scorer(pool_source_for_scores).squeeze(-1)

        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask, -1e9)

        alpha_score = torch.softmax(s, dim=-1)

        if self.fuse_mode == "convex":
            m = torch.sigmoid(self.fuse_mix_logit)
            alpha = (1.0 - m) * alpha_attn + m * alpha_score
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            return alpha, alpha_score

        # logit_add fusion: alpha = softmax(log(alpha_attn) + lambda*s)
        lam_val = self._score_scale_value(device=s.device, dtype=s.dtype)
        logits = (alpha_attn.clamp_min(self.eps)).log() + lam_val * s
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask, -1e9)
        alpha = torch.softmax(logits, dim=-1)
        return alpha, alpha_score

    # --- forward ---------------------------------------------------------------

    def forward(
        self,
        h: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
        external_queries: Optional[torch.Tensor] = None,
        return_entropy: bool = True,
        return_attn: bool = False,
    ):
        # Ensure batch
        if h.dim() == 2:
            h_seq = h.unsqueeze(0)   # [1,N,D]
        else:
            h_seq = h                # [B,N,D]
        B, N, D = h_seq.shape

        extras_attn = None
        alpha = None
        alpha_std = None
        alpha_topk = None
        alpha_topk_idx = None
        external_queries_used = False

        # for reg
        head_div_pen = torch.tensor(0.0, device=h_seq.device)

        if self.use_cls_token:
            # =========================
            # CLS attention (2-step)
            # =========================
            normed_h = self.pre_ln(h_seq)  # [B,N,D]

            # pooling/scoring source (keep consistent with your current best: tie_mha_v on normed inputs)
            if self.pool_from == "inputs":
                pool_source = self._project_for_pool(h_seq)
            else:
                pool_source = self._project_for_pool(normed_h)

            # -------- Step 1: CLS -> alpha1
            queries1, external_queries_used = self._resolve_cls_queries(
                B=B,
                D=D,
                h_seq=h_seq,
                external_queries=external_queries,
            )
            query_count = queries1.size(1)

            cls_out1, attn_w1 = self.mha(
                query=queries1,
                key=normed_h,
                value=normed_h,
                need_weights=True,
                average_attn_weights=False,
                key_padding_mask=key_padding_mask,
            )  # cls_out1 [B,K,D], attn_w1 [B,H,K,N]
            self.last_attn_step1 = attn_w1.detach()

            alpha_q1 = attn_w1.mean(dim=1)  # [B,K,N]
            if key_padding_mask is not None:
                alpha_q1 = alpha_q1.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            alpha_q1 = alpha_q1 / alpha_q1.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            alpha1, alpha_std = self._mix_query_attention(alpha_q1)

            # preliminary summary from step1
            summary1 = torch.bmm(alpha1.unsqueeze(1), normed_h).squeeze(1)  # [B,D]

            # -------- Step 2: refined CLS -> alpha_attn2 (used for pooling)
            if self.two_step:
                dq = self.refine_q(summary1).unsqueeze(1)  # [B,1,D]
                if self.refine_query_mode == "replace_with_summary":
                    queries2 = dq.expand(-1, query_count, -1)
                else:  # "add_summary"
                    queries2 = queries1 + dq.expand(-1, query_count, -1)

                if self.use_temperature:
                    queries2 = queries2 / self._tau()

                # optional token reweighting using alpha1
                if self.token_reweight_step2 and self.logit_gamma is not None:
                    gamma = torch.sigmoid(self.logit_gamma)  # [0,1]
                    normed_h2 = normed_h * (1.0 + gamma * alpha1.unsqueeze(-1))
                else:
                    normed_h2 = normed_h

                cls_out2, attn_w2 = self.mha(
                    query=queries2,
                    key=normed_h2,
                    value=normed_h2,
                    need_weights=True,
                    average_attn_weights=False,
                    key_padding_mask=key_padding_mask,
                )  # attn_w2 [B,H,K,N]
            else:
                cls_out2, attn_w2 = cls_out1, attn_w1

            self.last_attn = attn_w2.detach()

            alpha_q2 = attn_w2.mean(dim=1)  # [B,K,N]
            if key_padding_mask is not None:
                alpha_q2 = alpha_q2.masked_fill(key_padding_mask.unsqueeze(1), 0.0)
            alpha_q2 = alpha_q2 / alpha_q2.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            alpha_attn2, _ = self._mix_query_attention(alpha_q2)

            # fuse attention with instance scorer -> final alpha
            alpha, alpha_score = self._fuse_alpha_with_scores(alpha_attn2, pool_source, key_padding_mask)

            # record top-k for extras (pre any top-k renorm inside _topk_pool)
            if self.topk_n > 0 and self.topk_n < N:
                k = max(1, min(self.topk_n, N))
                alpha_topk, alpha_topk_idx = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)

            # pooling
            attn_pool = self._topk_pool(alpha, pool_source)  # [B,D] or [D]

            if self.residual_pooling:
                if self.residual_pool_from == "same":
                    res_source = pool_source
                elif self.residual_pool_from == "inputs":
                    res_source = self._project_for_pool(h_seq)
                else:
                    res_source = self._project_for_pool(self.pre_ln(h_seq))
                mean_pool = res_source.mean(dim=1)
                if self.residual_mix_learnable and self.residual_mix is not None:
                    m = torch.sigmoid(self.residual_mix)
                    bag_repr = (1.0 - m) * attn_pool + m * mean_pool
                else:
                    bag_repr = attn_pool + mean_pool
            else:
                bag_repr = attn_pool

            if return_attn:
                # keep prior contract: [B,H,1,N]
                extras_attn = self.last_attn.mean(dim=2, keepdim=True)

            # head diversity penalty (Step2 attention)
            head_div_pen = self._compute_head_div_penalty(attn_w2, key_padding_mask)

            # stash for debugging
            self.last_reg_loss = self.head_diversity_coeff * head_div_pen

            # squeeze for single bag handled later
            alpha_attn_for_extras = alpha_attn2
            alpha_step1_for_extras = alpha1

        else:
            # (unchanged from V4) self-attn path (no CLS)
            # You can extend this similarly later if you want 2-step without CLS.
            if self.use_checkpoint and self.training:
                def block_only(x, mask):
                    normed_h = self.pre_ln(x)
                    q_in = normed_h / self._tau() if self.use_temperature else normed_h
                    out, _ = self.mha(
                        q_in, normed_h, normed_h,
                        need_weights=True, average_attn_weights=False,
                        key_padding_mask=mask,
                    )
                    out = self.post_dropout(out) + x
                    out = self.post_ln(out)
                    out = self.output_projection(out) + out
                    return out

                attn_out = checkpoint.checkpoint(block_only, h_seq, key_padding_mask, use_reentrant=False)
                with torch.no_grad():
                    normed_h = self.pre_ln(h_seq)
                    q_in = normed_h / self._tau() if self.use_temperature else normed_h
                    _, attn_w = self.mha(
                        q_in, normed_h, normed_h,
                        need_weights=True, average_attn_weights=False,
                        key_padding_mask=key_padding_mask,
                    )
            else:
                normed_h = self.pre_ln(h_seq)
                q_in = normed_h / self._tau() if self.use_temperature else normed_h
                attn_out, attn_w = self.mha(
                    q_in, normed_h, normed_h,
                    need_weights=True, average_attn_weights=False,
                    key_padding_mask=key_padding_mask,
                )
                attn_out = self.post_dropout(attn_out) + h_seq
                attn_out = self.post_ln(attn_out)
                attn_out = self.output_projection(attn_out) + attn_out

            self.last_attn = attn_w.detach()  # [B,H,N,N]

            attn_head_mean = attn_w.mean(dim=1)  # [B,N,N]
            if key_padding_mask is not None:
                key_valid_f = (~key_padding_mask).float()
                attn_head_mean = attn_head_mean * key_valid_f.unsqueeze(1)
                query_valid_f = (~key_padding_mask).float()
                sum_over_q = (attn_head_mean * query_valid_f.unsqueeze(-1)).sum(dim=1)
                denom_q = query_valid_f.sum(dim=-1, keepdim=True).clamp_min(self.eps)
                alpha_raw = sum_over_q / denom_q
                alpha_raw = alpha_raw.masked_fill(key_padding_mask, 0.0)
            else:
                alpha_raw = attn_head_mean.mean(dim=1)

            alpha_attn = alpha_raw / alpha_raw.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            alpha_std = (
                attn_head_mean.std(dim=1, correction=self.std_correction)
                if attn_head_mean.size(1) > 1 else torch.zeros_like(alpha_attn)
            )

            if self.use_weighted_sum:
                if self.pool_from == "attn_out":
                    pool_source = attn_out
                elif self.pool_from == "normed_inputs":
                    pool_source = self._project_for_pool(self.pre_ln(h_seq))
                else:
                    pool_source = self._project_for_pool(h_seq)

                # fuse (optional)
                alpha, alpha_score = self._fuse_alpha_with_scores(alpha_attn, pool_source, key_padding_mask)

                if self.topk_n > 0 and self.topk_n < N:
                    k = max(1, min(self.topk_n, N))
                    alpha_topk, alpha_topk_idx = torch.topk(alpha, k, dim=-1, largest=True, sorted=True)

                attn_pool = self._topk_pool(alpha, pool_source)
                if h.dim() == 2:
                    attn_pool = attn_pool.squeeze(0)

                if self.residual_pooling:
                    if self.residual_pool_from == "same":
                        res_source = pool_source
                    elif self.residual_pool_from == "inputs":
                        res_source = self._project_for_pool(h_seq)
                    else:
                        res_source = self._project_for_pool(self.pre_ln(h_seq))
                    mean_pool = res_source.mean(dim=1)
                    if self.residual_mix_learnable and self.residual_mix is not None:
                        m = torch.sigmoid(self.residual_mix)
                        bag_repr = (1.0 - m) * attn_pool + m * mean_pool
                    else:
                        bag_repr = attn_pool + mean_pool
                else:
                    bag_repr = attn_pool
            else:
                alpha = alpha_attn
                bag_repr = attn_out.mean(dim=1)

            if return_attn:
                extras_attn = self.last_attn

            # head diversity on self-attn if enabled
            head_div_pen = self._compute_head_div_penalty(attn_w, key_padding_mask)
            self.last_reg_loss = self.head_diversity_coeff * head_div_pen

            alpha_attn_for_extras = alpha_attn
            alpha_step1_for_extras = None
            alpha_score = alpha_score if "alpha_score" in locals() else None

        # Final safety: mask + renorm
        if key_padding_mask is not None:
            km = key_padding_mask.squeeze(0) if (alpha.dim() == 1 and key_padding_mask.dim() == 2) else key_padding_mask
            alpha = alpha.masked_fill(km, 0.0)
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        a = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        s = a.sum(dim=-1, keepdim=True)
        a = torch.where(s > 0, a / s.clamp_min(self.eps), a)
        a = a.clamp(0.0, 1.0)
        s = a.sum(dim=-1, keepdim=True)
        alpha = torch.where(s > 0, a / s.clamp_min(self.eps), a)

        # squeeze for single-bag input
        if h.dim() == 2:
            bag_repr = bag_repr.squeeze(0)
            alpha = alpha.squeeze(0)
            alpha_std = alpha_std.squeeze(0)
            if alpha_topk is not None:
                alpha_topk = alpha_topk.squeeze(0)
                alpha_topk_idx = alpha_topk_idx.squeeze(0)
            if alpha_step1_for_extras is not None:
                alpha_step1_for_extras = alpha_step1_for_extras.squeeze(0)
            if alpha_attn_for_extras is not None and alpha_attn_for_extras.dim() == 2:
                alpha_attn_for_extras = alpha_attn_for_extras.squeeze(0)
            if alpha_score is not None and alpha_score.dim() == 2:
                alpha_score = alpha_score.squeeze(0)
            head_div_pen = head_div_pen.squeeze() if head_div_pen.numel() == 1 else head_div_pen

        # extras (preserve old keys)
        extras = {"alpha": alpha, "alpha_std": alpha_std}
        extras["external_queries_used"] = external_queries_used

        if return_entropy:
            alpha_safe = alpha.clamp_min(self.eps)
            log_alpha = alpha_safe.log()
            extras["entropy"] = (-(alpha_safe * log_alpha).sum()
                                 if alpha.dim() == 1 else
                                 -(alpha_safe * log_alpha).sum(dim=1))

        if return_attn and extras_attn is not None:
            extras["attn"] = extras_attn

        if alpha_topk is not None:
            extras["alpha_topk"] = alpha_topk
            extras["alpha_topk_idx"] = alpha_topk_idx

        # V5 additions (safe to ignore)
        if alpha_step1_for_extras is not None:
            extras["alpha_step1"] = alpha_step1_for_extras
        if alpha_attn_for_extras is not None:
            extras["alpha_attn_step2"] = alpha_attn_for_extras
        if alpha_score is not None:
            extras["alpha_score"] = alpha_score

        extras["head_div_penalty"] = head_div_pen
        extras["reg_loss"] = self.head_diversity_coeff * head_div_pen  # add this to your total loss

        return bag_repr, extras

    # --- properties & resizing -------------------------------------------------

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
        self._input_dim = int(value)
        self.out_dim = int(value)

        if value % self.num_heads != 0:
            for i in range(self.num_heads, 0, -1):
                if value % i == 0:
                    self.num_heads = i
                    break
        self.head_dim = value // self.num_heads

        self.mha = nn.MultiheadAttention(
            embed_dim=value,
            num_heads=self.num_heads,
            dropout=self.attn_dropout,
            bias=True,
            batch_first=True,
        )

        self.pre_ln = nn.LayerNorm(value) if self.use_layernorm else nn.Identity()
        self.post_ln = nn.LayerNorm(value) if (self.use_layernorm and not self.pre_layer_norm) else nn.Identity()
        self.output_projection = nn.Sequential(
            nn.Linear(value, value),
            nn.Dropout(self.dropout * 0.5),
        )

        if self.use_cls_token:
            self.cls_tokens = nn.Parameter(torch.randn(1, self.num_cls_tokens, value) * 0.02)
            self.cls_ln = nn.LayerNorm(value) if self.use_layernorm else nn.Identity()

        if self.pool_v_mode == "linear":
            self.pool_v = nn.Linear(value, value, bias=getattr(self.pool_v, "bias", None) is not None)
            nn.init.xavier_uniform_(self.pool_v.weight)
            if self.pool_v.bias is not None:
                nn.init.zeros_(self.pool_v.bias)
        else:
            self.pool_v = nn.Identity()

        if self.two_step and isinstance(self.refine_q, nn.Linear):
            self.refine_q = nn.Linear(value, value, bias=True)
            nn.init.xavier_uniform_(self.refine_q.weight)
            nn.init.zeros_(self.refine_q.bias)

        if self.use_instance_scorer and self.instance_scorer is not None:
            # rebuild scorer with same hidden dim
            hidden = self.instance_scorer[1].out_features
            self.instance_scorer = nn.Sequential(
                nn.LayerNorm(value) if self.use_layernorm else nn.Identity(),
                nn.Linear(value, int(hidden)),
                nn.GELU(),
                self.instance_scorer[3],
                nn.Linear(int(hidden), 1),
            )

        self._init_weights()

    # --- describe --------------------------------------------------------------

    def describe(self) -> dict:
        res_mix = float(torch.sigmoid(self.residual_mix).detach().cpu()) if (
            self.residual_mix_learnable and self.residual_mix is not None
        ) else None

        cls_mix = None
        if self.use_cls_token and self.num_cls_tokens > 1 and self.multi_cls_mixer and self.cls_mix_logits is not None:
            w = torch.softmax(self.cls_mix_logits.detach().cpu(), dim=0)
            cls_mix = [float(x) for x in w.tolist()]

        score_scale = None
        if self.use_instance_scorer and self.score_scale is not None:
            score_scale = float(
                self._score_scale_value(
                    device=self.score_scale.device,
                    dtype=self.score_scale.dtype,
                ).detach().cpu()
            )

        gamma = None
        if self.logit_gamma is not None:
            gamma = float(torch.sigmoid(self.logit_gamma).detach().cpu())

        return {
            "class": self.__class__.__name__,
            "input_dim": self.input_dim,
            "output_dim": self.out_dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "use_layernorm": self.use_layernorm,
            "pre_layer_norm": self.pre_layer_norm,
            "use_checkpoint": self.use_checkpoint,
            "use_cls_token": self.use_cls_token,
            "num_cls_tokens": self.num_cls_tokens,
            "multi_cls_mixer": self.multi_cls_mixer,
            "multi_cls_weights": cls_mix,
            "use_weighted_sum": self.use_weighted_sum,
            "residual_pooling": self.residual_pooling,
            "residual_pool_from": self.residual_pool_from,
            "residual_mix_learnable": self.residual_mix_learnable,
            "residual_mix_sigma": res_mix,
            "pool_from": self.pool_from,
            "pool_v_mode": self.pool_v_mode,
            "use_temperature": self.use_temperature,
            "tau": float(self._tau().detach().cpu()) if self.use_temperature else None,
            "out_proj_init": self.out_proj_init,
            "topk_n": self.topk_n,
            "topk_strategy": self.topk_strategy,
            # V5
            "two_step": self.two_step,
            "token_reweight_step2": self.token_reweight_step2,
            "token_reweight_gamma": gamma,
            "refine_query_mode": self.refine_query_mode,
            "use_instance_scorer": self.use_instance_scorer,
            "fuse_mode": getattr(self, "fuse_mode", None),
            "score_scale": score_scale,
            "score_scale_nonnegative": self.score_scale_nonnegative,
            "head_diversity_coeff": self.head_diversity_coeff,
        }
