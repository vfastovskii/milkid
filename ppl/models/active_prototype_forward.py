"""Active-prototype orchestration for MILCore.

These methods were part of MILCore (core.py); they are split into a mixin purely
to keep that file navigable. Bodies and ``self`` are unchanged, so behavior is
identical -- MILCore inherits ActivePrototypeMixin and resolves ``self.<method>``
via the MRO.
"""
from __future__ import annotations

import logging
import math
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from ppl.models.active_prototype_memory import (
    ActivePrototypeQuery,
    DynamicActivePrototypeBank,
    select_active_conformer_candidates,
)

LOGGER = logging.getLogger(__name__)


class ActivePrototypeMixin:
    """Active-conformer prototype memory: config, update, query gating, extras."""

    def _infer_active_prototype_dim(self, cfg: Dict[str, Any]) -> int:
        if cfg.get("dim") is not None:
            return int(cfg["dim"])

        embedder = getattr(self, "embedder", None)
        for attr in ("output_dim", "out_dim", "hidden_dim", "embed_dim"):
            value = getattr(embedder, attr, None)
            if value is not None:
                return int(value)

        aggregator = getattr(self, "aggregator", None)
        for attr in ("input_dim", "_input_dim", "out_dim", "output_dim"):
            value = getattr(aggregator, attr, None)
            if value is not None:
                return int(value)

        raise ValueError(
            "Could not infer active prototype dimension. Set "
            "model.active_prototype_kwargs.dim explicitly."
        )

    def _configure_active_prototypes(self, cfg: Dict[str, Any]) -> None:
        self.active_prototype_enabled = bool(cfg.get("enabled", False))
        self.active_prototype_warmup_epochs = int(cfg.get("warmup_epochs", 5))
        self.active_prototype_query_start_epoch = int(
            cfg.get("query_start_epoch", self.active_prototype_warmup_epochs)
        )
        self.active_prototype_query_ramp_epochs = max(
            1,
            int(cfg.get("query_ramp_epochs", 1)),
        )
        self.active_prototype_query_max_weight = float(
            cfg.get("query_max_weight", 1.0)
        )
        if not 0.0 <= self.active_prototype_query_max_weight <= 1.0:
            raise ValueError(
                "query_max_weight must be in [0, 1], got "
                f"{self.active_prototype_query_max_weight}"
            )
        self.active_prototype_min_active = int(cfg.get("min_active_prototypes", 3))
        self.active_prototype_active_threshold = float(
            cfg.get("active_threshold", 7.0)
        )
        self.active_prototype_top_m_candidates = int(
            cfg.get("top_m_candidates", 3)
        )
        self.active_prototype_candidate_selection = str(
            cfg.get("candidate_selection", "attention")
        ).lower()
        if self.active_prototype_candidate_selection not in {
            "attention",
            "attention_ablation",
            "gradient",
        }:
            raise ValueError(
                "candidate_selection must be 'attention', 'attention_ablation', or "
                f"'gradient', got {self.active_prototype_candidate_selection!r}"
            )
        self.active_prototype_ablation_candidate_pool = int(
            cfg.get(
                "ablation_candidate_pool",
                max(4, 4 * self.active_prototype_top_m_candidates),
            )
        )
        self.active_prototype_ablation_attention_weight = float(
            cfg.get("ablation_attention_weight", 0.5)
        )
        self.active_prototype_ablation_impact_weight = float(
            cfg.get("ablation_impact_weight", 0.5)
        )
        self.active_prototype_ablation_positive_only = bool(
            cfg.get("ablation_positive_only", True)
        )
        self.active_prototype_candidate_alpha_key = str(
            cfg.get("candidate_alpha_key", "alpha")
        )
        max_per_series = cfg.get("max_prototypes_per_series", None)
        self.active_prototype_max_per_series = (
            None if max_per_series is None else int(max_per_series)
        )
        self.active_prototype_use_on_eval = bool(cfg.get("use_on_eval", True))
        self._active_prototype_status_logged_epoch: Optional[int] = None
        # Per-epoch candidate accumulation; the bank is rebuilt (order-invariant)
        # from these at epoch end. Transient — not a buffer, not checkpointed.
        self._active_candidate_buffer: list = []
        self._active_candidate_series: list = []

        if not self.active_prototype_enabled:
            self.active_prototype_bank = None
            self.active_query_builder = None
            return

        dim = self._infer_active_prototype_dim(cfg)
        self.active_prototype_bank = DynamicActivePrototypeBank(
            dim=dim,
            max_prototypes=int(cfg.get("max_prototypes", 64)),
            min_count_to_keep=int(cfg.get("min_count_to_keep", 3)),
            eps=float(cfg.get("eps", 1e-8)),
        )
        self.active_query_builder = ActivePrototypeQuery(
            dim=dim,
            top_m_prototypes=int(
                cfg.get("query_top_m_prototypes", cfg.get("top_m_prototypes", 8))
            ),
            temperature=float(
                cfg.get("query_temperature", cfg.get("temperature", 0.2))
            ),
            prefer_same_series=bool(cfg.get("query_prefer_same_series", True)),
            fallback_to_all_series=bool(cfg.get("query_fallback_to_all_series", True)),
        )
        self.register_buffer(
            "active_prototype_num_updates",
            torch.zeros((), dtype=torch.long),
        )

        LOGGER.info(
            "[MODEL] Active prototype memory enabled: dim=%d max_prototypes=%d "
            "warmup_epochs=%d query_start_epoch=%d query_ramp_epochs=%d "
            "query_max_weight=%.3f active_threshold=%.3f min_active=%d "
            "top_m_candidates=%d candidate_selection=%s "
            "ablation_candidate_pool=%d candidate_alpha_key=%s "
            "max_prototypes_per_series=%s use_on_eval=%s "
            "query_prefer_same_series=%s",
            dim,
            self.active_prototype_bank.max_prototypes,
            self.active_prototype_warmup_epochs,
            self.active_prototype_query_start_epoch,
            self.active_prototype_query_ramp_epochs,
            self.active_prototype_query_max_weight,
            self.active_prototype_active_threshold,
            self.active_prototype_min_active,
            self.active_prototype_top_m_candidates,
            self.active_prototype_candidate_selection,
            self.active_prototype_ablation_candidate_pool,
            self.active_prototype_candidate_alpha_key,
            self.active_prototype_max_per_series,
            self.active_prototype_use_on_eval,
            self.active_query_builder.prefer_same_series,
        )

    def _epoch_ready_for_active_memory(
        self,
        current_epoch: Optional[int],
    ) -> bool:
        if current_epoch is None:
            return False
        return int(current_epoch) >= self.active_prototype_warmup_epochs

    def _active_query_weight(
        self,
        current_epoch: Optional[int],
    ) -> float:
        if current_epoch is None:
            return 0.0
        epoch = int(current_epoch)
        if epoch < self.active_prototype_query_start_epoch:
            return 0.0
        progress = epoch - self.active_prototype_query_start_epoch + 1
        ramp = min(
            1.0,
            progress / float(self.active_prototype_query_ramp_epochs),
        )
        return float(self.active_prototype_query_max_weight * ramp)

    def _effective_active_query_weight(
        self,
        current_epoch: Optional[int],
    ) -> float:
        """Return the active-query weight after training-phase overrides."""
        if bool(getattr(self, "active_query_force_disabled", False)):
            return 0.0

        forced_weight = getattr(self, "active_query_forced_weight", None)
        if forced_weight is not None:
            return float(max(0.0, min(1.0, float(forced_weight))))

        return self._active_query_weight(current_epoch)

    def _should_update_active_prototypes(
        self,
        labels: Optional[torch.Tensor],
        stage: Optional[str],
        current_epoch: Optional[int],
    ) -> bool:
        if not self.active_prototype_enabled:
            return False
        if self.active_prototype_bank is None:
            return False
        if labels is None:
            return False
        if not self.training:
            return False
        if stage is not None and stage != "train":
            return False
        return self._epoch_ready_for_active_memory(current_epoch)

    def _aggregator_supports_external_queries(self, aggregator: nn.Module) -> bool:
        return self._forward_accepts(aggregator, "external_queries")

    def _should_use_active_query(
        self,
        aggregator: nn.Module,
        stage: Optional[str],
        current_epoch: Optional[int],
    ) -> bool:
        if not self.active_prototype_enabled:
            return False
        if self.active_prototype_bank is None or self.active_query_builder is None:
            return False
        if not self._aggregator_supports_external_queries(aggregator):
            return False
        if not getattr(aggregator, "use_cls_token", False):
            return False
        if self.active_prototype_bank.num_active() < self.active_prototype_min_active:
            return False
        if self._effective_active_query_weight(current_epoch) <= 0.0:
            return False

        if stage == "train":
            return True

        if stage in {"val", "test"} or not self.training:
            return self.active_prototype_use_on_eval

        return True

    def _log_active_prototype_forward_status(
        self,
        stage: Optional[str],
        current_epoch: Optional[int],
        needs_bank_update: bool,
        use_active_query_now: bool,
        labels: Optional[torch.Tensor],
        aggregator: nn.Module,
    ) -> None:
        """Emit one train-time status line per epoch before prediction/loss.

        This is intentionally in the forward path rather than only at epoch end:
        if a run crashes early, the log still shows whether the prototype bank
        was enabled, warming up, updating, or query-ready for that epoch.
        """
        if not self.active_prototype_enabled or self.active_prototype_bank is None:
            return
        if stage != "train":
            return

        epoch = -1 if current_epoch is None else int(current_epoch)
        if self._active_prototype_status_logged_epoch == epoch:
            return
        self._active_prototype_status_logged_epoch = epoch

        num_active = self.active_prototype_bank.num_active()
        support = float(
            self.active_prototype_bank.counts[
                self.active_prototype_bank.active_mask
            ].sum().item()
        )
        warmup_done = self._epoch_ready_for_active_memory(current_epoch)
        supports_external = self._aggregator_supports_external_queries(aggregator)
        active_count = 0
        if labels is not None:
            active_count = int(
                (labels.detach() >= self.active_prototype_active_threshold)
                .sum()
                .item()
            )
        query_weight = self._effective_active_query_weight(current_epoch)

        # User-facing announcements on state transitions (via the "milk" narrative
        # logger so they show at INFO). Per-epoch detail stays at DEBUG below.
        run_log = logging.getLogger("milk")
        if warmup_done and not getattr(self, "_active_proto_warmup_announced", False):
            self._active_proto_warmup_announced = True
            run_log.info(
                "Active-prototype memory: warmup complete at epoch %s — bank now "
                "collecting prototypes (queries begin at epoch %s).",
                epoch,
                self.active_prototype_query_start_epoch,
            )
        if use_active_query_now and not getattr(self, "_active_proto_query_announced", False):
            self._active_proto_query_announced = True
            run_log.info(
                "Active-prototype query engaged at epoch %s — mixing prototype context "
                "into attention (query weight %.2f, %d active prototypes).",
                epoch,
                query_weight,
                num_active,
            )

        LOGGER.debug(
            "[ACTIVE_PROTO] epoch=%s warmup_done=%s update_bank=%s "
            "use_query=%s query_weight=%.3f active_prototypes=%d/%d support=%.0f "
            "active_labels_in_first_batch=%d min_required=%d "
            "alpha_key=%s external_query_supported=%s",
            epoch,
            warmup_done,
            needs_bank_update,
            use_active_query_now,
            query_weight,
            num_active,
            self.active_prototype_bank.max_prototypes,
            support,
            active_count,
            self.active_prototype_min_active,
            self.active_prototype_candidate_alpha_key,
            supports_external,
        )

    def _active_prototype_alpha(
        self,
        agg_info: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        alpha_key = self.active_prototype_candidate_alpha_key
        alpha = agg_info.get(alpha_key)
        if alpha is None and alpha_key != "alpha":
            LOGGER.debug(
                "[ACTIVE_PROTO] extras has no %s; falling back to alpha",
                alpha_key,
            )
            alpha = agg_info.get("alpha")
        return alpha

    @staticmethod
    def _normalise_positive_scores(scores: torch.Tensor) -> torch.Tensor:
        scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
        scores = scores.clamp_min(0.0)
        total = scores.sum()
        if float(total.item()) <= 0.0:
            return torch.zeros_like(scores)
        return scores / total.clamp_min(1e-8)

    def _select_gradient_refined_active_candidates(
        self,
        z: torch.Tensor,
        agg_info: Dict[str, Any],
        labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        cluster_ids: Optional[torch.Tensor],
        series_labels: Optional[list[str]],
        aggregator: nn.Module,
        predictor: nn.Module,
    ) -> tuple[torch.Tensor, Optional[list[str]]]:
        """Select active candidates by attention blended with gradient×input saliency.

        For each active bag the prediction gradient w.r.t. every conformer embedding
        is obtained in ONE backward pass on the feedback-free BASE path (no
        external_queries — same anti-self-selection guard as the ablation variant),
        giving d(pred)/d(z_i) for ALL valid conformers. Unlike the ablation variant
        there is no top-attention pre-filter, so a high-impact but low-attention
        conformer can still surface. Per-conformer importance is the gradient×input
        attribution grad_i · z_i (the first-order counterpart of the ablation delta:
        positive = the conformer pushes the prediction up), blended with the
        base-path attention. One forward+backward per active bag replaces 1+pool
        forwards.
        """
        alpha = self._active_prototype_alpha(agg_info)
        if alpha is None:
            return z.new_zeros((0, z.size(-1))), None
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() != 2:
            raise ValueError(
                f"Active prototype alpha must have shape [B, N], got {alpha.shape}"
            )

        B, N, D = z.shape
        y = labels.flatten().to(device=z.device)
        active_rows = y >= self.active_prototype_active_threshold
        if active_rows.sum() == 0:
            return z.new_zeros((0, D)), None

        if key_padding_mask is None:
            valid_mask = torch.ones(B, N, device=z.device, dtype=torch.bool)
        else:
            valid_mask = ~key_padding_mask.to(device=z.device).bool()

        attn_w = self.active_prototype_ablation_attention_weight
        grad_w = self.active_prototype_ablation_impact_weight
        selected: list[torch.Tensor] = []
        selected_series: Optional[list[str]] = [] if series_labels is not None else None
        overlaps: list[float] = []

        was_agg_training = aggregator.training
        was_pred_training = predictor.training
        aggregator.eval()
        predictor.eval()
        try:
            for b_t in torch.nonzero(active_rows, as_tuple=False).flatten():
                b = int(b_t.item())
                valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue

                mask_full = (
                    torch.zeros(1, N, dtype=torch.bool, device=z.device)
                    if key_padding_mask is None
                    else key_padding_mask[b : b + 1].to(device=z.device).bool()
                )
                cluster_b = None if cluster_ids is None else cluster_ids[b : b + 1]

                # d(pred)/d(z_i) for every conformer, one backward on the base path.
                z_b = z[b : b + 1].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    repr_b, _ = self._call_aggregator(
                        aggregator,
                        z_b,
                        key_padding_mask=mask_full,
                        cluster_ids=cluster_b,
                        return_attn=False,
                    )
                    pred_b = predictor(repr_b).flatten()[0]
                    grad_b = torch.autograd.grad(pred_b, z_b)[0][0]  # [N, D]

                # gradient×input attribution: positive = conformer inflates the pred.
                contrib = (grad_b * z_b.detach()[0]).sum(dim=-1)  # [N]
                if self.active_prototype_ablation_positive_only:
                    impact = contrib.clamp_min(0.0)
                else:
                    impact = contrib.abs()

                alpha_valid = alpha[b, valid_idx].to(device=z.device)
                impact_valid = impact[valid_idx]
                alpha_norm = self._normalise_positive_scores(alpha_valid)
                impact_norm = self._normalise_positive_scores(impact_valid)
                score = attn_w * alpha_norm + grad_w * impact_norm
                if float(score.sum().item()) <= 0.0:
                    score = alpha_norm
                if float(score.sum().item()) <= 0.0:
                    score = torch.ones_like(score) / max(1, score.numel())

                keep_k = min(
                    self.active_prototype_top_m_candidates, int(valid_idx.numel())
                )
                keep_local = torch.topk(
                    score, k=keep_k, largest=True, sorted=False
                ).indices
                keep_idx = valid_idx[keep_local]
                selected.append(z[b, keep_idx].detach())
                if selected_series is not None:
                    selected_series.extend(
                        [str(series_labels[b])] * int(keep_idx.numel())
                    )

                # Instrument: overlap with pure-attention top-k — does the gradient
                # term actually change the selection, or just reproduce attention?
                attn_local = torch.topk(
                    alpha_norm, k=keep_k, largest=True, sorted=False
                ).indices
                overlaps.append(
                    len(set(keep_local.tolist()) & set(attn_local.tolist()))
                    / max(1, keep_k)
                )
        finally:
            aggregator.train(was_agg_training)
            predictor.train(was_pred_training)

        if overlaps:
            LOGGER.debug(
                "[ACTIVE_PROTO] gradient selection: mean top-k overlap with "
                "attention-only = %.2f over %d active bags (1.0 = gradient changed "
                "nothing; low = it adds independent signal)",
                sum(overlaps) / len(overlaps),
                len(overlaps),
            )
        if not selected:
            return z.new_zeros((0, D)), selected_series
        return torch.cat(selected, dim=0), selected_series

    @torch.no_grad()
    def _select_ablation_refined_active_candidates(
        self,
        z: torch.Tensor,
        agg_info: Dict[str, Any],
        labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        cluster_ids: Optional[torch.Tensor],
        series_labels: Optional[list[str]],
        aggregator: nn.Module,
        predictor: nn.Module,
    ) -> tuple[torch.Tensor, Optional[list[str]]]:
        """Select active candidates using attention plus prediction ablation.

        The bank is refined from the base learned-query path, not the
        active-conditioned path, to avoid a feedback loop where the prototype
        bank selects candidates created by itself.
        """
        alpha = self._active_prototype_alpha(agg_info)
        if alpha is None:
            return z.new_zeros((0, z.size(-1))), None
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(0)
        if alpha.dim() != 2:
            raise ValueError(
                f"Active prototype alpha must have shape [B, N], got {alpha.shape}"
            )

        B, N, D = z.shape
        y = labels.flatten().to(device=z.device)
        active_rows = y >= self.active_prototype_active_threshold
        if active_rows.sum() == 0:
            return z.new_zeros((0, D)), None

        if key_padding_mask is None:
            valid_mask = torch.ones(B, N, device=z.device, dtype=torch.bool)
        else:
            valid_mask = ~key_padding_mask.to(device=z.device).bool()

        selected: list[torch.Tensor] = []
        selected_series: Optional[list[str]] = [] if series_labels is not None else None
        pool_limit = max(
            self.active_prototype_top_m_candidates,
            self.active_prototype_ablation_candidate_pool,
        )
        attn_w = self.active_prototype_ablation_attention_weight
        impact_w = self.active_prototype_ablation_impact_weight

        was_agg_training = aggregator.training
        was_pred_training = predictor.training
        aggregator.eval()
        predictor.eval()
        try:
            for b_t in torch.nonzero(active_rows, as_tuple=False).flatten():
                b = int(b_t.item())
                valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue

                alpha_valid = alpha[b, valid_idx].to(device=z.device)
                pool_k = min(int(pool_limit), int(valid_idx.numel()))
                if pool_k <= 0:
                    continue
                if pool_k < valid_idx.numel():
                    pool_local = torch.topk(
                        alpha_valid,
                        k=pool_k,
                        largest=True,
                        sorted=False,
                    ).indices
                else:
                    pool_local = torch.arange(
                        valid_idx.numel(),
                        device=z.device,
                    )
                pool_idx = valid_idx[pool_local]
                alpha_pool = alpha_valid[pool_local]

                mask_full = (
                    torch.zeros(1, N, dtype=torch.bool, device=z.device)
                    if key_padding_mask is None
                    else key_padding_mask[b : b + 1].to(device=z.device).bool()
                )
                cluster_b = None if cluster_ids is None else cluster_ids[b : b + 1]
                full_repr, _ = self._call_aggregator(
                    aggregator,
                    z[b : b + 1],
                    key_padding_mask=mask_full,
                    cluster_ids=cluster_b,
                    return_attn=False,
                )
                full_pred = predictor(full_repr).flatten()[0]

                impacts = []
                for idx_t in pool_idx:
                    if valid_idx.numel() <= 1:
                        impacts.append(z.new_zeros(()))
                        continue
                    mask_b = mask_full.clone()
                    mask_b[0, int(idx_t.item())] = True
                    ablated_repr, _ = self._call_aggregator(
                        aggregator,
                        z[b : b + 1],
                        key_padding_mask=mask_b,
                        cluster_ids=cluster_b,
                        return_attn=False,
                    )
                    ablated_pred = predictor(ablated_repr).flatten()[0]
                    delta = full_pred - ablated_pred
                    if self.active_prototype_ablation_positive_only:
                        impact = delta.clamp_min(0.0)
                    else:
                        impact = delta.abs()
                    impacts.append(impact)

                impact_pool = torch.stack(impacts)
                alpha_norm = self._normalise_positive_scores(alpha_pool)
                impact_norm = self._normalise_positive_scores(impact_pool)
                score = attn_w * alpha_norm + impact_w * impact_norm
                if float(score.sum().item()) <= 0.0:
                    score = alpha_norm
                if float(score.sum().item()) <= 0.0:
                    score = torch.ones_like(score) / max(1, score.numel())

                keep_k = min(
                    self.active_prototype_top_m_candidates,
                    int(pool_idx.numel()),
                )
                keep_local = torch.topk(
                    score,
                    k=keep_k,
                    largest=True,
                    sorted=False,
                ).indices
                keep_idx = pool_idx[keep_local]
                selected.append(z[b, keep_idx])
                if selected_series is not None:
                    selected_series.extend(
                        [str(series_labels[b])] * int(keep_idx.numel())
                    )
        finally:
            aggregator.train(was_agg_training)
            predictor.train(was_pred_training)

        if not selected:
            return z.new_zeros((0, D)), selected_series
        return torch.cat(selected, dim=0), selected_series

    def _update_active_prototypes_from_batch(
        self,
        z: torch.Tensor,
        agg_info: Dict[str, Any],
        labels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        cluster_ids: Optional[torch.Tensor] = None,
        series_labels: Optional[list[str]] = None,
        aggregator: Optional[nn.Module] = None,
        predictor: Optional[nn.Module] = None,
    ) -> int:
        if self.active_prototype_bank is None:
            return 0

        alpha = self._active_prototype_alpha(agg_info)
        if alpha is None:
            LOGGER.info(
                "[ACTIVE_PROTO] Skipping update because extras has neither %s nor alpha",
                self.active_prototype_candidate_alpha_key,
            )
            return 0

        with torch.no_grad():
            num_before = self.active_prototype_bank.num_active()
            support_before = float(
                self.active_prototype_bank.counts[
                    self.active_prototype_bank.active_mask
                ].sum().item()
            )
            sel = self.active_prototype_candidate_selection
            if (
                sel == "gradient"
                and aggregator is not None
                and predictor is not None
            ):
                candidates, candidate_series = (
                    self._select_gradient_refined_active_candidates(
                        z=z.detach(),
                        agg_info=agg_info,
                        labels=labels.detach(),
                        key_padding_mask=key_padding_mask,
                        cluster_ids=cluster_ids,
                        series_labels=series_labels,
                        aggregator=aggregator,
                        predictor=predictor,
                    )
                )
            elif (
                sel == "attention_ablation"
                and aggregator is not None
                and predictor is not None
            ):
                candidates, candidate_series = (
                    self._select_ablation_refined_active_candidates(
                        z=z.detach(),
                        agg_info=agg_info,
                        labels=labels.detach(),
                        key_padding_mask=key_padding_mask,
                        cluster_ids=cluster_ids,
                        series_labels=series_labels,
                        aggregator=aggregator,
                        predictor=predictor,
                    )
                )
            else:
                candidates, candidate_series = select_active_conformer_candidates(
                    z=z.detach(),
                    alpha=alpha.detach(),
                    y_pIC50=labels.detach(),
                    key_padding_mask=key_padding_mask,
                    series_labels=series_labels,
                    active_threshold=self.active_prototype_active_threshold,
                    top_m=self.active_prototype_top_m_candidates,
                    return_series=True,
                )
            # Accumulate this batch's candidates; the bank is rebuilt once per
            # epoch (order-invariant) in rebuild_active_prototypes() at epoch end.
            if candidates.size(0) > 0:
                self._active_candidate_buffer.append(candidates.detach().cpu())
                if candidate_series is not None:
                    self._active_candidate_series.extend(
                        str(s) for s in candidate_series
                    )
                LOGGER.debug(
                    "[ACTIVE_PROTO] buffered candidates=%d (epoch total=%d); "
                    "bank=%d active support=%.0f, rebuilt at epoch end",
                    int(candidates.size(0)),
                    sum(int(c.size(0)) for c in self._active_candidate_buffer),
                    num_before,
                    support_before,
                )

        return int(candidates.size(0))

    @torch.no_grad()
    def rebuild_active_prototypes(self) -> None:
        """Rebuild the bank from this epoch's accumulated active candidates.

        Order-invariant per-epoch rebuild (deterministic per-series clustering).
        Train-only — call at epoch end. Clears the candidate buffer afterwards.
        """
        bank = getattr(self, "active_prototype_bank", None)
        buf = getattr(self, "_active_candidate_buffer", None) or []
        if bank is None or not buf:
            self._reset_active_candidate_buffer()
            return
        candidates = torch.cat(buf, dim=0)
        series = getattr(self, "_active_candidate_series", None) or []
        candidate_series = series if len(series) == int(candidates.size(0)) else None
        bank.rebuild_from_candidates(
            candidates,
            candidate_series=candidate_series,
            max_per_series=self.active_prototype_max_per_series,
        )
        if hasattr(self, "active_prototype_num_updates"):
            self.active_prototype_num_updates.add_(1)
        LOGGER.debug(
            "[ACTIVE_PROTO] epoch rebuild from %d candidates -> %d active prototypes",
            int(candidates.size(0)),
            bank.num_active(),
        )
        self._reset_active_candidate_buffer()

    def _reset_active_candidate_buffer(self) -> None:
        self._active_candidate_buffer = []
        self._active_candidate_series = []

    def _add_active_prototype_extras(
        self,
        agg_info: Dict[str, Any],
        update_candidates: int = 0,
    ) -> Dict[str, Any]:
        if not self.active_prototype_enabled or self.active_prototype_bank is None:
            return agg_info

        num_active = self.active_prototype_bank.num_active()
        support = float(
            self.active_prototype_bank.counts[
                self.active_prototype_bank.active_mask
            ].sum().item()
        )
        agg_info.setdefault("active_query_used", False)
        agg_info.setdefault("active_query_weight", 0.0)
        agg_info.setdefault("num_active_prototypes", num_active)
        agg_info["active_prototype_num_active"] = num_active
        agg_info["active_prototype_total_support"] = support
        agg_info["active_prototype_ready"] = (
            num_active >= self.active_prototype_min_active
        )
        agg_info["active_prototype_update_candidates"] = int(update_candidates)
        agg_info["active_prototype_min_active"] = self.active_prototype_min_active
        agg_info["active_prototype_warmup_epochs"] = (
            self.active_prototype_warmup_epochs
        )
        agg_info["active_prototype_query_start_epoch"] = (
            self.active_prototype_query_start_epoch
        )
        agg_info["active_prototype_alpha_key"] = (
            self.active_prototype_candidate_alpha_key
        )
        if hasattr(self.active_prototype_bank, "series_histogram"):
            agg_info["active_prototype_series_histogram"] = (
                self.active_prototype_bank.series_histogram()
            )
        if hasattr(self.active_prototype_bank, "series_seen_histogram"):
            agg_info["active_prototype_seen_series_histogram"] = (
                self.active_prototype_bank.series_seen_histogram()
            )
        if hasattr(self.active_prototype_bank, "missing_seen_series"):
            agg_info["active_prototype_missing_seen_series"] = (
                self.active_prototype_bank.missing_seen_series()
            )
        return agg_info
