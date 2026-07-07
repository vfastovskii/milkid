from __future__ import annotations

import logging
import inspect
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Callable, Optional

from ppl.models.active_prototype_memory import (
    ActivePrototypeQuery,
    DynamicActivePrototypeBank,
    select_active_conformer_candidates,
)

LOGGER = logging.getLogger(__name__)


# Core MIL network
class MILCore(nn.Module):
    """Flexible MIL pipeline that supports different component structures."""

    def __init__(
        self,
        components: List[nn.Module],
        component_names: List[str],
        forward_func: Optional[Callable] = None,
        task: str = "regression",
        active_prototype_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the MIL core with flexible component structure.
        
        Parameters
        ----------
        components : List[nn.Module]
            List of model components in the order they should be applied
        component_names : List[str]
            Names of the components for logging and debugging
        forward_func : Optional[Callable], optional
            Custom forward function to use instead of the default, by default None.
            If None, will use the default forward function based on the component structure.
        task : str, optional
            Task type (classification or regression), by default "regression"
        """
        super().__init__()
        
        if len(components) != len(component_names):
            raise ValueError(f"Number of components ({len(components)}) must match number of names ({len(component_names)})")
        
        # Register components as module attributes
        for i, (component, name) in enumerate(zip(components, component_names)):
            setattr(self, f"component_{i}", component)
            setattr(self, f"{name}", component)  # Also register with the actual name for backward compatibility
        
        self.components = components
        self.component_names = component_names
        self.custom_forward = forward_func
        self.task = task.lower()
        self.active_query_force_disabled = False
        self.active_query_forced_weight: Optional[float] = None
        self._configure_active_prototypes(active_prototype_kwargs or {})
        
        LOGGER.info(f"[MODEL] Created MILCore with components: {component_names}")

    @staticmethod
    def _embed_with_optional_mask(
        embedder: nn.Module,
        batch_data: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Call mask-aware embedders with a valid-token mask.

        Data loaders and aggregators use PyTorch's convention where True means
        padding. Contextualized embedders use True for valid conformers, so the
        mask is inverted here.
        """
        if key_padding_mask is None:
            return embedder(batch_data)

        try:
            forward_sig = inspect.signature(embedder.forward)
        except (TypeError, ValueError):
            return embedder(batch_data)

        if "mask" not in forward_sig.parameters:
            return embedder(batch_data)

        valid_mask = ~key_padding_mask.bool()
        return embedder(batch_data, mask=valid_mask)

    @staticmethod
    def _forward_accepts(module: nn.Module, parameter_name: str) -> bool:
        try:
            forward_sig = inspect.signature(module.forward)
        except (TypeError, ValueError):
            return False
        return parameter_name in forward_sig.parameters

    @classmethod
    def _call_aggregator(
        cls,
        aggregator: nn.Module,
        h: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        external_queries: Optional[torch.Tensor] = None,
        external_query_weight: Optional[float] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        kwargs: Dict[str, Any] = {}

        if key_padding_mask is not None and cls._forward_accepts(
            aggregator, "key_padding_mask"
        ):
            kwargs["key_padding_mask"] = key_padding_mask

        if return_attn and cls._forward_accepts(aggregator, "return_attn"):
            kwargs["return_attn"] = True

        if cluster_ids is not None and cls._forward_accepts(aggregator, "cluster_ids"):
            kwargs["cluster_ids"] = cluster_ids

        if external_queries is not None:
            if not cls._forward_accepts(aggregator, "external_queries"):
                raise ValueError(
                    f"{aggregator.__class__.__name__} does not accept external_queries"
                )
            kwargs["external_queries"] = external_queries

        if external_query_weight is not None and cls._forward_accepts(
            aggregator,
            "external_query_weight",
        ):
            kwargs["external_query_weight"] = external_query_weight

        return aggregator(h, **kwargs)

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
        }:
            raise ValueError(
                "candidate_selection must be 'attention' or "
                f"'attention_ablation', got {self.active_prototype_candidate_selection!r}"
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
        default_prune_interval = 100 if self.active_prototype_min_active > 1 else 0
        self.active_prototype_prune_every_n_updates = int(
            cfg.get("prune_every_n_updates", default_prune_interval)
        )

        if not self.active_prototype_enabled:
            self.active_prototype_bank = None
            self.active_query_builder = None
            return

        dim = self._infer_active_prototype_dim(cfg)
        self.active_prototype_bank = DynamicActivePrototypeBank(
            dim=dim,
            max_prototypes=int(cfg.get("max_prototypes", 64)),
            create_sim_threshold=float(cfg.get("create_sim_threshold", 0.80)),
            merge_sim_threshold=float(cfg.get("merge_sim_threshold", 0.95)),
            ema_momentum=float(cfg.get("ema_momentum", 0.05)),
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
                    if (~mask_b[0]).sum() == 0:
                        impacts.append(z.new_zeros(()))
                        continue
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
            if (
                self.active_prototype_candidate_selection == "attention_ablation"
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
            self.active_prototype_bank.update(
                candidates,
                candidate_series=candidate_series,
                max_per_series=self.active_prototype_max_per_series,
            )
            num_after = self.active_prototype_bank.num_active()
            support_after = float(
                self.active_prototype_bank.counts[
                    self.active_prototype_bank.active_mask
                ].sum().item()
            )

            if candidates.size(0) > 0 or num_after != num_before:
                LOGGER.debug(
                    "[ACTIVE_PROTO] update candidates=%d active_prototypes=%d->%d "
                    "support=%.0f->%.0f max=%d",
                    int(candidates.size(0)),
                    num_before,
                    num_after,
                    support_before,
                    support_after,
                    self.active_prototype_bank.max_prototypes,
                )

            if hasattr(self, "active_prototype_num_updates"):
                self.active_prototype_num_updates.add_(1)
                prune_every = self.active_prototype_prune_every_n_updates
                if prune_every > 0:
                    num_updates = int(self.active_prototype_num_updates.item())
                    if num_updates % prune_every == 0:
                        self.active_prototype_bank.prune_weak_prototypes()

        return int(candidates.size(0))

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

    @staticmethod
    def _add_mask_extras(
        agg_info: Dict[str, Any],
        key_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if key_padding_mask is None:
            return agg_info

        valid_counts = (~key_padding_mask.bool()).sum(dim=-1)
        agg_info["valid_counts"] = valid_counts
        return agg_info
    
    def forward(
        self,
        batch_data,
        key_padding_mask: Optional[torch.Tensor] = None,
        cluster_ids: Optional[torch.Tensor] = None,
        series_labels: Optional[list[str]] = None,
        labels: Optional[torch.Tensor] = None,
        stage: Optional[str] = None,
        current_epoch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass through the MIL pipeline with padding before embedding.
        
        If a custom forward function was provided during initialization, it will be used.
        Otherwise, a default forward function based on the component structure will be used.
        
        Parameters
        ----------
        batch_data : torch.Tensor
            Input tensor with pre-padded data [B, N_max, D_input]
        key_padding_mask : Optional[torch.Tensor]
            Padding mask for batched inputs [B, N_max] where True = padding
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, Any]]
            Tuple of (output tensor, additional info)
        """
        # Log forward pass start
        batch_shape = batch_data.shape if hasattr(batch_data, 'shape') else 'unknown'
        mask_shape = key_padding_mask.shape if key_padding_mask is not None else 'None'
        LOGGER.debug(f"[MIL_CORE] Starting forward pass - batch_shape={batch_shape}, mask_shape={mask_shape}")
        LOGGER.debug(f"[MIL_CORE] Using padding-before-embedding approach (simpler, faster)")
        
        if self.custom_forward is not None:
            LOGGER.debug(f"[MIL_CORE] Using custom forward function: {self.custom_forward.__name__}")
            return self.custom_forward(self, batch_data, key_padding_mask)
        
        # Log component structure being used
        LOGGER.debug(f"[MIL_CORE] Using default forward with components: {self.component_names}")
        
        # Default forward function based on component structure
        if len(self.components) == 3 and "aggregator" in self.component_names:
            # Standard bag-attention pipeline: embedder -> aggregator -> predictor
            LOGGER.debug(f"[MIL_CORE] Using 3-component pipeline: embedder → aggregator → predictor")
            
            embedder_idx = self.component_names.index("embedder")
            aggregator_idx = self.component_names.index("aggregator")
            predictor_idx = self.component_names.index("predictor")
            
            embedder = self.components[embedder_idx]
            aggregator = self.components[aggregator_idx]
            predictor = self.components[predictor_idx]
            
            # Process pre-padded tensor through embedder (including padded positions)
            LOGGER.debug(f"[MIL_CORE] Processing pre-padded tensor through embedder")
            LOGGER.debug(f"[MIL_CORE] Input shape: {batch_data.shape}")
            
            h = self._embed_with_optional_mask(
                embedder, batch_data, key_padding_mask
            )  # [B, N_max, D_embed]
            
            LOGGER.debug(f"[MIL_CORE] After embedder: h.shape={h.shape}")
            if key_padding_mask is not None:
                LOGGER.debug(f"[MIL_CORE] Using padding mask: {key_padding_mask.shape}")
            else:
                LOGGER.debug(f"[MIL_CORE] No padding mask provided")
            
            # Continue with aggregator and predictor. If active prototype memory
            # is enabled, update it from training batches after warm-up, then use
            # it to build a bag-specific external CLS query when enough
            # prototypes exist.
            LOGGER.debug(f"[MIL_CORE] Calling aggregator with h.shape={h.shape}")
            update_candidates = 0
            needs_bank_update = self._should_update_active_prototypes(
                labels=labels,
                stage=stage,
                current_epoch=current_epoch,
            )
            use_active_query_now = self._should_use_active_query(
                aggregator=aggregator,
                stage=stage,
                current_epoch=current_epoch,
            )
            active_query_weight = self._effective_active_query_weight(current_epoch)
            self._log_active_prototype_forward_status(
                stage=stage,
                current_epoch=current_epoch,
                needs_bank_update=needs_bank_update,
                use_active_query_now=use_active_query_now,
                labels=labels,
                aggregator=aggregator,
            )

            if use_active_query_now:
                active_query, proto_info = self.active_query_builder(
                    h,
                    prototype_bank=self.active_prototype_bank,
                    key_padding_mask=key_padding_mask,
                    series_labels=series_labels,
                )
                bag_repr, agg_info = self._call_aggregator(
                    aggregator,
                    h,
                    key_padding_mask=key_padding_mask,
                    cluster_ids=cluster_ids,
                    external_queries=active_query,
                    external_query_weight=active_query_weight,
                    return_attn=True,
                )
                agg_info.update(proto_info)
                agg_info["active_query_used"] = True
                agg_info["active_query_weight"] = active_query_weight

                if needs_bank_update:
                    update_info = agg_info
                    if agg_info.get("alpha_base") is not None:
                        update_info = dict(agg_info)
                        update_info["alpha"] = agg_info["alpha_base"]
                        update_info["alpha_final"] = agg_info["alpha_base"]
                    update_candidates = self._update_active_prototypes_from_batch(
                        z=h,
                        agg_info=update_info,
                        labels=labels,
                        key_padding_mask=key_padding_mask,
                        cluster_ids=cluster_ids,
                        series_labels=series_labels,
                        aggregator=aggregator,
                        predictor=predictor,
                    )
            else:
                bag_repr, agg_info = self._call_aggregator(
                    aggregator,
                    h,
                    key_padding_mask=key_padding_mask,
                    cluster_ids=cluster_ids,
                    return_attn=needs_bank_update,
                )

                if needs_bank_update:
                    update_candidates = self._update_active_prototypes_from_batch(
                        z=h,
                        agg_info=agg_info,
                        labels=labels,
                        key_padding_mask=key_padding_mask,
                        cluster_ids=cluster_ids,
                        series_labels=series_labels,
                        aggregator=aggregator,
                        predictor=predictor,
                    )

                agg_info["active_query_used"] = False
                agg_info["active_query_weight"] = 0.0

            agg_info = self._add_active_prototype_extras(
                agg_info,
                update_candidates=update_candidates,
            )
            agg_info = self._add_mask_extras(agg_info, key_padding_mask)
            if key_padding_mask is not None:
                LOGGER.debug(f"[MIL_CORE] Aggregator called with padding mask")
            else:
                LOGGER.debug(f"[MIL_CORE] Aggregator called without padding mask")
            
            LOGGER.debug(f"[MIL_CORE] After aggregator: bag_repr.shape={bag_repr.shape}")
            LOGGER.debug(f"[MIL_CORE] Aggregator info keys: {list(agg_info.keys())}")
            
            logit = predictor(bag_repr)  # raw score
            LOGGER.debug(f"[MIL_CORE] After predictor: logit.shape={logit.shape}, logit.numel()={logit.numel()}")
            
            # Handle both single bag and batch outputs correctly
            if logit.numel() == 1:
                # Single prediction - reshape to scalar
                result_logit = logit.view(())
                LOGGER.debug(f"[MIL_CORE] Returning scalar output: {result_logit.shape}")
                return result_logit, agg_info
            else:
                # Batch predictions - keep as 1D tensor
                result_logit = logit.flatten()
                LOGGER.debug(f"[MIL_CORE] Returning batch output: {result_logit.shape}")
                return result_logit, agg_info
            
        elif len(self.components) == 2:
            if "embedder" in self.component_names and "predictor" in self.component_names:
                # Embedder-predictor pipeline: embedder -> predictor
                embedder_idx = self.component_names.index("embedder")
                predictor_idx = self.component_names.index("predictor")
                
                embedder = self.components[embedder_idx]
                predictor = self.components[predictor_idx]
                
                h = self._embed_with_optional_mask(
                    embedder, batch_data, key_padding_mask
                )
                logit = predictor(h)
                # Handle both single bag and batch outputs correctly
                if logit.numel() == 1:
                    return logit.view(()), {}
                else:
                    return logit.flatten(), {}
                
            elif "aggregator" in self.component_names and "predictor" in self.component_names:
                # Aggregator-only pipeline: aggregator -> predictor
                aggregator_idx = self.component_names.index("aggregator")
                predictor_idx = self.component_names.index("predictor")
                
                aggregator = self.components[aggregator_idx]
                predictor = self.components[predictor_idx]
                
                if key_padding_mask is not None:
                    if self._forward_accepts(aggregator, "cluster_ids"):
                        bag_repr, agg_info = aggregator(
                            batch_data,
                            key_padding_mask=key_padding_mask,
                            cluster_ids=cluster_ids,
                        )
                    else:
                        bag_repr, agg_info = aggregator(batch_data, key_padding_mask=key_padding_mask)
                else:
                    if cluster_ids is not None and self._forward_accepts(aggregator, "cluster_ids"):
                        bag_repr, agg_info = aggregator(batch_data, cluster_ids=cluster_ids)
                    else:
                        bag_repr, agg_info = aggregator(batch_data)
                logit = predictor(bag_repr)
                # Handle both single bag and batch outputs correctly
                if logit.numel() == 1:
                    return logit.view(()), agg_info
                else:
                    return logit.flatten(), agg_info
        
        # Generic fallback for any other component structure
        # Chain components in order, assuming the last component returns the final output
        h = batch_data
        agg_info = {}
        
        for i, component in enumerate(self.components):
            if i == len(self.components) - 1:
                # Last component, assume it's the predictor
                logit = component(h)
                # Handle both single bag and batch outputs correctly
                if logit.numel() == 1:
                    return logit.view(()), agg_info
                else:
                    return logit.flatten(), agg_info
            else:
                # Check if this component returns additional info (like aggregator)
                try:
                    result = component(h)
                    if isinstance(result, tuple) and len(result) == 2:
                        h, component_info = result
                        agg_info.update(component_info)
                    else:
                        h = result
                except Exception as e:
                    LOGGER.error(f"Error in component {self.component_names[i]}: {e}")
                    raise
        
        # Should never reach here
        raise RuntimeError("Forward pass failed to return a result")
