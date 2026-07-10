from __future__ import annotations

import logging
import inspect
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any, Callable, Optional

from ppl.models.active_prototype_forward import ActivePrototypeMixin

LOGGER = logging.getLogger(__name__)


# Core MIL network
class MILCore(ActivePrototypeMixin, nn.Module):
    """Flexible MIL pipeline that supports different component structures."""

    def __init__(
        self,
        components: List[nn.Module],
        component_names: List[str],
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
