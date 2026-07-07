"""Bag-Attention template specification.

This module provides a template specification for the Bag-Attention model architecture,
which uses a combination of an embedder, attention-based aggregator, and predictor.
"""

import logging
import traceback
from typing import Any, Tuple, List

from ppl.utils.model_builder.template_registry import (
    Template, TemplateSpec, TemplateDefinition, ComponentRequirement, register_template
)
from ppl.utils.model_builder.model_components_registry import get_component

LOGGER = logging.getLogger(__name__)


def _merge_self_attention_kwargs(cfg: Any, embedder_kwargs: dict) -> dict:
    """Merge optional self-attention kwargs into contextualized embedder kwargs."""
    self_attention_kwargs = getattr(cfg, "self_attention_kwargs", {}) or {}
    if not self_attention_kwargs:
        return embedder_kwargs

    overlap = set(embedder_kwargs).intersection(self_attention_kwargs)
    if overlap:
        raise ValueError(
            "Duplicate keys in embedder_kwargs and self_attention_kwargs: "
            f"{sorted(overlap)}. Keep each hyperparameter in only one group."
        )

    merged = embedder_kwargs.copy()
    merged.update(self_attention_kwargs)
    LOGGER.info(
        "[MODEL] Applying self_attention_kwargs to contextualized embedder: %s",
        sorted(self_attention_kwargs),
    )
    return merged

def _validate_attention_aggregator(component: Any, component_name: str) -> None:
    """Custom validator for attention-based aggregators."""
    # Check if it's an attention-based aggregator
    class_name = component.__class__.__name__.lower()
    if 'attention' not in class_name and 'att' not in class_name and 'vit' not in class_name:
        raise ValueError(
            f"{component_name} should be attention-based for Bag-Attention template. "
            f"Got {component.__class__.__name__}"
        )

@register_template(Template.BAG_ATTENTION)
class BagAttentionSpec(TemplateSpec):
    """Template specification for Bag-Attention model architecture."""
    
    def get_template_definition(self) -> TemplateDefinition:
        """Get the template definition for Bag-Attention architecture."""
        return TemplateDefinition(
            name="bag_attention",
            components=[
                ComponentRequirement(
                    component_type="embedder",
                    required_attributes=["input_dim", "output_dim"],
                    min_output_dim=16,  # Minimum for meaningful attention
                    max_output_dim=512  # Reasonable upper limit
                ),
                ComponentRequirement(
                    component_type="aggregator",
                    required_attributes=["input_dim", "output_dim"],
                    allowed_classes=["MultiHeadAttentionAggregator", "AttentionAggregator", "MultiHeadAttentionAggregatorV3", "VITAggregator", "MultiHeadAttentionAggregatorV4", "MultiHeadAttentionAggregatorV5", "ClusterHierarchicalAttentionAggregator"],
                    custom_validators=[_validate_attention_aggregator]
                ),
                ComponentRequirement(
                    component_type="predictor",
                    required_attributes=["input_dim", "output_dim"]
                )
            ],
            component_order=["embedder", "aggregator", "predictor"],
            task_compatibility=["regression", "classification"],
            description="Bag-Attention architecture with embedder, attention aggregator, and predictor"
        )
    
    def build_components(self, cfg: Any, input_dim: int) -> Tuple[Any, Any, Any]:
        """Build model components for Bag-Attention architecture."""
        LOGGER.info(f"[MODEL] Using template: {Template.BAG_ATTENTION.name}")
        
        # Get classes from the global registry
        try:
            embedder_cls = get_component("embedder_type", cfg.embedder_type)
            LOGGER.info(f"[MODEL] Using embedder: {cfg.embedder_type}")
        except Exception as e:
            LOGGER.error(f"Failed to get embedder component: {e}")
            raise ValueError(f"Invalid embedder_type: {cfg.embedder_type}") from e

        try:
            aggregator_cls = get_component("aggregator_type", cfg.aggregator_type)
            LOGGER.info(f"[MODEL] Using aggregator: {cfg.aggregator_type}")
        except Exception as e:
            LOGGER.error(f"Failed to get aggregator component: {e}")
            raise ValueError(f"Invalid aggregator_type: {cfg.aggregator_type}") from e

        try:
            predictor_cls = get_component("predictor_type", cfg.predictor_type)
            LOGGER.info(f"[MODEL] Using predictor: {cfg.predictor_type}")
        except Exception as e:
            LOGGER.error(f"Failed to get predictor component: {e}")
            raise ValueError(f"Invalid predictor_type: {cfg.predictor_type}") from e

        # Build components with dimension chaining
        try:
            # Create embedder
            embedder_kwargs = getattr(cfg, "embedder_kwargs", {}).copy()
            embedder_kwargs = _merge_self_attention_kwargs(cfg, embedder_kwargs)
            embedder = embedder_cls(input_dim=input_dim, **embedder_kwargs)
            self._log_tensor_initialization(embedder, "Embedder")
            
            # Create aggregator with embedder output dimension
            aggregator_kwargs = getattr(cfg, "aggregator_kwargs", {}).copy()
            aggregator = aggregator_cls(input_dim=embedder.output_dim, **aggregator_kwargs)
            self._log_tensor_initialization(aggregator, "Aggregator")
            
            # Create predictor with aggregator output dimension
            predictor_kwargs = getattr(cfg, "predictor_kwargs", {}).copy()
            if cfg.task == "regression" and "output_dim" not in predictor_kwargs:
                predictor_kwargs["output_dim"] = 1
            predictor = predictor_cls(input_dim=aggregator.output_dim, **predictor_kwargs)
            self._log_tensor_initialization(predictor, "Predictor")
            
            return embedder, aggregator, predictor
            
        except Exception as e:
            LOGGER.error(f"Failed to create components: {e}")
            LOGGER.error(traceback.format_exc())
            raise RuntimeError(f"Component initialization failed: {e}") from e
    
    def _log_tensor_initialization(self, module: Any, component_name: str) -> None:
        """Log tensor initialization information."""
        try:
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            LOGGER.info(f"[MODEL] {component_name} initialized with {total_params} parameters "
                        f"({trainable_params} trainable)")
            
            if LOGGER.isEnabledFor(logging.DEBUG):
                for name, param in module.named_parameters():
                    LOGGER.debug(f"[MODEL] {component_name}.{name}: {param.shape}")
        except Exception as e:
            LOGGER.warning(f"Failed to log tensor initialization for {component_name}: {e}")
