"""Aggregator-Only template specification.

This template uses only an aggregator and predictor, skipping the embedder.
Useful for scenarios where input features are already well-represented.
"""

import logging
import traceback
from typing import Any, Tuple

from ppl.utils.model_builder.template_registry import (
    Template, TemplateSpec, TemplateDefinition, ComponentRequirement, register_template
)
from ppl.utils.model_builder.model_components_registry import get_component

LOGGER = logging.getLogger(__name__)

@register_template(Template.AGGREGATOR_ONLY)
class AggregatorOnlySpec(TemplateSpec):
    """Template specification for Aggregator-Only architecture."""
    
    def get_template_definition(self) -> TemplateDefinition:
        """Get the template definition for Aggregator-Only architecture."""
        return TemplateDefinition(
            name="aggregator_only",
            components=[
                ComponentRequirement(
                    component_type="aggregator",
                    required_attributes=["input_dim", "output_dim"]
                ),
                ComponentRequirement(
                    component_type="predictor",
                    required_attributes=["input_dim", "output_dim"]
                )
            ],
            component_order=["aggregator", "predictor"],
            task_compatibility=["regression", "classification"],
            description="Aggregator-only architecture that skips embedding"
        )
    
    def build_components(self, cfg: Any, input_dim: int) -> Tuple[Any, Any]:
        """Build model components for Aggregator-Only architecture."""
        LOGGER.info(f"[MODEL] Using template: {Template.AGGREGATOR_ONLY.name}")
        
        # Get aggregator and predictor classes
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
            # Create aggregator with input dimension
            aggregator_kwargs = getattr(cfg, "aggregator_kwargs", {}).copy()
            aggregator = aggregator_cls(input_dim=input_dim, **aggregator_kwargs)
            self._log_tensor_initialization(aggregator, "Aggregator")
            
            # Create predictor with aggregator output dimension
            predictor_kwargs = getattr(cfg, "predictor_kwargs", {}).copy()
            if cfg.task == "regression" and "output_dim" not in predictor_kwargs:
                predictor_kwargs["output_dim"] = 1
            predictor = predictor_cls(input_dim=aggregator.output_dim, **predictor_kwargs)
            self._log_tensor_initialization(predictor, "Predictor")
            
            return aggregator, predictor
            
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