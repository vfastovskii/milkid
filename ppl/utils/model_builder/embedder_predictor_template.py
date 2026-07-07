"""Embedder-Predictor template specification.

This template uses only an embedder and predictor, skipping the aggregator.
Useful for scenarios where aggregation is not needed or handled elsewhere.
"""

import logging
import traceback
from typing import Any, Tuple

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

@register_template(Template.EMBEDDER_PREDICTOR)
class EmbedderPredictorSpec(TemplateSpec):
    """Template specification for Embedder-Predictor architecture."""
    
    def get_template_definition(self) -> TemplateDefinition:
        """Get the template definition for Embedder-Predictor architecture."""
        return TemplateDefinition(
            name="embedder_predictor",
            components=[
                ComponentRequirement(
                    component_type="embedder",
                    required_attributes=["input_dim", "output_dim"]
                ),
                ComponentRequirement(
                    component_type="predictor",
                    required_attributes=["input_dim", "output_dim"]
                )
            ],
            component_order=["embedder", "predictor"],
            task_compatibility=["regression", "classification"],
            description="Embedder-predictor architecture that skips aggregation"
        )
    
    def build_components(self, cfg: Any, input_dim: int) -> Tuple[Any, Any]:
        """Build model components for Embedder-Predictor architecture."""
        LOGGER.info(f"[MODEL] Using template: {Template.EMBEDDER_PREDICTOR.name}")
        
        # Get embedder and predictor classes
        try:
            embedder_cls = get_component("embedder_type", cfg.embedder_type)
            LOGGER.info(f"[MODEL] Using embedder: {cfg.embedder_type}")
        except Exception as e:
            LOGGER.error(f"Failed to get embedder component: {e}")
            raise ValueError(f"Invalid embedder_type: {cfg.embedder_type}") from e

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
            
            # Create predictor with embedder output dimension
            predictor_kwargs = getattr(cfg, "predictor_kwargs", {}).copy()
            if cfg.task == "regression" and "output_dim" not in predictor_kwargs:
                predictor_kwargs["output_dim"] = 1
            predictor = predictor_cls(input_dim=embedder.output_dim, **predictor_kwargs)
            self._log_tensor_initialization(predictor, "Predictor")
            
            return embedder, predictor
            
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
