"""Utility modules for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.utils.model_builder.model_builder import ModelBuilder
from ppl.utils.model_builder.model_components_registry import register, get_component
from ppl.utils.adaptive_entropic_loss import AdaptiveEntropicLoss
from ppl.utils.pipeline_builder import build_pipeline_from_config, list_available_use_cases
