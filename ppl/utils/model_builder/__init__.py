"""Model builder package.

This package provides utilities for building MIL models.
"""

# Import template registry and implementations to ensure they are registered
from ppl.utils.model_builder.template_registry import Template, get_template
from ppl.utils.model_builder.bag_attention_template import BagAttentionSpec
