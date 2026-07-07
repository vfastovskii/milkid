"""Model builder package.

This package provides utilities for building MIL models.
"""

from ppl.utils.model_builder.model_builder import ModelBuilder
from ppl.utils.model_builder.component_catalog import (
    AGGREGATORS,
    EMBEDDERS,
    PREDICTORS,
    build_components,
)
