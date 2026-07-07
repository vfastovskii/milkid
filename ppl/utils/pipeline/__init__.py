"""Pipeline utilities for the Multi-Instance Learning Kit (MILK).

This package contains utilities for the MILK pipeline, including
- MLFlow logging utilities
- Pipeline configuration utilities
- Results visualization utilities
- Resource management
- Pipeline execution and initialization
"""

# Expose key modules at the package level
# Pipeline configuration
from ppl.utils.pipeline.config_manager import PipelineConfigManager

# Pipeline component factory
from ppl.utils.pipeline.component_factory import PipelineComponentFactory

# Pipeline resource management
from ppl.utils.pipeline.resource_manager import PipelineResourceManager

# MLFlow logging
from ppl.utils.pipeline.mlflow_utils import (
    create_mlflow_logger,
    log_metrics,
    SafeMLFlowLogger,
)

# Pipeline execution and initialization
from ppl.utils.pipeline.pipeline_execution import execute_pipeline

from ppl.utils.pipeline.pipeline_initialization import (
    validate_pipeline_configuration,
    validate_pipeline_components,
    initialize_pipeline_components,
    cleanup_resources
)

# Visualization
from ppl.utils.pipeline.visualization import log_split_distributions
