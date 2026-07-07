"""Initialization and validation utilities for the PipelineOrchestrator."""
import logging
import traceback
from typing import Tuple

from ppl.utils.modelling_configs.pipeline_config import PipelineConfig
from ppl.utils.pipeline.component_factory import PipelineComponentFactory
from ppl.utils.pipeline.config_manager import PipelineConfigManager
from ppl.utils.pipeline.resource_manager import PipelineResourceManager

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def validate_pipeline_configuration(task: str) -> None:
    """Validate critical configuration parameters.

    Parameters
    ----------
    task : str
        The task type (classification or regression)

    Raises
    ------
    ValueError
        If any configuration parameter has an invalid value
    """
    valid_tasks = ['classification', 'regression']
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}, got {task}")

def validate_pipeline_components(component_factory: PipelineComponentFactory) -> None:
    """Validate that all required components are properly initialized.

    Parameters
    ----------
    component_factory : PipelineComponentFactory
        The pipeline component factory

    Raises
    ------
    RuntimeError
        If any required component is missing
    """
    required_attrs = ['model_trainer', 'split_trainer']
    for attr in required_attrs:
        if not hasattr(component_factory, attr) or getattr(component_factory, attr) is None:
            raise RuntimeError(f"Component factory missing required attribute: {attr}")

def initialize_pipeline_components(cfg: PipelineConfig) -> Tuple[
    PipelineConfigManager,
    PipelineResourceManager,
    PipelineComponentFactory,
    str,
]:
    """Initialize all pipeline components.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object for the pipeline

    Returns
    -------
    Tuple[PipelineConfigManager, PipelineResourceManager, PipelineComponentFactory, str]
        Tuple containing the initialized components:
        - config_manager: The pipeline configuration manager
        - resource_manager: The pipeline resource manager
        - component_factory: The pipeline component factory
        - task: The task type (classification or regression)

    Raises
    ------
    ValueError
        If required configuration parameters are missing or invalid
    Exception
        If initialization of any component fails
    """
    # Set up pipeline configuration
    config_manager = PipelineConfigManager(cfg)

    # Extract commonly used values
    task = config_manager.task

    # Validate critical configuration parameters
    validate_pipeline_configuration(task)

    # Initialize MLflow experiment logger
    try:
        resource_manager = PipelineResourceManager(
            log_save_dir=config_manager.log_save_dir,
            experiment_name=config_manager.trainer_cfg.experiment_name,
            run_name=config_manager.trainer_cfg.run_name,
            tracking_uri=config_manager.trainer_cfg.tracking_uri,
        )
    except Exception as e:
        LOGGER.error(f"Failed to initialize PipelineResourceManager: {str(e)}")
        raise

    # Initialize pipeline component factory
    try:
        component_factory = PipelineComponentFactory(
            data_cfg=config_manager.data_cfg,
            model_cfg=config_manager.model_cfg,
            trainer_cfg=config_manager.trainer_cfg,
            log_save_dir=config_manager.log_save_dir,
            task=config_manager.task,
            seed=config_manager.seed,
        )
    except Exception as e:
        LOGGER.error(f"Failed to initialize PipelineComponentFactory: {str(e)}")
        raise

    # Create pipeline components
    component_factory.create_ppl_components()

    # Validate initialized pipeline components
    validate_pipeline_components(component_factory)
    LOGGER.info("Pipeline components initialized successfully")

    return config_manager, resource_manager, component_factory, task

def cleanup_resources(components: list) -> None:
    """Clean up resources to prevent memory leaks.

    This function attempts to call the cleanup_resources method on each component
    that has one. It handles exceptions gracefully to ensure all components get
    a chance to clean up, even if some fail.

    Parameters
    ----------
    components : list
        List of components that might have cleanup_resources method
    """
    if not components:
        return

    # Track any exceptions that occur during cleanup
    cleanup_errors = []

    for component in components:
        if component is None:
            continue

        # Only attempt cleanup if the component has the method
        if hasattr(component, 'cleanup_resources'):
            try:
                component.cleanup_resources()
                LOGGER.debug(f"Successfully cleaned up resources for {component.__class__.__name__}")
            except Exception as e:
                # Log the error but continue with other components
                error_msg = f"Error cleaning up {component.__class__.__name__}: {str(e)}"
                LOGGER.warning(error_msg)
                cleanup_errors.append(error_msg)

    # If any errors occurred, log a summary but don't raise
    if cleanup_errors:
        LOGGER.warning(f"Encountered {len(cleanup_errors)} errors during resource cleanup")
