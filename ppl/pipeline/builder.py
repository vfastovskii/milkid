"""Build a pipeline from a YAML configuration file.

This module provides a function to build a PipelineOrchestrator from a YAML configuration file
with robust error handling, timeout mechanisms, and configuration validation.
"""
import logging
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import yaml

# Forward reference for type hints to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ppl.pipeline.orchestrator import PipelineOrchestrator

from ppl.config import PipelineConfig

LOGGER = logging.getLogger(__name__)

# Type variable for the timeout decorator
T = TypeVar('T')


@contextmanager
def timeout(seconds: int, error_message: str = "Operation timed out"):
    """Context manager for timing out operations.

    Parameters
    ----------
    seconds : int
        Timeout in seconds
    error_message : str
        Error message to include in the TimeoutError

    Raises
    ------
    TimeoutError
        If the operation times out
    """
    def handle_timeout(signum, frame):
        raise TimeoutError(error_message)

    # Set the timeout handler
    original_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)

    try:
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def with_timeout(seconds: int, default_value: Optional[Any] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to apply a timeout to a function.

    Parameters
    ----------
    seconds : int
        Timeout in seconds
    default_value : Any, optional
        Value to return if the function times out

    Returns
    -------
    Callable
        Decorated function with timeout
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                with timeout(seconds):
                    return func(*args, **kwargs)
            except TimeoutError:
                LOGGER.error(f"Function {func.__name__} timed out after {seconds} seconds")
                if default_value is not None:
                    return default_value
                raise
        return wrapper
    return decorator


@with_timeout(60)  # 60-second timeout for config loading
def build_pipeline_from_config(yaml_path: str | Path) -> "PipelineOrchestrator":
    """Build a pipeline from a YAML configuration file with robust error handling.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file

    Returns
    -------
    PipelineOrchestrator
        A configured pipeline runner

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist
    ValueError
        If the configuration is invalid
    yaml.YAMLError
        If the YAML file is malformed
    TimeoutError
        If loading the configuration takes too long
    """
    start_time = time.time()
    
    try:
        # Load and validate configuration
        LOGGER.info(f"Loading pipeline configuration from {yaml_path}")
        ppl_cfg = PipelineConfig.from_yaml(yaml_path)

        # Import PipelineOrchestrator here to avoid circular imports
        from ppl.pipeline.orchestrator import PipelineOrchestrator

        # Create pipeline runner
        pipeline = PipelineOrchestrator(ppl_cfg)

        elapsed_time = time.time() - start_time
        LOGGER.info(f"Pipeline configuration loaded successfully in {elapsed_time:.2f} seconds")

        return pipeline
    except FileNotFoundError:
        LOGGER.error(f"Configuration file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        LOGGER.error(f"Invalid YAML format in configuration file: {e}")
        raise
    except ValueError as e:
        LOGGER.error(f"Invalid configuration: {e}")
        raise
    except Exception as e:
        LOGGER.error(f"Unexpected error building pipeline: {e}")
        raise
