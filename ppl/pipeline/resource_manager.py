"""resource_manager.py

This module handles resource management for the pipeline.
It provides a ResourceManager class that initializes and cleans up resources like MLflow.
"""

import logging
import traceback
from pathlib import Path

import mlflow

from ppl.pipeline.mlflow_utils import create_mlflow_logger

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class PipelineResourceManager:
    """Manages resources for the pipeline.
    
    This class is responsible for:
    - Initializing MLflow logger
    - Cleaning up resources to prevent memory leaks
    """
    
    def __init__(
        self,
        log_save_dir: Path,
        experiment_name: str = None,
        run_name: str = None,
        tracking_uri: str = None,
    ) -> None:
        """Initialize the resource manager.
        
        Parameters
        ----------
        log_save_dir : Path
            Directory for saving logs
        experiment_name : str, optional
            Name of the MLflow experiment
        run_name : str, optional
            Name of the MLflow run
        tracking_uri : str, optional
            URI for MLflow tracking server
            
        Raises
        ------
        ValueError
            If MLflow logger initialization fails
        """
        self.log_save_dir = log_save_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.mlflow_logger = None
        
        # Initialize MLflow logger
        self._initialize_mlflow_logger()
    
    def _initialize_mlflow_logger(self) -> None:
        """Initialize the MLflow logger.
        
        Raises
        ------
        ValueError
            If MLflow logger initialization fails
        """
        try:
            self.mlflow_logger = create_mlflow_logger(
                self.log_save_dir,
                experiment_name=self.experiment_name,
                run_name=self.run_name,
                tracking_uri=self.tracking_uri,
            )
            LOGGER.info(f"{self.mlflow_logger.__class__.__name__} initialized with experiment_name: {self.experiment_name}")
        except Exception as e:
            LOGGER.error(f"Failed to initialize {self.mlflow_logger.__class__.__name__}: {str(e)}")
            LOGGER.debug(f"{self.mlflow_logger.__class__.__name__} initialization error details: {traceback.format_exc()}")
            raise ValueError(f"{self.mlflow_logger.__class__.__name__} initialization failed: {str(e)}") from e
    
    def cleanup_resources(self) -> None:
        """Clean up resources to prevent memory leaks."""
        # Clean up MLflow resources if they were initialized
        if self.mlflow_logger is not None:
            try:
                # End the MLflow run if active
                if mlflow.active_run():
                    mlflow.end_run()
                LOGGER.info("MLflow resources cleaned up")
            except Exception as e:
                LOGGER.warning(f"Error during MLflow cleanup: {str(e)}")
        
    def __enter__(self):
        """Context manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point with resource cleanup."""
        self.cleanup_resources()
        return False  # Don't suppress exceptions