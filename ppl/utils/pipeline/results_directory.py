"""Utilities for creating and managing the results directory structure."""

from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

def create_results_directory(experiment_name: str = None) -> Path:
    """Create the results directory structure.
    
    Parameters
    ----------
    experiment_name : str, optional
        Name of the experiment, by default None
        
    Returns
    -------
    Path
        Path to the results directory
    """
    # Create the base results directory
    results_dir = Path("results")
    
    # If experiment_name is provided, create a subdirectory for the experiment
    if experiment_name:
        results_dir = results_dir / experiment_name
    
    # Create subdirectories for validation and test
    validation_dir = results_dir / "validation"
    test_dir = results_dir / "test"
    
    # Create the directories
    validation_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"Created results directory structure at {results_dir}")
    
    return results_dir