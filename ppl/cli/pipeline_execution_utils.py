"""Pipeline execution utils for the command-line interface."""
import argparse
import logging
from pathlib import Path

from ppl.pipeline.orchestrator import PipelineOrchestrator
from ppl.pipeline.builder import build_pipeline_from_config

from ppl.cli.pipeline_setup_utils import DEFAULT_SEED, configure_logging
from ppl.utils.reproducibility import (
    check_determinism,
    set_deterministic,
)


def configure_pipeline(config_path: Path, device: str | None = None) -> PipelineOrchestrator:
    """Configure and build the pipeline from the YAML config file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file
    device : str, optional
        Overrides ``trainer.device`` from the config (``auto`` / ``cpu`` / ``gpu``).

    Returns
    -------
    PipelineOrchestrator
        Configured pipeline runner instance

    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    Exception
        If pipeline configuration fails
    """
    try:
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logging.getLogger("milk").info("Config: %s", config_path.name)
        return build_pipeline_from_config(yaml_path=config_path, device=device)
    except Exception as e:
        logging.error("Failed to configure pipeline: %s", e)
        raise

def execute_pipeline(args: argparse.Namespace) -> None:
    """Execute the core pipeline logic.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments

    Raises
    ------
    Exception
        If pipeline execution fails
    """
    try:
        configure_logging(args.log_level)

        set_deterministic(seed=DEFAULT_SEED)
        check_determinism()

        # Configure pipeline from config file
        pipeline = configure_pipeline(
            config_path=args.ppl_cfg_path, device=getattr(args, "device", None)
        )
            
        logging.debug("Running pipeline ...")
        pipeline.run()
    except Exception as e:
        logging.error("Pipeline execution failed: %s", e)
        raise
