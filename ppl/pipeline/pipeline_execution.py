"""Execution utilities for the PipelineOrchestrator."""
import logging
import pprint

from ppl.config.pipeline_config import PipelineConfig
from ppl.training.split_trainer import SplitTrainer

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def execute_pipeline(cfg: PipelineConfig, split_trainer: SplitTrainer) -> None:
    """Run the pipeline: train on train, validate on val, test on test.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object for the pipeline.
    split_trainer : SplitTrainer
        Component that trains and evaluates on the predefined split.

    Raises
    ------
    Exception
        If pipeline execution fails.
    """
    LOGGER.info("Pipeline started …")
    LOGGER.info("[EXP CFG] Experiment configuration:\n%s", pprint.pformat(cfg))

    split_trainer.run()

    LOGGER.info("Pipeline finished successfully.")
