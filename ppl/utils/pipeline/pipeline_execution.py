"""Execution utilities for the PipelineOrchestrator."""
import logging
import pprint
import traceback

from ppl.utils.modelling_configs.pipeline_config import PipelineConfig
from ppl.utils.model_trainer.cross_validator import CrossValidator
from ppl.utils.model_trainer.model_evaluator import ModelEvaluator

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def execute_pipeline(
    cfg: PipelineConfig,
    num_folds: int,
    cross_validator: CrossValidator,
    model_evaluator: ModelEvaluator
) -> None:
    """Execute the pipeline with cross-validation and final evaluation.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object for the pipeline
    num_folds : int
        Number of cross-validation folds
    cross_validator : CrossValidator
        Cross-validation component
    model_evaluator : ModelEvaluator
        Model evaluation component

    Raises
    ------
    Exception
        If pipeline execution fails
    """
    LOGGER.info("Pipeline started …")

    try:
        # Log experiment configuration
        LOGGER.info("[EXP CFG] Experiment configuration:\n%s", pprint.pformat(cfg))

        # Stage 1: always perform a train/validation run (CV if num_folds>1, single split otherwise)
        if num_folds >= 1:
            if num_folds == 1:
                LOGGER.info("Running Stage 1: train/validation on predefined/partitioned validation split")
            else:
                LOGGER.info(f"Running {num_folds}-fold cross-validation")
            try:
                cross_validator.run_cross_validation()
                LOGGER.info("Stage 1 completed successfully")
            except Exception as e:
                LOGGER.error(f"Stage 1 failed: {str(e)}")
                LOGGER.debug(f"Stage 1 error details: {traceback.format_exc()}")
                # Preserve the original exception
                raise

        # Run final evaluation (Stage 2) only if enabled
        try:
            stage2_enabled = True
            try:
                stage2_enabled = getattr(cfg.trainer, "stage_2_launch", True)
            except Exception:
                stage2_enabled = True

            if stage2_enabled:
                LOGGER.info("Running final model training and evaluation (Stage 2)")
                model_evaluator.final_fit_test()
                LOGGER.info("Final evaluation completed successfully")
            else:
                LOGGER.info("Stage 2 launch disabled by config (trainer.stage_2_launch=False). Finishing after Stage 1.")
        except Exception as e:
            LOGGER.error(f"Final evaluation failed: {str(e)}")
            LOGGER.debug(f"Final evaluation error details: {traceback.format_exc()}")
            # Preserve the original exception
            raise

        LOGGER.info("Pipeline finished successfully.")

    except Exception as e:
        LOGGER.error(f"Pipeline execution failed: {str(e)}")
        raise