"""Execution utilities for the PipelineOrchestrator."""
import logging
import pprint
import traceback

from ppl.utils.modelling_configs.pipeline_config import PipelineConfig
from ppl.utils.model_trainer.split_trainer import SplitTrainer
from ppl.utils.model_trainer.model_evaluator import ModelEvaluator

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def execute_pipeline(
    cfg: PipelineConfig,
    split_trainer: SplitTrainer,
    model_evaluator: ModelEvaluator
) -> None:
    """Train on the predefined split, then optionally run final evaluation.

    Parameters
    ----------
    cfg : PipelineConfig
        Configuration object for the pipeline
    split_trainer : SplitTrainer
        Stage 1 train/validation/test component
    model_evaluator : ModelEvaluator
        Stage 2 final-model evaluation component

    Raises
    ------
    Exception
        If pipeline execution fails
    """
    LOGGER.info("Pipeline started …")

    try:
        # Log experiment configuration
        LOGGER.info("[EXP CFG] Experiment configuration:\n%s", pprint.pformat(cfg))

        # Stage 1: train on split=0, validate on split=1, test the best checkpoint
        LOGGER.info("Running Stage 1: train/validation on the predefined split")
        try:
            split_trainer.run()
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