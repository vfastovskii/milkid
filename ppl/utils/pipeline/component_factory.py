"""component_factory.py

This module handles the creation of pipeline components.
It provides a ComponentFactory class that initializes and configures the components needed for the pipeline.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict

from ppl.utils.mil_data_handling.data_loader import DataLoaderConfig
from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.modelling_configs.trainer_config import TrainerConfig, TrainerBuilder
from ppl.utils.model_trainer.cross_validator import CrossValidator
from ppl.utils.model_trainer.model_evaluator import ModelEvaluator
from ppl.utils.model_trainer.model_trainer import ModelTrainer

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class PipelineComponentFactory:
    """Factory for creating pipeline components.

    This class is responsible for:
    - Creating and initializing the ModelTrainer
    - Creating and initializing the CrossValidator
    - Creating and initializing the ModelEvaluator
    """

    def __init__(
        self,
        data_cfg: DataLoaderConfig,
        model_cfg: ModelBuilderConfig,
        trainer_cfg: TrainerConfig,
        log_save_dir: Path,
        task: str,
        cv_seed: int,
    ) -> None:
        """Initialize the component factory.

        Parameters
        ----------
        data_cfg : DataLoaderConfig
            Data configuration
        model_cfg : ModelBuilderConfig
            Model configuration
        trainer_cfg : TrainerConfig
            Trainer configuration
        log_save_dir : Path
            Directory for saving logs
        task : str
            Task type (e.g., 'classification', 'regression')
        cv_seed : int
            Seed for cross-validation

        Raises
        ------
        ValueError
            If component initialization fails
        """
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.log_save_dir = log_save_dir
        self.task = task
        self.cv_seed = cv_seed

        # Components
        self.model_trainer = None
        self.model_cross_validator = None
        self.model_evaluator = None

    def create_ppl_components(self) -> None:
        """Create all pipeline components.

        This method initializes the ModelTrainer, CrossValidator, and ModelEvaluator.

        Raises
        ------
        ValueError
            If component initialization fails
        """
        try:
            # Create a shared configuration dictionary to reduce redundant parameter passing
            shared_config = self._get_shared_config()

            # Initialize components
            self._create_model_trainer()
            self._create_cross_validator(shared_config)
            self._create_model_evaluator(shared_config)
        except Exception as e:
            LOGGER.error(f"Failed to create components: {str(e)}")
            LOGGER.debug(f"Component creation error details: {traceback.format_exc()}")
            raise ValueError(f"Component creation failed: {str(e)}") from e

    def _get_shared_config(self) -> Dict[str, Any]:
        """Get shared configuration parameters for pipeline components.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing shared configuration parameters
        """
        return {
            'log_save_dir': self.log_save_dir,
            'experiment_name': self.trainer_cfg.experiment_name,
            'run_name': self.trainer_cfg.run_name,
            'tracking_uri': self.trainer_cfg.tracking_uri,
        }

    def _create_model_trainer(self) -> None:
        """Initialize the model trainer.

        Raises
        ------
        ValueError
            If model trainer initialization fails
        """
        try:
            # Get shared config to access experiment_name
            shared_config = self._get_shared_config()

            self.model_trainer = ModelTrainer(
                model_cfg=self.model_cfg,
                trainer_cfg=self.trainer_cfg,
                log_save_dir=self.log_save_dir,
                cv_seed=self.cv_seed,
                task=self.task,
                experiment_name=shared_config.get('experiment_name'),
            )
            LOGGER.info(f"{self.model_trainer.__class__.__name__} initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize {self.model_trainer.__class__.__name__}: {str(e)}")
            LOGGER.debug(f"{self.model_trainer.__class__.__name__}r initialization error details: {traceback.format_exc()}")
            raise ValueError(f"{self.model_trainer.__class__.__name__} initialization failed: {str(e)}") from e

    def _create_cross_validator(self, shared_config: Dict[str, Any]) -> None:
        """Initialize the cross-validator.

        Parameters
        ----------
        shared_config : Dict[str, Any]
            Shared configuration parameters

        Raises
        ------
        ValueError
            If cross-validator initialization fails
        """
        try:
            self.model_cross_validator = CrossValidator(
                data_cfg=self.data_cfg,
                model_trainer=self.model_trainer,
                **shared_config
            )
            LOGGER.info(f" {self.model_cross_validator.__class__.__name__} initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize {self.model_cross_validator.__class__.__name__}: {str(e)}")
            LOGGER.debug(f"{self.model_cross_validator.__class__.__name__} initialization error details: {traceback.format_exc()}")
            raise ValueError(f"{self.model_cross_validator.__class__.__name__} initialization failed: {str(e)}") from e

    def _create_model_evaluator(self, shared_config: Dict[str, Any]) -> None:
        """Initialize the model evaluator.

        Parameters
        ----------
        shared_config : Dict[str, Any]
            Shared configuration parameters

        Raises
        ------
        ValueError
            If model evaluator initialization fails
        """
        try:
            self.model_evaluator = ModelEvaluator(
                data_cfg=self.data_cfg,
                model_trainer=self.model_trainer,
                **shared_config
            )
            LOGGER.info(f"{self.model_evaluator.__class__.__name__} initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize {self.model_evaluator.__class__.__name__}: {str(e)}")
            LOGGER.debug(f"{self.model_evaluator.__class__.__name__} initialization error details: {traceback.format_exc()}")
            raise ValueError(f"{self.model_evaluator.__class__.__name__} initialization failed: {str(e)}") from e
