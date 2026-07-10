"""component_factory.py

This module handles the creation of pipeline components.
It provides a ComponentFactory class that initializes and configures the components needed for the pipeline.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict

from ppl.data.data_loader import DataLoaderConfig
from ppl.config.model_builder_config import ModelBuilderConfig
from ppl.config.trainer_config import TrainerConfig
from ppl.training.split_trainer import SplitTrainer
from ppl.training.model_trainer import ModelTrainer

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class PipelineComponentFactory:
    """Factory for creating pipeline components.

    This class is responsible for:
    - Creating and initializing the ModelTrainer
    - Creating and initializing the SplitTrainer
    """

    def __init__(
        self,
        data_cfg: DataLoaderConfig,
        model_cfg: ModelBuilderConfig,
        trainer_cfg: TrainerConfig,
        results_dir: Path,
        task: str,
        seed: int,
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
        results_dir : Path
            Directory for saving logs
        task : str
            Task type (e.g., 'classification', 'regression')
        seed : int
            Global RNG seed

        Raises
        ------
        ValueError
            If component initialization fails
        """
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg
        self.results_dir = results_dir
        self.task = task
        self.seed = seed

        # Components
        self.model_trainer = None
        self.split_trainer = None

    def create_ppl_components(self) -> None:
        """Create all pipeline components.

        This method initializes the ModelTrainer and SplitTrainer.

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
            self._create_split_trainer(shared_config)
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
            'results_dir': self.results_dir,
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
                results_dir=self.results_dir,
                seed=self.seed,
                task=self.task,
                experiment_name=shared_config.get('experiment_name'),
            )
            LOGGER.info(f"{self.model_trainer.__class__.__name__} initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize ModelTrainer: {str(e)}")
            LOGGER.debug(f"ModelTrainer initialization error details: {traceback.format_exc()}")
            raise ValueError(f"ModelTrainer initialization failed: {str(e)}") from e

    def _create_split_trainer(self, shared_config: Dict[str, Any]) -> None:
        """Initialize the single-split trainer.

        Parameters
        ----------
        shared_config : Dict[str, Any]
            Shared configuration parameters

        Raises
        ------
        ValueError
            If split-trainer initialization fails
        """
        try:
            self.split_trainer = SplitTrainer(
                data_cfg=self.data_cfg,
                model_trainer=self.model_trainer,
                **shared_config
            )
            LOGGER.info(f"{self.split_trainer.__class__.__name__} initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize SplitTrainer: {str(e)}")
            LOGGER.debug(f"SplitTrainer initialization error details: {traceback.format_exc()}")
            raise ValueError(f"SplitTrainer initialization failed: {str(e)}") from e
