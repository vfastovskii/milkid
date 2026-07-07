"""config_manager.py

This module handles configuration validation and setup for the pipeline.
It provides a ConfigManager class that validates and configures data, model, and trainer settings.
"""

import logging
from pathlib import Path

from ppl.utils.modelling_configs.data_loader_config import DataLoaderConfig
from ppl.utils.modelling_configs import (
    ModelBuilderConfig,
    PipelineConfig,
    TrainerOptimConfig,
)
from ppl.utils.modelling_configs.trainer_config import TrainerConfig, TrainerBuilder
from ppl.utils.pipeline.config_override_utils import override_dataclass
from ppl.utils.reproducibility import set_deterministic

# Global logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

class PipelineConfigManager:
    """Manages configuration validation and setup for the pipeline.

    This class is responsible for:
    - Validating configuration parameters
    - Setting up data, model, and trainer configurations
    - Configuring reproducibility settings
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        """Initialize the configuration manager with the given configuration.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration object for the pipeline

        Raises
        ------
        ValueError
            If required configuration parameters are missing or invalid
        """
        self.cfg = cfg
        self.data_cfg = None
        self.model_cfg = None
        self.trainer_cfg = None
        self.task = None
        self.num_folds = None
        self.cv_seed = None
        self.log_save_dir = None

        # Validate and configure
        self.validate_config()
        self.configure_data()
        self.configure_model()
        self.configure_trainer()
        self.configure_reproducibility()

    def validate_config(self) -> None:
        """Validate all critical configuration parameters."""
        # Validate data configuration
        if not hasattr(self.cfg, 'data') or self.cfg.data is None:
            raise ValueError("Missing data configuration section")

        if not hasattr(self.cfg.data, 'csv_path') or self.cfg.data.csv_path is None:
            raise ValueError("Missing required 'csv_path' in data configuration")

        # Validate model configuration
        if not hasattr(self.cfg, 'model') or self.cfg.model is None:
            raise ValueError("Missing model configuration section")

        # Validate trainer configuration
        if not hasattr(self.cfg, 'trainer') or self.cfg.trainer is None:
            raise ValueError("Missing trainer configuration section")

        # Validate log directory
        if not hasattr(self.cfg.trainer, 'log_save_dir') or not self.cfg.trainer.log_save_dir:
            raise ValueError("Missing required 'log_save_dir' in trainer configuration")

    def configure_data(self) -> None:
        """Set up data configuration."""
        # csv_path is already validated in validate_config
        base_data_cfg = DataLoaderConfig(csv_path=self.cfg.data.csv_path)
        self.data_cfg: DataLoaderConfig = override_dataclass(base_data_cfg, self.cfg.data)

        # Extract commonly used values
        self.task: str = self.data_cfg.task.lower()
        self.num_folds: int = self.data_cfg.num_folds
        self.cv_seed: int = self.data_cfg.cv_seed

        # Set experiment_name from trainer config if available
        if hasattr(self.cfg.trainer, 'experiment_name') and self.cfg.trainer.experiment_name:
            self.data_cfg.experiment_name = self.cfg.trainer.experiment_name
            LOGGER.info(f"Using experiment_name '{self.data_cfg.experiment_name}' for data splits")

            # Set default cache_dir if not specified
            if not self.data_cfg.cache_dir:
                # Use the experiment_name to create a default cache directory
                self.data_cfg.cache_dir = Path(self.cfg.trainer.experiment_name)
                LOGGER.info(f"Setting default cache_dir to '{self.data_cfg.cache_dir}' for data splits")

        # Validate path
        if isinstance(self.data_cfg.csv_path, str):
            self.data_cfg.csv_path = Path(self.data_cfg.csv_path).expanduser()

    def configure_model(self) -> None:
        """Set up model and optimization configuration."""
        base_model_cfg = ModelBuilderConfig(task=self.task)
        self.model_cfg: ModelBuilderConfig = override_dataclass(base_model_cfg, self.cfg.model)

        base_optim_cfg = TrainerOptimConfig()
        self.model_cfg.optim = override_dataclass(base_optim_cfg, self.cfg.model.optim)

    def configure_trainer(self) -> None:
        """Set up trainer configuration and logging."""
        base_trainer_cfg = TrainerConfig()
        self.trainer_cfg: TrainerConfig = override_dataclass(base_trainer_cfg, self.cfg.trainer)

        # Ensure the log directory is a Path and exists
        self.log_save_dir: Path = Path(self.trainer_cfg.log_save_dir).expanduser()
        self.log_save_dir.mkdir(parents=True, exist_ok=True)

    def configure_reproducibility(self) -> None:
        """Configure deterministic behavior for reproducibility."""
        # Use the seed from data config or default to 42
        seed = getattr(self.data_cfg, 'cv_seed', 42)
        LOGGER.info(f"Setting global seed to {seed}")
        set_deterministic(seed=seed)

    def get_shared_config(self) -> dict:
        """Get shared configuration parameters for pipeline components.

        Returns
        -------
        dict
            Dictionary containing shared configuration parameters
        """
        return {
            'log_save_dir': self.log_save_dir,
            'experiment_name': self.trainer_cfg.experiment_name,
            'run_name': self.trainer_cfg.run_name,
            'tracking_uri': self.trainer_cfg.tracking_uri,
        }
