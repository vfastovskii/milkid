"""config_manager.py

This module handles configuration validation and setup for the pipeline.
It provides a ConfigManager class that validates and configures data, model, and trainer settings.
"""

import logging
import shutil
from pathlib import Path

from ppl.config.data_loader_config import DataLoaderConfig
from ppl.config import (
    ModelBuilderConfig,
    PipelineConfig,
    TrainerOptimConfig,
)
from ppl.config.trainer_config import TrainerConfig
from ppl.pipeline.config_override_utils import override_dataclass
from ppl.pipeline.results_directory import create_results_directory
from ppl.utils.reproducibility import resolve_device, set_deterministic

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
        self.seed = None
        self.results_dir = None

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

    def configure_data(self) -> None:
        """Set up data configuration."""
        # csv_path is already validated in validate_config
        base_data_cfg = DataLoaderConfig(csv_path=self.cfg.data.csv_path)
        self.data_cfg: DataLoaderConfig = override_dataclass(base_data_cfg, self.cfg.data)

        # Extract commonly used values
        self.task: str = self.data_cfg.task.lower()
        self.seed: int = self.data_cfg.seed

        # Set experiment_name from trainer config if available
        if hasattr(self.cfg.trainer, 'experiment_name') and self.cfg.trainer.experiment_name:
            self.data_cfg.experiment_name = self.cfg.trainer.experiment_name
            LOGGER.info(f"Using experiment_name '{self.data_cfg.experiment_name}' for results")

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

        # Resolve the device now (prefer CUDA when available) so every downstream
        # consumer sees a concrete device. "auto"/"mps"/"cuda" -> cuda if present.
        resolved = resolve_device(self.trainer_cfg.device)
        if resolved != self.trainer_cfg.device:
            LOGGER.info("Resolved device '%s' -> '%s'", self.trainer_cfg.device, resolved)
        self.trainer_cfg.device = resolved

        # Consolidate every run output under results/<experiment_name>/ so the run
        # is self-contained and nothing is written to the project root. Without an
        # experiment name, outputs go directly under results/.
        self.results_dir: Path = create_results_directory(self.trainer_cfg.experiment_name)

        # Save a copy of the exact config used for this run next to the results.
        self._save_run_config(self.results_dir)

    def _save_run_config(self, results_dir: Path) -> None:
        """Copy the source YAML config into the results dir for reproducibility."""
        src = getattr(self.cfg, "source_path", None)
        if not src:
            return
        try:
            src = Path(src)
            if src.exists():
                dst = results_dir / "config.yaml"
                shutil.copyfile(src, dst)
                LOGGER.info("Saved run config to %s", dst)
        except Exception as e:
            LOGGER.warning("Failed to save run config: %s", e)

    def configure_reproducibility(self) -> None:
        """Configure deterministic behavior for reproducibility."""
        seed = self.seed  # resolved in configure_data
        LOGGER.info(f"Setting global seed to {seed}")
        set_deterministic(seed=seed)
