import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from ppl.utils.modelling_configs.data_loader_config import DataLoaderConfig
from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.modelling_configs.trainer_config import TrainerConfig
from ppl.utils.modelling_configs.config_registry import ConfigRegistry

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class PipelineConfig:
    """Main pipeline configuration with use case support."""
    
    # Core configuration sections
    data: Optional[DataLoaderConfig] = None
    model: Optional[ModelBuilderConfig] = None
    trainer: Optional[TrainerConfig] = None
    
    # Use case support
    use_case: Optional[str] = None
    
    def __post_init__(self):
        """Initialize use case-specific configurations."""
        if self.use_case:
            self._setup_use_case_configs()
    
    def _setup_use_case_configs(self):
        """Set up use case-specific configurations."""
        # Create configurations using the registry
        if not self.data:
            # We need to extract csv_path from the raw YAML data
            # This is a temporary workaround until we have a better solution
            try:
                # Try to get the path from the current working directory
                import os
                from pathlib import Path
                default_csv_path = Path(os.getcwd()) / "ppl/data/trypsin_usrcat.csv"
                self.data = ConfigRegistry.create_config(self.use_case, "data_loader", csv_path=default_csv_path)
                LOGGER.info(f"Using default csv_path: {default_csv_path}")
            except Exception as e:
                LOGGER.error(f"Failed to create data loader config: {e}")
                raise ValueError(f"csv_path is required for data loader configuration") from e
        
        if not self.model:
            self.model = ConfigRegistry.create_config(self.use_case, "model_builder")
        
        if not self.trainer:
            self.trainer = ConfigRegistry.create_config(self.use_case, "model_trainer")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file with validation.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file

        Returns
        -------
        PipelineConfig
            A validated pipeline configuration object

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist
        ValueError
            If the configuration is invalid or missing required fields
        yaml.YAMLError
            If the YAML file is malformed
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path_obj) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            LOGGER.error(f"Failed to parse YAML file {path}: {e}")
            raise

        # Check if use_case is specified
        use_case = raw.get("use_case")
        
        # If use_case is specified, we don't require all sections
        if use_case:
            # Create a config with the use case
            config = cls(use_case=use_case)
            
            # If specific sections are provided, override the use case defaults
            if "data" in raw:
                config.data = cls._create_data_config(raw["data"])
            if "model" in raw:
                config.model = cls._create_model_config(raw["model"])
            if "trainer" in raw:
                config.trainer = cls._create_trainer_config(raw["trainer"])
                
            # Validate inter-dependent configuration parameters
            cls._validate_config_compatibility(config.data, config.model, config.trainer)
            
            return config
        else:
            # Traditional approach - validate required top-level sections
            required_sections = ["data", "model", "trainer"]
            missing_sections = [section for section in required_sections if section not in raw]
            if missing_sections:
                raise ValueError(f"Missing required configuration sections: {missing_sections}")

            # If an `optim:` block exists at the top level, move it under `model`
            if "optim" in raw:
                raw.setdefault("model", {})["optim"] = raw.pop("optim")

            try:
                # Create configuration objects with validation
                data_config = cls._create_data_config(raw["data"])
                model_config = cls._create_model_config(raw["model"])
                trainer_config = cls._create_trainer_config(raw["trainer"])

                # Validate inter-dependent configuration parameters
                cls._validate_config_compatibility(data_config, model_config, trainer_config)

                return cls(
                    data=data_config,
                    model=model_config,
                    trainer=trainer_config,
                )
            except (ValueError, TypeError, KeyError) as e:
                LOGGER.error(f"Invalid configuration in {path}: {e}")
                raise ValueError(f"Configuration validation failed: {e}") from e
                
    @classmethod
    def from_use_case(cls, use_case: str) -> "PipelineConfig":
        """Create configuration directly from use case.
        
        Parameters
        ----------
        use_case : str
            Use case identifier
            
        Returns
        -------
        PipelineConfig
            A pipeline configuration object for the specified use case
        """
        return cls(use_case=use_case)

    @staticmethod
    def _create_data_config(data_config: Dict[str, Any]) -> DataLoaderConfig:
        """Create and validate DataLoaderConfig."""
        try:
            return DataLoaderConfig(**data_config)
        except TypeError as e:
            raise ValueError(f"Invalid data configuration: {e}")

    @staticmethod
    def _create_model_config(model_config: Dict[str, Any]) -> ModelBuilderConfig:
        """Create and validate ModelBuilderConfig."""
        try:
            return ModelBuilderConfig(**model_config)
        except TypeError as e:
            raise ValueError(f"Invalid model configuration: {e}")

    @staticmethod
    def _create_trainer_config(trainer_config: Dict[str, Any]) -> TrainerConfig:
        """Create and validate TrainerConfig."""
        try:
            return TrainerConfig(**trainer_config)
        except TypeError as e:
            raise ValueError(f"Invalid trainer configuration: {e}")

    @staticmethod
    def _validate_config_compatibility(
        data_config: DataLoaderConfig, 
        model_config: ModelBuilderConfig, 
        trainer_config: TrainerConfig
    ) -> None:
        """Validate compatibility between different configuration components."""
        # Ensure task is consistent across configurations
        if data_config.task != model_config.task:
            raise ValueError(
                f"Task mismatch between data config ({data_config.task}) "
                f"and model config ({model_config.task})"
            )

        # Validate batch size against available memory (simple heuristic)
        if data_config.batch_size > 512:
            LOGGER.warning(
                f"Batch size ({data_config.batch_size}) is unusually large and may cause OOM errors"
            )

        # Validate num_workers against available CPU cores
        import os
        cpu_count = os.cpu_count() or 4
        if data_config.num_workers > cpu_count:
            LOGGER.warning(
                f"num_workers ({data_config.num_workers}) exceeds available CPU cores ({cpu_count})"
            )
