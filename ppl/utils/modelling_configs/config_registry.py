"""Configuration registry for use case-specific configurations."""

from typing import Dict, Type, Any
from ppl.utils.modelling_configs.data_loader_config import DataLoaderConfig
from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.modelling_configs.model_trainer_config import ModelTrainerConfig
from ppl.utils.modelling_configs.trainer_config import TrainerConfig
from ppl.utils.modelling_configs.trainer_optim_config import TrainerOptimConfig

# Import use case-specific configurations
from ppl.utils.modelling_configs.use_case_configs import (
    ConformersDataLoaderConfig,
    ConformersModelBuilderConfig,
    ConformersModelTrainerConfig,
)

# Configuration registry mapping
USE_CASE_CONFIGS: Dict[str, Dict[str, Type[Any]]] = {
    "confs": {
        "data_loader": ConformersDataLoaderConfig,
        "model_builder": ConformersModelBuilderConfig,
        "model_trainer": ConformersModelTrainerConfig,
        "trainer": TrainerConfig,  # Use default
        "trainer_optim": TrainerOptimConfig,  # Use default
    },
    "default": {
        "data_loader": DataLoaderConfig,
        "model_builder": ModelBuilderConfig,
        "model_trainer": ModelTrainerConfig,
        "trainer": TrainerConfig,
        "trainer_optim": TrainerOptimConfig,
    }
}

class ConfigRegistry:
    """Registry for managing use case-specific configurations."""
    
    @staticmethod
    def get_config_classes(use_case: str) -> Dict[str, Type[Any]]:
        """Get configuration classes for a specific use case.
        
        Parameters
        ----------
        use_case : str
            Use case identifier
            
        Returns
        -------
        Dict[str, Type[Any]]
            Dictionary mapping config type to config class
        """
        if use_case not in USE_CASE_CONFIGS:
            available = list(USE_CASE_CONFIGS.keys())
            raise ValueError(f"Unknown use case '{use_case}'. Available: {available}")
        
        return USE_CASE_CONFIGS[use_case]
    
    @staticmethod
    def get_config_class(use_case: str, config_type: str) -> Type[Any]:
        """Get a specific configuration class for a use case.
        
        Parameters
        ----------
        use_case : str
            Use case identifier
        config_type : str
            Configuration type (e.g., 'data_loader', 'model_builder')
            
        Returns
        -------
        Type[Any]
            Configuration class
        """
        config_classes = ConfigRegistry.get_config_classes(use_case)
        
        if config_type not in config_classes:
            available = list(config_classes.keys())
            raise ValueError(f"Unknown config type '{config_type}'. Available: {available}")
        
        return config_classes[config_type]
    
    @staticmethod
    def create_config(use_case: str, config_type: str, **kwargs) -> Any:
        """Create a configuration instance for a use case.
        
        Parameters
        ----------
        use_case : str
            Use case identifier
        config_type : str
            Configuration type
        **kwargs
            Additional configuration parameters
            
        Returns
        -------
        Any
            Configuration instance
        """
        config_class = ConfigRegistry.get_config_class(use_case, config_type)
        return config_class(**kwargs)
    
    @staticmethod
    def list_use_cases() -> list[str]:
        """List all available use cases.
        
        Returns
        -------
        list[str]
            List of use case names
        """
        return [uc for uc in USE_CASE_CONFIGS.keys() if uc != "default"]
    
    @staticmethod
    def list_config_types() -> list[str]:
        """List all available configuration types.
        
        Returns
        -------
        list[str]
            List of configuration types
        """
        return list(USE_CASE_CONFIGS["default"].keys())