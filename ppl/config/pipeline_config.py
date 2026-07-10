import logging
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from ppl.config.data_loader_config import DataLoaderConfig
from ppl.config.model_builder_config import ModelBuilderConfig
from ppl.config.trainer_config import TrainerConfig
from ppl.config.trainer_optim_config import TrainerOptimConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=False)
class PipelineConfig:
    """Top-level pipeline configuration: ``data``, ``model``, and ``trainer``."""

    data: DataLoaderConfig
    model: ModelBuilderConfig
    trainer: TrainerConfig
    # Path to the YAML this config was loaded from (copied into the results dir).
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate configuration from a YAML file.

        The file must define ``data``, ``model``, and ``trainer`` sections. An
        optional top-level ``optim`` block is folded into ``model``.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        PipelineConfig
            A validated pipeline configuration object.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the configuration is invalid or missing required sections.
        yaml.YAMLError
            If the YAML file is malformed.
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

        if not isinstance(raw, dict):
            raise ValueError(f"Configuration file {path} must contain a top-level mapping")

        # If an `optim:` block exists at the top level, move it under `model`.
        if "optim" in raw:
            raw.setdefault("model", {})["optim"] = raw.pop("optim")

        required_sections = ["data", "model", "trainer"]
        missing_sections = [s for s in required_sections if s not in raw]
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")

        try:
            data_config = cls._create_data_config(raw["data"])
            model_config = cls._create_model_config(raw["model"])
            trainer_config = cls._create_trainer_config(raw["trainer"])

            cls._validate_config_compatibility(data_config, model_config, trainer_config)

            return cls(
                data=data_config,
                model=model_config,
                trainer=trainer_config,
                source_path=path_obj,
            )
        except (ValueError, TypeError, KeyError) as e:
            LOGGER.error(f"Invalid configuration in {path}: {e}")
            raise ValueError(f"Configuration validation failed: {e}") from e

    @staticmethod
    def _build_section(config_cls, section: Dict[str, Any], label: str):
        """Construct a config dataclass from a YAML section with a clear error."""
        try:
            return config_cls(**section)
        except TypeError as e:
            raise ValueError(f"Invalid {label} configuration: {e}")

    @classmethod
    def _create_data_config(cls, data_config: Dict[str, Any]) -> DataLoaderConfig:
        return cls._build_section(DataLoaderConfig, data_config, "data")

    @classmethod
    def _create_model_config(cls, model_config: Dict[str, Any]) -> ModelBuilderConfig:
        # Coerce a raw `optim` dict into TrainerOptimConfig so `.model.optim` is a
        # real dataclass, not a dict. Keep only known fields so unknown/typo keys
        # stay tolerated (they are dropped-with-warning later by override_dataclass).
        opt = model_config.get("optim")
        if isinstance(opt, dict):
            valid = {f.name for f in dc_fields(TrainerOptimConfig)}
            model_config = {
                **model_config,
                "optim": TrainerOptimConfig(**{k: v for k, v in opt.items() if k in valid}),
            }
        return cls._build_section(ModelBuilderConfig, model_config, "model")

    @classmethod
    def _create_trainer_config(cls, trainer_config: Dict[str, Any]) -> TrainerConfig:
        return cls._build_section(TrainerConfig, trainer_config, "trainer")

    @staticmethod
    def _validate_config_compatibility(
        data_config: DataLoaderConfig,
        model_config: ModelBuilderConfig,
        trainer_config: TrainerConfig,
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
