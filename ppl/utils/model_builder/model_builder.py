from __future__ import annotations

import dataclasses as dc
import logging
import pprint
import traceback

import pytorch_lightning as pl

from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.model_builder.model_impl import ModelFactory
from ppl.utils.model_builder.template_registry import Template, get_template

LOGGER = logging.getLogger(__name__)


# MIL Model Builder
class ModelBuilder:
    """High-level factory for creating MIL models.

    This class provides an interface for building MIL models using templates.
    """

    def __init__(
        self, 
        cfg: ModelBuilderConfig | None = None, 
        template: Template | str | None = None,
        **legacy_kwargs
    ):
        """Initialize the model builder with configuration and template.

        Parameters
        ----------
        cfg : ModelBuilderConfig, optional
            Model configuration
        template : Template | str, optional
            Model template to use. If provided, overrides template in cfg.
        **legacy_kwargs
            Legacy keyword arguments for backward compatibility

        Raises
        ------
        ValueError
            If both cfg and legacy_kwargs are provided or if task is invalid
        """
        # Initialize configuration
        if cfg is None:
            try:
                cfg = ModelBuilderConfig().update(**legacy_kwargs)
                LOGGER.info(
                    "[MODEL] Building model from legacy kwargs:\n%s", 
                    pprint.pformat(dc.asdict(cfg))
                )
            except Exception as e:
                LOGGER.error(f"Failed to create ModelBuilderConfig from legacy kwargs: {e}")
                LOGGER.error(traceback.format_exc())
                raise ValueError(f"Invalid model configuration: {e}") from e
        elif legacy_kwargs:
            raise ValueError("Pass either cfg or individual kwargs, not both.")
        else:
            LOGGER.info(
                "[MODEL] Building model from config:\n%s", 
                pprint.pformat(dc.asdict(cfg))
            )

        # Handle explicit template parameter
        if template is not None:
            if isinstance(template, str):
                # Convert string to Template enum
                template_enum = None
                for t in Template:
                    if t.value == template:
                        template_enum = t
                        break
                if template_enum is None:
                    raise ValueError(f"Unknown template: {template}")
                template = template_enum
            
            # Override template in config
            cfg = dc.replace(cfg, template=template)
            LOGGER.info(f"[MODEL] Using explicitly provided template: {template.value}")

        self.cfg = cfg
        self.template = cfg.template
        self.input_dim = cfg.input_dim
        self.task = cfg.task.lower()

        # Validate task
        if self.task not in ["classification", "regression"]:
            LOGGER.error(f"Invalid task: {self.task}. Must be 'classification' or 'regression'")
            raise ValueError(f"Invalid task: {self.task}. Must be 'classification' or 'regression'")

        # Validate template compatibility
        try:
            template_spec = get_template(self.template.value)
            template_def = template_spec.get_template_definition()
            if self.task not in template_def.task_compatibility:
                raise ValueError(
                    f"Template '{self.template.value}' does not support task '{self.task}'. "
                    f"Supported tasks: {template_def.task_compatibility}"
                )
        except Exception as e:
            LOGGER.error(f"Template validation failed: {e}")
            raise ValueError(f"Invalid template configuration: {e}") from e

        # Log initialization
        LOGGER.info(f"[MODEL] Initialized ModelBuilder with template: {self.template.value}")
        ModelFactory.log_memory_usage("ModelBuilder_init")

    def build(self) -> pl.LightningModule:
        """Build the model using the factory and template.

        Returns
        -------
        pl.LightningModule
            Lightning module ready for training

        Raises
        ------
        ValueError
            If input_dim is None or model components are incompatible
        RuntimeError
            If model construction fails
        """
        # Validate input dimension
        if self.input_dim is None:
            raise ValueError(
                "input_dim is None – ensure PipelineOrchestrator injects the number of descriptors "
                "before calling ModelBuilder.build()."
            )

        LOGGER.info(f"[MODEL] Building model with template: {self.template.value}")
        
        # Use the factory to build the model
        return ModelFactory.build_model(self.cfg, self.input_dim, self.task)

    @classmethod
    def from_template(
        cls,
        template: Template | str,
        embedder_type: str = "mlp_embedder",
        aggregator_type: str = "mha_att",
        predictor_type: str = "mlp_predictor",
        task: str = "regression",
        **kwargs
    ) -> "ModelBuilder":
        """Create a ModelBuilder directly from a template.

        Parameters
        ----------
        template : Template | str
            Model template to use
        embedder_type : str, default="mlp_embedder"
            Type of embedder to use
        aggregator_type : str, default="mha_att"
            Type of aggregator to use
        predictor_type : str, default="mlp_predictor"
            Type of predictor to use
        task : str, default="regression"
            Task type
        **kwargs
            Additional configuration parameters

        Returns
        -------
        ModelBuilder
            Configured model builder instance

        Example
        -------
        >>> builder = ModelBuilder.from_template(
        ...     Template.BAG_ATTENTION,
        ...     embedder_type="mlp_embedder",
        ...     task="regression"
        ... )
        """
        # Convert string to Template enum if needed
        if isinstance(template, str):
            template_enum = None
            for t in Template:
                if t.value == template:
                    template_enum = t
                    break
            if template_enum is None:
                raise ValueError(f"Unknown template: {template}")
            template = template_enum

        # Create config with template
        cfg = ModelBuilderConfig(
            template=template,
            embedder_type=embedder_type,
            aggregator_type=aggregator_type,
            predictor_type=predictor_type,
            task=task,
            **kwargs
        )

        return cls(cfg=cfg)

    def get_template_info(self) -> dict:
        """Get information about the current template.

        Returns
        -------
        dict
            Template information including name, components, and compatibility
        """
        try:
            template_spec = get_template(self.template.value)
            template_def = template_spec.get_template_definition()
            
            return {
                "name": template_def.name,
                "description": template_def.description,
                "components": template_def.component_order,
                "task_compatibility": template_def.task_compatibility,
                "current_config": {
                    "embedder_type": self.cfg.embedder_type,
                    "aggregator_type": self.cfg.aggregator_type,
                    "predictor_type": self.cfg.predictor_type,
                    "task": self.task
                }
            }
        except Exception as e:
            LOGGER.error(f"Failed to get template info: {e}")
            return {"error": str(e)}
