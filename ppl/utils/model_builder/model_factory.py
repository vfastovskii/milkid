from __future__ import annotations

import gc
import logging
import os
import pprint
import traceback

import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ppl.utils.model_builder.mil_core import MILCore
from ppl.utils.model_builder.mil_lightning_wrapper import MILModelLightningWrapper
from ppl.utils.model_builder.template_registry import get_template

LOGGER = logging.getLogger(__name__)

# Model Factory for component creation and validation
class ModelFactory:
    """Factory for creating and validating model components."""

    @staticmethod
    def log_memory_usage(step: str) -> None:
        """Log current memory usage.

        Parameters
        ----------
        step : str
            Current step name for logging
        """
        if torch.cuda.is_available():
            # GPU memory
            current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            LOGGER.info(f"[{step}] GPU Memory: Current={current_gpu_memory:.2f}GB, Peak={max_gpu_memory:.2f}GB")

        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory_gb = process.memory_info().rss / (1024**3)
        LOGGER.info(f"[{step}] CPU Memory: {cpu_memory_gb:.2f}GB")

    @staticmethod
    def _log_tensor_initialization(module: nn.Module, component_name: str) -> None:
        """Log tensor initialization information for a module.

        Parameters
        ----------
        module : nn.Module
            The module to log tensor information for
        component_name : str
            Name of the component (for logging)
        """
        LOGGER.info(f"[TENSOR_INIT] {component_name} tensor initialization details:")
        total_params = 0
        for name, param in module.named_parameters():
            shape_str = 'x'.join(str(dim) for dim in param.shape)
            mem_mb = param.nelement() * param.element_size() / (1024 * 1024)
            total_params += param.nelement()

            # Get initialization info if available
            init_info = "unknown"
            if hasattr(param, '_init_method'):
                init_info = param._init_method

            LOGGER.info(f"[TENSOR_INIT] - {name}: shape={shape_str}, dtype={param.dtype}, "
                       f"memory={mem_mb:.2f}MB, init={init_info}")

        LOGGER.info(f"[TENSOR_INIT] {component_name} total parameters: {total_params:,}")

        # Log buffers (non-parameter tensors like running_mean in BatchNorm)
        for name, buffer in module.named_buffers():
            if buffer is not None:
                shape_str = 'x'.join(str(dim) for dim in buffer.shape)
                mem_mb = buffer.nelement() * buffer.element_size() / (1024 * 1024)
                LOGGER.info(f"[TENSOR_INIT] - Buffer {name}: shape={shape_str}, "
                           f"dtype={buffer.dtype}, memory={mem_mb:.2f}MB")

    @staticmethod
    def validate_model_architecture(components, task: str, cfg=None, template_name=None, component_names=None) -> None:
        """Validate the model architecture for compatibility.

        This method performs a thorough validation of the model architecture,
        ensuring that all components have the required dimension attributes
        and that the dimensions are compatible between components.

        Parameters
        ----------
        components : tuple
            Tuple of model components
        task : str
            Task type (classification or regression)
        cfg : ModelBuilderConfig, optional
            Model configuration, by default None
        template_name : str, optional
            Name of the template to use for validation, by default None.
            If None, will try to extract from cfg.template, and if not available,
            will default to "bag_attention"
        component_names : list, optional
            Names of the components, by default None.
            If None, will try to determine from the template definition.

        Raises
        ------
        ValueError
            If the model architecture is invalid
        """
        LOGGER.info("[MODEL] Validating model architecture...")

        # Determine template name to use
        if template_name is None:
            if cfg is not None and hasattr(cfg, "template"):
                template_name = cfg.template.value
                template_display_name = cfg.template.name
            else:
                # Default to bag_attention template if none specified
                template_name = "bag_attention"
                template_display_name = "BAG_ATTENTION"
        else:
            template_display_name = template_name.upper()

        # Get template specification
        try:
            spec = get_template(template_name)
            template_def = spec.get_template_definition()
        except Exception as e:
            LOGGER.error(f"Failed to get template definition: {e}")
            raise ValueError(f"Failed to get template definition: {e}") from e

        # Determine component names if not provided
        if component_names is None:
            component_names = template_def.component_order

        # Log component information
        LOGGER.info(f"[MODEL] Template: {template_display_name}")
        LOGGER.info(f"[MODEL] Component structure: {component_names}")
        
        # Log the dimensions of each component
        for i, (component, name) in enumerate(zip(components, component_names)):
            if hasattr(component, "input_dim") and hasattr(component, "output_dim"):
                LOGGER.info(f"[MODEL] {name.capitalize()}: input_dim={component.input_dim}, output_dim={component.output_dim}")
            elif hasattr(component, "input_dim"):
                LOGGER.info(f"[MODEL] {name.capitalize()}: input_dim={component.input_dim}")
            elif hasattr(component, "output_dim"):
                LOGGER.info(f"[MODEL] {name.capitalize()}: output_dim={component.output_dim}")

        # Check dimension compatibility between adjacent components
        for i in range(len(components) - 1):
            current = components[i]
            next_comp = components[i + 1]
            
            if hasattr(current, "output_dim") and hasattr(next_comp, "input_dim"):
                if current.output_dim != next_comp.input_dim:
                    LOGGER.warning(
                        f"Dimension mismatch: {component_names[i]}.output_dim ({current.output_dim}) "
                        f"!= {component_names[i+1]}.input_dim ({next_comp.input_dim}). "
                        f"Automatically adjusting {component_names[i+1]}.input_dim to match."
                    )
                    # Set next component's input_dim to match current component's output_dim
                    next_comp.input_dim = current.output_dim
                    
                    # Update out_dim for next component if it's the same as input_dim
                    if hasattr(next_comp, "out_dim") and hasattr(next_comp, "_input_dim") and next_comp.out_dim == next_comp._input_dim:
                        next_comp.out_dim = current.output_dim
                    
                    # Log the updated dimensions
                    LOGGER.info(f"[MODEL] Updated {component_names[i+1].capitalize()}: input_dim={next_comp.input_dim}, output_dim={next_comp.output_dim}")

        # Check predictor output dimension for task compatibility
        predictor = components[-1]  # Assume last component is the predictor
        if hasattr(predictor, "output_dim"):
            if task == "classification" and predictor.output_dim != 1:
                LOGGER.warning(
                    f"Predictor output_dim ({predictor.output_dim}) != 1 for classification task. "
                    f"This may cause issues with the loss function."
                )
            elif task == "regression" and predictor.output_dim != 1:
                LOGGER.warning(
                    f"Predictor output_dim ({predictor.output_dim}) != 1 for regression task. "
                    f"This may cause issues with the loss function."
                )

        # Perform template-specific validation
        try:
            spec.validate_architecture(components, task)
            LOGGER.info(f"[MODEL] Template-specific validation for {template_display_name} passed ✓")
        except Exception as e:
            LOGGER.error(f"Template-specific validation failed: {e}")
            raise ValueError(f"Template-specific validation failed: {e}") from e

        LOGGER.info("[MODEL] Model architecture validation passed ✓")

    @staticmethod
    def ensure_dimension_compatibility(cfg, input_dim: int):
        """Ensure dimension compatibility between model components.

        This high-level method explicitly sets and validates the dimensions
        between embedder, aggregator, and predictor components.

        Parameters
        ----------
        cfg : ModelBuilderConfig
            Model configuration
        input_dim : int
            Input dimension for the embedder

        Returns
        -------
        dict
            Dictionary with validated dimensions for each component
        """
        # Start with the input dimension for the embedder
        dimensions = {
            "embedder_input_dim": input_dim
        }

        # Get embedder output dimension from config or use default
        if hasattr(cfg, "embedder_output_dim") and cfg.embedder_output_dim is not None:
            embedder_output_dim = cfg.embedder_output_dim
        elif hasattr(cfg, "embedder_kwargs") and "hidden_dims" in cfg.embedder_kwargs:
            # If hidden_dims is specified, use the last dimension as output
            hidden_dims = cfg.embedder_kwargs["hidden_dims"]
            if hidden_dims and isinstance(hidden_dims, list):
                embedder_output_dim = hidden_dims[-1]
            else:
                # Default for MLPEmbedder
                embedder_output_dim = 32
        else:
            # Default value
            embedder_output_dim = 32

        dimensions["embedder_output_dim"] = embedder_output_dim

        # Aggregator input dimension must match embedder output dimension
        dimensions["aggregator_input_dim"] = embedder_output_dim

        # Get aggregator output dimension (usually same as input for attention-based aggregators)
        if hasattr(cfg, "aggregator_output_dim") and cfg.aggregator_output_dim is not None:
            aggregator_output_dim = cfg.aggregator_output_dim
        else:
            # For most aggregators, output_dim = input_dim
            aggregator_output_dim = embedder_output_dim

        dimensions["aggregator_output_dim"] = aggregator_output_dim

        # Predictor input dimension must match aggregator output dimension
        dimensions["predictor_input_dim"] = aggregator_output_dim

        # Get predictor output dimension (usually 1 for classification/regression)
        if hasattr(cfg, "predictor_output_dim") and cfg.predictor_output_dim is not None:
            predictor_output_dim = cfg.predictor_output_dim
        else:
            # Default is 1 for both classification and regression
            predictor_output_dim = 1

        dimensions["predictor_output_dim"] = predictor_output_dim

        # Log the dimension chain for clarity
        LOGGER.info("[MODEL] Dimension chain explicitly set:")
        LOGGER.info(f"[MODEL] Embedder: input_dim={dimensions['embedder_input_dim']} → output_dim={dimensions['embedder_output_dim']}")
        LOGGER.info(f"[MODEL] Aggregator: input_dim={dimensions['aggregator_input_dim']} → output_dim={dimensions['aggregator_output_dim']}")
        LOGGER.info(f"[MODEL] Predictor: input_dim={dimensions['predictor_input_dim']} → output_dim={dimensions['predictor_output_dim']}")

        return dimensions

    @staticmethod
    def create_model_components(cfg, input_dim: int):
        """Create model components based on configuration.

        Parameters
        ----------
        cfg : ModelBuilderConfig
            Model configuration
        input_dim : int
            Input dimension for the first component

        Returns
        -------
        tuple
            Tuple of model components according to the template specification

        Raises
        ------
        ValueError
            If component types are invalid
        RuntimeError
            If component initialization fails
        """
        # Get the template specification from the registry
        try:
            LOGGER.info(f"[MODEL] Using template: {cfg.template.name}")
            spec = get_template(cfg.template.value)
            components = spec.build_components(cfg, input_dim)
            template_def = spec.get_template_definition()
            LOGGER.info(f"[MODEL] Created components: {template_def.component_order}")
            return components
        except Exception as e:
            LOGGER.error(f"Failed to create model components: {e}")
            LOGGER.error(traceback.format_exc())
            raise RuntimeError(f"Model component creation failed: {e}") from e

    @staticmethod
    def build_model(cfg, input_dim: int, task: str) -> pl.LightningModule:
        """Build the complete model with memory management.

        Parameters
        ----------
        cfg : ModelBuilderConfig
            Model configuration
        input_dim : int
            Input dimension for the first component
        task : str
            Task type (classification or regression)

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
        ModelFactory.log_memory_usage("build_start")

        try:
            # Create components
            components = ModelFactory.create_model_components(cfg, input_dim)

            # Get template definition to determine component names
            spec = get_template(cfg.template.value)
            template_def = spec.get_template_definition()
            component_names = template_def.component_order

            # Validate model architecture
            ModelFactory.validate_model_architecture(components, task, cfg, component_names=component_names)

            # Compose MIL model
            LOGGER.info("[MODEL] Creating core MIL model")
            core = MILCore(
                components,
                component_names,
                task=task,
                active_prototype_kwargs=getattr(
                    cfg, "active_prototype_kwargs", {}
                ),
            )

            # Log original model initialization
            LOGGER.info("[MODEL_INIT] Original model initialization details:")
            LOGGER.info(f"[MODEL_INIT] Task: {task}")
            LOGGER.info(f"[MODEL_INIT] Core model type: {core.__class__.__name__}")
            LOGGER.info(f"[MODEL_INIT] Template: {cfg.template.name}")
            LOGGER.info(f"[MODEL_INIT] Component structure: {component_names}")
            
            # Log component details
            for i, (component, name) in enumerate(zip(components, component_names)):
                LOGGER.info(f"[MODEL_INIT] {name.capitalize()}: {component.__class__.__name__}")

            # Log model architecture details
            total_params = sum(p.numel() for p in core.parameters())
            trainable_params = sum(p.numel() for p in core.parameters() if p.requires_grad)
            LOGGER.info(f"[MODEL_INIT] Total parameters: {total_params:,}")
            LOGGER.info(f"[MODEL_INIT] Trainable parameters: {trainable_params:,}")
            LOGGER.info(f"[MODEL_INIT] Non-trainable parameters: {total_params - trainable_params:,}")

            LOGGER.info("[MODEL] Creating Lightning wrapper")
            model = MILModelLightningWrapper(
                core, task=task, optim_cfg=cfg.optim
            )

            # Log Lightning model initialization
            LOGGER.info("[MODEL_INIT] Lightning model wrapper details:")
            LOGGER.info(f"[MODEL_INIT] Lightning model type: {model.__class__.__name__}")

            # Log optimizer configuration
            LOGGER.info(f"[MODEL_INIT] Learning rate: {cfg.optim.lr}")
            LOGGER.info(f"[MODEL_INIT] Weight decay: {cfg.optim.weight_decay}")

            # Log final model structure summary
            LOGGER.info("[MODEL_INIT] Final model structure summary:")
            model_info = {}
            for name, module in model.named_modules():
                if name:  # Skip the root module
                    module_type = module.__class__.__name__
                    if module_type not in model_info:
                        model_info[module_type] = 0
                    model_info[module_type] += 1

            for module_type, count in model_info.items():
                LOGGER.info(f"[MODEL_INIT] - {module_type}: {count} instances")

            # Log model configuration
            LOGGER.info("[MODEL] Constructed model configuration:")
            LOGGER.info("[MODEL] Input dim: %d", input_dim)

            # Log component descriptions
            for component, name in zip(components, component_names):
                if hasattr(component, "describe"):
                    LOGGER.info(
                        "[MODEL] %s description:\n%s",
                        name.capitalize(),
                        pprint.pformat(component.describe(), compact=True),
                    )
                else:
                    LOGGER.info(
                        "[MODEL] %s class: %s", name.capitalize(), component.__class__.__name__
                    )

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ModelFactory.log_memory_usage("build_end")
            return model

        except Exception as e:
            LOGGER.error(f"Unexpected error in model building: {e}")
            LOGGER.error(traceback.format_exc())

            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise RuntimeError(f"Model building failed: {e}") from e
