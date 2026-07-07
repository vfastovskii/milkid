"""Template registry for model architectures.

This module provides a registry for model templates, which define specific
combinations of embedders, aggregators, and predictors that work well together.

Templates are registered using the @register_template decorator and can be
retrieved using the get_template function.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, Any, List, Optional
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

@dataclass
class ComponentRequirement:
    """Defines requirements for a component in a template."""
    component_type: str  # 'embedder', 'aggregator', 'predictor'
    required_attributes: List[str]  # Required attributes like 'input_dim', 'output_dim'
    allowed_classes: Optional[List[str]] = None  # Allowed component class names
    min_output_dim: Optional[int] = None  # Minimum output dimension
    max_output_dim: Optional[int] = None  # Maximum output dimension
    custom_validators: Optional[List[callable]] = None  # Custom validation functions

@dataclass
class TemplateDefinition:
    """Defines the structure and requirements of a template."""
    name: str
    components: List[ComponentRequirement]
    component_order: List[str]  # Order of components in the pipeline
    task_compatibility: List[str]  # Supported tasks
    description: Optional[str] = None

class Template(enum.Enum):
    """Officially supported model templates."""
    BAG_ATTENTION = "bag_attention"
    AGGREGATOR_ONLY = "aggregator_only"  # Example: only aggregator + predictor
    EMBEDDER_PREDICTOR = "embedder_predictor"  # Example: skip aggregator
    # Add more templates as needed

class TemplateSpec(ABC):
    """Abstract base class for template specifications.
    
    A template specification defines how to build model components
    for a specific architecture template.
    """
    
    @abstractmethod
    def get_template_definition(self) -> TemplateDefinition:
        """Get the template definition that describes the structure and requirements.
        
        Returns
        -------
        TemplateDefinition
            Template definition with components and their requirements
        """
        pass
    
    @abstractmethod
    def build_components(self, cfg: Any, input_dim: int) -> tuple:
        """Build model components according to the template.
        
        Parameters
        ----------
        cfg : Any
            Model configuration
        input_dim : int
            Input dimension for the embedder
            
        Returns
        -------
        tuple
            Tuple of built components in the order specified by component_order
        """
        pass
    
    def validate_architecture(self, components: tuple, task: str) -> None:
        """Validate that the components satisfy template-specific requirements.
        
        Parameters
        ----------
        components : tuple
            Tuple of components in the order specified by component_order
        task : str
            Task type (e.g., "regression", "classification")
            
        Raises
        ------
        ValueError
            If the components don't satisfy template requirements
        """
        template_def = self.get_template_definition()
        
        # Check if task is supported
        if task not in template_def.task_compatibility:
            raise ValueError(
                f"Task '{task}' is not supported by template '{template_def.name}'. "
                f"Supported tasks: {template_def.task_compatibility}"
            )
        
        # Check if we have the right number of components
        if len(components) != len(template_def.components):
            raise ValueError(
                f"Template '{template_def.name}' expects {len(template_def.components)} "
                f"components, got {len(components)}"
            )
        
        # Validate each component
        for i, (component, requirement) in enumerate(zip(components, template_def.components)):
            self._validate_component(component, requirement, f"Component {i} ({requirement.component_type})")
        
        # Validate component chain (dimension compatibility)
        self._validate_component_chain(components, template_def)
    
    def _validate_component(self, component: Any, requirement: ComponentRequirement, component_name: str) -> None:
        """Validate a single component against its requirements.
        
        Parameters
        ----------
        component : Any
            Component to validate
        requirement : ComponentRequirement
            Requirements for the component
        component_name : str
            Name of the component for error messages
        """
        # Check required attributes
        for attr in requirement.required_attributes:
            if not hasattr(component, attr):
                raise ValueError(f"{component_name} must have '{attr}' attribute")
        
        # Check allowed classes
        if requirement.allowed_classes:
            class_name = component.__class__.__name__
            if class_name not in requirement.allowed_classes:
                raise ValueError(
                    f"{component_name} must be one of {requirement.allowed_classes}, "
                    f"got {class_name}"
                )
        
        # Check output dimension constraints
        if hasattr(component, 'output_dim'):
            output_dim = component.output_dim
            if requirement.min_output_dim and output_dim < requirement.min_output_dim:
                raise ValueError(
                    f"{component_name} output_dim ({output_dim}) must be >= "
                    f"{requirement.min_output_dim}"
                )
            if requirement.max_output_dim and output_dim > requirement.max_output_dim:
                raise ValueError(
                    f"{component_name} output_dim ({output_dim}) must be <= "
                    f"{requirement.max_output_dim}"
                )
        
        # Run custom validators
        if requirement.custom_validators:
            for validator in requirement.custom_validators:
                validator(component, component_name)
    
    def _validate_component_chain(self, components: tuple, template_def: TemplateDefinition) -> None:
        """Validate dimension compatibility between components.
        
        Parameters
        ----------
        components : tuple
            Tuple of components
        template_def : TemplateDefinition
            Template definition
        """
        # Check dimension chain for adjacent components
        for i in range(len(components) - 1):
            current_component = components[i]
            next_component = components[i + 1]
            
            current_name = template_def.component_order[i]
            next_name = template_def.component_order[i + 1]
            
            # Check if both components have the required dimension attributes
            if hasattr(current_component, 'output_dim') and hasattr(next_component, 'input_dim'):
                if current_component.output_dim != next_component.input_dim:
                    raise ValueError(
                        f"Dimension mismatch between {current_name} and {next_name}: "
                        f"{current_name}.output_dim ({current_component.output_dim}) != "
                        f"{next_name}.input_dim ({next_component.input_dim})"
                    )

# Registry for template specifications
_TEMPLATE_REGISTRY: Dict[str, Type[TemplateSpec]] = {}

def register_template(template: Template):
    """Register a template specification class.
    
    Example:
        @register_template(Template.BAG_ATTENTION)
        class BagAttentionSpec(TemplateSpec):
            ...
    """
    def _decorator(cls):
        if not issubclass(cls, TemplateSpec):
            raise TypeError(f"Template spec must inherit from TemplateSpec, got {cls}")
        
        _TEMPLATE_REGISTRY[template.value] = cls
        LOGGER.debug(f"Registered template: {template.name}")
        return cls
    
    return _decorator

def get_template(template_name: str) -> TemplateSpec:
    """Get a template specification by name.
    
    Parameters
    ----------
    template_name : str
        Name of the template
        
    Returns
    -------
    TemplateSpec
        Template specification instance
        
    Raises
    ------
    ValueError
        If the template is not found
    """
    if template_name not in _TEMPLATE_REGISTRY:
        available = list(_TEMPLATE_REGISTRY.keys())
        raise ValueError(
            f"Unknown template '{template_name}'. "
            f"Available: {available}"
        )
    
    return _TEMPLATE_REGISTRY[template_name]()

def all_templates() -> list[str]:
    """Get all registered template names.
    
    Returns
    -------
    list[str]
        List of template names
    """
    return list(_TEMPLATE_REGISTRY.keys())