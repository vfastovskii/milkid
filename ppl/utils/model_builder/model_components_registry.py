"""Global plug-in registry for model components with lazy importing."""

import pkgutil
import sys
from collections import defaultdict
from importlib import import_module
from typing import Callable, Dict, Type

# Which package to scan for each kind of component
_KIND_TO_PKG = {
    "embedder_type":   "ppl.utils.model_builder.model_components_factory.embedders",
    "aggregator_type": "ppl.utils.model_builder.model_components_factory.aggregators",
    "predictor_type":  "ppl.utils.model_builder.model_components_factory.predictors",
}

# Registry structure: kind -> name -> class
_REGISTRY: Dict[str, Dict[str, Type]] = defaultdict(dict)

def register(kind: str, name: str) -> Callable[[Type], Type]:
    """Register a component class.

    Example:
        @register("embedder_type", "mlp_embedder")
        class MLPEmbedder: ...
    """
    kind = kind.lower()
    name = name.lower()

    def _decorator(cls):
        _REGISTRY[kind][name] = cls
        return cls
    return _decorator

def _lazy_import(kind: str) -> None:
    """Import all modules in the kind's package to populate the registry."""
    pkg_name = _KIND_TO_PKG.get(kind)
    if not pkg_name or pkg_name in sys.modules:
        return  # already imported or unknown kind

    pkg = import_module(pkg_name)
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        import_module(modname)  # this runs @register on import

def get_component(kind: str, name: str) -> Type:
    """Retrieve a component class by kind and name, lazy-loading if needed."""
    kind = kind.lower()
    name = name.lower()

    cls = _REGISTRY.get(kind, {}).get(name)
    if cls is not None:
        return cls

    _lazy_import(kind)

    cls = _REGISTRY.get(kind, {}).get(name)
    if cls is not None:
        return cls

    raise ValueError(
        f"Unknown {kind} '{name}'. "
        f"Available: {list(_REGISTRY.get(kind, {}))}"
    ) from None

def all(kind: str):
    """Return all registered names for a given kind."""
    return tuple(_REGISTRY[kind.lower()].keys())

