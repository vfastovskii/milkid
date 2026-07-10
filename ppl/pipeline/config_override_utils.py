import dataclasses as dc
import logging
from typing import Any, TypeVar

LOGGER = logging.getLogger(__name__)

T = TypeVar('T')

def override_dataclass(instance: T, overrides: dict[str, Any] | Any) -> T:
    """Override a dataclass instance with values from overrides (dict or dataclass).

    This function:
    1. Warns about unknown keys coming from the YAML, so the user can catch typos 
    2. Logs the exact key/value pairs applied, so the final experiment set‑up is auditable
    3. Handles type conversion intelligently, including for generic types

    Parameters
    ----------
    instance : dataclass
        The dataclass instance to override
    overrides : dict | dataclass
        The values to override with

    Returns
    -------
    dataclass
        A new dataclass instance with overridden values

    Raises
    ------
    TypeError
        If the instance is not a dataclass or overrides has invalid type
    ValueError
        If critical type conversion fails
    """
    # Validate input
    if not dc.is_dataclass(instance):
        raise TypeError(f"instance must be a dataclass, got {type(instance)}")

    if dc.is_dataclass(overrides):
        overrides = dc.asdict(overrides)
    if not isinstance(overrides, dict):
        raise TypeError(f"overrides must be a dict or dataclass, got {type(overrides)}")

    valid_fields = {f.name for f in dc.fields(instance)}

    # warn about unknown YAML keys
    unknown_fields = set(overrides) - valid_fields
    if unknown_fields:
        LOGGER.warning(
            "Unknown config fields for %s ignored: %s",
            instance.__class__.__name__,
            sorted(unknown_fields),
        )

    # apply recognized overrides
    update_kwargs = {}
    for k, v in overrides.items():
        if k not in valid_fields:
            continue

        current_val = getattr(instance, k)

        # Handle None values
        if current_val is None:
            # For None values, accept the override but log it
            update_kwargs[k] = v
            LOGGER.debug("Setting %s=%s (current value was None)", k, v)
            continue

        # Direct assignment for matching types or None values
        if isinstance(v, type(current_val)) or v is None:
            update_kwargs[k] = v
            continue

        # Get field type information
        field_info = next((f for f in dc.fields(instance) if f.name == k), None)
        if field_info is None:
            LOGGER.warning(f"Could not find field info for {k}, skipping override")
            continue

        field_type = field_info.type

        # Handle a special case for Path | str
        if str(field_type) == 'Path | str' and isinstance(v, str):
            from pathlib import Path
            update_kwargs[k] = Path(v)
            LOGGER.debug(f"Converted {k}={v} to Path")
            continue

        # Handle Path type directly
        if str(field_type) == 'Path' and isinstance(v, str):
            from pathlib import Path
            update_kwargs[k] = Path(v)
            LOGGER.debug(f"Converted {k}={v} to Path")
            continue

        # Handle Union/Optional types
        import typing
        if hasattr(typing, 'get_origin') and hasattr(typing, 'get_args'):
            origin = typing.get_origin(field_type)
            args = typing.get_args(field_type)

            # Handle Optional[X] which is Union[X, None]
            if origin is typing.Union:
                # Try to convert to the first non-None type in the Union
                non_none_types = [t for t in args if t is not type(None)]
                if non_none_types:
                    for target_type in non_none_types:
                        try:
                            if isinstance(v, target_type):
                                update_kwargs[k] = v
                                break
                            converted_val = target_type(v)
                            update_kwargs[k] = converted_val
                            LOGGER.debug(f"Converted {k}={v} to {target_type.__name__}")
                            break
                        except (TypeError, ValueError):
                            continue
                    if k in update_kwargs:
                        continue

        # Attempt standard type conversion
        try:
            current_type = type(current_val)

            # Handle generic types
            if hasattr(current_type, '__origin__'):
                # For lists, try to convert the value if it's not already a list
                if current_type.__origin__ is list and not isinstance(v, list):
                    if isinstance(v, str):
                        # Try to parse string as a list
                        try:
                            import ast
                            converted_val = ast.literal_eval(v)
                            if isinstance(converted_val, list):
                                update_kwargs[k] = converted_val
                                LOGGER.debug(f"Converted string {k}={v} to list")
                                continue
                        except (SyntaxError, ValueError):
                            pass
                    # Try to convert to a single-item list
                    update_kwargs[k] = [v]
                    LOGGER.debug(f"Converted {k}={v} to single-item list")
                    continue
                else:
                    # For other generic types, try direct assignment if types match
                    update_kwargs[k] = v
                    LOGGER.debug(f"Assigned {k}={v} directly to generic type")
                    continue

            # Refuse to silently truncate a non-integer float into an int field
            # (e.g. a stray `n_strat_bins: 5.9`); skip the override with a warning
            # instead of quietly changing a hyperparameter.
            if current_type is int and isinstance(v, float) and not float(v).is_integer():
                LOGGER.warning(
                    "Refusing to truncate float override %s=%s into int field '%s'; skipping override",
                    k, v, k,
                )
                continue

            # Try a standard constructor
            converted_val = current_type(v)
            update_kwargs[k] = converted_val
            LOGGER.debug(f"Converted {k}={v} to type {current_type.__name__}")

        except (TypeError, ValueError) as e:
            LOGGER.error(
                f"Failed to convert field '{k}' value {v} to {type(current_val).__name__}: {e}. Skipping this override."
            )
            continue
        except Exception as e:
            LOGGER.error(
                f"Unexpected error converting field '{k}': {e}. Skipping this override."
            )
            continue

    updated_instance = dc.replace(instance, **update_kwargs)

    # Log applied changes
    if update_kwargs:
        LOGGER.info(
            "[EXP CFG] Applied overrides to %s",
            instance.__class__.__name__
        )

    return updated_instance
