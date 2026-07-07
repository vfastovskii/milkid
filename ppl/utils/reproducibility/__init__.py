"""Reproducibility utilities for the Multi-Instance Learning Kit (MILK)."""

# Expose key modules at the package level
from ppl.utils.reproducibility.deterministic_setup import (
    set_deterministic,
    check_determinism,
    detect_best_device,
    get_device,
    DEFAULT_SEED
)
