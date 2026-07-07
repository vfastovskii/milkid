import logging
import os
import random
import warnings

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Default seed value
DEFAULT_SEED = 42


class DeterminismError(RuntimeError):
    """Raised when deterministic behavior cannot be verified."""
    pass


def detect_best_device() -> torch.device:
    """Detect the best available device for PyTorch operations."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_deterministic(seed: int = DEFAULT_SEED, *, limit_threads: bool = False) -> None:
    """
    Configure deterministic behavior across all relevant libraries.
    
    Parameters
    ----------
    seed : int
        Random seed to use across all libraries
    limit_threads : bool
        Whether to limit thread count to 1 for maximum determinism.
        Warning: This significantly impacts performance.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch seeds (for both CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as e:
        warnings.warn(f"Could not enable deterministic algorithms: {e}")

    # CuBLAS determinism (requires CUDA ≥ 10.2 and PyTorch ≥ 1.8)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # CuDNN deterministic behavior
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Optional: limit CPU parallelism for maximum determinism
    if limit_threads:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        LOGGER.warning("Thread count limited to 1 - this will impact performance")

    LOGGER.info("Deterministic setup complete with seed: %d", seed)
    _log_determinism_status()


def check_determinism() -> None:
    """
    Verify that deterministic behavior is properly configured.
    
    Raises
    ------
    DeterminismError
        If deterministic behavior cannot be verified
    """
    issues = []

    # Check CuDNN settings
    if torch.backends.cudnn.is_available():
        if not torch.backends.cudnn.deterministic:
            issues.append("CuDNN deterministic mode is disabled")
        if torch.backends.cudnn.benchmark:
            issues.append("CuDNN benchmark mode is enabled (introduces non-determinism)")

    # Check PyTorch deterministic algorithms
    if not torch.are_deterministic_algorithms_enabled():
        issues.append("PyTorch deterministic algorithms are disabled")

    if issues:
        error_msg = "Deterministic setup verification failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        raise DeterminismError(error_msg)

    LOGGER.info("Deterministic setup verified successfully")


def _log_determinism_status() -> None:
    """Log current determinism configuration status."""
    LOGGER.info("CUBLAS_WORKSPACE_CONFIG = %s", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "Not set"))

    if torch.backends.cudnn.is_available():
        LOGGER.info("CuDNN deterministic: %s", torch.backends.cudnn.deterministic)
        LOGGER.info("CuDNN benchmark disabled: %s", not torch.backends.cudnn.benchmark)

    LOGGER.info("PyTorch deterministic algorithms enabled: %s", torch.are_deterministic_algorithms_enabled())
    LOGGER.info("Num threads: %d", torch.get_num_threads())
    LOGGER.info("Interop threads: %d", torch.get_num_interop_threads())


def get_device() -> torch.device:
    """Get the best available device for PyTorch operations."""
    return detect_best_device()


# Initialize a device (but don't print at import time)
_device = None


def device() -> torch.device:
    """Get the cached device instance."""
    global _device
    if _device is None:
        _device = detect_best_device()
        LOGGER.info("Using device: %s", _device)
    return _device


__all__ = [
    "set_deterministic",
    "check_determinism",
    "detect_best_device",
    "get_device",
    "device",
    "DeterminismError",
    "DEFAULT_SEED"
]
