"""Small stateless helpers for the Lightning wrapper.

These were previously two zero-public-method ``nn.Module`` mixins
(``MemoryManagement``, ``ParameterManagement``) folded into the wrapper via
multiple inheritance; they hold no state, so they are plain functions.
"""

import logging
import os

import psutil
import torch

LOGGER = logging.getLogger(__name__)


def log_memory_usage(step: str, log_level: str = "info") -> None:
    """Log current GPU (when available) and CPU memory for a pipeline step."""
    # During training steps, use debug level to avoid cluttering the console.
    if step.endswith("_step") and log_level == "info":
        log_level = "debug"
    log = LOGGER.debug if log_level == "debug" else LOGGER.info

    if torch.cuda.is_available():
        current = torch.cuda.memory_allocated() / (1024**3)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        log(f"[{step}] GPU Memory: Current={current:.2f}GB, Peak={peak:.2f}GB")

    cpu_gb = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    log(f"[{step}] CPU Memory: {cpu_gb:.2f}GB")


def log_model_size(module) -> None:
    """Log parameter count and a rough memory estimate for a module."""
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)
    grad_size_mb = sum(
        p.numel() * p.element_size() for p in module.parameters() if p.requires_grad
    ) / (1024**2)
    optimizer_size_mb = grad_size_mb * 2  # Rough estimate for Adam optimizer

    LOGGER.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
    LOGGER.info(
        f"Estimated memory usage: Parameters={param_size_mb:.2f}MB, "
        f"Gradients={grad_size_mb:.2f}MB, Optimizer={optimizer_size_mb:.2f}MB, "
        f"Total={(param_size_mb + grad_size_mb + optimizer_size_mb):.2f}MB"
    )


def split_params_for_weight_decay(module):
    """Split a module's trainable params into ``(weights, biases_and_norms)``.

    Biases and normalization-layer params are returned separately so the caller
    can exclude them from weight decay.
    """
    weights = []
    biases_norms = []
    for name, param in module.named_parameters():
        if param.requires_grad:
            if 'bias' in name:
                biases_norms.append(param)
            elif 'norm' in name or 'ln' in name or 'layernorm' in name.lower():
                biases_norms.append(param)
            elif 'bn' in name or 'batchnorm' in name.lower():
                biases_norms.append(param)
            else:
                weights.append(param)
    return weights, biases_norms
