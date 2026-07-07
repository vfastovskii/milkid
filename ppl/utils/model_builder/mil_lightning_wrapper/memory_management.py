"""Memory management utilities for MIL model."""

import gc
import logging
import os
import time
import psutil
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class MemoryManagement(nn.Module):
    """Memory management mixin for MIL model."""

    def _log_memory_usage(self, step: str, log_level: str = "info") -> None:
        """Log current memory usage.

        Parameters
        ----------
        step : str
            Current step name for logging
        log_level : str
            Logging level to use ("info", "debug", "warning")
        """
        # During training steps, use debug level to avoid cluttering the console
        if step.endswith("_step") and log_level == "info":
            log_level = "debug"

        if torch.cuda.is_available():
            # GPU memory
            current_gpu_memory = torch.cuda.memory_allocated() / (1024**3)
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)

            # Log at the appropriate level
            if log_level == "debug":
                LOGGER.debug(f"[{step}] GPU Memory: Current={current_gpu_memory:.2f}GB, Peak={max_gpu_memory:.2f}GB")
            else:
                LOGGER.info(f"[{step}] GPU Memory: Current={current_gpu_memory:.2f}GB, Peak={max_gpu_memory:.2f}GB")

            # Update peak memory
            self._peak_memory_gb = max(self._peak_memory_gb, max_gpu_memory)

            # Record memory snapshot
            if not hasattr(self, "_memory_snapshots"):
                self._memory_snapshots = []
            self._memory_snapshots.append({
                'step': step,
                'gpu_current': current_gpu_memory,
                'gpu_peak': max_gpu_memory,
                'timestamp': time.time()
            })

        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory_gb = process.memory_info().rss / (1024**3)

        # Log at the appropriate level
        if log_level == "debug":
            LOGGER.debug(f"[{step}] CPU Memory: {cpu_memory_gb:.2f}GB")
        else:
            LOGGER.info(f"[{step}] CPU Memory: {cpu_memory_gb:.2f}GB")

    def _log_model_size(self) -> None:
        """Log the model size in parameters and memory."""
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Estimate memory usage (parameters + gradients + optimizer state)
        param_size_mb = sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2)
        grad_size_mb = sum(p.numel() * p.element_size() for p in self.parameters() if p.requires_grad) / (1024**2)
        optimizer_size_mb = grad_size_mb * 2  # Rough estimate for Adam optimizer

        LOGGER.info(f"Model size: {total_params:,} parameters ({trainable_params:,} trainable)")
        LOGGER.info(f"Estimated memory usage: Parameters={param_size_mb:.2f}MB, "
                   f"Gradients={grad_size_mb:.2f}MB, Optimizer={optimizer_size_mb:.2f}MB, "
                   f"Total={(param_size_mb + grad_size_mb + optimizer_size_mb):.2f}MB")
