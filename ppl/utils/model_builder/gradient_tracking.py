"""
Gradient tracking module for monitoring and visualizing gradients during training.

This module provides functionality to:
1. Register hooks on model components to capture gradients
2. Calculate gradient statistics (norms, histograms)
3. Log gradient information to MLflow
4. Visualize gradient distributions for interpretability

The module is designed to work with the Bag-Attention Net architecture
(embedder → attention/aggregator → predictor) and is compatible with PyTorch Lightning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import io
from PIL import Image

class GradientTracker:
    """
    Tracks gradients of model components during training.

    This class registers hooks on model components to capture gradients
    during the backward pass, calculates statistics, and logs them to MLflow.
    """

    def __init__(self, model: nn.Module, track_embedder: bool = True, 
                 track_aggregator: bool = True, track_predictor: bool = True):
        """
        Initialize the gradient tracker.

        Parameters
        ----------
        model : nn.Module
            The model to track gradients for (expected to be a MILCore instance)
        track_embedder : bool, optional
            Whether to track gradients for the embedder, by default True
        track_aggregator : bool, optional
            Whether to track gradients for the aggregator, by default True
        track_predictor : bool, optional
            Whether to track gradients for the predictor, by default True
        """
        self.model = model
        self.track_embedder = track_embedder
        self.track_aggregator = track_aggregator
        self.track_predictor = track_predictor

        # Storage for gradient statistics
        self.grad_stats = {}

        # Hooks
        self.hooks = []

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks on model components to capture gradients."""
        # Clear any existing hooks
        self.remove_hooks()

        # Register hooks for embedder
        if self.track_embedder and hasattr(self.model, 'embedder'):
            # Track input gradients (first layer weights)
            if hasattr(self.model.embedder, 'layers'):
                # Handle both ModuleDict and ModuleList cases
                if hasattr(self.model.embedder.layers, 'keys'):
                    # ModuleDict case
                    if len(list(self.model.embedder.layers.keys())) > 0:
                        first_layer_name = list(self.model.embedder.layers.keys())[0]
                        first_layer = self.model.embedder.layers[first_layer_name]
                elif len(self.model.embedder.layers) > 0:
                    # ModuleList case
                    first_layer = self.model.embedder.layers[0]
                if hasattr(first_layer, 'linear') and hasattr(first_layer.linear, 'weight'):
                    hook = first_layer.linear.weight.register_hook(
                        lambda grad: self._save_grad_stats('embedder_first_layer_weight', grad)
                    )
                    self.hooks.append(hook)

            # Track output gradients
            def embedder_output_hook(module, grad_input, grad_output):
                if grad_output and len(grad_output) > 0:
                    self._save_grad_stats('embedder_output', grad_output[0])
                return None

            hook = self.model.embedder.register_full_backward_hook(embedder_output_hook)
            self.hooks.append(hook)

        # Register hooks for aggregator
        if self.track_aggregator and hasattr(self.model, 'aggregator'):
            # Track attention weights gradients if available
            if hasattr(self.model.aggregator, 'mha'):
                if hasattr(self.model.aggregator.mha, 'out_proj') and hasattr(self.model.aggregator.mha.out_proj, 'weight'):
                    hook = self.model.aggregator.mha.out_proj.weight.register_hook(
                        lambda grad: self._save_grad_stats('aggregator_attention_weights', grad)
                    )
                    self.hooks.append(hook)

            # Track output gradients
            def aggregator_output_hook(module, grad_input, grad_output):
                if grad_output and len(grad_output) > 0:
                    self._save_grad_stats('aggregator_output', grad_output[0])
                return None

            hook = self.model.aggregator.register_full_backward_hook(aggregator_output_hook)
            self.hooks.append(hook)

        # Register hooks for predictor
        if self.track_predictor and hasattr(self.model, 'predictor'):
            # Track input gradients
            def predictor_input_hook(module, grad_input, grad_output):
                if grad_input and len(grad_input) > 0 and grad_input[0] is not None:
                    self._save_grad_stats('predictor_input', grad_input[0])
                return None

            hook = self.model.predictor.register_full_backward_hook(predictor_input_hook)
            self.hooks.append(hook)

            # Track output gradients (final layer)
            if hasattr(self.model.predictor, 'layers'):
                # Handle both ModuleDict and ModuleList cases
                if hasattr(self.model.predictor.layers, 'keys'):
                    # ModuleDict case
                    if len(list(self.model.predictor.layers.keys())) > 0:
                        last_layer_name = list(self.model.predictor.layers.keys())[-1]
                        last_layer = self.model.predictor.layers[last_layer_name]
                elif len(self.model.predictor.layers) > 0:
                    # ModuleList case
                    last_layer = self.model.predictor.layers[-1]
                if hasattr(last_layer, 'linear') and hasattr(last_layer.linear, 'weight'):
                    hook = last_layer.linear.weight.register_hook(
                        lambda grad: self._save_grad_stats('predictor_last_layer_weight', grad)
                    )
                    self.hooks.append(hook)

    def _save_grad_stats(self, name: str, grad: torch.Tensor):
        """
        Save gradient statistics for the given tensor.

        Parameters
        ----------
        name : str
            Name of the gradient (used as key in grad_stats)
        grad : torch.Tensor
            Gradient tensor
        """
        if grad is None:
            return

        # Detach and move to CPU for statistics calculation
        grad_cpu = grad.detach().cpu()

        # Calculate statistics
        # Handle std calculation for small tensors to avoid warning
        if grad_cpu.numel() <= 1:
            # Use unbiased=False for single element tensors
            std_value = 0.0
        else:
            std_value = grad_cpu.std().item()

        stats = {
            'norm': torch.norm(grad_cpu).item(),
            'mean': grad_cpu.mean().item(),
            'std': std_value,
            'min': grad_cpu.min().item(),
            'max': grad_cpu.max().item(),
            'histogram': grad_cpu.flatten().numpy()
        }

        # Store statistics
        if name not in self.grad_stats:
            self.grad_stats[name] = []

        self.grad_stats[name].append(stats)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset_stats(self):
        """Reset gradient statistics."""
        self.grad_stats = {}

    def log_stats_to_mlflow(self, logger):
        """
        Log gradient statistics to MLflow.

        Parameters
        ----------
        logger : pytorch_lightning.loggers.MLFlowLogger
            PyTorch Lightning MLflow logger
        """
        # Calculate epoch-level statistics
        epoch_stats = {}

        for name, stats_list in self.grad_stats.items():
            if not stats_list:
                continue

            # Calculate average statistics
            avg_norm = np.mean([s['norm'] for s in stats_list])
            avg_mean = np.mean([s['mean'] for s in stats_list])
            avg_std = np.mean([s['std'] for s in stats_list])
            avg_min = np.mean([s['min'] for s in stats_list])
            avg_max = np.mean([s['max'] for s in stats_list])

            # Log scalar metrics
            logger.log_metrics({
                f'grad/{name}/norm': avg_norm,
                f'grad/{name}/mean': avg_mean,
                f'grad/{name}/std': avg_std,
                f'grad/{name}/min': avg_min,
                f'grad/{name}/max': avg_max
            })

            # Create and log histogram
            all_values = np.concatenate([s['histogram'] for s in stats_list])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(all_values, bins=50)
            ax.set_title(f'Gradient Distribution: {name}')
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Frequency')

            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)

            # Log image
            logger.experiment.log_figure(
                run_id=logger.run_id,
                figure=fig,
                artifact_file=f'grad_histograms/{name}.png'
            )

            plt.close(fig)

        # Reset statistics after logging
        self.reset_stats()

def add_gradient_tracking_to_model(model: nn.Module) -> GradientTracker:
    """
    Add gradient tracking to a model.

    Parameters
    ----------
    model : nn.Module
        The model to track gradients for (expected to be a MILCore instance)

    Returns
    -------
    GradientTracker
        The gradient tracker instance
    """
    return GradientTracker(model)
