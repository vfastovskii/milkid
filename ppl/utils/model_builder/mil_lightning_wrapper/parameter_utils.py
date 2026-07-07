"""Parameter management utilities for MIL model."""

import logging
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class ParameterManagement(nn.Module):
    """Parameter management mixin for MIL model."""

    def _split_params_for_weight_decay(self, module):
        """
        Split parameters into two groups:
        - weights that should have weight decay applied
        - biases and normalization layers that should not have weight decay

        Parameters
        ----------
        module : nn.Module
            Module to split parameters for

        Returns
        -------
        tuple
            (weights_params, biases_norm_params)
        """
        weights = []
        biases_norms = []

        for name, param in module.named_parameters():
            if param.requires_grad:
                # Skip frozen parameters
                if 'bias' in name:
                    # All biases don't get weight decay
                    biases_norms.append(param)
                elif 'norm' in name or 'ln' in name or 'layernorm' in name.lower():
                    # Normalization layers don't get weight decay
                    biases_norms.append(param)
                elif 'bn' in name or 'batchnorm' in name.lower():
                    # Batch normalization layers don't get weight decay
                    biases_norms.append(param)
                else:
                    # Everything else (weights) gets weight decay
                    weights.append(param)

        return weights, biases_norms
