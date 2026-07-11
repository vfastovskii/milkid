from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Loss"]


class Loss(nn.Module):
    """The model's loss.

    Regression uses MSE; classification uses BCEWithLogits. Attention/aggregator
    regularizers (e.g. cluster compactness) are added by the caller, not here.

    An optional per-sample ``weight`` enables importance-weighted training: with a
    group-balanced batch sampler the batch distribution Q differs from the data
    distribution P, and weighting each sample by P/Q recovers an unbiased estimate
    of the true-distribution loss (self-normalised, so absolute scale is irrelevant).
    """

    def __init__(self, *, task: str = "regression") -> None:
        super().__init__()
        self.task = task.lower()
        if self.task == "classification":
            self.base = nn.BCEWithLogitsLoss()
        elif self.task == "regression":
            self.base = nn.MSELoss()
        else:
            raise ValueError(task)

    def _per_sample(self, y_hat: torch.Tensor, y_raw: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return F.binary_cross_entropy_with_logits(y_hat, y_raw, reduction="none")
        return F.mse_loss(y_hat, y_raw, reduction="none")

    def forward(
        self,
        y_hat: torch.Tensor,
        y_raw: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y_raw = y_raw.to(device=y_hat.device, dtype=y_hat.dtype)

        if not torch.isfinite(y_hat).all():
            raise FloatingPointError("Non-finite predictions passed to loss")
        if not torch.isfinite(y_raw).all():
            raise FloatingPointError("Non-finite labels passed to loss")

        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y_raw.dim() == 0:
            y_raw = y_raw.unsqueeze(0)

        if weight is None:
            loss = self.base(y_hat, y_raw)
        else:
            per = self._per_sample(y_hat, y_raw)
            w = weight.to(device=per.device, dtype=per.dtype).flatten()
            # Self-normalised importance-weighted mean.
            loss = (w * per).sum() / w.sum().clamp_min(1e-8)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite supervised loss")
        return loss
