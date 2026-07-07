from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

__all__ = ["AdaptiveEntropicLoss", "SupervisedLoss"]

LOGGER = logging.getLogger(__name__)


class SupervisedLoss(nn.Module):
    """Pure supervised MIL loss.

    The historical ``AdaptiveEntropicLoss`` name is kept as an alias for
    compatibility, but the loss no longer uses attention entropy, KL, or any
    aggregator extras. Regression uses MSE; classification uses BCEWithLogits.
    """

    def __init__(self, *, task: str = "regression", **_: object) -> None:
        super().__init__()
        self.task = task.lower()
        if self.task == "classification":
            self.base = nn.BCEWithLogitsLoss()
        elif self.task == "regression":
            self.base = nn.MSELoss()
        else:
            raise ValueError(task)

    def forward(
        self,
        y_hat: torch.Tensor,
        y_raw: torch.Tensor,
        extras: Dict | None = None,
    ) -> torch.Tensor:
        del extras

        y_raw = y_raw.to(device=y_hat.device, dtype=y_hat.dtype)

        if not torch.isfinite(y_hat).all():
            raise FloatingPointError("Non-finite predictions passed to loss")
        if not torch.isfinite(y_raw).all():
            raise FloatingPointError("Non-finite labels passed to loss")

        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(0)
        if y_raw.dim() == 0:
            y_raw = y_raw.unsqueeze(0)

        loss = self.base(y_hat, y_raw)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite supervised loss")
        return loss


AdaptiveEntropicLoss = SupervisedLoss
