from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["Loss"]


class Loss(nn.Module):
    """The model's loss.

    Regression uses MSE; classification uses BCEWithLogits. Attention/aggregator
    regularizers (e.g. cluster compactness) are added by the caller, not here.
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

    def forward(self, y_hat: torch.Tensor, y_raw: torch.Tensor) -> torch.Tensor:
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
