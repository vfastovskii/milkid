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

    def __init__(self, *, task: str = "regression", ccc_weight: float = 0.0) -> None:
        super().__init__()
        self.task = task.lower()
        # CCC auxiliary term weight (regression only); 0 disables it (pure MSE).
        self.ccc_weight = float(ccc_weight) if self.task == "regression" else 0.0
        if self.task == "classification":
            self.base = nn.BCEWithLogitsLoss()
        elif self.task == "regression":
            self.base = nn.MSELoss()
        else:
            raise ValueError(task)

    @staticmethod
    def _ccc_term(y_hat: torch.Tensor, y_raw: torch.Tensor) -> torch.Tensor:
        """1 − Concordance Correlation Coefficient over the batch (0 when undefined).

        CCC = 2·cov / (var_hat + var_raw + (mean_hat − mean_raw)^2). It rewards the
        predictions matching the spread AND the identity line of y, so minimising
        (1 − CCC) pushes the best-fit slope toward 1 — a direct counter to shrinkage.
        Needs ≥2 samples with non-degenerate variance; returns 0 otherwise.
        """
        a = y_hat.flatten()
        b = y_raw.flatten()
        if a.numel() < 2:
            return y_hat.new_zeros(())
        ma, mb = a.mean(), b.mean()
        va = ((a - ma) ** 2).mean()
        vb = ((b - mb) ** 2).mean()
        cov = ((a - ma) * (b - mb)).mean()
        denom = va + vb + (ma - mb) ** 2
        if float(denom) <= 1e-8:
            return y_hat.new_zeros(())
        ccc = (2.0 * cov) / denom
        return 1.0 - ccc

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
        if self.ccc_weight > 0.0:
            loss = loss + self.ccc_weight * self._ccc_term(y_hat, y_raw)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Non-finite supervised loss")
        return loss
