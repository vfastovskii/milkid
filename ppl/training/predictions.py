"""Batch unpacking, inference, and regression metrics for artifact export."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)


def unpack_batch(batch):
    """Return (x, y, bag_ids, key_padding_mask, cluster_ids, series_labels)."""
    if isinstance(batch, dict):
        bags = batch.get("bags", batch.get("x"))
        y = batch.get("labels", batch.get("y"))
        bag_ids = batch.get("bag_ids", batch.get("ids"))
        key_padding_mask = batch.get("padding_mask")
        cluster_ids = batch.get("cluster_ids")
        series_labels = batch.get("series_labels")
    else:
        bags, y, bag_ids = batch[0], batch[1], batch[2]
        key_padding_mask = batch[3] if len(batch) > 3 else None
        cluster_ids = batch[4] if len(batch) > 4 else None
        series_labels = batch[5] if len(batch) > 5 else None
    x = bags[0] if isinstance(bags, list) else bags
    return x, y, bag_ids, key_padding_mask, cluster_ids, series_labels


def predict_rows(
    model: pl.LightningModule,
    dataloader,
    task: str,
    *,
    stage: Optional[str] = None,
    eval_epoch: Optional[int] = None,
    use_series_labels: bool = False,
) -> List[tuple]:
    """Run inference over a dataloader and collect (bag_id, true, pred) rows."""
    rows: List[tuple] = []
    if dataloader is None:
        return rows

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            x, y, bag_ids, key_padding_mask, cluster_ids, series_labels = unpack_batch(
                batch
            )
            x = x.to(device)
            if key_padding_mask is not None:
                try:
                    key_padding_mask = key_padding_mask.to(x.device)
                except Exception:
                    key_padding_mask = None
            if cluster_ids is not None:
                try:
                    cluster_ids = cluster_ids.to(x.device)
                except Exception:
                    cluster_ids = None

            core_kwargs = {}
            if key_padding_mask is not None:
                core_kwargs["key_padding_mask"] = key_padding_mask
            if cluster_ids is not None:
                core_kwargs["cluster_ids"] = cluster_ids
            if use_series_labels and series_labels is not None:
                core_kwargs["series_labels"] = series_labels
            if stage is not None:
                core_kwargs["stage"] = stage
            if eval_epoch is not None:
                core_kwargs["current_epoch"] = int(eval_epoch)

            try:
                logit, _ = model.core(x, **core_kwargs)
            except TypeError:
                logit, _ = model.core(x)

            y_pred = (
                torch.sigmoid(logit) if task.lower() == "classification" else logit
            )
            y_np = np.asarray(y.cpu().numpy()).reshape(-1)
            y_pred_np = np.asarray(y_pred.detach().cpu().numpy()).reshape(-1)
            for i, mol_id in enumerate(bag_ids):
                rows.append((str(mol_id), float(y_np[i]), float(y_pred_np[i])))
    return rows


def regression_metrics(rows, prefix: str, task: str):
    """Compute {loss, mae, rmse, r2, pearson, spearman} from prediction rows."""
    if not rows or task.lower() != "regression":
        return None
    y_true = np.asarray([row[1] for row in rows], dtype=np.float64)
    y_pred = np.asarray([row[2] for row in rows], dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size == 0:
        return None

    err = y_pred - y_true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denom <= 0.0 else float(1.0 - np.sum(err * err) / denom)

    def _corr(a, b):
        if a.size < 2:
            return float("nan")
        if float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    true_rank = pd.Series(y_true).rank(method="average").to_numpy(dtype=np.float64)
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy(dtype=np.float64)

    return {
        f"{prefix}_loss": mse,
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2": r2,
        f"{prefix}_pearson": _corr(y_true, y_pred),
        f"{prefix}_spearman": _corr(true_rank, pred_rank),
    }
