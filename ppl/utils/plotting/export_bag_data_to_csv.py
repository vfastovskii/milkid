"""Export bag data to CSV files."""

from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
import logging

LOGGER = logging.getLogger(__name__)


def _normalize_attention_1d(alpha: np.ndarray) -> np.ndarray:
    """Normalize a 1D attention vector to be non-negative and sum to 1.

    - Replaces NaN/Inf with 0.
    - If any negatives or non-positive sum: applies softmax for stability.
    - Else divides by sum.
    - Clips to [0, 1] to avoid tiny numerical drifts.
    """
    if alpha is None:
        return alpha
    a = np.asarray(alpha, dtype=float).flatten()
    if a.size == 0:
        return a

    # Sanitize
    a[~np.isfinite(a)] = 0.0

    # All zeros guard
    if np.allclose(a, 0.0):
        return a

    s = float(a.sum())
    if np.any(a < 0.0) or s <= 0.0:
        # Softmax
        m = float(np.max(a))
        exp_a = np.exp(a - m)
        denom = float(exp_a.sum())
        if denom > 0.0:
            a = exp_a / denom
        else:
            a = np.zeros_like(a)
    else:
        a = a / s

    # Numerical safety
    a = np.clip(a, 0.0, 1.0)
    # Renormalize after clipping if needed
    total = a.sum()
    if total > 0:
        a = a / total
    return a


def export_bag_data_to_csv(
    model: Any,
    dataloader: Any,
    save_dir: str,
    device: Optional[str] = None,
    max_bags: int = 1500,
    conf_ids: Optional[Dict[str, list]] = None,
    task: str = "regression",
    stage: Optional[str] = None,
    current_epoch: Optional[int] = None,
) -> None:
    """
    Export true values, predicted values, and attention weights for each bag to CSV files.

    Parameters
    ----------
    model : Any
        Trained model with an aggregator that has attention weights.
    dataloader : Any
        DataLoader containing the bags to export.
    save_dir : str
        Directory to save the CSV files.
    device : str, optional
        Device to run the model on. If None, uses the model's device.
    max_bags : int, default 1000
        Maximum number of bags to export.
    conf_ids : Dict[str, list], optional
        Dictionary mapping bag IDs to lists of conformer IDs. If provided, these will be used
        as instance IDs in the CSV files instead of simple indices.
    task : str, default "regression"
        Task type (regression or classification).
    """
    LOGGER.info(f"Exporting bag data to CSV files for up to {max_bags} bags to {save_dir}")

    if current_epoch is None:
        current_epoch = getattr(
            model,
            "_evaluation_epoch_override",
            getattr(
                model,
                "_instance_importance_epoch",
                getattr(model, "current_epoch", None),
            ),
        )
    if stage is None:
        save_parts = {part.lower() for part in Path(save_dir).parts}
        if "test" in save_parts:
            stage = "test"
        elif "validation" in save_parts or "val" in save_parts:
            stage = "val"
        elif "train" in save_parts:
            stage = "train"

    # Check if dataloader is empty
    if not dataloader:
        LOGGER.warning(f"Dataloader is empty. Cannot export bag data for {save_dir}")
        # Create a placeholder file to indicate no data was available
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        placeholder_path = Path(save_dir) / "no_data_available.txt"
        with open(placeholder_path, "w") as f:
            f.write("No data available for bag data export")
        LOGGER.info(f"Created placeholder file at {placeholder_path}")
        return

    # Ensure the save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()

    # Get the device
    if device is None:
        device = next(model.parameters()).device

    # Counter for the number of bags processed
    bag_count = 0

    # Process each batch
    with torch.no_grad():
        for batch in dataloader:
            # Handle dict and tuple/list batch formats
            key_padding_mask = None
            cluster_ids = None
            series_labels = None
            if isinstance(batch, dict):
                bags = batch.get('bags') if 'bags' in batch else batch.get('x')
                y = batch.get('labels') if 'labels' in batch else batch.get('y')
                bag_ids = batch.get('bag_ids') if 'bag_ids' in batch else batch.get('ids')
                key_padding_mask = batch.get('padding_mask')
                cluster_ids = batch.get('cluster_ids')
                series_labels = batch.get('series_labels')
            else:
                # tuple/list: (bags, labels, bag_ids, [padding_mask,...])
                bags = batch[0]
                y = batch[1]
                bag_ids = batch[2]
                key_padding_mask = batch[3] if len(batch) > 3 else None
                cluster_ids = batch[4] if len(batch) > 4 else None
                series_labels = batch[5] if len(batch) > 5 else None
            x = bags[0] if isinstance(bags, list) else bags

            # Move to the appropriate device
            x = x.to(device)
            # Ensure mask is on the same device
            if key_padding_mask is not None:
                try:
                    if hasattr(key_padding_mask, "device") and key_padding_mask.device != x.device:
                        LOGGER.debug(f"[PLOT] Moving key_padding_mask from {key_padding_mask.device} to {x.device}")
                    key_padding_mask = key_padding_mask.to(x.device)
                except Exception as e:
                    LOGGER.warning(f"[PLOT] Could not move key_padding_mask to device {x.device}: {e}. Proceeding without mask movement.")
            if cluster_ids is not None:
                try:
                    cluster_ids = cluster_ids.to(x.device)
                except Exception as e:
                    LOGGER.warning(f"[PLOT] Could not move cluster_ids to device {x.device}: {e}. Proceeding without cluster IDs.")
                    cluster_ids = None

            # Forward pass
            try:
                core_kwargs = {}
                if key_padding_mask is not None:
                    core_kwargs["key_padding_mask"] = key_padding_mask
                if cluster_ids is not None:
                    core_kwargs["cluster_ids"] = cluster_ids
                if y is not None:
                    core_kwargs["labels"] = y.to(x.device) if hasattr(y, "to") else y
                if series_labels is not None:
                    core_kwargs["series_labels"] = series_labels
                if stage is not None:
                    core_kwargs["stage"] = stage
                if current_epoch is not None:
                    core_kwargs["current_epoch"] = int(current_epoch)
                logit, extras = model.core(x, **core_kwargs)
            except TypeError:
                fallback_kwargs = {}
                if key_padding_mask is not None:
                    fallback_kwargs["key_padding_mask"] = key_padding_mask
                try:
                    logit, extras = model.core(x, **fallback_kwargs)
                except TypeError:
                    logit, extras = model.core(x)

            # Convert to predictions based on task
            if task.lower() == "classification":
                y_pred = torch.sigmoid(logit)
            else:
                y_pred = logit

            # Get per-bag attention weights aligned to true bag lengths
            alphas_per_bag: list[np.ndarray] | None = None
            cluster_ids_per_bag: list[np.ndarray] | None = None
            cluster_query_mass_per_bag: list[np.ndarray] | None = None
            cluster_final_mass_per_bag: list[np.ndarray] | None = None

            alpha_key = "alpha_final" if "alpha_final" in extras else "alpha"
            if alpha_key in extras:
                # Prefer alpha_final when present: it is the exact final
                # instance distribution used to form the prediction
                # representation. Older aggregators expose the same concept as
                # alpha only.
                alpha = extras[alpha_key]

                alphas_per_bag = []
                cluster_ids_per_bag = [] if cluster_ids is not None else None
                cluster_alpha_query = extras.get("cluster_alpha")
                cluster_alpha_final = extras.get("cluster_alpha_final")
                cluster_query_mass_per_bag = (
                    []
                    if cluster_ids is not None and cluster_alpha_query is not None
                    else None
                )
                cluster_final_mass_per_bag = (
                    []
                    if cluster_ids is not None and cluster_alpha_final is not None
                    else None
                )
                if alpha.dim() == 2 and alpha.size(0) == x.shape[0]:
                    # [B, N]
                    for b in range(alpha.size(0)):
                        alpha_b = alpha[b]
                        cluster_b = cluster_ids[b] if cluster_ids is not None else None
                        if key_padding_mask is not None and hasattr(key_padding_mask, 'shape') and key_padding_mask.ndim == 2:
                            valid = ~key_padding_mask[b].to(torch.bool)
                            if valid.numel() == alpha_b.numel():
                                alpha_b = alpha_b[valid]
                                if cluster_b is not None:
                                    cluster_b = cluster_b[valid]
                            else:
                                L = int(valid.sum().item())
                                alpha_b = alpha_b[:L]
                                if cluster_b is not None:
                                    cluster_b = cluster_b[:L]
                        alphas_per_bag.append(alpha_b.detach().cpu().flatten().numpy())
                        if cluster_ids_per_bag is not None and cluster_b is not None:
                            cluster_ids_per_bag.append(cluster_b.detach().cpu().flatten().numpy())
                        if (
                            cluster_query_mass_per_bag is not None
                            and cluster_b is not None
                            and cluster_alpha_query is not None
                        ):
                            c = cluster_b.long().clamp(0, cluster_alpha_query.size(1) - 1)
                            cluster_query_mass_per_bag.append(
                                cluster_alpha_query[b, c].detach().cpu().flatten().numpy()
                            )
                        if (
                            cluster_final_mass_per_bag is not None
                            and cluster_b is not None
                            and cluster_alpha_final is not None
                        ):
                            c = cluster_b.long().clamp(0, cluster_alpha_final.size(1) - 1)
                            cluster_final_mass_per_bag.append(
                                cluster_alpha_final[b, c].detach().cpu().flatten().numpy()
                            )
                else:
                    # Fallback: flatten and reuse for all
                    alpha_1d = alpha.flatten()
                    alpha_np = alpha_1d.detach().cpu().numpy()
                    B = len(bag_ids)
                    alphas_per_bag = [alpha_np for _ in range(B)]
                    cluster_ids_per_bag = None
                    cluster_query_mass_per_bag = None
                    cluster_final_mass_per_bag = None

            elif hasattr(model.core.aggregator, 'last_attn') and model.core.aggregator.last_attn is not None:
                attn = model.core.aggregator.last_attn

                B = x.shape[0] if hasattr(x, 'shape') else len(bag_ids)
                # Infer [B, N] attention (per bag) from common shapes
                if attn.dim() == 4:
                    # [B, H, 1, N] or [B, H, N, N]
                    if attn.size(2) == 1:
                        # [B, H, 1, N] -> [B, H, N] -> mean over heads -> [B, N]
                        weights = attn.squeeze(2).mean(dim=1)
                    elif attn.size(1) == 1:
                        # [B, 1, N, N] -> [B, N, N] -> mean over queries -> [B, N]
                        weights = attn.squeeze(1).mean(dim=1)
                    else:
                        # [B, H, N, N] -> mean over heads -> [B, N, N] -> mean over queries -> [B, N]
                        weights = attn.mean(dim=1).mean(dim=1)
                elif attn.dim() == 3:
                    if attn.size(0) == (x.shape[0] if hasattr(x, 'shape') else attn.size(0)):
                        # Likely [B, H, N] or [B, N, N]
                        weights = attn.mean(dim=1)
                    else:
                        # [H, N, N] – fallback: same weights for each bag
                        w = attn.mean(dim=0).mean(dim=0)  # [N]
                        weights = w.unsqueeze(0).repeat(B, 1)
                elif attn.dim() == 2:
                    if attn.size(0) == (x.shape[0] if hasattr(x, 'shape') else attn.size(0)):
                        # [B, N]
                        weights = attn
                    else:
                        # [H, N] or [N, N] – fallback to mean over first dim
                        w = attn.mean(dim=0)
                        weights = w.unsqueeze(0).repeat(B, 1)
                else:
                    LOGGER.warning(f"Unexpected attention shape: {attn.shape}, cannot parse per-bag weights")
                    weights = None

                # Slice to true bag lengths using padding mask when available
                alphas_per_bag = []
                if weights is not None:
                    for b in range(len(bag_ids)):
                        alpha_b = weights[b]
                        if key_padding_mask is not None and hasattr(key_padding_mask, 'shape') and key_padding_mask.ndim == 2:
                            valid = ~key_padding_mask[b].to(torch.bool)
                            L = int(valid.sum().item())
                            alpha_b = alpha_b[:L]
                        alphas_per_bag.append(alpha_b.detach().cpu().flatten().numpy())
                else:
                    alphas_per_bag = None

            else:
                LOGGER.warning("No attention weights found in the model")
                return

            # Convert true and predicted values to numpy
            y_np = y.cpu().numpy()
            y_pred_np = y_pred.cpu().numpy()

            # Ensure arrays are 1D
            if y_np.ndim > 1:
                y_np = y_np.flatten()
            elif y_np.ndim == 0:  # Handle scalar values (0D arrays)
                y_np = np.array([y_np])  # Convert to 1D array with single element

            if y_pred_np.ndim > 1:
                y_pred_np = y_pred_np.flatten()
            elif y_pred_np.ndim == 0:  # Handle scalar values (0D arrays)
                y_pred_np = np.array([y_pred_np])  # Convert to 1D array with single element

            # Export data for each bag
            for i, bag_id in enumerate(bag_ids):
                alpha_np = alphas_per_bag[i]
                cluster_np = (
                    cluster_ids_per_bag[i]
                    if cluster_ids_per_bag is not None and i < len(cluster_ids_per_bag)
                    else None
                )
                cluster_query_mass_np = (
                    cluster_query_mass_per_bag[i]
                    if cluster_query_mass_per_bag is not None and i < len(cluster_query_mass_per_bag)
                    else None
                )
                cluster_final_mass_np = (
                    cluster_final_mass_per_bag[i]
                    if cluster_final_mass_per_bag is not None and i < len(cluster_final_mass_per_bag)
                    else None
                )
                # Generate instance IDs
                # If conformer IDs are provided for this bag, use them, otherwise use indices
                if conf_ids is not None and bag_id in conf_ids:
                    # Use provided conformer IDs
                    inst_ids = list(map(str, conf_ids[bag_id]))
                    # Ensure we have the right number of IDs
                    if len(inst_ids) != len(alpha_np):
                        L = min(len(inst_ids), len(alpha_np))
                        LOGGER.warning(
                            f"Number of conformer IDs ({len(inst_ids)}) doesn't match number of attention weights ({len(alpha_np)}) for bag {bag_id}. Aligning to {L}."
                        )
                        inst_ids = inst_ids[:L]
                        alpha_np = alpha_np[:L]
                        if cluster_np is not None:
                            cluster_np = cluster_np[:L]
                        if cluster_query_mass_np is not None:
                            cluster_query_mass_np = cluster_query_mass_np[:L]
                        if cluster_final_mass_np is not None:
                            cluster_final_mass_np = cluster_final_mass_np[:L]
                else:
                    # Use indices as fallback
                    inst_ids = list(range(len(alpha_np)))

                # Normalize attention after final alignment to IDs
                alpha_np = _normalize_attention_1d(alpha_np)

                # Final safety before writing: enforce invariants on attention weights
                alpha_np = np.asarray(alpha_np, dtype=float).flatten()
                s = float(np.sum(alpha_np))
                if s > 0.0 and not np.isclose(s, 1.0, atol=1e-8):
                    LOGGER.debug(f"[ATTN-CSV] Re-normalizing alpha for bag {bag_id} at write time (sum={s:.12f})")
                    alpha_np = alpha_np / s
                alpha_np = np.clip(alpha_np, 0.0, 1.0)
                s = float(np.sum(alpha_np))
                if s > 0.0:
                    alpha_np = alpha_np / s
                maxv = float(np.max(alpha_np)) if alpha_np.size else 0.0
                if maxv > 1.0 + 1e-12:
                    LOGGER.warning(f"[ATTN-CSV] Clipped alpha > 1 for bag {bag_id} at write time (max={maxv:.12f})")
                    alpha_np = np.clip(alpha_np, 0.0, 1.0)
                    s = float(np.sum(alpha_np))
                    if s > 0.0:
                        alpha_np = alpha_np / s
                # Round to stable decimals to avoid CSV textual artifacts
                alpha_np = np.round(alpha_np, 12)

                try:
                    # Create a DataFrame with instance IDs, true values, predicted values, and attention weights
                    csv_data = {
                        "instance_id": inst_ids,
                        "true_value": y_np[i],  # Same true value for all instances in the bag
                        "predicted_value": y_pred_np[i],  # Same predicted value for all instances in the bag
                        "attention_weight": alpha_np
                    }
                    if cluster_np is not None and len(cluster_np) == len(inst_ids):
                        csv_data["cluster_id"] = cluster_np.astype(int)
                    if (
                        cluster_query_mass_np is not None
                        and len(cluster_query_mass_np) == len(inst_ids)
                    ):
                        csv_data["cluster_attention_query_mass"] = np.round(
                            cluster_query_mass_np,
                            12,
                        )
                    if (
                        cluster_final_mass_np is not None
                        and len(cluster_final_mass_np) == len(inst_ids)
                    ):
                        csv_data["cluster_attention_final_mass"] = np.round(
                            cluster_final_mass_np,
                            12,
                        )
                    df = pd.DataFrame(csv_data)

                    # Save to CSV
                    csv_path = Path(save_dir) / f"{bag_id}.csv"
                    df.to_csv(csv_path, index=False)
                    LOGGER.info(f"Saved bag data to {csv_path}")

                    bag_count += 1
                    if bag_count >= max_bags:
                        LOGGER.info(f"Reached maximum number of bags ({max_bags})")
                        return
                except Exception as e:
                    LOGGER.error(f"Error exporting bag data for bag {bag_id}: {e}")
                    LOGGER.error(f"Alpha shape: {alpha_np.shape}, inst_ids length: {len(inst_ids)}")
                    import traceback
                    LOGGER.debug(traceback.format_exc())
