from pathlib import Path
from typing import Iterable, Sequence, Dict, Any, Optional

import matplotlib

matplotlib.use("Agg")   # ← sidestep Qt entirely
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import pandas as pd

# Optional centralized style helpers.  HPO/cluster scratch copies may include
# only the ppl package, so plotting must still import without the sibling style
# package.
try:
    from style.style import (
        BAR_EDGE_LINEWIDTH,
        apply_mpl_style,
        get_palette,
        style_axes,
    )
except ModuleNotFoundError:
    BAR_EDGE_LINEWIDTH = 0.8

    def apply_mpl_style() -> None:
        plt.style.use("default")

    def get_palette(n: int):
        cmap = plt.get_cmap("tab20")
        fills = [cmap(i % 20) for i in range(max(1, n))]
        edges = ["black"] * max(1, n)
        return fills, edges

    def style_axes(
        ax,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        grid: bool = True,
        y_grid: bool | None = None,
        **_: object,
    ) -> None:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        # Pass line props (alpha) only when enabling the grid — matplotlib warns and
        # force-enables if you supply styling alongside grid(False).
        show_grid = grid if y_grid is None else y_grid
        if show_grid:
            ax.grid(True, alpha=0.25)
        else:
            ax.grid(False)

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


def plot_alpha(
    alpha: Sequence[float],
    bag_id: str | int,
    inst_ids: Iterable[str] | Iterable[int],  # kept for backward-compatibility
    save_dir: str = "./alpha",
) -> None:
    """
    Plot and save the distribution of attention weights inside a bag.

    Publication-ready version using style.py.

    Parameters
    ----------
    alpha : array-like
        Attention weights (length = number of instances in the bag).
    bag_id : str | int
        Identifier of the bag; becomes the file name.
    inst_ids : iterable[str | int]
        Per-instance identifiers – shown on the y axis.
    save_dir : str, default "./alpha"
        Folder where the PNG is written. Created automatically if missing.
    """
    # --- style --------------------------------------------------------------
    apply_mpl_style()

    # -- sanitise ---------------------------------------------
    alpha = np.asarray(alpha, dtype=float)
    inst_ids = [str(i) for i in inst_ids]

    if alpha.shape[0] != len(inst_ids):
        raise ValueError(
            f"alpha length ({alpha.shape[0]}) and inst_ids length ({len(inst_ids)}) mismatch"
        )

    # -- ensure output dir exists ------------------------------
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # -- plotting ----------------------------------------------
    n = len(alpha)
    fig_h = max(2.5, min(14, 0.35 * n))
    fig, ax = plt.subplots(figsize=(7.0, fig_h))

    y_pos = np.arange(n)

    # Use unified pastel palette + standard edge linewidth
    fills, edges = get_palette(n)
    bars = ax.barh(
        y_pos,
        alpha,
        align="center",
        color=fills,
        edgecolor=edges,
        linewidth=BAR_EDGE_LINEWIDTH,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(inst_ids)
    ax.invert_yaxis()  # highest weight on top
    ax.set_xlabel("Attention Weight")
    ax.set_title(f"Bag {bag_id} – Attention Weights")

    # light x-grid only (keep y clean for categorical labels)
    ax.xaxis.grid(True, which="major", linestyle="--", alpha=0.35)

    # x-limits: [0, slightly above max]
    xmax = float(np.nanmax(alpha)) if np.isfinite(np.nanmax(alpha)) else 1.0
    ax.set_xlim(0.0, max(1.0, xmax * 1.05))

    # publication-style axes (but don't touch y tick locators for categories)
    style_axes(
        ax,
        y_grid=False,
        y_major_step=None,
        y_minor_step=None,
    )

    # annotate with exact values only if not too crowded
    if n <= 40:
        for y, w in zip(y_pos, alpha):
            ax.text(
                w + max(0.005, 0.01 * xmax if np.isfinite(xmax) and xmax > 0 else 0.01),
                y,
                f"{w:.3f}",
                va="center",
                fontsize=7,
                color="#222222",
            )

    fig.tight_layout()
    outfile = Path(save_dir) / f"{bag_id}.png"
    fig.savefig(outfile, bbox_inches="tight")
    LOGGER.info(f"Saved attention weights plot to {outfile}")

    plt.close(fig)


def plot_attention_weights_from_model(
    model: Any,
    dataloader: Any,
    save_dir: str,
    device: Optional[str] = None,
    max_bags: int = 1000,
    conf_ids: Optional[Dict[str, list]] = None,
    stage: Optional[str] = None,
    current_epoch: Optional[int] = None,
) -> None:
    """
    Plot attention weights for all bags in a dataloader using a trained model.

    Parameters
    ----------
    model : Any
        Trained model with an aggregator that has attention weights.
    dataloader : Any
        DataLoader containing the bags to plot.
    save_dir : str
        Directory to save the plots.
    device : str, optional
        Device to run the model on. If None, uses the model's device.
    max_bags : int, default 1000
        Maximum number of bags to plot.
    conf_ids : Dict[str, list], optional
        Dictionary mapping bag IDs to lists of conformer IDs. If provided, these will be used
        as instance IDs in the plots instead of simple indices.
    """
    LOGGER.info(f"Plotting attention weights for up to {max_bags} bags to {save_dir}")

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
        LOGGER.warning(f"Dataloader is empty. Cannot generate attention weight plots for {save_dir}")
        # Create a placeholder file to indicate no data was available
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        placeholder_path = Path(save_dir) / "no_data_available.png"

        # Publication-style placeholder
        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.text(
            0.5,
            0.5,
            "No data available for attention weight plotting",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(placeholder_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        LOGGER.info(f"Created placeholder image at {placeholder_path}")
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
            # Initialize optional mask per batch
            key_padding_mask = None
            cluster_ids = None
            series_labels = None
            # Handle the batch structure (dictionary format from collate_mil)
            if isinstance(batch, dict):
                bags = batch.get('bags') if 'bags' in batch else batch.get('x')
                y = batch.get('labels') if 'labels' in batch else batch.get('y')
                bag_ids = batch.get('bag_ids') if 'bag_ids' in batch else batch.get('ids')
                key_padding_mask = batch.get('padding_mask')
                cluster_ids = batch.get('cluster_ids')
                series_labels = batch.get('series_labels')
            else:
                # Fallback for tuple/list batch format: (bags, labels, bag_ids, [padding_mask, ...])
                try:
                    bags = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                    bag_ids = batch[2]
                    key_padding_mask = batch[3] if len(batch) > 3 else None
                    cluster_ids = batch[4] if len(batch) > 4 else None
                    series_labels = batch[5] if len(batch) > 5 else None
                except Exception as e:
                    LOGGER.error(f"Unexpected batch structure in attention plotter: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'NA'}; error={e}")
                    raise
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
                # Backward compatibility: model.core may not accept masks/cluster IDs.
                fallback_kwargs = {}
                if key_padding_mask is not None:
                    fallback_kwargs["key_padding_mask"] = key_padding_mask
                try:
                    logit, extras = model.core(x, **fallback_kwargs)
                except TypeError:
                    logit, extras = model.core(x)

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
                LOGGER.debug(
                    "Original %s shape from extras: %s",
                    alpha_key,
                    alpha.shape,
                )

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
                LOGGER.debug(f"Original attention shape: {attn.shape}")

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
                        if attn.size(1) == (x.shape[1] if hasattr(x, 'shape') else attn.size(1)) and attn.size(2) == (x.shape[1] if hasattr(x, 'shape') else attn.size(2)):
                            # [B, N, N]
                            weights = attn.mean(dim=1)
                        else:
                            # [B, H, N]
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
                        # Ensure 1D numpy
                        alpha_np_b = alpha_b.detach().cpu().flatten().numpy()
                        alphas_per_bag.append(alpha_np_b)
                else:
                    alphas_per_bag = None
            else:
                LOGGER.warning("No attention weights found in the model")
                return

            # Compute predictions based on task, and prepare true labels if available
            try:
                task = getattr(model, 'task', getattr(getattr(model, 'core', None), 'task', 'regression'))
                if isinstance(task, str):
                    task = task.lower()
                else:
                    task = 'regression'
            except Exception:
                task = 'regression'

            try:
                if task == 'classification':
                    y_pred = torch.sigmoid(logit)
                else:
                    y_pred = logit
            except Exception:
                y_pred = logit

            # Convert to numpy arrays (1D per bag)
            try:
                y_pred_np = y_pred.detach().cpu().numpy()
                if y_pred_np.ndim > 1:
                    y_pred_np = y_pred_np.flatten()
                elif y_pred_np.ndim == 0:
                    y_pred_np = np.array([y_pred_np])
            except Exception:
                y_pred_np = np.full((len(bag_ids),), np.nan, dtype=float)

            if y is not None:
                try:
                    y_np = y.detach().cpu().numpy() if hasattr(y, 'detach') else (y.cpu().numpy() if hasattr(y, 'cpu') else np.asarray(y))
                    if y_np.ndim > 1:
                        y_np = y_np.flatten()
                    elif y_np.ndim == 0:
                        y_np = np.array([y_np])
                except Exception:
                    y_np = np.full((len(bag_ids),), np.nan, dtype=float)
            else:
                y_np = np.full((len(bag_ids),), np.nan, dtype=float)

            # If still None, nothing to plot
            if alphas_per_bag is None:
                LOGGER.warning("Could not derive per-bag attention weights – skipping plots")
                return

            # Plot the attention weights per bag with ID alignment and export CSV
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

                # Determine instance IDs for this bag
                if conf_ids is not None and bag_id in conf_ids:
                    inst_ids = list(map(str, conf_ids[bag_id]))
                    if len(inst_ids) != len(alpha_np):
                        # Never silently truncate: a length mismatch means the IDs are
                        # not aligned with the attention weights, so labeling would
                        # misassign attention to the wrong conformer. Fail loudly (the
                        # per-bag handler skips this bag rather than mislabel it).
                        raise ValueError(
                            f"[ATTN] conformer-id / attention length mismatch for bag "
                            f"{bag_id}: {len(inst_ids)} ids vs {len(alpha_np)} weights. "
                            "Refusing to guess an alignment."
                        )
                else:
                    inst_ids = list(range(len(alpha_np)))

                # Normalize attention after final alignment to IDs
                alpha_np = _normalize_attention_1d(alpha_np)

                try:
                    # -------- Full-bag plot (publication style) -----------------
                    plot_alpha(
                        alpha=alpha_np,
                        bag_id=bag_id,
                        inst_ids=inst_ids,
                        save_dir=save_dir,
                    )

                    # Additionally, write per-bag CSV with true/pred and attention weights
                    try:
                        true_val = y_np[i] if i < len(y_np) else np.nan
                        pred_val = y_pred_np[i] if i < len(y_pred_np) else np.nan

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

                        csv_data = {
                            "instance_id": inst_ids,
                            "true_value": [true_val] * len(inst_ids),
                            "predicted_value": [pred_val] * len(inst_ids),
                            "attention_weight": alpha_np,
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
                        csv_path = Path(save_dir) / f"{bag_id}.csv"
                        df.to_csv(csv_path, index=False)
                        LOGGER.info(f"Saved bag CSV to {csv_path}")

                        # === Top-20 export (renormalized to sum=1) ===
                        k = min(20, len(alpha_np))
                        if k > 0:
                            order = np.argsort(-alpha_np)[:k]
                            top_ids = [str(inst_ids[j]) for j in order]
                            top_weights = alpha_np[order]
                            top_sum = float(top_weights.sum())
                            if top_sum > 0:
                                top_weights_renorm = top_weights / top_sum
                            else:
                                LOGGER.warning(f"Top-{k} attention weights sum to zero for bag {bag_id}; leaving renormed weights as zeros.")
                                top_weights_renorm = np.zeros_like(top_weights)

                            # Build and save CSV for top-20
                            top_data = {
                                "rank": np.arange(1, k + 1),
                                "instance_id": top_ids,
                                "attention_weight": np.round(top_weights, 12),
                                "attention_weight_top20_renorm": np.round(top_weights_renorm, 12),
                                "true_value": [true_val] * k,
                                "predicted_value": [pred_val] * k,
                            }
                            if cluster_np is not None and len(cluster_np) == len(inst_ids):
                                top_data["cluster_id"] = cluster_np[order].astype(int)
                            if (
                                cluster_query_mass_np is not None
                                and len(cluster_query_mass_np) == len(inst_ids)
                            ):
                                top_data["cluster_attention_query_mass"] = np.round(
                                    cluster_query_mass_np[order],
                                    12,
                                )
                            if (
                                cluster_final_mass_np is not None
                                and len(cluster_final_mass_np) == len(inst_ids)
                            ):
                                top_data["cluster_attention_final_mass"] = np.round(
                                    cluster_final_mass_np[order],
                                    12,
                                )
                            top_df = pd.DataFrame(top_data)
                            top_csv_path = Path(save_dir) / f"{bag_id}_top20.csv"
                            top_df.to_csv(top_csv_path, index=False)
                            LOGGER.info(f"Saved top-{k} renorm CSV to {top_csv_path}")

                            # -------- Top-k plot (publication style) --------------
                            apply_mpl_style()
                            fig_h = max(2.5, min(10, 0.45 * k))
                            fig, ax = plt.subplots(figsize=(7.0, fig_h))
                            y = np.arange(k)

                            fills, edges = get_palette(k)
                            ax.barh(
                                y,
                                top_weights_renorm,
                                color=fills,
                                edgecolor=edges,
                                linewidth=BAR_EDGE_LINEWIDTH,
                            )
                            ax.set_yticks(y)
                            ax.set_yticklabels(top_ids)
                            ax.invert_yaxis()
                            ax.set_xlabel(f"Attention weight (renormed over top-{k}; sum = 1)")
                            ax.set_title(f"Bag {bag_id} – Top-{k} Attention")

                            ax.xaxis.grid(True, linestyle="--", alpha=0.35)
                            ax.set_xlim(0.0, 1.05)

                            style_axes(
                                ax,
                                y_grid=False,
                                y_major_step=None,
                                y_minor_step=None,
                            )

                            # Annotate values if not too many
                            if k <= 30:
                                xmax_top = float(np.nanmax(top_weights_renorm)) if np.isfinite(np.nanmax(top_weights_renorm)) else 1.0
                                for yi, w in zip(y, top_weights_renorm):
                                    ax.text(
                                        w + max(0.003, 0.01 * xmax_top if xmax_top > 0 else 0.01),
                                        yi,
                                        f"{w:.3f}",
                                        va="center",
                                        fontsize=7,
                                        color="#222222",
                                    )

                            fig.tight_layout()
                            top_png_path = Path(save_dir) / f"{bag_id}_top20.png"
                            fig.savefig(top_png_path, bbox_inches="tight")
                            plt.close(fig)
                            LOGGER.info(f"Saved top-{k} renorm plot to {top_png_path}")
                    except Exception as csv_e:
                        LOGGER.error(f"Error saving CSV for bag {bag_id}: {csv_e}")
                        import traceback as _tb
                        LOGGER.debug(_tb.format_exc())

                    bag_count += 1
                    if bag_count >= max_bags:
                        LOGGER.info(f"Reached maximum number of bags ({max_bags})")
                        return
                except Exception as e:
                    LOGGER.error(f"Error plotting attention weights for bag {bag_id}: {e}")
                    LOGGER.error(f"Alpha shape: {alpha_np.shape}, inst_ids length: {len(inst_ids)}")
                    import traceback
                    LOGGER.debug(traceback.format_exc())
