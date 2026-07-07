import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional centralized style helpers.  Cluster scratch copies may include only
# the ppl package, so plotting should fall back gracefully when style/ is absent.
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
        cmap = plt.get_cmap("tab10")
        fills = [cmap(i % 10) for i in range(max(1, n))]
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
        ax.grid(grid if y_grid is None else y_grid, alpha=0.25)

LOGGER = logging.getLogger(__name__)


def plot_true_vs_pred(
    df: pd.DataFrame,
    mask,
    title: str,
    out_path: str,
    annotate_pdb: bool = False,
    x_label: str = "Experimental Endpoint Value",
    y_label: str = "Predicted Endpoint Value",
    endpoint_col: str = "endpoint",
) -> None:
    """Draw a *True vs. Predicted* scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns specified by endpoint_col (true values), ``'y_pred'``
        (model predictions) and optionally ``'pdb_id'``.
    mask : array-like
        Boolean mask selecting the rows to plot.
    title : str
        Title for the figure.
    out_path : str
        Destination file name (e.g. *true_vs_pred.png*).
    annotate_pdb : bool, default False
        Annotate only the points with a non-empty / non-NaN PDB identifier.
    x_label : str, default "True pKi"
        Label for the x-axis.
    y_label : str, default "Predicted pKi"
        Label for the y-axis.
    endpoint_col : str, default "endpoint"
        Name of the column containing the true values.
    """
    # ---- style & data -------------------------------------------------------
    apply_mpl_style()

    x = df.loc[mask, endpoint_col].values
    y_pred = df.loc[mask, "y_pred"].values
    # Round predictions to 2 decimals for visualization
    y_pred = np.round(y_pred, 2)

    pdb_ids = (
        df.loc[mask, "pdb_id"].values
        if "pdb_id" in df.columns
        else np.full_like(x, fill_value="", dtype=object)
    )

    # Fixed plot limits as requested: strictly from 3 to 10 on both axes
    lo = 3.0
    hi = 10.0

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    fills, edges = get_palette(1)
    point_color = fills[0]
    edge_color = edges[0]

    ax.scatter(
        x,
        y_pred,
        alpha=0.8,
        s=18,
        facecolor=point_color,
        edgecolor=edge_color,
        linewidth=BAR_EDGE_LINEWIDTH * 0.6,
    )

    # Identity (ideal prediction) line in specified color
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        color="#8F1600",
        linewidth=BAR_EDGE_LINEWIDTH * 0.8,
    )

    # ±1 log-error support lines in specified color
    ax.plot(
        [lo, hi],
        [lo + 1, hi + 1],
        linestyle="--",
        color="#4B1C80",
        linewidth=BAR_EDGE_LINEWIDTH * 0.7,
    )
    ax.plot(
        [lo, hi],
        [lo - 1, hi - 1],
        linestyle="--",
        color="#4B1C80",
        linewidth=BAR_EDGE_LINEWIDTH * 0.7,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Light y-grid, keep default tick locators (pKi ranges, etc.)
    style_axes(
        ax,
        y_grid=True,
        y_major_step=None,
        y_minor_step=None,
    )

    # Optional PDB annotations
    if annotate_pdb:
        for xi, yi, pdb in zip(x, y_pred, pdb_ids):
            if isinstance(pdb, str) and pdb.strip() and pdb.strip().lower() != "noval":
                ax.text(
                    xi + 0.01,
                    yi + 0.01,
                    pdb,
                    fontsize=5,
                    alpha=0.8,
                )

    fig.tight_layout()

    # Ensure the directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info(f"Scatter saved to {out_path}")


def plot_true_vs_pred_from_model(
    model: Any,
    dataloader: Any,
    save_path: str,
    title: str = "Experimental vs. Predicted Endpoint Value",
    device: Optional[str] = None,
    task: str = "regression",
    x_label: str = "Experimental Endpoint Value",
    y_label: str = "Predicted Endpoint Value",
) -> None:
    """
    Generate a true vs. predicted plot from model predictions on a dataloader.

    Parameters
    ----------
    model : Any
        Trained model to use for predictions.
    dataloader : Any
        DataLoader containing the data to predict on.
    save_path : str
        Path to save the plot.
    title : str, default "True vs. Predicted"
        Title for the plot.
    device : str, optional
        Device to run the model on. If None, uses the model's device.
    task : str, default "regression"
        Task type (regression or classification).
    x_label : str, default "True Value"
        Label for the x-axis.
    y_label : str, default "Predicted Value"
        Label for the y-axis.
    """
    LOGGER.info(f"Generating true vs. predicted plot to {save_path}")

    # Check if dataloader is empty
    if not dataloader:
        LOGGER.warning(f"Dataloader is empty. Cannot generate plot for {save_path}")
        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.text(
            0.5,
            0.5,
            "No data available for plotting",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        return

    # Ensure the directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Set the model to evaluation mode
    model.eval()

    # Get the device
    if device is None:
        device = next(model.parameters()).device

    # Lists to store true values and predictions
    true_values: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    bag_ids: List[Any] = []

    # Process each batch
    with torch.no_grad():
        for batch in dataloader:
            # Initialize optional mask per batch
            key_padding_mask = None
            cluster_ids = None
            # Handle the batch structure (dictionary format from collate_mil)
            if isinstance(batch, dict):
                bags = batch.get('bags') if 'bags' in batch else batch.get('x')
                y = batch.get('labels') if 'labels' in batch else batch.get('y')
                batch_bag_ids = batch.get('bag_ids') if 'bag_ids' in batch else batch.get('ids')
                key_padding_mask = batch.get('padding_mask')
                cluster_ids = batch.get('cluster_ids')
            else:
                # Fallback for tuple/list batch format: (bags, labels, bag_ids, [padding_mask, ...])
                bags = batch[0]
                y = batch[1]
                batch_bag_ids = batch[2]
                key_padding_mask = batch[3] if len(batch) > 3 else None
                cluster_ids = batch[4] if len(batch) > 4 else None

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
                logit, _ = model.core(x, **core_kwargs)
            except TypeError:
                fallback_kwargs = {}
                if key_padding_mask is not None:
                    fallback_kwargs["key_padding_mask"] = key_padding_mask
                try:
                    logit, _ = model.core(x, **fallback_kwargs)
                except TypeError:
                    logit, _ = model.core(x)

            # Convert to predictions based on task
            if task.lower() == "classification":
                y_pred = torch.sigmoid(logit)
            else:
                y_pred = logit

            # Append to lists - ensure they are flattened to 1D arrays
            y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else y.cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            if y_np.ndim > 1:
                y_np = y_np.flatten()
            elif y_np.ndim == 0:
                y_np = np.array([y_np])

            if y_pred_np.ndim > 1:
                y_pred_np = y_pred_np.flatten()
            elif y_pred_np.ndim == 0:
                y_pred_np = np.array([y_pred_np])

            true_values.append(y_np)
            predictions.append(y_pred_np)
            bag_ids.extend(batch_bag_ids)

    # Check if we collected any data
    if not true_values:
        LOGGER.warning(f"No data collected from dataloader. Cannot generate plot for {save_path}")
        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.text(
            0.5,
            0.5,
            "No data available for plotting",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        return

    # Log array information for debugging
    LOGGER.debug(f"Number of arrays to concatenate: {len(true_values)}")
    LOGGER.debug(f"Sample true value array shape: {true_values[0].shape if true_values else 'No arrays'}")
    LOGGER.debug(f"Sample prediction array shape: {predictions[0].shape if predictions else 'No arrays'}")

    # Concatenate lists
    try:
        true_values_arr = np.concatenate(true_values)
        predictions_arr = np.concatenate(predictions)
        LOGGER.debug(f"Successfully concatenated arrays. Result shapes: true={true_values_arr.shape}, pred={predictions_arr.shape}")
    except Exception as e:
        LOGGER.error(f"Error concatenating arrays: {e}")
        LOGGER.error(f"True values shapes: {[arr.shape for arr in true_values]}")
        LOGGER.error(f"Prediction shapes: {[arr.shape for arr in predictions]}")
        LOGGER.error(f"True values dtypes: {[arr.dtype for arr in true_values]}")
        LOGGER.error(f"Prediction dtypes: {[arr.dtype for arr in predictions]}")

        apply_mpl_style()
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ax.text(
            0.5,
            0.5,
            f"Error generating plot:\n{e}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=7,
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        return

    # Create a DataFrame
    df = pd.DataFrame({
        "endpoint": true_values_arr,
        "y_pred": predictions_arr,
        "bag_id": bag_ids
    })

    # Create a mask for all rows
    mask = np.ones(len(df), dtype=bool)

    # Plot
    plot_true_vs_pred(
        df=df,
        mask=mask,
        title=title,
        out_path=save_path,
        x_label=x_label,
        y_label=y_label,
        endpoint_col="endpoint",
    )
