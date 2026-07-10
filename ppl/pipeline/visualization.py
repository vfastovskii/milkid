"""Visualization utilities for the pipeline.

This module provides utilities for visualizing data distributions and metrics,
including functions for logging label distributions to MLFlow.
"""
import logging
import os
import tempfile
from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from ppl.data.data_loader import MILDataModule

LOGGER = logging.getLogger(__name__)


def log_split_distributions(dm: MILDataModule, *, stage: str):
    """Log label distributions for each split as MLflow artifacts.
    
    Parameters
    ----------
    dm : MILDataModule
        Data module containing the data loaders
    stage : str
        Stage name for the artifact path (e.g., 'train', 'final')
    """
    loader_map = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    fig, ax = plt.subplots()
    for split, dl in loader_map.items():
        if dl is None:
            continue  # e.g. no val loader for the final model
        values: List[np.ndarray] = []
        for batch in dl:
            # Standard tuple format: (bags, labels, bag_ids, padding_mask)
            y = batch[1]
            LOGGER.debug(f"Processing batch with tuple format for {split} split")
            values.append(y.cpu().numpy())
        if values:
            ax.hist(
                np.concatenate(values),
                bins=5,
                histtype="step",
                density=True,
                label=split,
                alpha=0.75,
            )

    ax.set_title(f"Label distribution – {stage}")
    ax.set_xlabel("Label value")
    ax.set_ylabel("Density")
    ax.legend()

    # Write the figure into a temp dir that is auto-removed (no leaked PNG).
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"{stage}_label_dist.png")
        fig.savefig(path)
        mlflow.log_artifact(path, artifact_path=f"{stage}_label_dists")
    plt.close(fig)