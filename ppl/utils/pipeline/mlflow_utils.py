"""MLFlow utilities for experiment tracking and logging.

This module provides utilities for MLFlow integration, including a custom MLFlow logger
that fixes issues with artifact paths, functions for creating loggers, and utilities
for logging metrics.
"""
import logging
from pathlib import Path
from typing import Dict

import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities import rank_zero_only

LOGGER = logging.getLogger(__name__)


class SafeMLFlowLogger(MLFlowLogger):
    """Bug‑fix for MLFlowLogger (#15111): ensure artifact_path is str."""
    
    def _scan_and_log_checkpoints(self, checkpoint_callback):  # noqa: D401
        """Scan and log checkpoints, ensuring artifact_path is a string."""
        if checkpoint_callback is None:
            return
        for ckpt_path in checkpoint_callback.best_k_models:
            artifact_path = str(Path(ckpt_path).stem)
            self.experiment.log_artifact(self._run_id, str(ckpt_path), artifact_path)


def create_mlflow_logger(
    save_dir: Path,
    *,
    experiment_name: str,
    run_name: str | None,
    tracking_uri: str | Path | None,
) -> SafeMLFlowLogger:
    """Create an MLFlow logger with proper configuration.
    
    Parameters
    ----------
    save_dir : Path
        Directory to save MLFlow logs
    experiment_name : str
        Name of the MLFlow experiment
    run_name : str | None
        Name of the MLFlow run
    tracking_uri : str | Path | None
        URI for MLFlow tracking server
        
    Returns
    -------
    SafeMLFlowLogger
        Configured MLFlow logger
    """
    save_dir = save_dir.expanduser()
    tracking_uri = str(tracking_uri or save_dir / "mlflow_logs")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # avoid double‑logging
    mlflow.autolog(disable=True)

    return SafeMLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        log_model=True,
    )


@rank_zero_only
def log_metrics(metrics: Dict[str, float], *, stage: str, prefix: str | None = None) -> None:
    """Log metrics to the console, ensuring only the rank-zero process logs.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    stage : str
        Stage name (e.g., 'train', 'val', 'test')
    prefix : str | None
        Optional prefix for the log message
    """
    tag = f"{prefix + ' ' if prefix else ''}{stage}"
    for k, v in metrics.items():
        LOGGER.info("%s %-15s = %.4f", tag, k, v)