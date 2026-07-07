"""pipeline_orchestrator.py

Experiment tracking/logging is handled via MLflow.
To launch the MLflow UI, open a new terminal window (from the milk/ root directory) and run:

    $ mlflow ui --backend-store-uri exp_log/mlflow_logs

Once running, the UI will be accessible at http://127.0.0.1:5000.
"""

import logging
from ppl.pipeline.pipeline_initialization import initialize_pipeline_components, cleanup_resources
from ppl.pipeline.pipeline_execution import execute_pipeline
from ppl.config.pipeline_config import PipelineConfig

LOGGER = logging.getLogger(__name__)

# A list of attributes that may hold resources requiring explicit cleanup.
_SAFE_CLEANUP_ATTRS: tuple[str, ...] = (
    "resource_manager",
    "component_factory",
    "config_manager",
    "model_trainer",
    "split_trainer",
)

class PipelineOrchestrator:
    """Coordinate the end-to-end machine-learning pipeline.

    This class focuses on high-level orchestration
    and delegates all heavy lifting to helper classes created by
    `initialize_pipeline_components`.

    Parameters
    ----------
    cfg : PipelineConfig
        Pipeline configuration.
    """

    # Construction helpers
    def __init__(self, cfg: PipelineConfig) -> None:
        self.config: PipelineConfig = cfg
        self._components_ready: bool = False

        try:
            self._initialize_components()
            self._components_ready = True
        except Exception as exc:
            self._cleanup_resources()
            LOGGER.error("Pipeline initialization failed: %s", exc, exc_info=True)
            raise

    def _initialize_components(self) -> None:
        """Create all pipeline sub-components."""
        (
            self.config_manager,
            self.resource_manager,
            self.component_factory,
            self.task,
        ) = initialize_pipeline_components(self.config)

        self.model_trainer = self.component_factory.model_trainer
        self.split_trainer = self.component_factory.split_trainer

    # Resource management
    def _cleanup_resources(self) -> None:
        """Release memory and clean up resources.

        This method collects all components that might need cleanup and
        passes them to the cleanup_resources function. It then marks the
        pipeline as not ready for execution.

        This method is idempotent - calling it multiple times will only
        perform cleanup once.
        """
        # If components are already cleaned up, return early
        if not self._components_ready:
            return

        # Collect components that might need cleanup
        components = [
            getattr(self, attr)
            for attr in _SAFE_CLEANUP_ATTRS
            if hasattr(self, attr)
        ]

        # Clean up resources
        cleanup_resources(components)

        # Mark the pipeline as not ready for execution
        self._components_ready = False

    # Public API
    def run(self) -> None:
        """Train on train, validate on val, and test on test (predefined split)."""
        if not self._components_ready:
            raise RuntimeError("Attempted to run pipeline before it was ready.")

        try:
            execute_pipeline(self.config, self.split_trainer)
        finally:
            # Always attempt cleanup – even if execute_pipeline raised.
            self._cleanup_resources()

    # Context-manager helpers
    def __enter__(self) -> "PipelineOrchestrator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._cleanup_resources()
        # Propagate exceptions – caller decides what to do.
        return False
