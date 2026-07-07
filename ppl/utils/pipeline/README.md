# Pipeline Utilities for MILK

This package contains utilities for the Multi-Instance Learning Kit (MILK) pipeline. It provides a modular and maintainable structure for the pipeline components.

## Package Structure

The package is organized into the following modules:

### `config_utils.py`

Utilities for handling configuration objects, including:
- `override_dataclass`: Override a dataclass instance with values from dictionaries or other dataclasses

### `mlflow_utils.py`

Utilities for MLFlow integration, including:
- `SafeMLFlowLogger`: A custom MLFlow logger that fixes issues with artifact paths
- `create_mlflow_logger`: Create an MLFlow logger with proper configuration
- `log_metrics`: Log metrics to the console

### `visualization.py`

Utilities for visualizing data distributions and metrics, including:
- `log_split_distributions`: Log label distributions for each split as MLFlow artifacts

## Usage

These utilities are used by the main `PipelineOrchestrator` class to provide a clean and modular implementation of the pipeline. They can also be used independently for other purposes.

Example:

```python
from ppl.utils.pipeline.config_override_utils import override_dataclass
from ppl.utils.pipeline.mlflow_utils import create_mlflow_logger, log_metrics
from ppl.utils.pipeline.visualization import log_split_distributions

# Override a dataclass instance
config = override_dataclass(base_config, overrides)

# Create an MLFlow logger
logger = create_mlflow_logger(
    save_dir=Path("./logs"),
    experiment_name="my_experiment",
    run_name="my_run",
    tracking_uri=None,
)

# Log metrics
log_metrics(metrics, stage="val", prefix="fold0")

# Log label distributions
log_split_distributions(data_module, stage="fold0")
```
