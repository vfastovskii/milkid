from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True, frozen=False)
class ModelTrainerConfig:
    """Holds all training/runtime parameters for the Lightning Trainer."""
    max_epochs: int = 150
    device: str = "cuda"
    log_save_dir: str = "exp_log"

    experiment_name: str = "mil_exp"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None