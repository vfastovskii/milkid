import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass(slots=True, frozen=False)
class DataLoaderConfig:
    """Container for all pipeline configuration‑related parameters."""

    csv_path: Path | str
    task: str = "regression"  # or "classification"
    descriptor_cols: Optional[List[str]] = None  # auto-detected if None

    # column names
    bag_id_col: str = "mol_id"
    inst_id_col: str = "conf_id"
    endpoint_value_col: str = "endpoint"
    energy_col: str = "Energy"
    pdb_id_col: str = "mol_id"
    smiles_col: str = "smiles"
    split_col: str = "split"  # 0=train,1=val,2=test
    series_col: Optional[str] = None

    # splits
    predefined_split: bool = True  # use the [split_col] column: 0=train, 1=val, 2=test
    test_size: float = 0.1  # only used when predefined_split=False
    n_strat_bins: int = 5
    seed: int = 42  # single RNG seed for splits and dataloaders

    # loader hyper‑parameters
    batch_size: int = 1  # variable‑length bags, stick to bs=1
    num_workers: int = os.cpu_count()
    pin_memory: bool = True
    balance_train_batches_by_series: bool = False

    # optional per-bag conformer clustering in scaled descriptor space
    cluster_instances: bool = False
    cluster_selection_method: str = "silhouette"  # silhouette, distance, or fixed
    cluster_max_clusters: int = 6
    cluster_min_clusters: int = 1
    cluster_min_silhouette: float = 0.05
    cluster_distance_threshold: Optional[float] = None
    cluster_linkage: str = "average"
    cluster_metric: str = "euclidean"

    # memory management and caching
    cache_dir: Optional[Union[Path, str]] = None  # directory to cache processed bags
    memory_limit: Optional[float] = None  # maximum memory usage in GB
    on_demand_loading: bool = False  # load bags from disk when needed
    experiment_name: Optional[str] = None  # experiment name for saving splits and models

    # when predefined_split=False, carve a stratified validation set out of train
    val_partition: bool = True
