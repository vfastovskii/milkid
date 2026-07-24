import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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

    # drop bags whose endpoint is below this value (e.g. pIC50 < 4), from every
    # split. None disables the filter.
    min_endpoint_value: Optional[float] = None  # None = keep full activity range (no low-endpoint drop)

    # splits
    predefined_split: bool = True  # use the [split_col] column: 0=train, 1=val, 2=test
    test_size: float = 0.1  # only used when predefined_split=False
    n_strat_bins: int = 5
    seed: int = 42  # single RNG seed for splits and dataloaders

    # loader hyper‑parameters
    batch_size: int = 1  # variable‑length bags, stick to bs=1
    num_workers: int = os.cpu_count() or 1
    pin_memory: bool = True
    balance_train_batches_by_series: bool = False
    # When balanced batching is on, importance-weight the train loss by
    # w(s)=|s|·n_series/N to undo the sampler's group oversampling (unbiased
    # objective). Set False for a deliberately macro-averaged (per-series) objective.
    series_balance_importance_weight: bool = True

    # Label-Distribution-Smoothing (LDS) target-density loss weighting. Multiplies
    # each train bag's loss by w_lds(y) = density(y)^(-alpha) (Gaussian-smoothed
    # histogram of pIC50, mean-normalised, clipped to [lds_min, lds_max]) so the
    # sparse activity extremes get gradient share proportional to their rarity,
    # countering the MSE-toward-mean shrinkage. Combines multiplicatively with the
    # series importance weight. Off by default.
    target_density_weighting: bool = False
    lds_alpha: float = 0.5       # 0 = uniform, 1 = full inverse-density; temper on small tails
    lds_sigma: float = 1.0       # Gaussian smoothing kernel width, in histogram bins
    lds_bin_width: float = 0.5   # pIC50 histogram bin width
    lds_min: float = 0.5         # per-sample weight floor (after mean-normalisation)
    lds_max: float = 4.0         # per-sample weight cap — keeps a handful of extremes from dominating

    # optional per-bag conformer clustering in scaled descriptor space
    cluster_instances: bool = False
    cluster_selection_method: str = "silhouette"  # silhouette, distance, or fixed
    cluster_max_clusters: int = 6
    cluster_min_clusters: int = 1
    cluster_min_silhouette: float = 0.05
    cluster_distance_threshold: Optional[float] = None
    cluster_linkage: str = "average"
    cluster_metric: str = "euclidean"

    experiment_name: Optional[str] = None  # experiment name for saving results/models

    # when predefined_split=False, carve a stratified validation set out of train
    val_partition: bool = True
