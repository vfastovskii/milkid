"""Use case-specific configuration variants."""

from dataclasses import dataclass
from ppl.config.data_loader_config import DataLoaderConfig
from ppl.config.model_builder_config import ModelBuilderConfig
from ppl.config.model_trainer_config import ModelTrainerConfig

# Conformers Use Case (existing default)
@dataclass(slots=True, frozen=False)
class ConformersDataLoaderConfig(DataLoaderConfig):
    """Data loader configuration for conformers use case."""
    bag_id_col: str = "mol_id"
    inst_id_col: str = "conf_id"
    endpoint_value_col: str = "endpoint"
    energy_col: str = "Energy"
    pdb_id_col: str = "mol_id"
    smiles_col: str = "smiles"
    batch_size: int = 1
    test_size: float = 0.1

@dataclass(slots=True, frozen=False)
class ConformersModelBuilderConfig(ModelBuilderConfig):
    """Model builder configuration for conformers use case."""
    embedder_type: str = "contextualized_mlp_embedder"
    aggregator_type: str = "cluster_hier_mha"
    predictor_type: str = "mlp_predictor"
    input_dim: int = 128
    task: str = "regression"

@dataclass(slots=True, frozen=False)
class ConformersModelTrainerConfig(ModelTrainerConfig):
    """Model trainer configuration for conformers use case."""
    max_epochs: int = 150
    device: str = "cuda"
    log_save_dir: str = "exp_log"
    experiment_name: str = "mil_exp"