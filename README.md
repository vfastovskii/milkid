# MILK — Multi-Instance Learning Kit

**MILK** trains multiple-instance-learning (MIL) models on **molecular conformer bags**:
each molecule is a *bag*, its 3D conformers are the *instances*, and the model predicts a
single molecule-level endpoint (e.g. activity/affinity) by learning which conformers matter.

Built on **PyTorch Lightning**, with **MLflow** experiment tracking and a single-command CLI.

- **Task:** regression (or classification) over bags of conformer descriptors.
- **Data:** one CSV where each row is a conformer; rows are grouped into bags by a molecule id.
- **Split:** predefined `split` column — `0 = train`, `1 = val`, `2 = test`. One run: train on
  train, select the best checkpoint on val, evaluate on test.

---

## Concept

```
Molecule (bag)                          Model
────────────────                        ─────────────────────────────
mol_id = "M1"                           per-conformer descriptors
 ├─ conf_id M1_c0  ─ descriptors ─┐
 ├─ conf_id M1_c1  ─ descriptors ─┤──►  embed each conformer
 ├─ conf_id M1_c2  ─ descriptors ─┤──►  attention-pool over conformers ──►  ŷ  (one value per molecule)
 └─ conf_id M1_c3  ─ descriptors ─┘──►  predict
endpoint = 6.7  (label, per molecule)
```

The descriptor columns are fingerprint blocks (e.g. `GETAWAYFingerprint_*`, `WHIMFingerprint_*`,
`USRCATFingerprint_*`, …), auto-detected from the CSV.

---

## Pipeline

```mermaid
flowchart LR
    CSV[CSV of conformers] --> DATA["1 · Data<br/>split → bags → scale → cluster"]
    DATA --> MODEL["2 · Model<br/>embedder → aggregator → predictor"]
    MODEL --> TRAIN["3 · Train + validate<br/>Lightning Trainer · monitor val_loss"]
    TRAIN --> TEST["4 · Test best checkpoint"]
    TEST --> OUT["5 · Artifacts + MLflow"]
```

**Stages**

1. **Data** (`ppl/data`) — load CSV → apply predefined split → build bags → per-fingerprint-block
   `StandardScaler` (fit on **train only**) → optional per-bag conformer clustering (agglomerative,
   silhouette-selected) → sqrt-block-size normalization. Processed bags are held in memory
   for the run.
2. **Model** (`ppl/models`) — assemble `embedder → aggregator → predictor` (see below).
3. **Train/validate** (`ppl/training`) — `Trainer.fit` on train, monitor **`val_loss`**, keep the
   best checkpoint (with optional attention-refinement schedule).
4. **Test** — evaluate the best checkpoint on the held-out test split.
5. **Artifacts** — prediction CSVs, a `res.txt` summary, attention-weight plots, and MLflow metrics.

### Model architecture

```mermaid
flowchart LR
    X[conformer descriptors<br/>input_dim] --> E[Embedder<br/>contextualized MLP<br/>+ bag self-attention]
    E --> A[Aggregator<br/>cluster-hierarchical<br/>multi-head attention]
    A --> P[Predictor<br/>MLP head]
    P --> Y[endpoint ŷ]
    A -. query .-> M[(Active-prototype<br/>memory · optional)]
```

The three components are chosen by name in the config and instantiated directly from a small
catalog (`ppl/models/component_catalog.py`) — no template/registry indirection.

| Slot | Config key | Value |
|------|------------|-------|
| Embedder | `embedder_type` | `contextualized_mlp_embedder` |
| Aggregator | `aggregator_type` | `cluster_hier_mha` |
| Predictor | `predictor_type` | `mlp_predictor` |

---

## Installation

Requires **Python 3.11 or 3.12** and [Poetry](https://python-poetry.org/).

```bash
# 1. clone
git clone https://github.com/vfastovskii/milkid.git
cd milkid

# 2. fetch the dataset (tracked with Git LFS)
git lfs install
git lfs pull                 # materializes ppl/data/*.csv (~340 MB)

# 3. install dependencies + the `milk` CLI
poetry install
```

> If `git lfs pull` is skipped, `ppl/data/*.csv` remain small pointer files and a run fails with
> `CSV missing required columns`.

Key dependencies (pinned in `pyproject.toml`): `torch`, `lightning`, `torchmetrics`,
`scikit-learn`, `mlflow`, `rdkit`, `optuna`, `pandas`, `matplotlib`.

---

## Quickstart

```bash
# run with the default config (ppl/config/experiment_configs/run_config.yaml)
poetry run milk

# run with a custom config
poetry run milk -c path/to/your_config.yaml

# verbose logging
poetry run milk --log-level DEBUG
```

At the default `INFO` level the output is a minimal, staged narrative:

```
Config: run_config.yaml
Experiment: my_experiment
Preparing data…
Data ready: 647 train / 324 val / 0 test bags · 472 features
Model: contextualized_mlp_embedder → cluster_hier_mha → mlp_predictor (3.8M params)
Training up to 100 epochs on mps…
Epoch 0 · train loss 0.904 rmse 1.115 mae 0.925 · val loss 0.907 rmse 0.953 mae 0.760
Epoch 1 · train loss 0.645 rmse 0.953 mae 0.770 · val loss 0.686 rmse 0.828 mae 0.664
…
Evaluating best model on the test split…
Test: rmse 0.812 mae 0.640
Results saved to results/my_experiment
Pipeline finished.
```

Use `--log-level DEBUG` to see every technical detail (feature scaling, optimizer groups,
per-block stats, memory, etc.).

---

## Configuration

A run is fully described by one YAML with three sections: `data`, `model`, `trainer`
(see `ppl/config/experiment_configs/run_config.yaml`). Highlights:

### `data`
```yaml
data:
  csv_path: ppl/data/<your>.csv
  task: regression

  # column names
  bag_id_col: mol_id        # groups rows into bags (molecules)
  inst_id_col: conf_id      # instance id (conformer)
  endpoint_value_col: endpoint
  split_col: split          # 0=train, 1=val, 2=test
  series_col: Series        # optional congeneric-series label

  predefined_split: true    # use split_col; if false, stratified split is created
  seed: 42                  # single RNG seed for splits + dataloaders
  batch_size: 13
  balance_train_batches_by_series: true   # each batch covers every series

  cluster_instances: true   # per-bag conformer clustering in scaled space
  cluster_selection_method: silhouette
  cluster_max_clusters: 6
```

### `model`
```yaml
model:
  embedder_type: contextualized_mlp_embedder
  aggregator_type: cluster_hier_mha
  predictor_type: mlp_predictor
  task: regression
  embedder_kwargs:   { ... }   # per-component hyperparameters
  aggregator_kwargs: { ... }
  predictor_kwargs:  { ... }
  active_prototype_kwargs: { enabled: true, ... }   # optional prototype memory
  optim: { lr: 1.0e-4, weight_decay: 3.0e-3, scheduler: none, ... }   # aggregator-focus curriculum owns the LR
```

### `trainer`
```yaml
trainer:
  max_epochs: 100
  device: mps                 # "cuda" | "mps" | "cpu"
  precision: "32-true"
  experiment_name: my_experiment
  log_save_dir: my_experiment_log
  checkpoint_monitor: val_loss
  checkpoint_min_epoch: 20
  attention_refinement_enabled: true   # keep training past overfit with a query ramp
  save_attention_artifacts: true
```

> **Splitting:** with `predefined_split: true` the pipeline reads the `split` column directly.
> With `predefined_split: false` it creates a stratified test split (`test_size`) and, if
> `val_partition: true`, a stratified validation split from the remaining train.

---

## Outputs

For `experiment_name = my_experiment`, results land under `results/my_experiment/`:

```
results/my_experiment/
├── train_fit.csv          # mol_id, true, predicted, abs_error  (train)
├── val.csv                # …                                    (validation)
├── test.csv               # …                                    (test)
├── res.txt                # run summary: val/train metrics, best epoch, schedule
├── train/  validation/  test/
│                          # attention-weight plots, true-vs-predicted figures
├── preprocessing.joblib   # per-block StandardScalers + preprocessing metadata
└── models/                # best checkpoint (.ckpt)
```

Metrics, params, and the model are also logged to **MLflow**. Browse them with:

```bash
mlflow ui --backend-store-uri <log_save_dir>
# then open http://127.0.0.1:5000
```

---

## Package layout

```
ppl/
├── cli/         command-line entry point + logging setup
├── config/      dataclass configs + experiment_configs/*.yaml
├── data/        MILDataModule, bag building, feature scaling, clustering, samplers
├── models/      model factory + component catalog
│   ├── components/   embedders / aggregators / predictors
│   └── lightning/    LightningModule (training/validation/test steps, optimizers)
├── training/    trainer, callbacks, checkpoints, artifacts, predictions
├── pipeline/    orchestrator, builder, component factory, MLflow utils
├── plotting/    attention-weight and true-vs-predicted plots
├── hpo/         Optuna hyperparameter search
└── utils/       reproducibility helpers
```

The pipeline is a single, one-directional flow (no circular imports) and Lightning owns the
training loop; checkpointing/early-stopping/logging are handled via callbacks.

---

## Notebooks

`notebooks/` mirrors the pipeline step-by-step (they call the same classes as the CLI):

| Notebook | Step |
|----------|------|
| `00_overview_and_config.ipynb` | Load & inspect the config |
| `01_data_preprocessing.ipynb` | `MILDataModule.setup()` — bags, clustering, scaling |
| `02_model_construction.ipynb` | Build the MIL model |
| `03_model_training.ipynb` | Train → validate → test |

Select the Poetry environment as the Jupyter kernel and run `git lfs pull` first.

---

## Hyperparameter optimization (optional)

Multi-objective Optuna search lives in `ppl/hpo/optuna_runner.py` — it maximizes
`val_rmsd_top1` and minimizes `val_rmse` (read from each trial's `run_metrics.json`):

```bash
poetry run python -m ppl.hpo.optuna_runner \
  --search-space ppl/config/experiment_configs/optuna_search_space.yaml \
  --n-trials 40
```

Each trial samples overrides from the search-space YAML (see that file for the format),
writes a concrete `config.yaml`, and runs the normal MILK CLI as a subprocess. Trial
configs/logs land under `ppl/optuna_trials/<study-name>/` by default (`--trial-root` to
override), and the non-dominated set is written to `<trial-root>/pareto_front.json` when
the study finishes. Pass `--dry-run` to sample and write trial configs without launching
training — useful for sanity-checking a search-space YAML.

`ppl/hpo/optuna_final_pipeline.py` (final retrain from search results) and the SLURM
launcher at `ppl/scripts/run_optuna_final_milk.slurm` run the multi-objective flow: they
select the best trial from `<trial-root>/pareto_front.json` and retrain it over the
confirmation seeds.
