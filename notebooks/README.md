# MILK pipeline notebooks

A family of Jupyter notebooks that launch each step of the MILK (Multi-Instance
Learning Kit) pipeline. They call the **same classes** as `poetry run milk` —
nothing is re-implemented — so results match the CLI.

| Notebook | Step | What it launches |
|----------|------|------------------|
| [`00_overview_and_config.ipynb`](00_overview_and_config.ipynb) | Setup | Loads & inspects the experiment YAML config |
| [`01_data_preprocessing.ipynb`](01_data_preprocessing.ipynb) | Data preprocessing | `MILDataModule.setup()` — build bags, cluster conformers, scale features |
| [`02_model_construction.ipynb`](02_model_construction.ipynb) | Model construction | `ModelTrainer.build_model()` — embedder → aggregator → predictor |
| [`03_model_training.ipynb`](03_model_training.ipynb) | Training & evaluation | Stage 1 (CV / train-val) + Stage 2 (final fit + test) |

## Running them

1. **Use the project's Poetry environment as the kernel** (it has torch, lightning,
   rdkit, etc.):
   ```bash
   poetry install
   poetry run python -m ipykernel install --user --name milk
   poetry run jupyter notebook        # or open the folder in VS Code / Jupyter
   ```
   Then select the **milk** kernel.

2. **Pull the data.** The CSV under `ppl/data/` is tracked with Git LFS. If it is
   only a pointer file, preprocessing/training will fail with a
   `CSV missing required columns` error until you run:
   ```bash
   git lfs pull
   ```

3. Run the notebooks in order (00 → 03). Each is also runnable standalone: they
   all read the same `CONFIG_PATH` defined in the second cell, and a bootstrap
   cell locates the project root and `chdir`s to it so paths resolve correctly.

## Changing the experiment

Edit `CONFIG_PATH` (second code cell) in each notebook to point at a different
YAML under `ppl/utils/experiment_configs/`. For a quick smoke test, lower
`trainer.max_epochs` in the YAML before running notebook 03.

## Where results go

- **MLflow**: `mlflow ui --backend-store-uri exp_log/mlflow_logs` (from repo root).
- **Results directory**: predictions (`train_fit.csv`, `val.csv`, `test.csv`),
  `res.txt`, attention-weight plots and true-vs-pred figures, under the
  experiment folder named by `trainer.experiment_name`.
