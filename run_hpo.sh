#!/usr/bin/env bash
# Local multi-objective HPO on macOS CPU, trials run SEQUENTIALLY.
#
# For this small model CPU beats MPS (no per-op kernel-launch overhead, and one sequential
# trial gets every CPU core via PyTorch intra-op parallelism) — so this is the fast path.
#   - N_TRIALS trials, one at a time (N_JOBS=1) -> a live Optuna trial progress bar
#   - attention weights + per-epoch dumps saved ONLY for the final best-params retrain
#     (HPO trials skip them for speed — optuna_search_space.yaml fixed_overrides)
#   - objectives: maximize val_rmsd_top1, minimize val_rmse; final pick = highest val_rmsd_top1
#
# Outputs -> results/<STUDY_NAME>_final_pipeline/ (pareto_front.json, final_pipeline_summary.json).
# The final model's attention weights land under ppl/results/<STUDY_NAME>..._final/seed42/.
#
# Overrides (env): N_TRIALS  N_JOBS  NUM_WORKERS  DEVICE  CONFIRMATION_SEEDS  STUDY_NAME  HPO_PYTHON
set -euo pipefail
cd "$(dirname "$0")"

# --- python with torch+optuna+rdkit+lightning (prefer the project poetry venv) ---
PY="${HPO_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in "$HOME"/Library/Caches/pypoetry/virtualenvs/milk-*/bin/python /opt/anaconda3/envs/*/bin/python; do
    [[ -x "$c" ]] || continue
    "$c" -c "import torch, optuna, rdkit, pytorch_lightning" 2>/dev/null && PY="$c" && break
  done
fi
PY="${PY:-python3}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# No OMP/MKL thread caps: let PyTorch use all CPU cores for intra-op parallelism (the CPU speedup).

# --- knobs ---
N_TRIALS="${N_TRIALS:-20}"
N_JOBS="${N_JOBS:-1}"              # sequential — each trial gets all cores; >1 just oversubscribes the CPU
NUM_WORKERS="${NUM_WORKERS:-0}"    # dataloader workers; data is tiny, 0 is fine (try 2-4 to overlap I/O)
DEVICE="${DEVICE:-cpu}"            # CPU is faster than MPS here; set DEVICE=mps to try the GPU
CONFIRMATION_SEEDS="${CONFIRMATION_SEEDS:-42}"
# Fresh study each run — the search space changed, so resuming an old study.db would clash.
STUDY_NAME="${STUDY_NAME:-milk_hpo_mac_$(date +%Y%m%d_%H%M%S)}"

echo "[hpo-mac] python=$PY  device=$DEVICE"
echo "[hpo-mac] trials=$N_TRIALS  n_jobs=$N_JOBS (sequential)  num_workers=$NUM_WORKERS  seeds=$CONFIRMATION_SEEDS"
echo "[hpo-mac] study=$STUDY_NAME  ->  results/${STUDY_NAME}_final_pipeline/"

exec "$PY" -m ppl.hpo.optuna_final_pipeline \
  --search-space ppl/config/experiment_configs/optuna_search_space.yaml \
  --study-name "$STUDY_NAME" \
  --n-trials "$N_TRIALS" \
  --n-jobs "$N_JOBS" \
  --confirmation-seeds ${CONFIRMATION_SEEDS} \
  --device "$DEVICE" --accelerator "$DEVICE" --devices 1 --precision 32-true \
  --num-workers "$NUM_WORKERS" \
  --output-dir "results/${STUDY_NAME}_final_pipeline" \
  --log-level INFO \
  "$@"
