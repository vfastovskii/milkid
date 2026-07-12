#!/usr/bin/env bash
# Fast local multi-objective HPO on Apple-Silicon MPS.
#
#   - N_TRIALS trials, N_JOBS of them running concurrently on the one MPS GPU
#   - attention weights + per-epoch dumps are saved ONLY for the final best-params retrain
#     (HPO trials skip them for speed — see optuna_search_space.yaml fixed_overrides)
#   - objectives: maximize val_rmsd_top1, minimize val_rmse; final pick = highest val_rmsd_top1
#
# Outputs land in results/<STUDY_NAME>_final_pipeline/ (pareto_front.json, final_pipeline_summary.json,
# and the final model's attention weights under results/<...>_final/seed42/).
#
# Overrides (env vars):  N_TRIALS  N_JOBS  NUM_WORKERS  CONFIRMATION_SEEDS  STUDY_NAME  HPO_PYTHON
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

# --- MPS hygiene: fall back to CPU for the few ops MPS lacks (else training crashes mid-run) ---
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# --- knobs ---
N_TRIALS="${N_TRIALS:-20}"
N_JOBS="${N_JOBS:-3}"              # concurrent trials sharing the single MPS GPU; drop to 2 if you OOM
NUM_WORKERS="${NUM_WORKERS:-0}"    # dataloader workers per trial; 0 is safest on macOS+MPS, try 2-4 to overlap I/O
CONFIRMATION_SEEDS="${CONFIRMATION_SEEDS:-42}"
# Fresh study each run — the search space changed, so resuming an old study.db would clash.
STUDY_NAME="${STUDY_NAME:-milk_hpo_mac_$(date +%Y%m%d_%H%M%S)}"

echo "[hpo-mac] python=$PY"
echo "[hpo-mac] trials=$N_TRIALS  n_jobs=$N_JOBS (concurrent on MPS)  num_workers=$NUM_WORKERS  seeds=$CONFIRMATION_SEEDS"
echo "[hpo-mac] study=$STUDY_NAME  ->  results/${STUDY_NAME}_final_pipeline/"

exec "$PY" -m ppl.hpo.optuna_final_pipeline \
  --search-space ppl/config/experiment_configs/optuna_search_space.yaml \
  --study-name "$STUDY_NAME" \
  --n-trials "$N_TRIALS" \
  --n-jobs "$N_JOBS" \
  --confirmation-seeds ${CONFIRMATION_SEEDS} \
  --device mps --accelerator mps --devices 1 --precision 32-true \
  --num-workers "$NUM_WORKERS" \
  --output-dir "results/${STUDY_NAME}_final_pipeline" \
  --log-level INFO \
  "$@"
