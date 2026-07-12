#!/bin/bash
#SBATCH --job-name=mlkJul
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''
#SBATCH --partition=qGPUr
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
shopt -s nullglob
ulimit -n 4096 || true

# -------------------- conda --------------------
CONDA_SH="${CONDA_SH:-/sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-milk_py311}"
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "[WARN] CONDA_SH not found: $CONDA_SH"
  echo "[WARN] Continuing with current python: $(command -v python3 || true)"
fi

# -------------------- C++ runtime compatibility --------------------
unset LD_PRELOAD || true
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
echo "[INFO] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

python3 - <<'PY'
import sys
required = (3, 11)
current = sys.version_info[:3]
print(f"[INFO] python={sys.version.split()[0]} executable={sys.executable}")
if current < required:
    raise SystemExit(
        "[ERROR] MILK requires Python >=3.11,<3.13. "
        f"Current Python is {sys.version.split()[0]} at {sys.executable}. "
        "Activate/create a conda environment with Python 3.11 or 3.12 before "
        "submitting this job."
    )
for name in ("torch", "pytorch_lightning", "optuna", "yaml"):
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "<no __version__>")
        print(f"[INFO] import OK: {name} version={version}")
    except Exception as exc:
        print(f"[ERROR] import failed: {name}: {exc!r}")
        sys.exit(3)
try:
    import torch
    print(f"[INFO] cuda_available={torch.cuda.is_available()}")
except Exception:
    pass
PY

# -------------------- python / threading hygiene --------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export MKL_DYNAMIC=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# -------------------- project / data paths --------------------
# Repository root on the shared filesystem.
REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/milk_udt}"
PKGDIR="${PKGDIR:-${REPO_DIR}/ppl}"
DATA_DIR="${DATA_DIR:-${PKGDIR}/data}"
DATA_CSV="${DATA_CSV:-${DATA_DIR}/bace809_3d_fp_5fp_200c_05prune_6kcal_no_Hs_june_concated_splited.csv}"

BASE_CONFIG="${BASE_CONFIG:-${PKGDIR}/config/experiment_configs/run_config.yaml}"
SEARCH_SPACE="${SEARCH_SPACE:-${PKGDIR}/config/experiment_configs/optuna_search_space_bace809_cluster_hier_mha.yaml}"

# -------------------- run knobs --------------------
STUDY_NAME="${STUDY_NAME:-bace809_cluster_hier_mha_optuna}"
TRIALS="${TRIALS:-40}"
METRIC="${METRIC:-val_rmse}"
LOAD_IF_EXISTS="${LOAD_IF_EXISTS:-0}"

FINAL_RUN_NAME="${FINAL_RUN_NAME:-final_best}"
FINAL_EXPERIMENT_NAME="${FINAL_EXPERIMENT_NAME:-}"
FINAL_LOG_SAVE_DIR="${FINAL_LOG_SAVE_DIR:-}"
FINAL_STAGE2="${FINAL_STAGE2:-1}"
SAVE_FINAL_ATTENTION="${SAVE_FINAL_ATTENTION:-1}"
FINAL_VALIDATION_INSTANCE_IMPORTANCE="${FINAL_VALIDATION_INSTANCE_IMPORTANCE:-0}"
FINAL_LOG_PER_EPOCH="${FINAL_LOG_PER_EPOCH:-1}"
FINAL_PROGRESS_BAR="${FINAL_PROGRESS_BAR:-1}"
FINAL_MODEL_SUMMARY="${FINAL_MODEL_SUMMARY:-1}"

DEVICE="${DEVICE:-cuda}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32-true}"
NUM_WORKERS="${NUM_WORKERS:-10}"
PIN_MEMORY="${PIN_MEMORY:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

STUDY_TIMEOUT="${STUDY_TIMEOUT:-}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-}"

# -------------------- node-local scratch --------------------
JOB_ID="${SLURM_JOB_ID:-local}"
WORKDIR="${SLURM_TMPDIR:-/tmp}/${JOB_ID}"
STUDY_DIR="${WORKDIR}/study_${JOB_ID}"
OUTDIR="${OUTDIR:-${REPO_DIR}/optuna_runs/study_${JOB_ID}}"

mkdir -p "$WORKDIR" "$STUDY_DIR" "$OUTDIR"

copy_outputs_back() {
  local rc=$?
  trap - EXIT
  set +e
  echo "[INFO] Copy study outputs back... exit_code=${rc}"
  rsync -av "$STUDY_DIR/" "$OUTDIR/"

  if [[ -d "$WORKDIR/ppl/results" ]]; then
    mkdir -p "$OUTDIR/pipeline_outputs/results"
    rsync -av "$WORKDIR/ppl/results/" "$OUTDIR/pipeline_outputs/results/"
  fi

  for d in "$WORKDIR"/ppl/*_log "$WORKDIR"/ppl/*_log_* "$WORKDIR"/ppl/lightning_logs; do
    if [[ -d "$d" ]]; then
      mkdir -p "$OUTDIR/pipeline_outputs/$(basename "$d")"
      rsync -av "$d/" "$OUTDIR/pipeline_outputs/$(basename "$d")/"
    fi
  done

  echo "[INFO] Done. Results in: $OUTDIR"
  exit "$rc"
}
trap copy_outputs_back EXIT

echo "[INFO] JobID: ${JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Study dir (node-local): $STUDY_DIR"
echo "[INFO] Output dir: $OUTDIR"
echo "[INFO] Package dir: $PKGDIR"
echo "[INFO] Data dir: $DATA_DIR"
echo "[INFO] Data CSV: $DATA_CSV"
echo "[INFO] Search space: $SEARCH_SPACE"
echo "[INFO] Base config: $BASE_CONFIG"
echo "[INFO] Single-study HPO + final retrain: TRIALS=$TRIALS METRIC=$METRIC"
echo "[INFO] Final run: run_name=$FINAL_RUN_NAME stage2=$FINAL_STAGE2 save_attention=$SAVE_FINAL_ATTENTION log_per_epoch=$FINAL_LOG_PER_EPOCH validation_instance_importance=$FINAL_VALIDATION_INSTANCE_IMPORTANCE"
echo "[INFO] Device: device=$DEVICE accelerator=$ACCELERATOR devices=$DEVICES precision=$PRECISION num_workers=$NUM_WORKERS pin_memory=$PIN_MEMORY"

[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }
[[ -f "$DATA_CSV" ]] || { echo "[ERROR] Missing data CSV: $DATA_CSV"; exit 2; }
[[ -f "$BASE_CONFIG" ]] || { echo "[ERROR] Missing base config: $BASE_CONFIG"; exit 2; }
[[ -f "$SEARCH_SPACE" ]] || { echo "[ERROR] Missing search space: $SEARCH_SPACE"; exit 2; }

echo "[INFO] Copy package and input CSV to node-local scratch..."
rsync -av \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".git" \
  --exclude="results" \
  --exclude="*_log" \
  --exclude="*_log_*" \
  --exclude="optuna_trials" \
  "$PKGDIR" "$WORKDIR/"

mkdir -p "$WORKDIR/data"
cp -v "$DATA_CSV" "$WORKDIR/data/"
DATA_CSV_LOCAL="$WORKDIR/data/$(basename "$DATA_CSV")"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORKDIR/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

cd "$WORKDIR"
python3 -c "import ppl; print('[INFO] ppl import OK from', ppl.__file__)"

# Objectives are fixed (maximize val_rmsd_top1, minimize val_rmse); the runner resumes
# an existing study automatically. The lean pipeline retrains the highest-rmsd_top1
# Pareto-front config over CONFIRMATION_SEEDS; the removed --metric/--final-*/timeout
# knobs live in the base config / search space now.
CMD=(
  python3 -m ppl.hpo.optuna_final_pipeline
  --search-space "$WORKDIR/ppl/config/experiment_configs/optuna_search_space.yaml"
  --base-config "$WORKDIR/ppl/config/experiment_configs/run_config.yaml"
  --data-csv "$DATA_CSV_LOCAL"
  --output-dir "$STUDY_DIR"
  --trial-root "$STUDY_DIR/trials"
  --study-name "$STUDY_NAME"
  --n-trials "$TRIALS"
  --device "$DEVICE"
  --accelerator "$ACCELERATOR"
  --devices "$DEVICES"
  --precision "$PRECISION"
  --num-workers "$NUM_WORKERS"
  --log-level "$LOG_LEVEL"
)

# Retrain the selected config over these seeds (stability). Space-separated, e.g. "42 43 44".
if [[ -n "${CONFIRMATION_SEEDS:-}" ]]; then
  CMD+=(--confirmation-seeds ${CONFIRMATION_SEEDS})
fi
# Set SKIP_HPO=1 to reuse an existing pareto_front.json and jump to selection + retrain.
if [[ "${SKIP_HPO:-0}" == "1" ]]; then
  CMD+=(--skip-hpo)
fi
printf '[INFO] CMD:'
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
