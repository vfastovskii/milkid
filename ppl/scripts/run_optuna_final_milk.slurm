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
# Repository root on persistent scratch. The job runs HERE in place — no node-local /tmp
# and no staging copy.
REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/milkid_opt_july}"
PKGDIR="${PKGDIR:-${REPO_DIR}/ppl}"
DATA_DIR="${DATA_DIR:-${PKGDIR}/data}"
DATA_CSV="${DATA_CSV:-${DATA_DIR}/bace809_3d_fp_5fp_200c_05prune_6kcal_no_Hs_june_concated_splited.csv}"

BASE_CONFIG="${BASE_CONFIG:-${PKGDIR}/config/experiment_configs/run_config.yaml}"
SEARCH_SPACE="${SEARCH_SPACE:-${PKGDIR}/config/experiment_configs/optuna_search_space.yaml}"

# -------------------- run knobs --------------------
STUDY_NAME="${STUDY_NAME:-bace809_cluster_hier_mha_optuna}"
TRIALS="${TRIALS:-50}"
CONFIRMATION_SEEDS="${CONFIRMATION_SEEDS:-42}"   # space-separated for repeats, e.g. "42 43 44"
SKIP_HPO="${SKIP_HPO:-0}"                         # 1 = reuse existing pareto_front.json

DEVICE="${DEVICE:-cuda}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-32-true}"
NUM_WORKERS="${NUM_WORKERS:-10}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# -------------------- run in place on persistent scratch --------------------
# The job runs directly from REPO_DIR (persistent scratch) — no node-local /tmp, no staging
# copy, no end-of-job copy-back. All outputs (study DB, trial configs, pipeline results, mpl
# cache) stay under REPO_DIR, so nothing is lost when the node is reclaimed and the study DB
# resumes across re-submitted jobs.
JOB_ID="${SLURM_JOB_ID:-local}"
WORKDIR="${WORKDIR:-${REPO_DIR}}"
STUDY_DIR="${WORKDIR}/optuna_study"
mkdir -p "$STUDY_DIR"

report_outputs() {
  local rc=$?
  trap - EXIT
  echo "[INFO] exit_code=${rc}. All outputs are in persistent scratch:"
  echo "[INFO]   study dir:        $STUDY_DIR"
  echo "[INFO]   pareto front:     $STUDY_DIR/trials/pareto_front.json"
  echo "[INFO]   pipeline summary: $STUDY_DIR/final_pipeline_summary.json"
  echo "[INFO]   training results: $PKGDIR/results/"
  exit "$rc"
}
trap report_outputs EXIT

echo "[INFO] JobID: ${JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Workdir (run in place on scratch): $WORKDIR"
echo "[INFO] Study dir: $STUDY_DIR"
echo "[INFO] Package dir: $PKGDIR"
echo "[INFO] Data CSV: $DATA_CSV"
echo "[INFO] Search space: $SEARCH_SPACE"
echo "[INFO] Base config: $BASE_CONFIG"
echo "[INFO] Multi-objective HPO + final retrain: TRIALS=$TRIALS (maximize val_rmsd_top1, minimize val_rmse), confirmation_seeds=$CONFIRMATION_SEEDS"
echo "[INFO] Device: device=$DEVICE accelerator=$ACCELERATOR devices=$DEVICES precision=$PRECISION num_workers=$NUM_WORKERS"

[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }
[[ -f "$DATA_CSV" ]] || { echo "[ERROR] Missing data CSV: $DATA_CSV"; exit 2; }
[[ -f "$BASE_CONFIG" ]] || { echo "[ERROR] Missing base config: $BASE_CONFIG"; exit 2; }
[[ -f "$SEARCH_SPACE" ]] || { echo "[ERROR] Missing search space: $SEARCH_SPACE"; exit 2; }

# Run in place: no staging copy, the data CSV is read directly from scratch.
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
  --search-space "$SEARCH_SPACE"
  --base-config "$BASE_CONFIG"
  --data-csv "$DATA_CSV"
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
