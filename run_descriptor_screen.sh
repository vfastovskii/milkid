#!/usr/bin/env bash
# End-to-end descriptor screen: one SDF of conformers+experimental poses in,
# a ranked list of descriptors (and concatenations) out, judged by BOTH
#   (1) clustering quality  -> descriptor_kid_cluster_screen.py
#   (2) MIL/LightGBM pIC50 prediction -> mil_mean_lgbm_descriptor_screen.py
#
# Usage:
#   ./run_descriptor_screen.sh <conformers.sdf> [output_dir] [-- extra cluster-screen flags]
#
#   ./run_descriptor_screen.sh ppl/kid_calculator/bace809_200confs_ls.sdf
#   ./run_descriptor_screen.sh my.sdf out/my_screen
#   ./run_descriptor_screen.sh my.sdf out/my_screen -- --sdf-remove-hs --concat-max-size 4
#
#   # Resume a crashed run from precalculated descriptors + pose_similarity.csv
#   # (no SDF needed; missing concats are regenerated, everything re-ranked):
#   RESUME=1 ./run_descriptor_screen.sh out/sch_descr_screen
#
#   # RESUME=2: repair a crashed/disk-full run. Regenerates truncated concat CSVs,
#   # re-clusters ONLY the affected sets, and patches the clustering rankings.
#   # LGBM is skipped (it's unaffected by concat-CSV truncation).
#   RESUME=2 ./run_descriptor_screen.sh out/sch_descr_screen
#
# Env overrides:
#   SCREEN_PYTHON=/path/to/python   # force interpreter (needs rdkit,skfp,lightgbm,sklearn,joblib)
#   WORKERS=8                       # parallel workers (default: all cores). 1 = serial/debug.
#   RESUME=1                        # reuse <out>/descriptors + <out>/pose_similarity.csv; no SDF needed
#   RESUME=2                        # repair truncated concats + re-cluster only those; skips LGBM
#   SKIP_CLUSTER=1                  # skip phase 1 entirely, run only the LGBM screen on <out>/descriptors
#   SKIP_LGBM=1                     # run only the clustering screen
set -euo pipefail
cd "$(dirname "$0")"

RESUME="${RESUME:-0}"
RESUME_ON=0; [[ "$RESUME" == "1" || "$RESUME" == "2" ]] && RESUME_ON=1

# Arg parsing. In RESUME mode the SDF is optional: if the first arg is an existing
# directory it is taken as the output dir and no SDF is used.
if [[ "$RESUME_ON" == "1" && "${1:-}" != "" && -d "${1:-}" && ! -f "${1:-}" ]]; then
  SDF="-"; OUT="$1"; shift || true
else
  SDF="${1:?usage: run_descriptor_screen.sh <conformers.sdf> [output_dir] [-- extra flags]}"
  shift || true
  OUT="descriptor_screen"
  if [[ $# -gt 0 && "$1" != "--" ]]; then OUT="$1"; shift || true; fi
fi
[[ "${1:-}" == "--" ]] && shift || true
EXTRA=("$@")   # forwarded to the cluster screen only

# SDF only needs to exist when we are actually going to read it (i.e. not a full resume).
if [[ "$RESUME_ON" != "1" ]]; then
  [[ -f "$SDF" ]] || { echo "SDF not found: $SDF" >&2; exit 1; }
  if head -1 "$SDF" | grep -q "git-lfs"; then
    echo "ERROR: $SDF is an unmaterialized git-lfs pointer. Run 'git lfs pull' first." >&2
    exit 1
  fi
fi

# Interpreter: env override, else this project's poetry venv, else first conda env with the stack.
PY="${SCREEN_PYTHON:-}"
if [[ -z "$PY" ]]; then
  cand="$(poetry env info -p 2>/dev/null)/bin/python"
  if [[ -x "$cand" ]] && "$cand" -c "import rdkit,skfp,lightgbm,sklearn,joblib" 2>/dev/null; then
    PY="$cand"
  fi
fi
if [[ -z "$PY" ]]; then
  for c in /opt/anaconda3/envs/*/bin/python; do
    "$c" -c "import rdkit,skfp,lightgbm,sklearn,joblib" 2>/dev/null && PY="$c" && break
  done
fi
PY="${PY:-python}"
if ! "$PY" -c "import rdkit,skfp,lightgbm,sklearn,joblib" 2>/dev/null; then
  echo "ERROR: interpreter '$PY' is missing required packages (rdkit,skfp,lightgbm,sklearn,joblib)." >&2
  echo "       Set SCREEN_PYTHON=/path/to/python explicitly." >&2
  exit 1
fi

WORKERS="${WORKERS:--1}"
DESC_DIR="$OUT/descriptors"
mkdir -p "$OUT"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUT/run_${STAMP}.log"

# Everything below is tee'd to both console and the timestamped run log.
exec > >(tee -a "$LOG") 2>&1

echo "=========================================================================="
echo "[descriptor-screen] start   : $(date)"
echo "[descriptor-screen] python  : $PY"
echo "[descriptor-screen] sdf     : $([[ "$SDF" == "-" ]] && echo '(none - resume)' || echo "$SDF")"
echo "[descriptor-screen] output  : $OUT"
echo "[descriptor-screen] workers : $WORKERS"
echo "[descriptor-screen] resume  : $RESUME"
echo "[descriptor-screen] log     : $LOG"
echo "[descriptor-screen] extra   : ${EXTRA[*]:-(none)}"
echo "=========================================================================="

# ---- Phase 1: clustering screen (SDF -> descriptors/*.csv + clustering ranking) ----
if [[ "${SKIP_CLUSTER:-0}" != "1" ]]; then
  echo; echo "[phase 1/2] clustering screen -> $OUT"
  # Resume: reuse precalculated descriptors + pose_similarity.csv, no SDF needed.
  # RESUME=2 additionally validates/regenerates truncated concat CSVs and only
  # re-clusters those (incremental), patching the rankings.
  RESUME_FLAGS=()
  [[ "$RESUME_ON" == "1" ]] && RESUME_FLAGS=(--reuse-descriptors --reuse-similarity)
  [[ "$RESUME" == "2" ]] && RESUME_FLAGS+=(--incremental)
  SDF_ARG=(); [[ "$SDF" != "-" ]] && SDF_ARG=("$SDF")
  # ${arr[@]+"${arr[@]}"} keeps empty arrays safe under `set -u` on bash 3.2 (macOS).
  "$PY" screens/descriptor_kid_cluster_screen.py ${SDF_ARG[@]+"${SDF_ARG[@]}"} \
    -o "$OUT" \
    --workers "$WORKERS" \
    ${RESUME_FLAGS[@]+"${RESUME_FLAGS[@]}"} \
    ${EXTRA[@]+"${EXTRA[@]}"}
else
  echo; echo "[phase 1/2] SKIPPED (SKIP_CLUSTER=1); reusing $DESC_DIR"
  [[ -d "$DESC_DIR" ]] || { echo "ERROR: $DESC_DIR does not exist; cannot skip clustering." >&2; exit 1; }
fi

# ---- Phase 2: MIL + LightGBM predictive screen (descriptors/*.csv -> R2 ranking) ----
# RESUME=2 skips LGBM: it builds concats from the single descriptors in memory, so
# concat-CSV truncation never affected its results -- re-running it is pure waste.
if [[ "$RESUME" == "2" ]]; then
  echo; echo "[phase 2/2] SKIPPED (RESUME=2: LGBM is unaffected by concat truncation)"
elif [[ "${SKIP_LGBM:-0}" != "1" ]]; then
  echo; echo "[phase 2/2] MIL + LightGBM screen -> $OUT/mil_lgbm"
  "$PY" screens/mil_mean_lgbm_descriptor_screen.py "$DESC_DIR" \
    -o "$OUT/mil_lgbm" \
    --workers "$WORKERS"
else
  echo; echo "[phase 2/2] SKIPPED (SKIP_LGBM=1)"
fi

echo
echo "=========================================================================="
echo "[descriptor-screen] done    : $(date)"
echo "[descriptor-screen] clustering ranking : $OUT/descriptor_ranking.csv"
echo "[descriptor-screen] predictive ranking : $OUT/mil_lgbm/mil_lgbm_results.csv"
echo "[descriptor-screen] full log           : $LOG"
echo "=========================================================================="
