#!/usr/bin/env bash
# Align each molecule's top-N attention conformers to its experimental crystal pose and
# score the fit (heavy-atom best-RMSD + Crippen O3A). Thin wrapper over
# align_top_attention_poses.py.
#
# Usage:
#   ./align_poses.sh <csv_dir> [top_n] [sdf]
#
#   csv_dir : directory holding the attention-weight CSVs (searched RECURSIVELY).
#             Default pattern '*__noexp.csv' = the validation eval bags (the ones the
#             KID metric scores). Override with PATTERN=... (e.g. PATTERN='*.csv' for train).
#   top_n   : top-attention conformers per molecule to align            (default 10)
#   sdf     : master SDF with conformers + experimental poses
#             (default: ppl/kid_calculator/bace809_200confs.sdf)
#
# Env overrides:  PATTERN=<glob>  OUT_DIR=<dir>  INCLUDE_REF=0|1  ALIGN_PYTHON=<python>
# Extra flags are passed straight through:  ./align_poses.sh <dir> 10 <sdf> --max-matches 5000
#
# Output (per CSV + MERGED/ALL_CSVS): two aligned SDFs — *_aligned_by_heavy_best_rmsd.sdf
# and *_aligned_by_crippen_o3a.sdf — under OUT_DIR (default <csv_dir>/aligned_top<N>).
set -euo pipefail
cd "$(dirname "$0")"

CSV_DIR="${1:?usage: $0 <csv_dir> [top_n] [sdf]   e.g. ./align_poses.sh results/<run>/validation/attention_weights 10}"
TOP_N="${2:-10}"
SDF="${3:-ppl/kid_calculator/bace809_200confs.sdf}"
PATTERN="${PATTERN:-*__noexp.csv}"
OUT_DIR="${OUT_DIR:-$CSV_DIR/aligned_top${TOP_N}}"
INCLUDE_REF="${INCLUDE_REF:-1}"

# python with rdkit + pandas (prefer the project poetry venv)
PY="${ALIGN_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in "$HOME"/Library/Caches/pypoetry/virtualenvs/milk-*/bin/python /opt/anaconda3/envs/*/bin/python; do
    [[ -x "$c" ]] || continue
    "$c" -c "import rdkit, pandas" 2>/dev/null && PY="$c" && break
  done
fi
PY="${PY:-python3}"

[[ -d "$CSV_DIR" ]] || { echo "csv_dir not found: $CSV_DIR" >&2; exit 2; }
[[ -f "$SDF" ]]     || { echo "SDF not found: $SDF" >&2; exit 2; }

ref_flag=(); [[ "$INCLUDE_REF" == "1" ]] && ref_flag=(--include-reference)

echo "[align] python : $PY"
echo "[align] sdf    : $SDF"
echo "[align] csv-dir: $CSV_DIR   pattern: $PATTERN   (recursive)"
echo "[align] top-n  : $TOP_N   include_ref=$INCLUDE_REF   ->   out: $OUT_DIR"

exec "$PY" align_top_attention_poses.py \
  --sdf "$SDF" \
  --csv-dir "$CSV_DIR" \
  --pattern "$PATTERN" \
  --recursive \
  --top-n "$TOP_N" \
  --out-dir "$OUT_DIR" \
  "${ref_flag[@]}" \
  "${@:4}"
