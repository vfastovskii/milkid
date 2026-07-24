#!/usr/bin/env bash
# Convenience runner for kid_retrieval_rate.py: auto-picks an rdkit-capable python
# and a default conformer SDF, so a KID retrieval-rate run is one command.
#
# Usage:
#   ./run_kid.sh <run-name-or-attention_weights_dir> [extra flags for kid_retrieval_rate.py...]
#
#   ./run_kid.sh bace809_jul8_run1                       # run-name shorthand
#   ./run_kid.sh results/bace809_jul8_run1/validation/attention_weights
#   ./run_kid.sh bace809_jul8_run1 --tag run1 --rmsd-threshold 2.0
#
# Overrides (env vars):
#   KID_PYTHON=/path/to/python      # force a specific interpreter
#   KID_SDF=/path/to/conformers.sdf # force a specific SDF
set -euo pipefail
cd "$(dirname "$0")"

ATTN="${1:?usage: run_kid.sh <run-name-or-attention_weights_dir> [extra flags...]}"
shift

# Resolve to an attention_weights dir. Accepts: the dir itself, a run-root path
# (results/<run> or <run>), or a bare run name — appending validation/attention_weights.
if   [[ -d "$ATTN/validation/attention_weights" ]]; then
  ATTN="$ATTN/validation/attention_weights"
elif [[ ! -d "$ATTN" && -d "results/$ATTN/validation/attention_weights" ]]; then
  ATTN="results/$ATTN/validation/attention_weights"
fi
if [[ ! -d "$ATTN" ]]; then
  echo "attention dir not found: $ATTN" >&2
  echo "pass a run name, e.g.:  ./run_kid.sh bace809_jul8_run16_paper" >&2
  echo "available runs:" >&2
  ls -1 results 2>/dev/null | sed 's/^/  /' >&2
  exit 1
fi

# python with the full stack: env override, else first conda env that imports it, else `python`.
PY="${KID_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in /opt/anaconda3/envs/*/bin/python; do
    "$c" -c "import rdkit,pandas,matplotlib,scipy" 2>/dev/null && PY="$c" && break
  done
fi
PY="${PY:-python}"

# SDF: KID_SDF override, else the canonical file if it's really present (not an LFS pointer),
# else the newest materialized conformer SDF anywhere in results/.
SDF="${KID_SDF:-}"
if [[ -z "$SDF" || ! -f "$SDF" ]]; then
  canon="ppl/kid_calculator/bace809_200confs_sch.sdf"
  if [[ -f "$canon" ]] && ! head -1 "$canon" | grep -q "git-lfs"; then
    SDF="$canon"
  else
    SDF="$(ls -t results/*/validation/attention_weights/act_cor_pred/pdb/*renamed_last.sdf 2>/dev/null | head -1 || true)"
  fi
fi
[[ -n "$SDF" && -f "$SDF" ]] || { echo "No SDF found; pass one via KID_SDF=/path/to.sdf" >&2; exit 1; }

echo "[run_kid] python: $PY"
echo "[run_kid] attn:   $ATTN"
echo "[run_kid] sdf:    $SDF"
exec "$PY" kid_retrieval_rate.py "$ATTN" "$SDF" "$@"
