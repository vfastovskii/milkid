#!/usr/bin/env python3
"""One-shot KID retrieval-rate pipeline: attention-weights dir + SDF -> top-k barplot + log.

Chains the four existing ppl/kid_calculator stages so you run one command instead of four:

  1. filter_actives : keep __noexp bags that are active (true >= 7) & correctly
                      predicted (|true - pred| <= pred_tol)
  2. filter_pdb     : keep only molecules with a PDB-like id (they have an experimental pose)
  3. add_sdf_prop   : left-join best_rmsd_to_exp / o3a_to_exp (+ Series, Resolution) from the SDF
  4. calc_topk_success_rate : top-k RMSD/O3A retrieval rate vs. hypergeometric baseline + barplot

This is only the glue: stages 1-4 are reused as-is, so the barplot/log are identical to
running calc_topk_success_rate.py by hand on a pdb_with_props/ dir.

Example
-------
    python kid_retrieval_rate.py \\
        results/bace809_jul8_run1/validation/attention_weights \\
        ppl/kid_calculator/bace809_200confs.sdf

    -> results/bace809_jul8_run1/validation/attention_weights/act_cor_pred/pdb/
       pdb_with_props/res_run1_barplot.png   (+ _log.txt, _per_bag.csv, _summary.csv, ...)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
KID = REPO / "ppl" / "kid_calculator"
sys.path.insert(0, str(KID))  # reuse the stage modules without packaging them

import add_sdf_prop  # noqa: E402  load_sdf_properties, augment_csv
import filter_actives  # noqa: E402  file_matches(csv, true_col, pred_col)
import filter_pdb  # noqa: E402  is_pdb_named_csv(path)


def select_bags(attn_dir: Path, true_col: str, pred_col: str) -> list[Path]:
    """Stages 1+2 as an in-memory filter: __noexp, active+correct, PDB-named CSVs."""
    picked = []
    for csv in sorted(attn_dir.glob("*.csv")):
        name = csv.name
        if "noexp" not in name or "top20" in name:  # filter_actives name rule
            continue
        if not filter_pdb.is_pdb_named_csv(csv):  # filter_pdb: experimental structure only
            continue
        if not filter_actives.file_matches(csv, true_col, pred_col):  # active + correct
            continue
        picked.append(csv)
    return picked


def derive_tag(attn_dir: Path) -> str:
    """Pull 'run<N>' out of the results path so the default output matches res_run<N>_*."""
    m = re.search(r"run\d+", str(attn_dir))
    return m.group(0) if m else "run"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "attention_dir",
        type=Path,
        help="Dir of per-molecule attention_weight CSVs "
        "(e.g. results/<run>/validation/attention_weights).",
    )
    ap.add_argument(
        "sdf",
        type=Path,
        help="SDF of conformers carrying best_rmsd_to_exp / o3a_to_exp, keyed by _Name = instance_id.",
    )
    ap.add_argument("--tag", default=None, help="Output prefix -> res_<tag>_barplot.png. Default: run<N> from the path.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir. Default: <attention_dir>/act_cor_pred/pdb/pdb_with_props",
    )
    ap.add_argument("--rmsd-threshold", type=float, default=2.0)
    ap.add_argument("--o3a-threshold", type=float, default=0.8, help="Normalized-O3A cutoff. Pass a negative value to skip O3A.")
    ap.add_argument("--pred-tol", type=float, default=1.0, help="Passed through to filter_actives' |true-pred| tolerance is fixed at 1.0 there; kept for visibility.")
    ap.add_argument("--props", nargs="+", default=["best_rmsd_to_exp", "o3a_to_exp", "Series", "Resolution"])
    ap.add_argument("--true-col", default="true_value")
    ap.add_argument("--pred-col", default="predicted_value")
    args = ap.parse_args()

    if not args.attention_dir.is_dir():
        sys.exit(f"attention_dir not found: {args.attention_dir}")
    if not args.sdf.exists():
        sys.exit(f"SDF not found: {args.sdf}")

    tag = args.tag or derive_tag(args.attention_dir)
    out_dir = args.out_dir or (args.attention_dir / "act_cor_pred" / "pdb" / "pdb_with_props")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stages 1+2 -----------------------------------------------------------
    picked = select_bags(args.attention_dir, args.true_col, args.pred_col)
    print(f"[KID] {len(picked)} active+correct PDB molecules selected from {args.attention_dir}")
    if not picked:
        sys.exit("[KID] no qualifying molecules (need __noexp, PDB-like id, true>=7, |true-pred|<=1).")

    # Stage 3: merge SDF props into each selected bag -> out_dir ------------
    id2props = add_sdf_prop.load_sdf_properties(
        args.sdf, sdf_id_prop="_Name", props=args.props, name_fallback=True
    )
    for csv in picked:
        add_sdf_prop.augment_csv(
            csv_path=csv,
            out_path=out_dir / csv.name,
            id_col="instance_id",
            id2props=id2props,
            props=args.props,
            strict=False,
            add_max_03a="o3a_to_exp",
        )

    # Stage 4: reuse the full calc CLI as-is. Glob only *noexp*.csv so calc's own
    # res_<tag>_*.csv outputs are never re-ingested as bags on a repeat run.
    prefix = out_dir / f"res_{tag}"
    cmd = [
        sys.executable,
        str(KID / "calc_topk_success_rate.py"),
        str(out_dir),
        "--pattern", "*noexp*.csv",
        "--rmsd-threshold", str(args.rmsd_threshold),
        "--output-prefix", str(prefix),
        "--series-col", "Series",
    ]
    if args.o3a_threshold >= 0:
        cmd += ["--norm-o3a-threshold", str(args.o3a_threshold)]
    print("[KID] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"\n[KID] done -> {prefix}_barplot.png   (log: {prefix}_log.txt)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
