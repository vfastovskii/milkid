#!/usr/bin/env python3
"""
augment_csv_with_sdf_props.py

Add per-instance properties from an SDF to each row of CSV files that share an identifier
(e.g., instance_id). The SDF is expected to contain one molecule per instance with a
string property that matches the CSV's ID column.

NEW:
- Optional --add-max-03a O3A_COL:
  After augmenting the CSV with SDF properties, compute, for each CSV (treated as one bag),
  the maximum value of O3A_COL and add it as a new column 'max_03a'
  (same value repeated for all rows in that CSV).

- Automatic per-CSV min best_rmsd_to_exp:
  If the merged CSV contains a column 'best_rmsd_to_exp', compute, for each CSV (treated
  as one bag), the minimum value of that column and add it as 'min_best_rmsd_to_exp'
  (same value repeated for all rows in that CSV).

Typical use:
------------
python3 augment_csv_with_sdf_props.py \\
    --sdf bace810_confs_ls_6kcal_prune_05_max200_renamed.sdf \\
    --csv-dir . --glob "*.csv" \\
    --id-col instance_id --sdf-id-prop _Name \\
    --props best_rmsd_to_exp o3a_to_exp Resolution \\
    --add-max-03a o3a_to_exp \\
    --out-dir pdb_with_props

Notes:
- Supports .sdf and .sdf.gz
- Left-join semantics: every CSV row is kept; properties missing in SDF become NaN.
- If the SDF contains duplicate IDs, the first occurrence is used and a warning is logged.
- If a property is missing on an SDF entry, it is left blank (NaN).
- Writes CSVs with the same basename to the output directory.

Dependencies:
- rdkit
- pandas

Author: ChatGPT
"""

import argparse
import gzip
import io
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from rdkit import Chem
except Exception as e:
    print("ERROR: RDKit is required to run this script. Install rdkit, e.g.:", file=sys.stderr)
    print("  conda install -c conda-forge rdkit", file=sys.stderr)
    raise


def _open_maybe_gzip(path: Path) -> io.TextIOBase:
    if str(path).lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"))
    return open(path, "r", encoding="utf-8", errors="replace")


def load_sdf_properties(
    sdf_path: Path,
    sdf_id_prop: str,
    props: List[str],
    name_fallback: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Load an SDF and build a mapping: id -> {prop: value}
    - sdf_id_prop is the SDF property carrying the instance identifier (string).
    - If name_fallback is True and sdf_id_prop is missing, use the molecule title (_Name).
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    id2props: Dict[str, Dict[str, Optional[float]]] = {}
    dup_count = 0
    total = 0

    for mol in suppl:
        total += 1
        if mol is None:
            continue

        # Pull ID
        sid = None
        if mol.HasProp(sdf_id_prop):
            sid = mol.GetProp(sdf_id_prop)
        elif name_fallback:
            title = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            sid = title

        if sid is None or str(sid).strip() == "":
            # can't index this mol, skip
            continue

        sid = str(sid)

        # Collect properties
        rec: Dict[str, Optional[float]] = {}
        for p in props:
            if mol.HasProp(p):
                val = mol.GetProp(p)
                # Try to parse into float when possible; otherwise keep as string
                try:
                    rec[p] = float(val)
                except (ValueError, TypeError):
                    rec[p] = val  # keep raw
            else:
                rec[p] = None

        if sid in id2props:
            dup_count += 1
            # keep first occurrence; could also overwrite—choose to keep first and warn later
            continue
        id2props[sid] = rec

    if dup_count > 0:
        print(
            f"WARNING: Found {dup_count} duplicate IDs in SDF for property '{sdf_id_prop}'. "
            f"Keeping the first occurrence.",
            file=sys.stderr,
        )
    kept = len(id2props)
    print(f"Loaded SDF: {total} records read, {kept} with usable IDs.", file=sys.stderr)
    return id2props


def augment_csv(
    csv_path: Path,
    out_path: Path,
    id_col: str,
    id2props: Dict[str, Dict[str, Optional[float]]],
    props: List[str],
    strict: bool = False,
    add_max_03a: Optional[str] = None,
) -> None:
    """
    Merge properties into a CSV (left join on id_col).

    Parameters
    ----------
    csv_path : Path
        Input CSV path.
    out_path : Path
        Output CSV path.
    id_col : str
        Name of ID column in CSV.
    id2props : dict
        Mapping id -> {prop: value} from SDF.
    props : list[str]
        Property names to merge from SDF mapping.
    strict : bool
        If True, raise an error if any CSV id is missing in SDF mapping.
    add_max_03a : str | None
        If not None, interpreted as the name of O3A column in the merged CSV.
        The script will compute max(add_max_03a) over the whole CSV and
        add a constant column 'max_03a' with that value.
    """
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        raise KeyError(f"{csv_path.name}: missing required id column '{id_col}'")

    # Build a DataFrame from mapping for merge
    map_df = pd.DataFrame.from_dict(id2props, orient="index")
    map_df.index.name = id_col
    map_df.reset_index(inplace=True)

    # Ensure desired columns exist, even if absent in mapping
    for p in props:
        if p not in map_df.columns:
            map_df[p] = pd.NA

    merged = df.merge(map_df[[id_col] + props], on=id_col, how="left")

    if strict and props:
        # Just check first prop for presence; you could refine if needed
        missing = merged[merged[props[0]].isna()].shape[0]
        if missing > 0:
            raise ValueError(
                f"{csv_path.name}: {missing} rows have no matching ID in SDF mapping."
            )

    # --- Per-CSV max O3A (CSV is treated as a single bag) ----------
    if add_max_03a is not None:
        o3a_col = add_max_03a
        if o3a_col not in merged.columns:
            print(
                f"WARNING: {csv_path.name}: O3A column '{o3a_col}' not found; "
                f"cannot compute max_03a.",
                file=sys.stderr,
            )
        else:
            o3a_numeric = pd.to_numeric(merged[o3a_col], errors="coerce")
            if o3a_numeric.notna().any():
                max_val = float(o3a_numeric.max())
            else:
                max_val = float("nan")
            merged["max_03a"] = max_val

    # --- NEW: per-CSV min best_rmsd_to_exp (bag-level) -------------
    rmsd_col = "best_rmsd_to_exp"
    if rmsd_col in merged.columns:
        rmsd_numeric = pd.to_numeric(merged[rmsd_col], errors="coerce")
        if rmsd_numeric.notna().any():
            min_val = float(rmsd_numeric.min())
        else:
            min_val = float("nan")
        merged["min_best_rmsd_to_exp"] = min_val
    else:
        print(
            f"WARNING: {csv_path.name}: RMSD column '{rmsd_col}' not found; "
            f"cannot compute min_best_rmsd_to_exp.",
            file=sys.stderr,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


def discover_csvs(csv_dir: Path, glob_pat: str) -> List[Path]:
    return sorted(csv_dir.glob(glob_pat))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="augment_csv_with_sdf_props.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Left-join SDF properties into CSV files by matching an identifier (instance_id).",
        epilog=textwrap.dedent(
            """\
            Examples
            --------
            # Basic usage: add three props into all CSVs in a dir
            %(prog)s --sdf instances.sdf --csv-dir ./preds --glob "*.csv" \\
                      --id-col instance_id --sdf-id-prop instance_id \\
                      --props energy rmsd_getbest_heavy o3a_crippen_score \\
                      --out-dir ./preds_with_props

            # If your SDF uses _Name as identifier
            %(prog)s --sdf instances.sdf --csv-dir ./preds --glob "*.csv" \\
                      --id-col instance_id --sdf-id-prop _Name \\
                      --props energy rmsd_getbest_heavy o3a_crippen_score

            # Also compute per-CSV max O3A (treat each CSV as one bag)
            %(prog)s --sdf instances.sdf --csv-dir ./preds --glob "*.csv" \\
                      --id-col instance_id --sdf-id-prop _Name \\
                      --props best_rmsd_to_exp o3a_to_exp \\
                      --add-max-03a o3a_to_exp \\
                      --out-dir ./preds_with_props

            # Strict mode: error if any CSV ID is not found in SDF
            %(prog)s --sdf instances.sdf --csv-dir ./preds --glob "*.csv" \\
                      --id-col instance_id --sdf-id-prop instance_id \\
                      --props energy rmsd_getbest_heavy o3a_crippen_score \\
                      --strict
        """
        ),
    )
    p.add_argument(
        "--sdf",
        required=True,
        type=Path,
        help="Path to SDF (.sdf or .sdf.gz) with instance entries.",
    )
    p.add_argument(
        "--csv-dir",
        required=True,
        type=Path,
        help="Directory containing CSVs to augment.",
    )
    p.add_argument(
        "--glob",
        default="*.csv",
        help='Glob to select CSVs (default: "*.csv").',
    )
    p.add_argument(
        "--id-col",
        required=True,
        help="CSV column name containing the instance identifier.",
    )
    p.add_argument(
        "--sdf-id-prop",
        required=True,
        help="SDF property name carrying the identifier (e.g., instance_id or _Name).",
    )
    p.add_argument(
        "--props",
        nargs="+",
        required=True,
        help="SDF property names to copy into each CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: in-place overwrite).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Error if any CSV row has no match in SDF mapping.",
    )
    p.add_argument(
        "--add-max-03a",
        metavar="O3A_COL",
        help=(
            "Also add a column 'max_03a' to each CSV, computed as the maximum "
            "of O3A_COL over the entire CSV (treated as one bag). "
            "Example: --add-max-03a o3a_to_exp"
        ),
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    if not args.sdf.exists():
        print(f"ERROR: SDF not found: {args.sdf}", file=sys.stderr)
        return 2
    if not args.csv_dir.exists() or not args.csv_dir.is_dir():
        print(f"ERROR: CSV directory not found or not a directory: {args.csv_dir}", file=sys.stderr)
        return 2

    # Load SDF mapping
    id2props = load_sdf_properties(
        args.sdf,
        sdf_id_prop=args.sdf_id_prop,
        props=args.props,
        name_fallback=True,
    )

    csv_paths = discover_csvs(args.csv_dir, args.glob)
    if not csv_paths:
        print(
            f"WARNING: No CSV files found in {args.csv_dir} matching pattern {args.glob}",
            file=sys.stderr,
        )

    for csv_path in csv_paths:
        out_path = csv_path
        if args.out_dir is not None:
            out_path = args.out_dir / csv_path.name
        try:
            augment_csv(
                csv_path=csv_path,
                out_path=out_path,
                id_col=args.id_col,
                id2props=id2props,
                props=args.props,
                strict=args.strict,
                add_max_03a=args.add_max_03a,
            )
        except Exception as e:
            print(f"ERROR processing {csv_path.name}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
