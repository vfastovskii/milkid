#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors


CONF_RE = re.compile(r"^(?P<base>.+)_conf_(?P<conf_id>\d+)$")


def mol_name(mol: Chem.Mol) -> str:
    if mol is not None and mol.HasProp("_Name"):
        return mol.GetProp("_Name")
    return ""


def base_from_instance_id(instance_id: str) -> str:
    m = CONF_RE.match(str(instance_id))
    if not m:
        raise ValueError(
            f"Cannot parse instance_id '{instance_id}'. "
            "Expected format: '<base>_conf_<n>'"
        )
    return m.group("base")


def reference_id_from_instance_id(instance_id: str) -> str:
    base = base_from_instance_id(instance_id)
    return f"{base}_experimental_pose"


def find_csv_files(args: argparse.Namespace) -> List[Path]:
    if args.csv:
        paths = [Path(p) for p in args.csv]
    else:
        pattern = f"**/{args.pattern}" if args.recursive else args.pattern
        paths = sorted(Path(args.csv_dir).glob(pattern))

    paths = [p for p in paths if p.is_file()]

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found with pattern '{args.pattern}' in '{args.csv_dir}'. "
            "Use --csv file1.csv file2.csv or --pattern '*.csv'."
        )

    return paths


def read_top_attention_records(
    csv_paths: Iterable[Path],
    id_col: str,
    att_col: str,
    top_n: int,
) -> Tuple[Dict[Path, List[dict]], Set[str]]:

    per_csv: Dict[Path, List[dict]] = {}
    required_sdf_names: Set[str] = set()

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        if id_col not in df.columns:
            raise KeyError(
                f"{csv_path} does not contain id column '{id_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        if att_col not in df.columns:
            raise KeyError(
                f"{csv_path} does not contain attention column '{att_col}'. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df[att_col] = pd.to_numeric(df[att_col], errors="coerce")
        df = df.dropna(subset=[id_col, att_col])
        df[id_col] = df[id_col].astype(str)

        top = (
            df.sort_values(att_col, ascending=False)
            .drop_duplicates(subset=[id_col], keep="first")
            .head(top_n)
        )

        records = []

        for rank, row in enumerate(top.to_dict(orient="records"), start=1):
            instance_id = str(row[id_col])
            reference_id = reference_id_from_instance_id(instance_id)

            record = {
                "rank": rank,
                "instance_id": instance_id,
                "reference_id": reference_id,
                "attention_weight": float(row[att_col]),
            }

            records.append(record)
            required_sdf_names.add(instance_id)
            required_sdf_names.add(reference_id)

        per_csv[csv_path] = records

    return per_csv, required_sdf_names


def load_sdf_subset(
    sdf_path: Path,
    required_names: Set[str],
) -> Tuple[Dict[str, Chem.Mol], Set[str]]:

    found: Dict[str, Chem.Mol] = {}
    missing = set(required_names)

    with open(sdf_path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(
            handle,
            removeHs=False,
            sanitize=True,
        )

        for mol in supplier:
            if mol is None:
                continue

            name = mol_name(mol)

            if name in missing:
                found[name] = Chem.Mol(mol)
                missing.remove(name)

                if not missing:
                    break

    return found, missing


def remove_hs_safe(mol: Chem.Mol) -> Chem.Mol:
    try:
        return Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    except TypeError:
        return Chem.RemoveHs(Chem.Mol(mol))


def heavy_atom_indices(mol: Chem.Mol) -> List[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]


def heavy_atom_maps(
    probe: Chem.Mol,
    ref: Chem.Mol,
    max_matches: int,
) -> List[List[Tuple[int, int]]]:

    probe_heavy_original = heavy_atom_indices(probe)
    ref_heavy_original = heavy_atom_indices(ref)

    probe_noh = remove_hs_safe(probe)
    ref_noh = remove_hs_safe(ref)

    maps: List[List[Tuple[int, int]]] = []

    matches = probe_noh.GetSubstructMatches(
        ref_noh,
        uniquify=False,
        maxMatches=max_matches,
    )

    for match in matches:
        atom_map = [
            (probe_heavy_original[probe_noh_idx], ref_heavy_original[ref_noh_idx])
            for ref_noh_idx, probe_noh_idx in enumerate(match)
        ]
        maps.append(atom_map)

    if maps:
        return maps

    if len(probe_heavy_original) == len(ref_heavy_original):
        same_atom_types = all(
            probe.GetAtomWithIdx(i).GetAtomicNum()
            == ref.GetAtomWithIdx(j).GetAtomicNum()
            for i, j in zip(probe_heavy_original, ref_heavy_original)
        )

        if same_atom_types:
            return [
                [
                    (i, j)
                    for i, j in zip(probe_heavy_original, ref_heavy_original)
                ]
            ]

    raise ValueError(
        f"Could not build heavy-atom map for "
        f"{mol_name(probe)} vs {mol_name(ref)}"
    )


def align_by_best_heavy_rmsd(
    probe: Chem.Mol,
    ref: Chem.Mol,
    max_matches: int,
) -> Tuple[Chem.Mol, float]:

    aligned = Chem.Mol(probe)

    maps = heavy_atom_maps(
        probe=aligned,
        ref=ref,
        max_matches=max_matches,
    )

    rmsd = rdMolAlign.GetBestRMS(
        aligned,
        ref,
        map=maps,
        maxMatches=max_matches,
    )

    return aligned, float(rmsd)


def align_by_crippen_o3a(
    probe: Chem.Mol,
    ref: Chem.Mol,
) -> Tuple[Chem.Mol, float, float]:

    aligned = Chem.Mol(probe)

    probe_contribs = rdMolDescriptors._CalcCrippenContribs(aligned)
    ref_contribs = rdMolDescriptors._CalcCrippenContribs(ref)

    o3a = rdMolAlign.GetCrippenO3A(
        aligned,
        ref,
        probe_contribs,
        ref_contribs,
    )

    o3a_score = float(o3a.Score())
    o3a_rmsd = float(o3a.Align())

    return aligned, o3a_score, o3a_rmsd


def set_common_props(
    mol: Chem.Mol,
    csv_path: Path,
    record: dict,
    method: str,
) -> None:

    mol.SetProp("record_type", "top_attention_conformer")
    mol.SetProp("source_csv", csv_path.name)
    mol.SetProp("instance_id", record["instance_id"])
    mol.SetProp("reference_pose_id", record["reference_id"])
    mol.SetProp("attention_rank", str(record["rank"]))
    mol.SetProp("attention_weight", str(record["attention_weight"]))
    mol.SetProp("alignment_method", method)


def set_metric_props(
    mol: Chem.Mol,
    heavy_rmsd: Optional[float],
    o3a_score: Optional[float],
    o3a_rmsd: Optional[float],
) -> None:

    if heavy_rmsd is not None:
        mol.SetProp("heavy_best_rmsd_to_reference", f"{heavy_rmsd:.8f}")

    if o3a_score is not None:
        mol.SetProp("crippen_o3a_score_to_reference", f"{o3a_score:.8f}")

    if o3a_rmsd is not None:
        mol.SetProp("crippen_o3a_alignment_rmsd", f"{o3a_rmsd:.8f}")


def make_reference_copy(
    ref: Chem.Mol,
    csv_path: Path,
    record: dict,
) -> Chem.Mol:

    out = Chem.Mol(ref)

    out.SetProp("record_type", "reference_experimental_pose")
    out.SetProp("source_csv", csv_path.name)
    out.SetProp("reference_pose_id", record["reference_id"])
    out.SetProp("linked_instance_id", record["instance_id"])
    out.SetProp("alignment_method", "reference_not_aligned")

    return out


def write_sdf(path: Path, mols: Sequence[Chem.Mol]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    writer = Chem.SDWriter(str(path))

    try:
        for mol in mols:
            writer.write(mol)
    finally:
        writer.close()

    path.touch(exist_ok=True)

    if not path.exists():
        raise RuntimeError(f"Failed to create output SDF: {path}")


def process_one_csv(
    csv_path: Path,
    records: List[dict],
    mols: Dict[str, Chem.Mol],
    max_matches: int,
    include_reference: bool,
) -> Tuple[List[Chem.Mol], List[Chem.Mol], int, int]:

    rmsd_mols: List[Chem.Mol] = []
    o3a_mols: List[Chem.Mol] = []

    n_written = 0
    n_failed = 0

    references_added: Set[str] = set()

    for record in records:
        instance_id = record["instance_id"]
        reference_id = record["reference_id"]

        probe = mols.get(instance_id)
        ref = mols.get(reference_id)

        if probe is None:
            print(f"[WARN] Missing conformer in SDF: {instance_id}", file=sys.stderr)
            n_failed += 1
            continue

        if ref is None:
            print(f"[WARN] Missing experimental pose in SDF: {reference_id}", file=sys.stderr)
            n_failed += 1
            continue

        if include_reference and reference_id not in references_added:
            ref_copy = make_reference_copy(ref, csv_path, record)
            rmsd_mols.append(Chem.Mol(ref_copy))
            o3a_mols.append(Chem.Mol(ref_copy))
            references_added.add(reference_id)

        heavy_rmsd = None
        o3a_score = None
        o3a_rmsd = None

        aligned_rmsd = None
        aligned_o3a = None

        try:
            aligned_rmsd, heavy_rmsd = align_by_best_heavy_rmsd(
                probe=probe,
                ref=ref,
                max_matches=max_matches,
            )
        except Exception as exc:
            print(
                f"[WARN] RMSD alignment failed for {instance_id}: {exc}",
                file=sys.stderr,
            )

        try:
            aligned_o3a, o3a_score, o3a_rmsd = align_by_crippen_o3a(
                probe=probe,
                ref=ref,
            )
        except Exception as exc:
            print(
                f"[WARN] O3A alignment failed for {instance_id}: {exc}",
                file=sys.stderr,
            )

        if aligned_rmsd is None and aligned_o3a is None:
            n_failed += 1
            continue

        if aligned_rmsd is not None:
            set_common_props(
                mol=aligned_rmsd,
                csv_path=csv_path,
                record=record,
                method="heavy_best_rmsd",
            )
            set_metric_props(
                mol=aligned_rmsd,
                heavy_rmsd=heavy_rmsd,
                o3a_score=o3a_score,
                o3a_rmsd=o3a_rmsd,
            )
            rmsd_mols.append(aligned_rmsd)

        if aligned_o3a is not None:
            set_common_props(
                mol=aligned_o3a,
                csv_path=csv_path,
                record=record,
                method="crippen_o3a",
            )
            set_metric_props(
                mol=aligned_o3a,
                heavy_rmsd=heavy_rmsd,
                o3a_score=o3a_score,
                o3a_rmsd=o3a_rmsd,
            )
            o3a_mols.append(aligned_o3a)

        n_written += 1

    return rmsd_mols, o3a_mols, n_written, n_failed


def file_report(path: Path) -> str:
    if path.exists():
        return f"{path} ({path.stat().st_size} bytes)"
    return f"{path} (missing)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each CSV, select top-N attention conformers, align them to "
            "their experimental poses, and write per-CSV plus merged SDF files."
        )
    )

    parser.add_argument(
        "--sdf",
        required=True,
        type=Path,
        help="Master SDF containing conformers and experimental poses.",
    )

    parser.add_argument(
        "--csv-dir",
        default=".",
        type=Path,
        help="Directory containing CSV files.",
    )

    parser.add_argument(
        "--csv",
        nargs="*",
        help="Explicit CSV file(s). Overrides --csv-dir and --pattern.",
    )

    parser.add_argument(
        "--pattern",
        default="*_noexp*.csv",
        help="CSV glob pattern. Default: '*_noexp*.csv'.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively inside --csv-dir.",
    )

    parser.add_argument(
        "--out-dir",
        default="final",
        type=Path,
        help="Output directory. Default: final.",
    )

    parser.add_argument(
        "--top-n",
        default=10,
        type=int,
        help="Number of top attention conformers per CSV.",
    )

    parser.add_argument(
        "--id-col",
        default="instance_id",
        help="CSV column containing conformer IDs.",
    )

    parser.add_argument(
        "--att-col",
        default="attention_weight",
        help="CSV column containing attention weights.",
    )

    parser.add_argument(
        "--max-matches",
        default=1000,
        type=int,
        help="Maximum symmetry matches for heavy-atom RMSD alignment.",
    )

    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="Also include experimental poses in output SDFs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = find_csv_files(args)

    print(f"Found {len(csv_paths)} CSV file(s):")
    for csv_path in csv_paths:
        print(f"  {csv_path}")

    per_csv, required_names = read_top_attention_records(
        csv_paths=csv_paths,
        id_col=args.id_col,
        att_col=args.att_col,
        top_n=args.top_n,
    )

    print(f"\nNeed {len(required_names)} SDF records.")
    mols, missing = load_sdf_subset(args.sdf, required_names)
    print(f"Loaded {len(mols)} matching SDF records.")

    if missing:
        print(f"\n[WARN] Missing {len(missing)} required SDF records:", file=sys.stderr)
        for name in sorted(missing):
            print(f"  {name}", file=sys.stderr)

    all_rmsd_mols: List[Chem.Mol] = []
    all_o3a_mols: List[Chem.Mol] = []

    total_written = 0
    total_failed = 0

    for csv_path, records in per_csv.items():
        rmsd_mols, o3a_mols, n_written, n_failed = process_one_csv(
            csv_path=csv_path,
            records=records,
            mols=mols,
            max_matches=args.max_matches,
            include_reference=args.include_reference,
        )

        stem = csv_path.stem

        rmsd_out = args.out_dir / f"{stem}_top{args.top_n}_aligned_by_heavy_best_rmsd.sdf"
        o3a_out = args.out_dir / f"{stem}_top{args.top_n}_aligned_by_crippen_o3a.sdf"

        write_sdf(rmsd_out, rmsd_mols)
        write_sdf(o3a_out, o3a_mols)

        # IMPORTANT: accumulate molecules for merged files
        all_rmsd_mols.extend(Chem.Mol(mol) for mol in rmsd_mols)
        all_o3a_mols.extend(Chem.Mol(mol) for mol in o3a_mols)

        total_written += n_written
        total_failed += n_failed

        print(f"\nProcessed: {csv_path.name}")
        print(f"  written conformers: {n_written}")
        print(f"  failed/skipped: {n_failed}")
        print(f"  RMSD output: {file_report(rmsd_out)}")
        print(f"  O3A output:  {file_report(o3a_out)}")

    # ============================================================
    # MERGED FILE CREATION
    # This is the part that creates the two merged SDF files.
    # ============================================================

    merged_rmsd = args.out_dir / f"MERGED_top{args.top_n}_aligned_by_heavy_best_rmsd.sdf"
    merged_o3a = args.out_dir / f"MERGED_top{args.top_n}_aligned_by_crippen_o3a.sdf"

    write_sdf(merged_rmsd, all_rmsd_mols)
    write_sdf(merged_o3a, all_o3a_mols)

    # Optional duplicate names in case you search for ALL_CSVS
    all_csvs_rmsd = args.out_dir / f"ALL_CSVS_top{args.top_n}_aligned_by_heavy_best_rmsd.sdf"
    all_csvs_o3a = args.out_dir / f"ALL_CSVS_top{args.top_n}_aligned_by_crippen_o3a.sdf"

    shutil.copyfile(merged_rmsd, all_csvs_rmsd)
    shutil.copyfile(merged_o3a, all_csvs_o3a)

    print("\nMERGED OUTPUTS CREATED:")
    print(f"  {file_report(merged_rmsd)}")
    print(f"  {file_report(merged_o3a)}")

    print("\nDuplicate ALL_CSVS outputs created:")
    print(f"  {file_report(all_csvs_rmsd)}")
    print(f"  {file_report(all_csvs_o3a)}")

    print("\nSummary:")
    print(f"  total selected conformers written: {total_written}")
    print(f"  total failed/skipped: {total_failed}")
    print(f"  output directory: {args.out_dir.resolve()}")

    # Hard check: fail loudly if merged files are absent
    for path in [merged_rmsd, merged_o3a, all_csvs_rmsd, all_csvs_o3a]:
        if not path.exists():
            raise RuntimeError(f"Expected merged output was not created: {path}")


if __name__ == "__main__":
    main()
