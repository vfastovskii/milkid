#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path


# PDB IDs contain four alphanumeric characters and conventionally
# begin with a digit, e.g. 4ACX, 4B00, 4J0P.
PDB_FILENAME_PATTERN = re.compile(
    r"^[0-9][A-Za-z0-9]{3}(?:_|$)",
    flags=re.IGNORECASE,
)


def is_pdb_named_csv(path: Path) -> bool:
    """Return True when a CSV filename starts with a PDB-like identifier."""
    return (
        path.is_file()
        and path.suffix.lower() == ".csv"
        and PDB_FILENAME_PATTERN.match(path.stem) is not None
    )


def copy_pdb_named_csvs(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
) -> None:
    """Copy PDB-named CSV files from input_dir into output_dir."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = input_dir.rglob("*.csv") if recursive else input_dir.glob("*.csv")

    matched = 0
    skipped = 0

    for source_path in csv_files:
        if not is_pdb_named_csv(source_path):
            skipped += 1
            continue

        # Preserve subdirectory structure when recursive mode is enabled.
        relative_path = source_path.relative_to(input_dir)
        destination_path = output_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source_path, destination_path)
        print(f"Copied: {relative_path}")
        matched += 1

    print("\nFinished")
    print(f"PDB-named CSV files copied: {matched}")
    print(f"Other CSV files skipped:     {skipped}")
    print(f"Output directory:            {output_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy CSV files beginning with PDB-like IDs into a new directory."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the CSV files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory into which matching files will be copied.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search inside subdirectories as well.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    copy_pdb_named_csvs(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()