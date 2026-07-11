#!/usr/bin/env python3
"""
filter_csv_simple.py

Find all CSV files in the input directory whose names contain '_noexp' and where:
    abs(true_value - predicted_value) <= 1.0  and  true_value >= 7.0

Copy only those files into the output directory.
"""

import argparse
from pathlib import Path
import shutil
import sys
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Filter CSV files (with '_noexp' in name) based on true/predicted values.")
    parser.add_argument("-i", "--input-dir", type=Path, required=True, help="Directory with input CSV files")
    parser.add_argument("-o", "--output-dir", type=Path, required=True, help="Directory to save matching CSVs")
    parser.add_argument("--true-col", default="true_value", help="Column name for true values")
    parser.add_argument("--pred-col", default="predicted_value", help="Column name for predicted values")
    return parser.parse_args()


def file_matches(csv_path, true_col, pred_col):
    try:
        df = pd.read_csv(csv_path)
        if true_col not in df.columns or pred_col not in df.columns:
            return False
        diff = (df[true_col] - df[pred_col]).abs()
        # mask = diff <= 1.0
        mask = (df[true_col] >= 7.0) & (diff <= 1.0)
        return mask.any()
    except Exception:
        return False


def main():
    args = parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir

    if not input_dir.is_dir():
        sys.exit(f"Input directory '{input_dir}' not found.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process only files that contain "_noexp" in their names
    # csv_files = [f for f in input_dir.glob("*.csv") if "_noexp_top20" in f.name]

    csv_files = [
        f
        for f in input_dir.glob("*.csv")
        if "noexp" in f.name and "top20" not in f.name
    ]

    print(f"[INFO] Found {len(csv_files)} CSV files with '_noexp_top20' in name.")

    copied = 0
    for csv_path in csv_files:
        if file_matches(csv_path, args.true_col, args.pred_col):
            dest_path = output_dir / csv_path.name
            shutil.copy2(csv_path, dest_path)
            print(f"[✓] Copied: {csv_path.name}")
            copied += 1

    print(f"\n[SUMMARY] Copied {copied}/{len(csv_files)} matching files.")
    print("Done.")


if __name__ == "__main__":
    main()
