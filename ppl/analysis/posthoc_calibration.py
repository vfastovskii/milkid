#!/usr/bin/env python
"""Post-hoc slope recalibration for the MIL pIC50 regressor.

Undoes prediction shrinkage (best-fit slope < 1) at INFERENCE by fitting a 1-D
calibrator ``pred' = a·pred + b`` (and optionally isotonic) on a held-out fold,
then applying it to the eval fold. Complements the training-time fixes; it masks
the underfit rather than curing it.

LEAKAGE RULE (critical): the calibrator MUST be fit on data DISJOINT from the
fold you report metrics on. This project's split has no test fold (split col is
0/1 only), so the honest option is: fit on TRAIN dumps, apply to VAL dumps — VAL
is never seen by the fit. NEVER fit on the same val set you report R² on.

Reads the per-bag (true_value, predicted_value) already dumped in a run's
attention CSVs, so no model reload is needed.

Usage:
    python -m ppl.analysis.posthoc_calibration results/<run>
    python -m ppl.analysis.posthoc_calibration --demo        # runnable self-check
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional

import numpy as np


def _metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """R², RMSE, MAE, and the best-fit shrinkage slope (predicted ≈ slope·true + b)."""
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    var_t = float(np.var(true))
    slope = float(np.cov(pred, true, bias=True)[0, 1] / var_t) if var_t > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae, "slope": slope}


def fit_linear(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Least-squares calibrator mapping pred → true: corrected = a·pred + b.

    Minimises RMSE (best R²), but note the corrected-vs-true slope becomes r², not 1
    — least squares is MSE-optimal, which is itself shrunk. Use variance-matching to
    restore the full dynamic range (slope → r).
    """
    a, b = np.polyfit(np.asarray(pred, float), np.asarray(true, float), deg=1)
    return float(a), float(b)


def apply_linear(pred: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    a, b = ab
    return a * np.asarray(pred, float) + b


def fit_variance_match(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    """Moment-matching calibrator: corrected = mean_t + (pred − mean_p)·(std_t/std_p).

    Forces the prediction spread to equal the target spread, restoring dynamic range
    (corrected-vs-true slope → Pearson r, closer to 1 than least squares). Returns
    (mean_p, std_p, mean_t, std_t) fit on the held-out fold.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    return float(pred.mean()), float(pred.std() + 1e-12), float(true.mean()), float(true.std())


def apply_variance_match(pred: np.ndarray, params: tuple[float, float, float, float]) -> np.ndarray:
    mean_p, std_p, mean_t, std_t = params
    return mean_t + (np.asarray(pred, float) - mean_p) * (std_t / std_p)


def load_bag_predictions(attn_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect one (true, pred) per bag from a run's attention_weights CSVs.

    Dedups the full/__noexp/_top20 variants of each bag by its base id and reads
    true_value/predicted_value (constant within a bag) from the first row.
    """
    import pandas as pd

    seen: dict[str, tuple[float, float]] = {}
    for path in sorted(glob.glob(os.path.join(attn_dir, "*.csv"))):
        name = os.path.basename(path)
        if name.endswith("_top20.csv"):
            continue  # a re-normalised subset of the same bag
        base = re.sub(r"(__noexp)?(_top20)?\.csv$", "", name)
        try:
            df = pd.read_csv(path, nrows=1)
            if "true_value" not in df.columns or "predicted_value" not in df.columns:
                continue
            seen[base] = (float(df["true_value"].iat[0]), float(df["predicted_value"].iat[0]))
        except Exception:
            continue
    if not seen:
        raise FileNotFoundError(f"no usable attention CSVs (true_value/predicted_value) in {attn_dir}")
    arr = np.array(list(seen.values()), float)
    return arr[:, 0], arr[:, 1]


def _attn_dir(run_root: str, split: str) -> str:
    for cand in (
        os.path.join(run_root, split, "attention_weights"),
        os.path.join(run_root, "validation" if split == "val" else split, "attention_weights"),
    ):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"no {split} attention_weights dir under {run_root}")


def calibrate_run(run_root: str) -> dict:
    """Fit on TRAIN dumps, apply to VAL dumps, report before/after (no leakage).

    Reports both calibrators: 'linear' (best R²/RMSE) and 'varmatch' (restores
    dynamic range, slope → r).
    """
    tr_true, tr_pred = load_bag_predictions(_attn_dir(run_root, "train"))
    va_true, va_pred = load_bag_predictions(_attn_dir(run_root, "val"))
    ab = fit_linear(tr_true, tr_pred)
    vm = fit_variance_match(tr_true, tr_pred)
    return {
        "n_train": int(tr_true.size), "n_val": int(va_true.size),
        "linear": {"params": {"a": ab[0], "b": ab[1]}},
        "varmatch": {"params": {"mean_p": vm[0], "std_p": vm[1], "mean_t": vm[2], "std_t": vm[3]}},
        "val_before": _metrics(va_true, va_pred),
        "val_after_linear": _metrics(va_true, apply_linear(va_pred, ab)),
        "val_after_varmatch": _metrics(va_true, apply_variance_match(va_pred, vm)),
    }


def _print_report(res: dict) -> None:
    a, b = res["linear"]["params"]["a"], res["linear"]["params"]["b"]
    print(f"fit on TRAIN (n={res['n_train']}), applied to VAL (n={res['n_val']}):")
    print(f"  linear   : corrected = {a:.3f}·pred + {b:.3f}   (best R²/RMSE; slope→r²)")
    print(f"  varmatch : match spread std_t/std_p            (restores dynamic range; slope→r)")
    hdr = f"  {'metric':7s} {'before':>9s} {'linear':>9s} {'varmatch':>9s}"
    print(hdr)
    for k in ("r2", "rmse", "mae", "slope"):
        print(f"  {k:7s} {res['val_before'][k]:9.4f} {res['val_after_linear'][k]:9.4f} "
              f"{res['val_after_varmatch'][k]:9.4f}")
    print("NOTE: complement, not a cure — fit on TRAIN so VAL stays untouched; still fix shrinkage in training.")


def _demo() -> None:
    """Self-check on synthetic shrunk predictions: linear lifts R², varmatch restores slope→1."""
    rng = np.random.default_rng(0)
    true = rng.normal(6.7, 1.1, size=400)
    pred = 0.6 * true + 1.2 + rng.normal(0, 0.4, size=400)    # shrunk slope + low bias
    before = _metrics(true, pred)
    lin = _metrics(true, apply_linear(pred, fit_linear(true, pred)))
    vm = _metrics(true, apply_variance_match(pred, fit_variance_match(true, pred)))
    assert before["slope"] < 0.75, before                    # shrunk to start
    assert lin["r2"] >= before["r2"] - 1e-9, (before, lin)    # least squares can only raise R²…
    assert lin["rmse"] <= before["rmse"] + 1e-9, (before, lin)  # …and lower RMSE
    assert vm["slope"] > before["slope"] + 0.1, (before, vm)  # varmatch restores dynamic range
    assert abs(vm["slope"] - 1.0) < abs(before["slope"] - 1.0)
    print(f"[demo] slope {before['slope']:.2f} -> linear {lin['slope']:.2f} / varmatch {vm['slope']:.2f}; "
          f"R² {before['r2']:.3f} -> linear {lin['r2']:.3f} / varmatch {vm['r2']:.3f}  OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Post-hoc slope recalibration (fit on train, apply to val).")
    p.add_argument("run_root", nargs="?", help="results/<run> dir with train/ and validation/ attention dumps")
    p.add_argument("--demo", action="store_true", help="run the self-check on synthetic data")
    args = p.parse_args()
    if args.demo or not args.run_root:
        _demo()
    else:
        _print_report(calibrate_run(args.run_root))
