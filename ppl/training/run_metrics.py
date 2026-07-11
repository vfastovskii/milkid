"""Structured final-model metrics: <results_dir>/run_metrics.json (errors + KID).

Emitted alongside res.txt so downstream tooling (HPO) reads metrics as data
instead of scraping text. KID is computed on the best model here because a
standalone trainer.validate() does not fire the per-epoch KID callback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ppl.training.kid_calculator import kid_metrics_for_model

LOGGER = logging.getLogger(__name__)

_ERROR_KEYS = ("rmse", "mae", "loss", "r2", "pearson", "spearman")
_KID_KEYS = tuple(
    f"{split}_{metric}_top{k}"
    for split in ("val", "train")
    for metric in ("rmsd", "o3a")
    for k in (1, 3, 5)
)


def _load_kid_calculator(mt):
    """Build a KidCalculator from trainer config, or None if unavailable."""
    tcfg = mt.trainer_cfg
    if not bool(getattr(tcfg, "kid_metric_enabled", False)):
        return None
    sdf = getattr(tcfg, "kid_sdf_path", None)
    if not sdf:
        return None
    from ppl.training.kid_calculator import KidCalculator
    return KidCalculator(
        sdf,
        top_k=getattr(tcfg, "kid_top_k", [1, 3, 5]),
        rmsd_threshold=getattr(tcfg, "kid_rmsd_threshold", 2.0),
        o3a_threshold=getattr(tcfg, "kid_o3a_threshold", 0.8),
        active_threshold=getattr(tcfg, "kid_active_threshold", 7.0),
        pred_tol=getattr(tcfg, "kid_pred_tol", 1.0),
        pdb_only=bool(getattr(tcfg, "kid_pdb_only", True)),
    )


def write_run_metrics(results_dir, mt, dm, best_model, val_metrics, train_metrics) -> dict:
    results_dir = Path(results_dir)
    out: dict = {}
    try:
        out["best_epoch"] = mt._get_best_epoch_from_trainer(mt._last_trainer)
    except Exception:
        out["best_epoch"] = None

    for key in _ERROR_KEYS:
        out[f"val_{key}"] = _num((val_metrics or {}).get(f"val_{key}"))
        out[f"train_{key}"] = _num((train_metrics or {}).get(f"train_{key}"))

    for key in (*_KID_KEYS, "kid_n_active_correct", "kid_n_rmsd_valid", "kid_n_o3a_valid"):
        out[key] = None
    try:
        calc = _load_kid_calculator(mt)
        if calc is not None:
            epoch = out["best_epoch"] if out["best_epoch"] is not None else 0
            out.update(kid_metrics_for_model(best_model, dm, calc, epoch=epoch))
    except Exception as e:  # KID is diagnostic — never break metric emission
        LOGGER.warning("[MODEL] run_metrics KID computation failed: %s", e)

    try:
        (results_dir / "run_metrics.json").write_text(json.dumps(out, indent=2, sort_keys=True))
        LOGGER.info("[MODEL] Wrote structured metrics to %s", results_dir / "run_metrics.json")
    except OSError as e:
        LOGGER.warning("[MODEL] Failed to write run_metrics.json: %s", e)
    return out


def _num(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
