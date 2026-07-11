"""Per-epoch pose-selection (KID) top-k success rates from model attention.

For active, correctly-predicted molecules that have an experimental pose, this
measures whether the model's top-k attention-weighted conformers recover a
near-experimental pose:

    RMSD success (top-k): a conformer among the k most-attended has
        best_rmsd_to_exp <= rmsd_threshold.
    O3A success (top-k):  a conformer among the k most-attended has
        norm_o3a_to_exp = o3a_to_exp / max(o3a_to_exp) >= o3a_threshold.

A molecule is RMSD-valid if any of its conformers is within the RMSD threshold,
and O3A-valid if any reaches the normalized-O3A threshold. This mirrors
ppl/kid_calculator/calc_topk_success_rate.py (evaluate_bag), computed inline each
epoch from the model's attention instead of from exported CSVs. The per-conformer
best_rmsd_to_exp / o3a_to_exp come from the SDF (keyed by conformer _Name), so no
RMSD/O3A alignment is recomputed here.
"""
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)

# A molecule has an experimental (crystal) structure iff its id is a PDB-like code:
# four alphanumerics starting with a digit, then "_" or end (e.g. 4LXA_5300, 3RSV).
# Matches ppl/kid_calculator/filter_pdb.py — KID is only meaningful for these.
_PDB_ID_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}(?:_|$)", re.IGNORECASE)


def is_pdb_molecule(mol_id: str) -> bool:
    return _PDB_ID_PATTERN.match(str(mol_id)) is not None


class KidCalculator:
    """Compute top-k pose-selection success rates from per-molecule attention."""

    def __init__(
        self,
        sdf_path,
        *,
        top_k: Sequence[int] = (1, 3, 5),
        rmsd_threshold: float = 2.0,
        o3a_threshold: float = 0.8,
        active_threshold: float = 7.0,
        pred_tol: float = 1.0,
        pdb_only: bool = True,
    ) -> None:
        self.top_k = tuple(int(k) for k in top_k)
        self.rmsd_threshold = float(rmsd_threshold)
        self.o3a_threshold = float(o3a_threshold)
        self.active_threshold = float(active_threshold)
        self.pred_tol = float(pred_tol)
        self.pdb_only = bool(pdb_only)
        self._rmsd, self._o3a = self._load_sdf(sdf_path)
        LOGGER.info(
            "[KID] loaded pose properties for %d conformers (rmsd) / %d (o3a) from %s",
            len(self._rmsd), len(self._o3a), sdf_path,
        )

    @staticmethod
    def _load_sdf(sdf_path):
        from rdkit import Chem

        rmsd: dict[str, float] = {}
        o3a: dict[str, float] = {}
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol in suppl:
            if mol is None or not mol.HasProp("_Name"):
                continue
            name = mol.GetProp("_Name")

            def _prop(p):
                if not mol.HasProp(p):
                    return None
                try:
                    return float(mol.GetProp(p))
                except ValueError:
                    return None

            r, o = _prop("best_rmsd_to_exp"), _prop("o3a_to_exp")
            if r is not None:
                rmsd[name] = r
            if o is not None:
                o3a[name] = o
        if not rmsd and not o3a:
            raise ValueError(
                f"No best_rmsd_to_exp / o3a_to_exp properties found in SDF {sdf_path}"
            )
        return rmsd, o3a

    def _topk_success(self, attention, values, threshold, mode):
        """Top-k success per molecule for one metric, or None if not valid.

        ``values`` are the per-conformer metric aligned with ``attention``; NaN
        entries (conformers absent from the SDF) are dropped. Returns {k: bool} of
        whether a hit appears among the k most-attended conformers, or None when
        the molecule has no hittable pose (not valid for this metric).
        """
        finite = np.isfinite(values)
        att, vals = attention[finite], values[finite]
        if vals.size == 0:
            return None
        hit = (vals <= threshold) if mode == "le" else (vals >= threshold)
        if not hit.any():
            return None  # no near-experimental pose exists -> molecule not valid
        order = np.argsort(-att, kind="mergesort")  # descending attention
        ranked = hit[order]
        return {k: bool(ranked[: min(k, ranked.size)].any()) for k in self.top_k}

    def compute(self, molecules) -> dict:
        """molecules: iterable of {conf_ids: list[str], attention: 1d array,
        true: float, pred: float}. Returns a flat metrics dict of success rates."""
        rmsd_hits = {k: 0 for k in self.top_k}
        o3a_hits = {k: 0 for k in self.top_k}
        rmsd_valid = o3a_valid = n_active_correct = 0

        for m in molecules:
            # KID is only defined for molecules with an experimental (PDB) structure.
            if self.pdb_only and not is_pdb_molecule(m.get("mol_id", "")):
                continue
            true, pred = m.get("true"), m.get("pred")
            if true is None or pred is None:
                continue
            # active + correctly predicted (filter_actives)
            if not (true >= self.active_threshold and abs(true - pred) <= self.pred_tol):
                continue
            conf_ids = [str(c) for c in m["conf_ids"]]
            att = np.asarray(m["attention"], dtype=float).flatten()
            if att.size == 0 or att.size != len(conf_ids):
                continue
            n_active_correct += 1

            rvals = np.array([self._rmsd.get(c, np.nan) for c in conf_ids], dtype=float)
            r = self._topk_success(att, rvals, self.rmsd_threshold, "le")
            if r is not None:
                rmsd_valid += 1
                for k in self.top_k:
                    rmsd_hits[k] += int(r[k])

            ovals = np.array([self._o3a.get(c, np.nan) for c in conf_ids], dtype=float)
            finite = np.isfinite(ovals)
            if finite.any():
                mx = float(np.nanmax(ovals))
                norm = ovals / mx if mx > 0 else np.full_like(ovals, np.nan)
                o = self._topk_success(att, norm, self.o3a_threshold, "ge")
                if o is not None:
                    o3a_valid += 1
                    for k in self.top_k:
                        o3a_hits[k] += int(o[k])

        out = {
            "n_active_correct": float(n_active_correct),
            "n_rmsd_valid": float(rmsd_valid),
            "n_o3a_valid": float(o3a_valid),
        }
        for k in self.top_k:
            out[f"rmsd_top{k}"] = rmsd_hits[k] / rmsd_valid if rmsd_valid else float("nan")
            out[f"o3a_top{k}"] = o3a_hits[k] / o3a_valid if o3a_valid else float("nan")
        return out


def _unpack_batch(batch):
    """collate_mil tuple -> (bags, y, bag_ids, kpm, cluster_ids, series_labels)."""
    cluster_ids = series_labels = None
    if len(batch) == 4:
        bags, y, bag_ids, kpm = batch
    elif len(batch) == 5:
        bags, y, bag_ids, kpm, cluster_ids = batch
    else:
        bags, y, bag_ids, kpm, cluster_ids, series_labels = batch
    return bags, y, bag_ids, kpm, cluster_ids, series_labels


@torch.no_grad()
def extract_molecule_attention(model, loader, conf_ids_map, *, stage, epoch, noexp_only):
    """Run the model over ``loader`` and return per-molecule attention records.

    Each record is {conf_ids, attention (aligned, non-padded), true, pred}. For
    validation we keep only "__noexp" bags (the KID target, matching filter_actives
    and the eval metrics); train bags are already experimental-pose-free.
    """
    core = getattr(model, "core", model)
    device = next(model.parameters()).device
    task = str(getattr(model, "task", "regression")).lower()
    records: list[dict] = []
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            bags, y, bag_ids, kpm, cluster_ids, series = _unpack_batch(batch)
            x = bags.to(device)
            kwargs = {"labels": y.to(device), "stage": stage, "current_epoch": int(epoch)}
            if kpm is not None:
                kwargs["key_padding_mask"] = kpm.to(device)
            if cluster_ids is not None:
                kwargs["cluster_ids"] = cluster_ids.to(device)
            if series is not None:
                kwargs["series_labels"] = series
            logit, extras = core(x, **kwargs)
            alpha = extras.get("alpha_final", extras.get("alpha"))
            if alpha is None:
                continue
            alpha = alpha.detach().cpu().numpy()
            pred = (torch.sigmoid(logit) if task == "classification" else logit)
            pred = pred.detach().cpu().numpy().reshape(-1)
            yv = y.detach().cpu().numpy().reshape(-1)
            valid = (~kpm.bool().cpu().numpy()) if kpm is not None else None

            for b, bag_id in enumerate(bag_ids):
                bid = str(bag_id)
                if noexp_only and not bid.endswith("__noexp"):
                    continue
                cids = conf_ids_map.get(bid) if conf_ids_map else None
                a = alpha[b][valid[b]] if valid is not None else alpha[b]
                a = np.asarray(a, dtype=float).reshape(-1)
                if cids is None or len(cids) != a.size:
                    continue  # cannot align ids to attention -> skip (never mislabel)
                mol_id = bid[: -len("__noexp")] if bid.endswith("__noexp") else bid
                records.append(
                    {"mol_id": mol_id, "conf_ids": list(cids), "attention": a,
                     "true": float(yv[b]), "pred": float(pred[b])}
                )
    finally:
        model.train(was_training)
    return records


class KidMetricCallback(pl.Callback):
    """Log per-epoch KID top-k success rates (RMSD + O3A) for train and val."""

    def __init__(self, calculator: KidCalculator) -> None:
        super().__init__()
        self.calc = calculator

    def _run(self, trainer, model, loader, stage, prefix, noexp_only):
        if loader is None or (hasattr(loader, "__len__") and len(loader) == 0):
            return
        try:
            dm = trainer.datamodule
            conf_ids = dict(getattr(dm, "bag_conf_ids", {}) or {})
            recs = extract_molecule_attention(
                model, loader, conf_ids, stage=stage,
                epoch=int(trainer.current_epoch), noexp_only=noexp_only,
            )
            metrics = self.calc.compute(recs)
            log = {f"kid_{prefix}_{k}": float(v) for k, v in metrics.items()
                   if not k.startswith("n_")}
            model.log_dict(log, sync_dist=True, add_dataloader_idx=False)
            logging.getLogger("milk").info(
                "KID/%s: active&correct=%d rmsd_valid=%d o3a_valid=%d · "
                "rmsd top%s=%s · o3a top%s=%s",
                prefix, int(metrics["n_active_correct"]), int(metrics["n_rmsd_valid"]),
                int(metrics["n_o3a_valid"]),
                self.calc.top_k, _fmt(metrics, "rmsd_top"),
                self.calc.top_k, _fmt(metrics, "o3a_top"),
            )
        except Exception as e:  # KID is diagnostic — never break training
            LOGGER.warning("[KID] %s computation failed: %s", prefix, e)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        dm = trainer.datamodule
        self._run(trainer, pl_module, dm.val_dataloader(), "val", "val", noexp_only=True)
        # Train KID over a sequential (unbalanced) pass, each molecule once.
        train_loader = dm._make_loader(dm._train, shuffle=False)
        self._run(trainer, pl_module, train_loader, "train", "train", noexp_only=False)


def _fmt(metrics, base):
    return "/".join(
        f"{metrics[k]:.2f}" if np.isfinite(metrics.get(k, np.nan)) else "n/a"
        for k in sorted(metrics) if k.startswith(base)
    )
