import types
import pytest
import ppl.training.kid_calculator as kc


def _fake_dm():
    dm = types.SimpleNamespace()
    dm.val_dataloader = lambda: ["val_batch"]
    dm._train = ["train_ds"]
    dm._make_loader = lambda ds, shuffle: ["train_batch"]
    dm.bag_conf_ids = {}
    return dm


class _FakeCalc:
    # returns different numbers for val vs train so we can tell them apart
    def __init__(self):
        self.calls = 0
    def compute(self, records):
        self.calls += 1
        base = 0.1 * self.calls
        return {
            "rmsd_top1": base, "rmsd_top3": base + 0.01, "rmsd_top5": base + 0.02,
            "o3a_top1": base + 0.03, "o3a_top3": base + 0.04, "o3a_top5": base + 0.05,
            "n_active_correct": 24.0, "n_rmsd_valid": 23.0, "n_o3a_valid": 24.0,
        }


def test_kid_metrics_for_model_flattens_val_and_train(monkeypatch):
    seen = []
    def fake_extract(model, loader, conf_ids_map, *, stage, epoch, noexp_only):
        seen.append((loader[0], stage, noexp_only))
        return [{"mol_id": "1ABC", "conf_ids": ["c0"], "attention": [1.0], "true": 8.0, "pred": 8.0}]
    monkeypatch.setattr(kc, "extract_molecule_attention", fake_extract)

    out = kc.kid_metrics_for_model(object(), _fake_dm(), _FakeCalc(), epoch=23)

    # val pass is call #1 (0.1), train pass is call #2 (0.2)
    assert out["val_rmsd_top1"] == pytest.approx(0.1)
    assert out["train_rmsd_top1"] == pytest.approx(0.2)
    assert out["val_o3a_top5"] == pytest.approx(0.15)
    assert out["kid_n_active_correct"] == pytest.approx(24.0)
    # val uses noexp_only=True, train uses noexp_only=False
    assert ("val_batch", "val", True) in seen
    assert ("train_batch", "train", False) in seen


import json
import ppl.training.run_metrics as rm


def _fake_mt(kid_enabled):
    tcfg = types.SimpleNamespace(
        kid_metric_enabled=kid_enabled, kid_sdf_path="x.sdf", kid_top_k=[1, 3, 5],
        kid_rmsd_threshold=2.0, kid_o3a_threshold=0.8, kid_active_threshold=7.0,
        kid_pred_tol=1.0, kid_pdb_only=True,
    )
    mt = types.SimpleNamespace(trainer_cfg=tcfg, _last_trainer=object())
    mt._get_best_epoch_from_trainer = lambda tr: 23
    return mt


def test_write_run_metrics_errors_and_kid(monkeypatch, tmp_path):
    monkeypatch.setattr(rm, "_load_kid_calculator", lambda mt: object())
    monkeypatch.setattr(
        rm, "kid_metrics_for_model",
        lambda model, dm, calc, *, epoch: {"val_rmsd_top1": 0.61, "train_rmsd_top1": 0.29},
    )
    val_metrics = {"val_rmse": 1.07, "val_mae": 0.86, "val_loss": 1.14}
    train_metrics = {"train_rmse": 1.17, "train_mae": 0.98}

    out = rm.write_run_metrics(tmp_path, _fake_mt(True), object(), object(), val_metrics, train_metrics)

    written = json.loads((tmp_path / "run_metrics.json").read_text())
    assert written == out
    assert out["val_rmse"] == 1.07 and out["train_rmse"] == 1.17
    assert out["val_rmsd_top1"] == 0.61 and out["best_epoch"] == 23


def test_write_run_metrics_kid_disabled_gives_null(monkeypatch, tmp_path):
    out = rm.write_run_metrics(tmp_path, _fake_mt(False), object(), object(),
                               {"val_rmse": 1.07}, {"train_rmse": 1.17})
    assert out["val_rmse"] == 1.07
    assert out["val_rmsd_top1"] is None


def test_resolve_sdf_path_is_cwd_independent(monkeypatch, tmp_path):
    from pathlib import Path
    from ppl.training.kid_calculator import _resolve_sdf_path
    rel = "ppl/training/kid_calculator.py"          # a real repo file (not the 480MB SDF)
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)                      # cwd where the relative path does NOT exist
    resolved = Path(_resolve_sdf_path(rel))
    assert resolved == repo_root / rel and resolved.exists()   # resolved via repo root, not cwd
    ap = str(repo_root / rel)
    assert _resolve_sdf_path(ap) == ap                          # absolute passes through
    assert _resolve_sdf_path("does/not/exist.sdf") == "does/not/exist.sdf"  # fallback as-given
