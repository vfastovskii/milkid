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
