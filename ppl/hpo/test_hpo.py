import json
import math
import pytest
from ppl.hpo.run_metrics_reader import RunMetrics, RunMetricsError


def test_run_metrics_objectives(tmp_path):
    (tmp_path / "run_metrics.json").write_text(json.dumps({"val_rmsd_top1": 0.61, "val_rmse": 1.07}))
    rm = RunMetrics.from_dir(tmp_path)
    assert rm.objectives() == (0.61, 1.07)


def test_run_metrics_missing_file_raises(tmp_path):
    with pytest.raises(RunMetricsError):
        RunMetrics.from_dir(tmp_path)


def test_run_metrics_null_objective_raises(tmp_path):
    (tmp_path / "run_metrics.json").write_text(json.dumps({"val_rmsd_top1": None, "val_rmse": 1.07}))
    with pytest.raises(RunMetricsError):
        RunMetrics.from_dir(tmp_path).objectives()
