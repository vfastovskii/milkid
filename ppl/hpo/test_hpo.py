import json
import math
import pytest
from optuna.trial import FixedTrial
from ppl.hpo.run_metrics_reader import RunMetrics, RunMetricsError
from ppl.hpo.optuna_runner import SearchSpace


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


def test_search_space_samples(tmp_path):
    ss_yaml = """
groups:
  optim:
    model.optim.lr:
      type: categorical
      choices: [0.001, 0.0005]
fixed_overrides:
  trainer.max_epochs: 5
"""
    p = tmp_path / "ss.yaml"
    p.write_text(ss_yaml)
    ss = SearchSpace(p, phase="all")
    sampled = ss.sample(FixedTrial({"model.optim.lr": 0.0005}), base_config={})
    assert sampled["model.optim.lr"] == 0.0005
    assert sampled["trainer.max_epochs"] == 5   # fixed override included


import yaml
from ppl.hpo.optuna_runner import TrialConfigBuilder


def test_trial_config_builder(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("trainer:\n  kid_metric_enabled: false\n  max_epochs: 100\n")
    builder = TrialConfigBuilder(base)
    out = builder.build({"trainer.max_epochs": 7}, tmp_path / "t0", "exp/t0", "t0")
    cfg = yaml.safe_load(out.read_text())
    assert cfg["trainer"]["max_epochs"] == 7
    assert cfg["trainer"]["kid_metric_enabled"] is True
    assert cfg["trainer"]["experiment_name"] == "exp/t0"
    assert cfg["trainer"]["run_name"] == "t0"
