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


import subprocess
import types
from pathlib import Path as _P
from ppl.hpo.optuna_runner import PipelineRunner


def test_pipeline_runner_success(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    runner = PipelineRunner(package_root=tmp_path, repo_root=tmp_path)
    results = runner.run(tmp_path / "config.yaml", "exp/t0", tmp_path / "t.log")
    assert results == tmp_path / "results" / "exp/t0"


def test_pipeline_runner_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1))
    (tmp_path / "t.log").write_text("boom")
    runner = PipelineRunner(package_root=tmp_path, repo_root=tmp_path)
    import pytest
    with pytest.raises(RuntimeError):
        runner.run(tmp_path / "config.yaml", "exp/t0", tmp_path / "t.log")


import json as _json
from optuna.trial import FixedTrial as _FT
from ppl.hpo.optuna_runner import MilkObjective


class _StubSS:
    def sample(self, trial, base_config): return {"a": 1}
class _StubBuilder:
    base_config = {}
    def build(self, sampled, trial_dir, exp, run):
        _P(trial_dir).mkdir(parents=True, exist_ok=True); return _P(trial_dir) / "config.yaml"
class _StubRunner:
    def __init__(self, results): self.results = results
    def run(self, config_path, experiment_name, log_path): return self.results


def test_milk_objective_returns_tuple(tmp_path):
    results = tmp_path / "res"; results.mkdir()
    (results / "run_metrics.json").write_text(_json.dumps({"val_rmsd_top1": 0.61, "val_rmse": 1.07}))
    obj = MilkObjective(_StubSS(), _StubBuilder(), _StubRunner(results),
                        trial_root=tmp_path, base_experiment="exp")
    assert obj(_FT({})) == (0.61, 1.07)


from ppl.hpo.optuna_runner import HpoStudy


def test_hpo_study_multiobjective(tmp_path):
    # objective callable that ignores the trial and returns a fixed Pareto-ish pair
    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x  # maximize x, minimize (1-x)
    study = HpoStudy(objective, study_name="t", out_dir=tmp_path)
    result = study.run(n_trials=5)
    assert result.directions[0].name == "MAXIMIZE"
    assert result.directions[1].name == "MINIMIZE"
    assert (tmp_path / "pareto_front.json").exists()
    front = __import__("json").loads((tmp_path / "pareto_front.json").read_text())
    assert isinstance(front, list) and len(front) >= 1


def test_hpo_study_continues_past_failed_trials(tmp_path):
    calls = {"n": 0}
    def flaky(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("simulated trial failure")
        return x, 1.0 - x
    study = HpoStudy(flaky, study_name="t_flaky", out_dir=tmp_path)
    result = study.run(n_trials=6)            # must NOT raise
    import optuna
    completed = [t for t in result.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed = [t for t in result.trials if t.state == optuna.trial.TrialState.FAIL]
    assert len(completed) >= 1 and len(failed) >= 1   # some failed, study continued
    assert (tmp_path / "pareto_front.json").exists()
