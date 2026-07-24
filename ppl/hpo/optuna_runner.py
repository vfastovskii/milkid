"""Optuna runner for MILK experiment configs.

The runner consumes a grouped YAML search-space file, samples config overrides,
writes a concrete trial config, launches the normal MILK CLI in a subprocess,
and reports the requested validation metric back to Optuna.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import optuna
import yaml

from ppl.hpo.run_metrics_reader import RunMetrics


LOGGER = logging.getLogger(__name__)


def _repo_root() -> Path:
    # this file is ppl/hpo/optuna_runner.py; parents[2] is the repo root
    return Path(__file__).resolve().parents[2]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str | Path, *, bases: Iterable[Path]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if not parts:
        raise ValueError("Empty config override path")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise TypeError(
                f"Cannot set {dotted_path!r}: {part!r} is not a mapping"
            )
        cursor = existing
    cursor[parts[-1]] = value


def _get_by_dotted_path(config: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_path, value in overrides.items():
        _set_by_dotted_path(config, str(dotted_path), value)


def _sample_value(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = str(spec.get("type", "categorical")).lower()
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{name}: categorical spec needs non-empty choices")
        return trial.suggest_categorical(name, choices)

    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        if log:
            return trial.suggest_int(name, low, high, log=True)
        return trial.suggest_int(name, low, high, step=step)

    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is None:
            return trial.suggest_float(name, low, high, log=log)
        return trial.suggest_float(name, low, high, step=float(step))

    raise ValueError(f"{name}: unsupported search-space type {kind!r}")


def _active_ignore_paths(
    search_space: dict[str, Any],
    config: dict[str, Any],
) -> set[str]:
    ignored: set[str] = set()
    for rule in search_space.get("conditional_rules", []) or []:
        when = str(rule.get("when", "")).strip()
        # Supported simple condition format:
        #   some.path == false/true/value
        if "==" not in when:
            continue
        lhs, rhs = [part.strip() for part in when.split("==", 1)]
        current = _get_by_dotted_path(config, lhs)
        rhs_lower = rhs.lower()
        if rhs_lower == "false":
            expected: Any = False
        elif rhs_lower == "true":
            expected = True
        elif rhs_lower in {"none", "null"}:
            expected = None
        else:
            rhs_clean = rhs.strip("\"'")
            try:
                if any(ch in rhs_clean for ch in (".", "e", "E")):
                    expected = float(rhs_clean)
                else:
                    expected = int(rhs_clean)
            except ValueError:
                expected = rhs_clean
        if current == expected:
            ignored.update(str(path) for path in rule.get("ignore", []) or [])
    return ignored


def _phase_settings(
    search_space: dict[str, Any],
    phase: str,
) -> tuple[set[str] | None, dict[str, Any]]:
    if phase == "all":
        return None, {}
    strategy = search_space.get("recommended_two_phase_strategy", {})
    phase_cfg = strategy.get(phase)
    if not isinstance(phase_cfg, dict):
        raise ValueError(
            f"Unknown phase {phase!r}; expected 'all' or one of "
            f"{sorted(strategy.keys())}"
        )
    include_groups = phase_cfg.get("include_groups")
    include = set(include_groups) if include_groups else None
    force_overrides = phase_cfg.get("force_overrides") or {}
    return include, dict(force_overrides)


def _sample_overrides(
    trial: optuna.Trial,
    search_space: dict[str, Any],
    *,
    include_groups: set[str] | None,
    force_overrides: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    groups = search_space.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("search_space.groups must be a mapping")

    for group_name, params in groups.items():
        if include_groups is not None and group_name not in include_groups:
            continue
        if not isinstance(params, dict):
            raise ValueError(f"search_space.groups.{group_name} must be a mapping")
        for dotted_path, spec in params.items():
            if dotted_path in force_overrides:
                continue
            if not isinstance(spec, dict):
                raise ValueError(f"{dotted_path}: parameter spec must be a mapping")
            overrides[dotted_path] = _sample_value(trial, dotted_path, spec)

    overrides.update(force_overrides)
    return overrides


def _filter_conditional_overrides(
    search_space: dict[str, Any],
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    # Apply current overrides to inspect conditions, then remove ignored sampled
    # paths.  The condition path itself remains in place.
    probe = deepcopy(config)
    _apply_overrides(probe, overrides)
    ignored = _active_ignore_paths(search_space, probe)
    return {k: v for k, v in overrides.items() if k not in ignored}


def _trial_name(study_name: str, trial_number: int) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name).strip("_")
    return f"{safe_study}_trial_{trial_number:04d}"


class SearchSpace:
    """Loads a grouped search-space YAML and samples per-trial overrides."""

    def __init__(self, path, phase: str = "all") -> None:
        self.path = Path(path)
        self.data = _load_yaml(self.path)
        self.phase = phase
        self.include_groups, self.force_overrides = _phase_settings(self.data, phase)

    def sample(self, trial, base_config: dict) -> dict:
        sampled = _sample_overrides(
            trial, self.data,
            include_groups=self.include_groups,
            force_overrides=self.force_overrides,
        )
        probe = deepcopy(base_config)
        _apply_overrides(probe, dict(self.data.get("fixed_overrides", {}) or {}))
        sampled = _filter_conditional_overrides(self.data, probe, sampled)
        merged = dict(self.data.get("fixed_overrides", {}) or {})
        merged.update(sampled)
        return merged


class TrialConfigBuilder:
    """Builds a concrete trial config.yaml from base + sampled overrides."""

    def __init__(self, base_config_path) -> None:
        self.base_config = _load_yaml(Path(base_config_path))

    def build(self, sampled: dict, trial_dir, experiment_name: str, run_name: str) -> Path:
        trial_dir = Path(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = deepcopy(self.base_config)
        _apply_overrides(config, sampled)
        _set_by_dotted_path(config, "trainer.kid_metric_enabled", True)
        _set_by_dotted_path(config, "trainer.experiment_name", experiment_name)
        _set_by_dotted_path(config, "trainer.run_name", run_name)
        config_path = trial_dir / "config.yaml"
        _write_yaml(config, config_path)
        return config_path


class PipelineRunner:
    """Runs the MILK pipeline as an isolated subprocess for one trial."""

    def __init__(self, package_root, repo_root, log_level: str = "INFO", timeout=None) -> None:
        self.package_root = Path(package_root)
        self.repo_root = Path(repo_root)
        self.log_level = log_level
        self.timeout = timeout

    def run(self, config_path, experiment_name: str, log_path) -> Path:
        cmd = [sys.executable, "-m", "ppl.cli.entry_point",
               "-c", str(config_path), "--log-level", self.log_level]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.repo_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("MPLCONFIGDIR", "/tmp")
        with Path(log_path).open("w") as log_file:
            completed = subprocess.run(
                cmd, cwd=self.package_root, env=env,
                stdout=log_file, stderr=subprocess.STDOUT,
                text=True, timeout=self.timeout, check=False,
            )
        results_dir = self.package_root / "results" / experiment_name
        if completed.returncode != 0:
            tail = ""
            lp = Path(log_path)
            if lp.exists():
                tail = "\n".join(lp.read_text(errors="replace").splitlines()[-60:])
            raise RuntimeError(
                f"pipeline exited {completed.returncode}; see {log_path}\n"
                f"--- log tail ---\n{tail}"
            )
        return results_dir


class MilkObjective:
    """One Optuna trial: sample -> build config -> run pipeline -> read (rmsd_top1, rmse)."""

    def __init__(self, search_space, config_builder, runner, *, trial_root) -> None:
        self.search_space = search_space
        self.config_builder = config_builder
        self.runner = runner
        self.trial_root = Path(trial_root)

    def __call__(self, trial) -> tuple[float, float]:
        trial_name = _trial_name(trial.study.study_name, trial.number) if hasattr(trial, "study") \
            else f"trial_{trial.number:04d}"
        # One directory per trial holds EVERYTHING (config, models, preprocessing, mlflow,
        # run_metrics) — the same dir the runner drops config.yaml + trial.log into, under
        # the study's pipeline output. Passing the ABSOLUTE trial dir as experiment_name
        # makes create_results_directory (and PipelineRunner.run's reader) resolve straight
        # to it: pathlib drops the "results" prefix when the right operand is absolute.
        trial_dir = (self.trial_root / trial_name).resolve()
        experiment_name = str(trial_dir)
        sampled = self.search_space.sample(trial, self.config_builder.base_config)
        config_path = self.config_builder.build(sampled, trial_dir, experiment_name, trial_name)
        LOGGER.info("[HPO] ▶ trial %d training — %s (log: %s)",
                    trial.number, experiment_name, trial_dir / "trial.log")
        results_dir = self.runner.run(config_path, experiment_name, trial_dir / "trial.log")
        metrics = RunMetrics.from_dir(results_dir)
        rmsd_top1, rmse = metrics.objectives()
        trial.set_user_attr("results_dir", str(results_dir))
        trial.set_user_attr("val_rmsd_top1", rmsd_top1)
        trial.set_user_attr("val_rmse", rmse)
        return rmsd_top1, rmse


def _count_complete(study) -> int:
    return sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)


class HpoStudy:
    """Multi-objective Optuna study: maximize val_rmsd_top1, minimize val_rmse."""

    OBJECTIVE_NAMES = ("val_rmsd_top1", "val_rmse")

    def __init__(self, objective, *, study_name: str, out_dir, storage=None, seed: int = 42) -> None:
        self.objective = objective
        self.study_name = study_name
        self.out_dir = Path(out_dir)
        self.storage = storage
        self.seed = seed

    def run(self, n_trials: int, n_jobs: int = 1) -> "optuna.Study":
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
            directions=["maximize", "minimize"],
            # Multi-objective TPE, not NSGA-II. NSGA-II is population-based with
            # population_size=50 by default, so at our trial budget (~30) it never
            # finishes its initial random population — i.e. it degenerates to random
            # search. TPE starts exploiting after n_startup_trials.
            # multivariate=True models parameter INTERACTIONS, which matters here:
            # component LR = base_lr × factor, so the base and the factors are
            # correlated by construction and must not be modelled independently.
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=8,   # ~1/4 of a 30-trial budget on random exploration
                multivariate=True,
            ),
        )
        prior = _count_complete(study)  # trials already done (when resuming a study)

        def _log_progress(study: "optuna.Study", trial) -> None:
            done = _count_complete(study) - prior
            if trial.values:
                status = f"val_rmsd_top1={trial.values[0]:.4f} val_rmse={trial.values[1]:.4f}"
            else:
                status = f"{trial.state.name} — no objectives (trial failed, study continues)"
            comp = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE and t.values]
            best = max(comp, key=lambda t: (t.values[0], -t.values[1])) if comp else None
            best_s = (f"best val_rmsd_top1={best.values[0]:.4f} @trial#{best.number}"
                      if best else "no completed trials yet")
            LOGGER.info("[HPO] ✓ trial %d finished — %d/%d this run — %s — %s",
                        trial.number, max(done, 0), n_trials, status, best_s)

        # n_jobs>1 runs trials concurrently (threads, each blocking on a training subprocess),
        # so several trainings share the one GPU. Failed trials are caught so the study continues.
        # show_progress_bar renders a trial bar in a real terminal; through the pipeline's log
        # pipe tqdm auto-disables, and the per-trial "✓ trial N — k/total" lines carry progress.
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs,
                       catch=(Exception,), gc_after_trial=True,
                       callbacks=[_log_progress], show_progress_bar=(n_jobs == 1))
        self._write_pareto(study)
        return study

    def _write_pareto(self, study) -> None:
        front = []
        for t in study.best_trials:  # non-dominated set
            front.append({
                "number": t.number,
                "params": t.params,
                self.OBJECTIVE_NAMES[0]: t.values[0] if t.values else None,
                self.OBJECTIVE_NAMES[1]: t.values[1] if t.values else None,
            })
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "pareto_front.json").write_text(json.dumps(front, indent=2, sort_keys=True))
        LOGGER.info("[OPTUNA] wrote %d Pareto trials to %s",
                    len(front), self.out_dir / "pareto_front.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-objective Optuna HPO for MILK.")
    p.add_argument("--search-space", required=True)
    p.add_argument("--base-config", default=None, help="Overrides base_config from the search-space YAML.")
    p.add_argument("--study-name", default="milk_hpo")
    p.add_argument("--storage", default=None, help="e.g. sqlite:///ppl/optuna_trials/study.db")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel trials (each a training subprocess); on a single GPU keep small (2-3).")
    p.add_argument("--phase", default="all")
    p.add_argument("--trial-root", default=None, help="Where trial dirs are written.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--trial-timeout", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args(argv)

    package_root, repo_root = _package_root(), _repo_root()
    search_space = SearchSpace(
        _resolve_path(args.search_space, bases=[package_root, repo_root]), phase=args.phase
    )
    base_config_path = args.base_config or search_space.data.get("base_config")
    if not base_config_path:
        raise SystemExit("No base_config given (pass --base-config or set it in the search space).")
    base_config_path = _resolve_path(base_config_path, bases=[package_root, repo_root])
    config_builder = TrialConfigBuilder(base_config_path)

    trial_root = (
        Path(args.trial_root) if args.trial_root else package_root / "optuna_trials" / args.study_name
    ).resolve()
    trial_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8, multivariate=True),
        )
        for _ in range(args.n_trials):
            trial = study.ask()
            name = _trial_name(args.study_name, trial.number)
            sampled = search_space.sample(trial, config_builder.base_config)
            config_builder.build(sampled, trial_root / name, str((trial_root / name).resolve()), name)
            study.tell(trial, (0.0, 0.0))
        LOGGER.info("[DRY-RUN] wrote %d trial configs under %s", args.n_trials, trial_root)
        return 0

    runner = PipelineRunner(package_root, repo_root, log_level=args.log_level, timeout=args.trial_timeout)
    objective = MilkObjective(search_space, config_builder, runner, trial_root=trial_root)
    HpoStudy(objective, study_name=args.study_name, out_dir=trial_root,
             storage=args.storage).run(args.n_trials, n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
