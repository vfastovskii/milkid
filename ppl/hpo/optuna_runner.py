"""Optuna runner for MILK experiment configs.

The runner consumes a grouped YAML search-space file, samples config overrides,
writes a concrete trial config, launches the normal MILK CLI in a subprocess,
and reports the requested validation metric back to Optuna.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import optuna
import yaml
from optuna.trial import TrialState

from ppl.hpo.run_metrics_reader import RunMetrics


LOGGER = logging.getLogger(__name__)
METRIC_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_./:-]+)\s*:\s*(?P<value>[-+0-9.eE]+)\s*$")
CV_METRIC_RE = re.compile(
    r"CV\s+(?P<key>[A-Za-z0-9_./:-]+)\s*:\s*(?P<value>[-+0-9.eE]+)"
)


def _repo_root() -> Path:
    # ppl/utils/optuna_runner.py -> milk_udt
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


def _res_metric_aliases(key: str, section: str | None) -> list[str]:
    """Return stable metric aliases for a key parsed from res.txt.

    res.txt is written in human sections.  The train section can be generated by
    running Lightning validation on the training loader, so older files may
    contain keys such as ``val_rmse`` under "Training metrics".  Normalize those
    to explicit train_* names and keep validation aliases objective-friendly.
    """
    key = str(key)
    if section == "val":
        base = key[4:] if key.startswith("val_") else key
        aliases = [key, f"best_val_{base}"]
        if not key.startswith("val_"):
            aliases.append(f"val_{key}")
        return aliases

    if section == "train":
        base = key
        if base.startswith("val_"):
            base = base[4:]
        elif base.startswith("train_"):
            base = base[6:]
        return [f"train_{base}", f"best_train_{base}"]

    return [key]


def _parse_metrics_from_res_txt(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    section: str | None = None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("validation metrics"):
            section = "val"
            continue
        if lower.startswith("training metrics"):
            section = "train"
            continue
        if lower.startswith(("training schedule", "metric source", "early stopping")):
            section = None
            continue

        match = METRIC_RE.match(line)
        if match:
            try:
                value = float(match.group("value"))
            except ValueError:
                continue
            for alias in _res_metric_aliases(match.group("key"), section):
                metrics[alias] = value
    return metrics


def _parse_metrics_from_log(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(errors="replace").splitlines():
        match = CV_METRIC_RE.search(line)
        if match:
            try:
                metrics[f"cv_{match.group('key')}_mean"] = float(match.group("value"))
            except ValueError:
                continue
    return metrics


def _objective_value(metrics: dict[str, float], metric_name: str) -> float:
    candidates = [
        metric_name,
        f"cv_{metric_name}_mean",
        metric_name.removeprefix("cv_").removesuffix("_mean"),
    ]
    for key in candidates:
        value = metrics.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise KeyError(
        f"Metric {metric_name!r} not found. Available metrics: {sorted(metrics)}"
    )


def _metric_value_or_none(metrics: dict[str, float], metric_name: str) -> float | None:
    candidates = [
        metric_name,
        f"val_{metric_name}",
        f"cv_val_{metric_name}_mean",
        f"cv_{metric_name}_mean",
        metric_name.removeprefix("val_"),
        metric_name.removeprefix("cv_val_").removesuffix("_mean"),
    ]
    for key in candidates:
        value = metrics.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def _require_best_checkpoint_metrics(
    metrics: dict[str, float],
    *,
    res_txt: Path,
) -> None:
    best_epoch = _metric_value_or_none(metrics, "best_epoch")
    best_score = _metric_value_or_none(metrics, "best_checkpoint_score")
    if best_epoch is not None and best_score is not None:
        return

    raise RuntimeError(
        "Trial finished without best-checkpoint metrics. Refusing to register "
        "last-epoch/fallback results. "
        f"best_epoch={best_epoch}, best_checkpoint_score={best_score}, "
        f"res_txt={res_txt}"
    )


def _format_metric(value: float | None, *, decimals: int = 6) -> str:
    if value is None:
        return "NA"
    if decimals <= 0:
        return str(int(round(float(value))))
    return f"{float(value):.{decimals}f}"


def _trial_name(study_name: str, trial_number: int) -> str:
    safe_study = re.sub(r"[^A-Za-z0-9_.-]+", "_", study_name).strip("_")
    return f"{safe_study}_trial_{trial_number:04d}"


class MilkOptunaObjective:
    def __init__(
        self,
        *,
        search_space_path: Path,
        base_config_path: Path,
        trial_root: Path,
        phase: str,
        metric_name: str,
        log_level: str,
        trial_timeout: int | None,
        dry_run: bool,
    ) -> None:
        self.search_space_path = search_space_path
        self.search_space = _load_yaml(search_space_path)
        self.base_config_path = base_config_path
        self.base_config = _load_yaml(base_config_path)
        self.trial_root = trial_root
        self.phase = phase
        self.metric_name = metric_name
        self.log_level = log_level
        self.trial_timeout = trial_timeout
        self.dry_run = dry_run
        self.package_root = _package_root()
        self.repo_root = _repo_root()

        self.include_groups, self.force_overrides = _phase_settings(
            self.search_space,
            phase,
        )

    def __call__(self, trial: optuna.Trial) -> float:
        trial_name = _trial_name(trial.study.study_name, trial.number)
        trial_dir = self.trial_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        sampled = _sample_overrides(
            trial,
            self.search_space,
            include_groups=self.include_groups,
            force_overrides=self.force_overrides,
        )
        config = deepcopy(self.base_config)
        fixed_overrides = dict(self.search_space.get("fixed_overrides", {}) or {})
        _apply_overrides(config, fixed_overrides)
        sampled = _filter_conditional_overrides(self.search_space, config, sampled)
        _apply_overrides(config, sampled)

        base_experiment = str(
            _get_by_dotted_path(self.base_config, "trainer.experiment_name")
            or "milk_optuna"
        )
        experiment_name = f"{base_experiment}_optuna/{trial_name}"
        _set_by_dotted_path(config, "trainer.experiment_name", experiment_name)
        _set_by_dotted_path(config, "trainer.run_name", trial_name)

        config_path = trial_dir / "config.yaml"
        summary_path = trial_dir / "summary.json"
        log_path = trial_dir / "trial.log"
        _write_yaml(config, config_path)

        trial.set_user_attr("config_path", str(config_path))
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("experiment_name", experiment_name)
        trial.set_user_attr("sampled_overrides", sampled)

        if self.dry_run:
            LOGGER.info("[DRY-RUN] Wrote sampled config to %s", config_path)
            summary_path.write_text(
                json.dumps(
                    {
                        "trial_number": trial.number,
                        "trial_name": trial_name,
                        "dry_run": True,
                        "sampled_overrides": sampled,
                        "fixed_overrides": fixed_overrides,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0.0

        cmd = [
            sys.executable,
            "-m",
            "ppl.cli.entry_point",
            "-c",
            str(config_path),
            "--log-level",
            self.log_level,
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.repo_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env.setdefault("MPLCONFIGDIR", "/tmp")

        LOGGER.info("[OPTUNA] trial=%d running %s", trial.number, " ".join(cmd))
        with log_path.open("w") as log_file:
            completed = subprocess.run(
                cmd,
                cwd=self.package_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=self.trial_timeout,
                check=False,
            )

        results_dir = self.package_root / "results" / experiment_name
        res_txt = results_dir / "res.txt"
        metrics = _parse_metrics_from_res_txt(res_txt)
        metrics.update(_parse_metrics_from_log(log_path))

        summary = {
            "trial_number": trial.number,
            "trial_name": trial_name,
            "returncode": completed.returncode,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "results_dir": str(results_dir),
            "res_txt": str(res_txt),
            "sampled_overrides": sampled,
            "fixed_overrides": fixed_overrides,
            "metrics": metrics,
        }

        if completed.returncode != 0:
            trial.set_user_attr("failed_log_path", str(log_path))
            trial.set_user_attr("failed_returncode", completed.returncode)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            log_tail = ""
            if log_path.exists():
                lines = log_path.read_text(errors="replace").splitlines()
                log_tail = "\n".join(lines[-80:])
            raise RuntimeError(
                f"Trial {trial.number} failed with return code {completed.returncode}. "
                f"See {log_path}\n--- trial.log tail ---\n{log_tail}"
            )

        _require_best_checkpoint_metrics(metrics, res_txt=res_txt)
        value = _objective_value(metrics, self.metric_name)
        trial_metric_summary = {
            "mae": _metric_value_or_none(metrics, "val_mae"),
            "rmse": _metric_value_or_none(metrics, "val_rmse"),
            "r2": _metric_value_or_none(metrics, "val_r2"),
            "best_val_loss": _metric_value_or_none(metrics, "val_loss"),
            "best_val_mae": _metric_value_or_none(metrics, "val_mae"),
            "best_val_rmse": _metric_value_or_none(metrics, "val_rmse"),
            "best_val_r2": _metric_value_or_none(metrics, "val_r2"),
            "best_val_pearson": _metric_value_or_none(metrics, "val_pearson"),
            "best_val_spearman": _metric_value_or_none(metrics, "val_spearman"),
            "best_train_loss": _metric_value_or_none(metrics, "train_loss"),
            "best_train_mae": _metric_value_or_none(metrics, "train_mae"),
            "best_train_rmse": _metric_value_or_none(metrics, "train_rmse"),
            "best_train_r2": _metric_value_or_none(metrics, "train_r2"),
            "best_train_pearson": _metric_value_or_none(metrics, "train_pearson"),
            "best_train_spearman": _metric_value_or_none(metrics, "train_spearman"),
            "best_epoch": _metric_value_or_none(metrics, "best_epoch"),
            "best_checkpoint_score": _metric_value_or_none(
                metrics,
                "best_checkpoint_score",
            ),
            "trained_epochs": _metric_value_or_none(metrics, "trained_epochs"),
            "max_epochs": _metric_value_or_none(metrics, "max_epochs"),
            "early_stopping_patience": _metric_value_or_none(
                metrics,
                "early_stopping_patience",
            ),
        }
        summary["objective_metric"] = self.metric_name
        summary["objective_value"] = value
        summary["trial_metric_summary"] = trial_metric_summary
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

        for key, metric_value in metrics.items():
            trial.set_user_attr(key, metric_value)
        for key, metric_value in trial_metric_summary.items():
            if metric_value is not None:
                trial.set_user_attr(f"trial_{key}", metric_value)

        LOGGER.info(
            "[OPTUNA] trial=%d %s=%.6f metric_source=best_checkpoint "
            "best_epoch=%s best_score=%s "
            "train_loss=%s train_mae=%s train_rmse=%s train_r2=%s "
            "train_pearson=%s train_spearman=%s "
            "val_loss=%s val_mae=%s val_rmse=%s val_r2=%s "
            "val_pearson=%s val_spearman=%s "
            "epochs=%s/%s patience=%s results=%s",
            trial.number,
            self.metric_name,
            value,
            _format_metric(trial_metric_summary["best_epoch"], decimals=0),
            _format_metric(trial_metric_summary["best_checkpoint_score"]),
            _format_metric(trial_metric_summary["best_train_loss"]),
            _format_metric(trial_metric_summary["best_train_mae"]),
            _format_metric(trial_metric_summary["best_train_rmse"]),
            _format_metric(trial_metric_summary["best_train_r2"]),
            _format_metric(trial_metric_summary["best_train_pearson"]),
            _format_metric(trial_metric_summary["best_train_spearman"]),
            _format_metric(trial_metric_summary["best_val_loss"]),
            _format_metric(trial_metric_summary["best_val_mae"]),
            _format_metric(trial_metric_summary["best_val_rmse"]),
            _format_metric(trial_metric_summary["best_val_r2"]),
            _format_metric(trial_metric_summary["best_val_pearson"]),
            _format_metric(trial_metric_summary["best_val_spearman"]),
            _format_metric(trial_metric_summary["trained_epochs"], decimals=0),
            _format_metric(trial_metric_summary["max_epochs"], decimals=0),
            _format_metric(trial_metric_summary["early_stopping_patience"], decimals=0),
            results_dir,
        )
        return value


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

    def __init__(self, search_space, config_builder, runner, *, trial_root, base_experiment) -> None:
        self.search_space = search_space
        self.config_builder = config_builder
        self.runner = runner
        self.trial_root = Path(trial_root)
        self.base_experiment = base_experiment

    def __call__(self, trial) -> tuple[float, float]:
        trial_name = _trial_name(trial.study.study_name, trial.number) if hasattr(trial, "study") \
            else f"trial_{trial.number:04d}"
        trial_dir = self.trial_root / trial_name
        experiment_name = f"{self.base_experiment}_optuna/{trial_name}"
        sampled = self.search_space.sample(trial, self.config_builder.base_config)
        config_path = self.config_builder.build(sampled, trial_dir, experiment_name, trial_name)
        results_dir = self.runner.run(config_path, experiment_name, trial_dir / "trial.log")
        metrics = RunMetrics.from_dir(results_dir)
        rmsd_top1, rmse = metrics.objectives()
        trial.set_user_attr("results_dir", str(results_dir))
        trial.set_user_attr("val_rmsd_top1", rmsd_top1)
        trial.set_user_attr("val_rmse", rmse)
        return rmsd_top1, rmse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Optuna optimization for a MILK YAML config.",
    )
    parser.add_argument(
        "--search-space",
        default="config/experiment_configs/optuna_search_space_bace809_cluster_hier_mha.yaml",
        help="Path to grouped Optuna search-space YAML.",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Override the base config path from the search-space YAML.",
    )
    parser.add_argument("--study-name", default="bace809_cluster_hier_mha_optuna")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL, e.g. sqlite:///ppl/optuna_trials/study.db.",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Alias for --study-timeout, kept for convenience.",
    )
    parser.add_argument(
        "--study-timeout",
        type=int,
        default=None,
        help="Maximum wall time for the whole Optuna study, in seconds.",
    )
    parser.add_argument(
        "--trial-timeout",
        type=int,
        default=None,
        help="Maximum wall time for one MILK trial subprocess, in seconds.",
    )
    parser.add_argument(
        "--trial-root",
        default="optuna_trials/bace809_cluster_hier_mha",
        help="Directory for sampled configs, logs, and trial summaries.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Objective metric to read from res.txt/log. Defaults to search-space objective.primary_metric.",
    )
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default=None,
        help="Study direction. Defaults to search-space objective.direction.",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Search phase: all, phase_1, or phase_2 from the search-space YAML.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level passed to each MILK trial subprocess.",
    )
    parser.add_argument(
        "--runner-log-level",
        default="INFO",
        help="Log level for this Optuna runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample/write configs but do not launch training.",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="Reuse an existing Optuna study with the same name/storage.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.runner_log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    repo_root = _repo_root()
    package_root = _package_root()
    search_space_path = _resolve_path(
        args.search_space,
        bases=[Path.cwd(), package_root, repo_root],
    )
    search_space = _load_yaml(search_space_path)

    base_config_arg = args.base_config or search_space.get("base_config")
    if not base_config_arg:
        raise ValueError("Base config must be provided by --base-config or search-space base_config")
    base_config_path = _resolve_path(
        base_config_arg,
        bases=[Path.cwd(), package_root, repo_root, search_space_path.parent],
    )

    direction = args.direction or search_space.get("objective", {}).get(
        "direction",
        "minimize",
    )
    metric_name = args.metric or search_space.get("objective", {}).get(
        "primary_metric",
        "val_loss",
    )
    trial_root = _resolve_path(
        args.trial_root,
        bases=[Path.cwd(), package_root, repo_root],
    )
    trial_root.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{trial_root / 'study.db'}"

    objective = MilkOptunaObjective(
        search_space_path=search_space_path,
        base_config_path=base_config_path,
        trial_root=trial_root,
        phase=args.phase,
        metric_name=metric_name,
        log_level=args.log_level,
        trial_timeout=args.trial_timeout,
        dry_run=args.dry_run,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=args.load_if_exists,
    )
    LOGGER.info(
        "Starting Optuna study=%s direction=%s metric=%s storage=%s phase=%s pruner=NopPruner",
        args.study_name,
        direction,
        metric_name,
        storage,
        args.phase,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.study_timeout if args.study_timeout is not None else args.timeout,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None
    ]
    if not completed_trials:
        raise RuntimeError(
            "All Optuna trials failed; no best trial can be selected. "
            f"Inspect trial logs under {trial_root}."
        )

    LOGGER.info("Best trial: %s", study.best_trial.number)
    LOGGER.info("Best value: %.6f", study.best_value)
    LOGGER.info("Best params: %s", study.best_params)

    best_path = trial_root / "best_trial.json"
    best_path.write_text(
        json.dumps(
            {
                "study_name": args.study_name,
                "direction": direction,
                "metric": metric_name,
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    LOGGER.info("Wrote %s", best_path)


if __name__ == "__main__":
    main()
