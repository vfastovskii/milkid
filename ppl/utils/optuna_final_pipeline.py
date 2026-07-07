"""Cluster entrypoint: Optuna HPO followed by final best-parameter training.

This module intentionally reuses the normal MILK CLI and Optuna runner.  The
only extra responsibility here is orchestration:

1. prepare runtime copies of the base config and search-space config,
2. run HPO with attention artifacts disabled,
3. extract the filtered best overrides,
4. write a final config with those overrides plus final-run artifact settings,
5. launch the normal MILK pipeline once more for the final model.
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


LOGGER = logging.getLogger(__name__)
METRIC_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_./:-]+)\s*:\s*(?P<value>[-+0-9.eE]+)\s*$")
CV_METRIC_RE = re.compile(
    r"CV\s+(?P<key>[A-Za-z0-9_./:-]+)\s*:\s*(?P<value>[-+0-9.eE]+)"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path_like: str | Path, *, bases: Iterable[Path]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    first_base = next(iter(bases), Path.cwd())
    return (first_base / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_by_dotted_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor: dict[str, Any] = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, dict):
            raise TypeError(f"Cannot set {dotted_path!r}: {part!r} is not a mapping")
        cursor = child
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


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else math.nan
    avg = _mean(values)
    var = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _parse_metrics_from_res_txt(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        try:
            metrics[match.group("key")] = float(match.group("value"))
        except ValueError:
            continue
    return metrics


def _parse_metrics_from_log(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(errors="replace").splitlines():
        match = CV_METRIC_RE.search(line)
        if not match:
            continue
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


def _run_and_tee(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    LOGGER.info("[CMD] cwd=%s %s", cwd, printable)
    if dry_run:
        log_path.write_text(f"[DRY-RUN] {printable}\n")
        return 0

    with log_path.open("w") as log_file:
        log_file.write(f"[CMD] cwd={cwd} {printable}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def _runtime_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp")
    return env


def _resolve_data_csv(args: argparse.Namespace, bases: list[Path]) -> Path | None:
    if args.data_csv is None:
        return None
    csv_path = Path(args.data_csv).expanduser()
    if csv_path.is_absolute():
        return csv_path.resolve()
    if args.data_dir is not None:
        return (Path(args.data_dir).expanduser() / csv_path).resolve()
    return _resolve_path(csv_path, bases=bases)


def _prepare_runtime_configs(
    *,
    search_space_path: Path,
    base_config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    base_cfg = _load_yaml(base_config_path)
    search_space = _load_yaml(search_space_path)

    data_csv = _resolve_data_csv(
        args,
        bases=[Path.cwd(), _package_root(), _repo_root(), base_config_path.parent],
    )
    if data_csv is not None:
        _set_by_dotted_path(base_cfg, "data.csv_path", str(data_csv))
        LOGGER.info("[DATA] Using CSV: %s", data_csv)
    else:
        LOGGER.info("[DATA] Using CSV from base config: %s", _get_by_dotted_path(base_cfg, "data.csv_path"))

    runtime_overrides: dict[str, Any] = {}
    if args.device is not None:
        runtime_overrides["trainer.device"] = args.device
    if args.accelerator is not None:
        runtime_overrides["trainer.accelerator"] = args.accelerator
    if args.devices is not None:
        runtime_overrides["trainer.devices"] = args.devices
    if args.precision is not None:
        runtime_overrides["trainer.precision"] = args.precision
    if args.num_workers is not None:
        runtime_overrides["data.num_workers"] = args.num_workers
    if args.pin_memory is not None:
        runtime_overrides["data.pin_memory"] = args.pin_memory

    _apply_overrides(base_cfg, runtime_overrides)

    runtime_base_path = output_dir / "runtime_base_config.yaml"
    runtime_search_path = output_dir / "runtime_search_space.yaml"

    search_space = deepcopy(search_space)
    search_space["base_config"] = str(runtime_base_path)
    _write_yaml(base_cfg, runtime_base_path)
    _write_yaml(search_space, runtime_search_path)

    return runtime_base_path, runtime_search_path, base_cfg, search_space


def _load_best_overrides(best_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    best = json.loads(best_path.read_text())
    attrs = best.get("best_user_attrs") or {}
    overrides = attrs.get("sampled_overrides")
    if overrides is None:
        overrides = best.get("best_params")
    if not isinstance(overrides, dict):
        raise ValueError(
            f"Could not find best overrides in {best_path}. Expected "
            "best_user_attrs.sampled_overrides or best_params."
        )
    return best, overrides


def _write_best_overrides(
    *,
    best_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    best, overrides = _load_best_overrides(best_path)
    payload = {
        "source_best_trial_json": str(best_path),
        "objective_metric": best.get("metric"),
        "objective_value": best.get("best_value"),
        "best_trial": best.get("best_trial"),
        "overrides": overrides,
    }
    out_path = output_dir / "best_overrides.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("[BEST] Wrote filtered best overrides: %s", out_path)
    return overrides, out_path


def _write_best_overrides_payload(
    *,
    payload: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    overrides = payload.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("Best-overrides payload must contain an overrides mapping")
    out_path = output_dir / "best_overrides.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("[BEST] Wrote selected overrides: %s", out_path)
    return overrides, out_path


def _phase_storage(args: argparse.Namespace, phase_root: Path) -> str:
    if args.storage:
        return args.storage
    return f"sqlite:///{phase_root / 'study.db'}"


def _with_phase_suffix(study_name: str, phase_name: str) -> str:
    if study_name.endswith(f"_{phase_name}"):
        return study_name
    return f"{study_name}_{phase_name}"


def _write_runtime_config_pair(
    *,
    base_cfg: dict[str, Any],
    search_space: dict[str, Any],
    output_dir: Path,
    phase_name: str,
) -> tuple[Path, Path]:
    base_path = output_dir / f"runtime_base_config_{phase_name}.yaml"
    search_path = output_dir / f"runtime_search_space_{phase_name}.yaml"
    phase_search = deepcopy(search_space)
    phase_search["base_config"] = str(base_path)
    _write_yaml(base_cfg, base_path)
    _write_yaml(phase_search, search_path)
    return base_path, search_path


def _run_optuna_phase(
    *,
    args: argparse.Namespace,
    phase_name: str,
    n_trials: int,
    base_config_path: Path,
    search_space_path: Path,
    trial_root: Path,
    output_dir: Path,
    package_root: Path,
    env: dict[str, str],
) -> tuple[Path, str, str, Path]:
    trial_root.mkdir(parents=True, exist_ok=True)
    study_name = _with_phase_suffix(args.study_name, phase_name)
    storage = _phase_storage(args, trial_root)
    hpo_cmd = [
        sys.executable,
        "-m",
        "ppl.utils.optuna_runner",
        "--search-space",
        str(search_space_path),
        "--base-config",
        str(base_config_path),
        "--study-name",
        study_name,
        "--storage",
        storage,
        "--trial-root",
        str(trial_root),
        "--n-trials",
        str(n_trials),
        "--phase",
        phase_name,
        "--log-level",
        args.log_level,
        "--runner-log-level",
        args.runner_log_level,
    ]
    if args.metric:
        hpo_cmd.extend(["--metric", args.metric])
    if args.direction:
        hpo_cmd.extend(["--direction", args.direction])
    if args.study_timeout is not None:
        hpo_cmd.extend(["--study-timeout", str(args.study_timeout)])
    if args.trial_timeout is not None:
        hpo_cmd.extend(["--trial-timeout", str(args.trial_timeout)])
    if args.load_if_exists:
        hpo_cmd.append("--load-if-exists")
    if args.dry_run:
        hpo_cmd.append("--dry-run")

    hpo_rc = _run_and_tee(
        hpo_cmd,
        cwd=package_root,
        env=env,
        log_path=output_dir / f"optuna_{phase_name}.log",
        dry_run=False,
    )
    if hpo_rc != 0:
        raise SystemExit(f"Optuna {phase_name} failed with exit code {hpo_rc}")

    return trial_root / "best_trial.json", storage, study_name, trial_root


def _load_top_trial_overrides(
    *,
    study_name: str,
    storage: str,
    top_k: int,
    direction: str,
) -> list[dict[str, Any]]:
    study = optuna.load_study(study_name=study_name, storage=storage)
    complete = [
        trial
        for trial in study.trials
        if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    reverse = direction == "maximize"
    complete.sort(key=lambda trial: float(trial.value), reverse=reverse)

    selected: list[dict[str, Any]] = []
    for rank, trial in enumerate(complete[: max(1, top_k)], start=1):
        overrides = trial.user_attrs.get("sampled_overrides")
        if overrides is None:
            overrides = trial.params
        if not isinstance(overrides, dict):
            continue
        selected.append(
            {
                "rank": rank,
                "trial_number": trial.number,
                "trial_value": float(trial.value),
                "trial_user_attrs": dict(trial.user_attrs),
                "overrides": dict(overrides),
            }
        )
    if not selected:
        raise ValueError(f"No complete trials found in study {study_name!r}")
    return selected


def _confirmation_experiment_name(
    base_cfg: dict[str, Any],
    run_index: int,
    config_rank: int,
    seed: int,
) -> str:
    base_name = _get_by_dotted_path(base_cfg, "trainer.experiment_name")
    if not base_name:
        base_name = "milk_optuna"
    return (
        f"{base_name}_phase3_confirmation/"
        f"rank{config_rank:02d}_seed{seed}_run{run_index:04d}"
    )


def _confirmation_log_save_dir(
    base_cfg: dict[str, Any],
    run_index: int,
    config_rank: int,
    seed: int,
) -> str:
    base_dir = _get_by_dotted_path(base_cfg, "trainer.log_save_dir")
    if not base_dir:
        base_dir = "milk_optuna_log"
    return (
        f"{base_dir}_phase3_confirmation/"
        f"rank{config_rank:02d}_seed{seed}_run{run_index:04d}"
    )


def _run_phase3_confirmation(
    *,
    args: argparse.Namespace,
    runtime_base_cfg: dict[str, Any],
    phase1_overrides: dict[str, Any],
    phase2_top_trials: list[dict[str, Any]],
    output_dir: Path,
    package_root: Path,
    env: dict[str, str],
    metric_name: str,
    direction: str,
) -> dict[str, Any]:
    confirmation_dir = output_dir / "phase3_confirmation"
    confirmation_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    grouped_values: dict[int, list[float]] = {
        int(candidate["rank"]): [] for candidate in phase2_top_trials
    }
    for run_index in range(int(args.phase3_reruns)):
        candidate = phase2_top_trials[run_index % len(phase2_top_trials)]
        config_rank = int(candidate["rank"])
        seed = int(args.phase3_seed_start) + run_index

        combined_overrides = dict(phase1_overrides)
        combined_overrides.update(dict(candidate["overrides"]))

        config = deepcopy(runtime_base_cfg)
        _apply_overrides(config, combined_overrides)
        confirmation_overrides = {
            "data.seed": seed,
            "trainer.experiment_name": _confirmation_experiment_name(
                runtime_base_cfg,
                run_index,
                config_rank,
                seed,
            ),
            "trainer.log_save_dir": _confirmation_log_save_dir(
                runtime_base_cfg,
                run_index,
                config_rank,
                seed,
            ),
            "trainer.run_name": f"phase3_rank{config_rank:02d}_seed{seed}",
            "trainer.stage_2_launch": False,
            "trainer.log_per_epoch": False,
            "trainer.save_attention_artifacts": False,
            "trainer.validation_instance_importance_enabled": False,
            "trainer.enable_progress_bar": False,
            "trainer.enable_model_summary": False,
            "model.optim.enable_gradient_tracking": False,
        }
        _apply_overrides(config, confirmation_overrides)

        run_dir = confirmation_dir / f"run_{run_index:04d}_rank{config_rank:02d}_seed{seed}"
        config_path = run_dir / "config.yaml"
        log_path = run_dir / "run.log"
        summary_path = run_dir / "summary.json"
        _write_yaml(config, config_path)

        cmd = [
            sys.executable,
            "-m",
            "ppl.cli_entry_point",
            "-c",
            str(config_path),
            "--log-level",
            args.log_level,
        ]
        returncode = _run_and_tee(
            cmd,
            cwd=package_root,
            env=env,
            log_path=log_path,
            dry_run=args.dry_run,
        )
        results_dir = package_root / "results" / confirmation_overrides["trainer.experiment_name"]
        metrics = _parse_metrics_from_res_txt(results_dir / "res.txt")
        metrics.update(_parse_metrics_from_log(log_path))

        if args.dry_run:
            objective = float(config_rank)
        else:
            if returncode != 0:
                raise SystemExit(
                    f"Phase 3 confirmation run {run_index} failed with exit code "
                    f"{returncode}; see {log_path}"
                )
            objective = _objective_value(metrics, metric_name)
        grouped_values[config_rank].append(objective)

        run_summary = {
            "run_index": run_index,
            "config_rank": config_rank,
            "phase2_trial_number": candidate["trial_number"],
            "phase2_trial_value": candidate["trial_value"],
            "seed": seed,
            "returncode": returncode,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "results_dir": str(results_dir),
            "metric": metric_name,
            "objective_value": objective,
            "metrics": metrics,
            "combined_overrides": combined_overrides,
        }
        summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True))
        runs.append(run_summary)

    per_config: list[dict[str, Any]] = []
    candidates_by_rank = {int(candidate["rank"]): candidate for candidate in phase2_top_trials}
    for rank, values in sorted(grouped_values.items()):
        if not values:
            continue
        candidate = candidates_by_rank[rank]
        combined_overrides = dict(phase1_overrides)
        combined_overrides.update(dict(candidate["overrides"]))
        per_config.append(
            {
                "rank": rank,
                "phase2_trial_number": candidate["trial_number"],
                "phase2_trial_value": candidate["trial_value"],
                "n_reruns": len(values),
                "mean": _mean(values),
                "std": _std(values),
                "values": values,
                "overrides": combined_overrides,
            }
        )

    reverse = direction == "maximize"
    per_config.sort(key=lambda item: float(item["mean"]), reverse=reverse)
    best_config = per_config[0]
    summary = {
        "metric": metric_name,
        "direction": direction,
        "requested_reruns": int(args.phase3_reruns),
        "top_k": len(phase2_top_trials),
        "best_config": best_config,
        "per_config": per_config,
        "runs": runs,
    }
    (confirmation_dir / "phase3_confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    LOGGER.info(
        "[PHASE3] best rank=%s phase2_trial=%s mean_%s=%.6f std=%.6f n=%d",
        best_config["rank"],
        best_config["phase2_trial_number"],
        metric_name,
        best_config["mean"],
        best_config["std"],
        best_config["n_reruns"],
    )
    return summary


def _final_experiment_name(base_cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.final_experiment_name:
        return args.final_experiment_name
    base_name = _get_by_dotted_path(base_cfg, "trainer.experiment_name")
    if not base_name:
        base_name = "milk_optuna"
    return f"{base_name}_final_best"


def _final_log_save_dir(base_cfg: dict[str, Any], args: argparse.Namespace) -> str:
    if args.final_log_save_dir:
        return args.final_log_save_dir
    base_dir = _get_by_dotted_path(base_cfg, "trainer.log_save_dir")
    if not base_dir:
        base_dir = "milk_optuna_log"
    return f"{base_dir}_final_best"


def _write_final_config(
    *,
    runtime_base_cfg: dict[str, Any],
    best_overrides: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    final_cfg = deepcopy(runtime_base_cfg)
    _apply_overrides(final_cfg, best_overrides)

    final_overrides = {
        "trainer.experiment_name": _final_experiment_name(runtime_base_cfg, args),
        "trainer.log_save_dir": _final_log_save_dir(runtime_base_cfg, args),
        "trainer.run_name": args.final_run_name,
        "trainer.stage_2_launch": bool(args.final_stage2),
        "trainer.log_per_epoch": bool(args.final_log_per_epoch),
        "trainer.save_attention_artifacts": bool(args.save_final_attention),
        "trainer.validation_instance_importance_enabled": bool(
            args.final_validation_instance_importance
        ),
        "trainer.enable_progress_bar": bool(args.final_progress_bar),
        "trainer.enable_model_summary": bool(args.final_model_summary),
    }
    _apply_overrides(final_cfg, final_overrides)

    final_config_path = output_dir / "final_config.yaml"
    _write_yaml(final_cfg, final_config_path)
    LOGGER.info("[FINAL] Wrote final optimized config: %s", final_config_path)
    LOGGER.info(
        "[FINAL] experiment_name=%s log_save_dir=%s stage_2_launch=%s "
        "save_attention_artifacts=%s validation_instance_importance=%s",
        final_overrides["trainer.experiment_name"],
        final_overrides["trainer.log_save_dir"],
        final_overrides["trainer.stage_2_launch"],
        final_overrides["trainer.save_attention_artifacts"],
        final_overrides["trainer.validation_instance_importance_enabled"],
    )
    return final_config_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MILK Optuna HPO and train the final best-parameter model.",
    )
    parser.add_argument(
        "--search-space",
        default="ppl/utils/experiment_configs/optuna_search_space_bace809_cluster_hier_mha.yaml",
    )
    parser.add_argument(
        "--base-config",
        default="ppl/utils/experiment_configs/nos_regression_experiment_config.yaml",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-csv", default=None)
    parser.add_argument("--output-dir", default="optuna_final_runs/latest")
    parser.add_argument("--trial-root", default=None)
    parser.add_argument("--study-name", default="bace809_cluster_hier_mha_optuna")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--study-timeout", type=int, default=None)
    parser.add_argument("--trial-timeout", type=int, default=None)
    parser.add_argument("--phase", default="all")
    parser.add_argument(
        "--three-phase",
        action="store_true",
        help=(
            "Run Phase 1 architecture/optimizer HPO, Phase 2 active-prototype "
            "HPO around the best Phase 1 config, then Phase 3 seed confirmation."
        ),
    )
    parser.add_argument("--phase1-trials", type=int, default=400)
    parser.add_argument("--phase2-trials", type=int, default=400)
    parser.add_argument("--phase3-reruns", type=int, default=100)
    parser.add_argument(
        "--phase3-top-k",
        type=int,
        default=5,
        help="Number of best Phase 2 configs to confirm with repeated seeds.",
    )
    parser.add_argument("--phase3-seed-start", type=int, default=1000)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--direction", choices=["minimize", "maximize"], default=None)
    parser.add_argument("--load-if-exists", action="store_true")
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--hpo-only", action="store_true")
    parser.add_argument("--best-trial-json", default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--device", default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true", dest="pin_memory")
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.set_defaults(pin_memory=None)

    parser.add_argument("--final-experiment-name", default=None)
    parser.add_argument("--final-log-save-dir", default=None)
    parser.add_argument("--final-run-name", default="final_best")
    parser.add_argument("--final-stage2", action="store_true", dest="final_stage2")
    parser.add_argument("--no-final-stage2", action="store_false", dest="final_stage2")
    parser.set_defaults(final_stage2=True)
    parser.add_argument("--final-log-per-epoch", action="store_true")
    parser.add_argument("--save-final-attention", action="store_true", dest="save_final_attention")
    parser.add_argument("--no-save-final-attention", action="store_false", dest="save_final_attention")
    parser.set_defaults(save_final_attention=True)
    parser.add_argument(
        "--final-validation-instance-importance",
        action="store_true",
        dest="final_validation_instance_importance",
    )
    parser.add_argument(
        "--no-final-validation-instance-importance",
        action="store_false",
        dest="final_validation_instance_importance",
    )
    parser.set_defaults(final_validation_instance_importance=False)
    parser.add_argument("--final-progress-bar", action="store_true")
    parser.add_argument("--final-model-summary", action="store_true")

    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--runner-log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.runner_log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    repo_root = _repo_root()
    package_root = _package_root()
    bases = [Path.cwd(), package_root, repo_root]
    output_dir = _resolve_path(args.output_dir, bases=bases)
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space_path = _resolve_path(args.search_space, bases=bases)
    base_config_path = _resolve_path(args.base_config, bases=bases)

    runtime_base_path, runtime_search_path, runtime_base_cfg, runtime_search_space = _prepare_runtime_configs(
        search_space_path=search_space_path,
        base_config_path=base_config_path,
        output_dir=output_dir,
        args=args,
    )

    trial_root = (
        _resolve_path(args.trial_root, bases=bases)
        if args.trial_root
        else output_dir / "trials"
    )
    trial_root.mkdir(parents=True, exist_ok=True)
    env = _runtime_env(repo_root)

    best_path = (
        _resolve_path(args.best_trial_json, bases=bases)
        if args.best_trial_json
        else trial_root / "best_trial.json"
    )
    selected_best_trial_json = best_path
    metric_name = args.metric or runtime_search_space.get("objective", {}).get(
        "primary_metric",
        "val_mae",
    )
    direction = args.direction or runtime_search_space.get("objective", {}).get(
        "direction",
        "minimize",
    )

    best_overrides_path: Path
    if args.three_phase and not args.skip_hpo:
        LOGGER.info(
            "[THREE-PHASE] phase1_trials=%d phase2_trials=%d "
            "phase3_reruns=%d phase3_top_k=%d metric=%s",
            args.phase1_trials,
            args.phase2_trials,
            args.phase3_reruns,
            args.phase3_top_k,
            metric_name,
        )

        phase1_root = trial_root / "phase_1"
        phase1_best_path, _, _, _ = _run_optuna_phase(
            args=args,
            phase_name="phase_1",
            n_trials=int(args.phase1_trials),
            base_config_path=runtime_base_path,
            search_space_path=runtime_search_path,
            trial_root=phase1_root,
            output_dir=output_dir,
            package_root=package_root,
            env=env,
        )
        phase1_best, phase1_overrides = _load_best_overrides(phase1_best_path)
        LOGGER.info(
            "[PHASE1] best_trial=%s %s=%s",
            phase1_best.get("best_trial"),
            metric_name,
            phase1_best.get("best_value"),
        )

        phase2_base_cfg = deepcopy(runtime_base_cfg)
        _apply_overrides(phase2_base_cfg, phase1_overrides)
        phase2_base_path, phase2_search_path = _write_runtime_config_pair(
            base_cfg=phase2_base_cfg,
            search_space=runtime_search_space,
            output_dir=output_dir,
            phase_name="phase_2",
        )
        phase2_root = trial_root / "phase_2"
        phase2_best_path, phase2_storage, phase2_study_name, _ = _run_optuna_phase(
            args=args,
            phase_name="phase_2",
            n_trials=int(args.phase2_trials),
            base_config_path=phase2_base_path,
            search_space_path=phase2_search_path,
            trial_root=phase2_root,
            output_dir=output_dir,
            package_root=package_root,
            env=env,
        )
        selected_best_trial_json = phase2_best_path
        phase2_best, phase2_overrides = _load_best_overrides(phase2_best_path)
        LOGGER.info(
            "[PHASE2] best_trial=%s %s=%s",
            phase2_best.get("best_trial"),
            metric_name,
            phase2_best.get("best_value"),
        )

        phase2_top_trials = _load_top_trial_overrides(
            study_name=phase2_study_name,
            storage=phase2_storage,
            top_k=int(args.phase3_top_k),
            direction=str(direction),
        )
        if int(args.phase3_reruns) > 0:
            phase3_summary = _run_phase3_confirmation(
                args=args,
                runtime_base_cfg=runtime_base_cfg,
                phase1_overrides=phase1_overrides,
                phase2_top_trials=phase2_top_trials,
                output_dir=output_dir,
                package_root=package_root,
                env=env,
                metric_name=metric_name,
                direction=str(direction),
            )
            best_payload = {
                "selection_mode": "three_phase_confirmed",
                "objective_metric": metric_name,
                "objective_direction": direction,
                "phase1_best_trial_json": str(phase1_best_path),
                "phase2_best_trial_json": str(phase2_best_path),
                "phase3_confirmation_summary": str(
                    output_dir
                    / "phase3_confirmation"
                    / "phase3_confirmation_summary.json"
                ),
                "phase1_best_trial": phase1_best.get("best_trial"),
                "phase1_best_value": phase1_best.get("best_value"),
                "phase2_best_trial": phase2_best.get("best_trial"),
                "phase2_best_value": phase2_best.get("best_value"),
                "confirmed_best": phase3_summary["best_config"],
                "overrides": phase3_summary["best_config"]["overrides"],
            }
        else:
            combined_overrides = dict(phase1_overrides)
            combined_overrides.update(phase2_overrides)
            best_payload = {
                "selection_mode": "two_phase_no_confirmation",
                "objective_metric": metric_name,
                "objective_direction": direction,
                "phase1_best_trial_json": str(phase1_best_path),
                "phase2_best_trial_json": str(phase2_best_path),
                "phase1_best_trial": phase1_best.get("best_trial"),
                "phase1_best_value": phase1_best.get("best_value"),
                "phase2_best_trial": phase2_best.get("best_trial"),
                "phase2_best_value": phase2_best.get("best_value"),
                "overrides": combined_overrides,
            }
        best_overrides, best_overrides_path = _write_best_overrides_payload(
            payload=best_payload,
            output_dir=output_dir,
        )
    elif not args.skip_hpo:
        hpo_cmd = [
            sys.executable,
            "-m",
            "ppl.utils.optuna_runner",
            "--search-space",
            str(runtime_search_path),
            "--base-config",
            str(runtime_base_path),
            "--study-name",
            args.study_name,
            "--trial-root",
            str(trial_root),
            "--n-trials",
            str(args.n_trials),
            "--phase",
            args.phase,
            "--log-level",
            args.log_level,
            "--runner-log-level",
            args.runner_log_level,
        ]
        if args.metric:
            hpo_cmd.extend(["--metric", args.metric])
        if args.direction:
            hpo_cmd.extend(["--direction", args.direction])
        if args.storage:
            hpo_cmd.extend(["--storage", args.storage])
        if args.study_timeout is not None:
            hpo_cmd.extend(["--study-timeout", str(args.study_timeout)])
        if args.trial_timeout is not None:
            hpo_cmd.extend(["--trial-timeout", str(args.trial_timeout)])
        if args.load_if_exists:
            hpo_cmd.append("--load-if-exists")
        if args.dry_run:
            hpo_cmd.append("--dry-run")

        hpo_rc = _run_and_tee(
            hpo_cmd,
            cwd=package_root,
            env=env,
            log_path=output_dir / "optuna_hpo.log",
            dry_run=False,
        )
        if hpo_rc != 0:
            raise SystemExit(f"Optuna HPO failed with exit code {hpo_rc}")

        if not best_path.exists():
            raise FileNotFoundError(
                f"Missing best trial JSON after HPO: {best_path}"
            )
        best_overrides, best_overrides_path = _write_best_overrides(
            best_path=best_path,
            output_dir=output_dir,
        )
    else:
        if not best_path.exists():
            raise FileNotFoundError(
                f"Missing best trial JSON: {best_path}. Run HPO first or pass --best-trial-json."
            )
        best_overrides, best_overrides_path = _write_best_overrides(
            best_path=best_path,
            output_dir=output_dir,
        )

    if args.hpo_only:
        LOGGER.info("[DONE] HPO-only mode; skipping final training")
        return
    final_config_path = _write_final_config(
        runtime_base_cfg=runtime_base_cfg,
        best_overrides=best_overrides,
        output_dir=output_dir,
        args=args,
    )

    final_cmd = [
        sys.executable,
        "-m",
        "ppl.cli_entry_point",
        "-c",
        str(final_config_path),
        "--log-level",
        args.log_level,
    ]
    final_rc = _run_and_tee(
        final_cmd,
        cwd=package_root,
        env=env,
        log_path=output_dir / "final_run.log",
        dry_run=args.dry_run,
    )
    summary = {
        "runtime_base_config": str(runtime_base_path),
        "runtime_search_space": str(runtime_search_path),
        "best_trial_json": str(selected_best_trial_json),
        "best_overrides_json": str(best_overrides_path),
        "final_config": str(final_config_path),
        "final_log": str(output_dir / "final_run.log"),
        "final_returncode": final_rc,
        "final_experiment_name": _final_experiment_name(runtime_base_cfg, args),
        "final_results_dir": str(
            package_root / "results" / _final_experiment_name(runtime_base_cfg, args)
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    if final_rc != 0:
        raise SystemExit(f"Final training failed with exit code {final_rc}")
    LOGGER.info("[DONE] Final optimized model run complete")


if __name__ == "__main__":
    main()
