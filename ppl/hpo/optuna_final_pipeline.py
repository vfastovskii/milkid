"""Cluster entrypoint: multi-objective Optuna HPO -> select best-KID config -> retrain.

Lean two-step driver around the multi-objective runner (``ppl.hpo.optuna_runner``):

  1. Run the runner (unless ``--skip-hpo``): it optimizes the Pareto front of
     (maximize ``val_rmsd_top1``, minimize ``val_rmse``) and writes
     ``<trial-root>/pareto_front.json`` plus per-trial ``config.yaml`` files.
  2. Select the non-dominated trial with the highest ``val_rmsd_top1`` (ties broken
     by lower ``val_rmse``), then retrain that trial's exact config over one or more
     seeds (a stability check) and record each run's metrics.

All metrics are read from each run's structured ``run_metrics.json`` — no text parsing.
The final-model choice is deliberately "best rmsd_top1 on the front"; change the
selection in ``_select_best_from_pareto`` if a different trade-off is wanted.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from ppl.hpo.optuna_runner import (
    _apply_overrides,
    _load_yaml,
    _package_root,
    _repo_root,
    _resolve_path,
    _set_by_dotted_path,
    _trial_name,
    _write_yaml,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# small numeric + process helpers
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _runtime_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp")
    return env


def _run_and_tee(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    """Run a subprocess, streaming combined stdout/stderr to console and a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(cmd)
    LOGGER.info("[CMD] cwd=%s %s", cwd, printable)
    with log_path.open("w") as log_file:
        log_file.write(f"[CMD] cwd={cwd} {printable}\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        return proc.wait()


def _read_run_metrics(results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / "run_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# HPO + selection
# --------------------------------------------------------------------------- #
def _run_hpo(
    *,
    search_space_path: Path,
    base_config_path: Path,
    study_name: str,
    storage: str,
    n_trials: int,
    n_jobs: int,
    trial_root: Path,
    log_level: str,
    env: dict[str, str],
    package_root: Path,
    log_path: Path,
    dry_run: bool,
) -> int:
    cmd = [
        sys.executable, "-m", "ppl.hpo.optuna_runner",
        "--search-space", str(search_space_path),
        "--base-config", str(base_config_path),
        "--study-name", study_name,
        "--storage", storage,
        "--trial-root", str(trial_root),
        "--n-trials", str(n_trials),
        "--n-jobs", str(n_jobs),
        "--log-level", log_level,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return _run_and_tee(cmd, cwd=package_root, env=env, log_path=log_path)


def _select_best_from_pareto(pareto_path: Path) -> dict[str, Any]:
    """Pick the Pareto-front trial with the highest val_rmsd_top1 (ties -> lower val_rmse)."""
    if not pareto_path.exists():
        raise SystemExit(f"pareto_front.json not found: {pareto_path} (did the HPO run?)")
    front = json.loads(pareto_path.read_text())
    usable = [
        t for t in front
        if isinstance(t.get("val_rmsd_top1"), (int, float))
        and isinstance(t.get("val_rmse"), (int, float))
    ]
    if not usable:
        raise SystemExit(f"No usable trials with both objectives in {pareto_path}")
    return max(usable, key=lambda t: (t["val_rmsd_top1"], -t["val_rmse"]))


def _selected_trial_config(trial_root: Path, study_name: str, trial_number: int) -> Path:
    """Path to the exact config.yaml the selected trial was built and run with."""
    path = trial_root / _trial_name(study_name, trial_number) / "config.yaml"
    if not path.exists():
        raise SystemExit(
            f"Selected trial config not found: {path}. The runner writes each trial's "
            "config under <trial-root>/<trial-name>/config.yaml; ensure --trial-root "
            "matches the HPO run."
        )
    return path


# --------------------------------------------------------------------------- #
# retrain / confirmation
# --------------------------------------------------------------------------- #
def _retrain_over_seeds(
    *,
    selected_cfg: dict[str, Any],
    seeds: list[int],
    final_base_name: str,
    output_dir: Path,
    package_root: Path,
    env: dict[str, str],
    log_level: str,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        experiment_name = f"{final_base_name}/seed{seed}"
        cfg = deepcopy(selected_cfg)
        _set_by_dotted_path(cfg, "data.seed", seed)
        _set_by_dotted_path(cfg, "trainer.experiment_name", experiment_name)
        _set_by_dotted_path(cfg, "trainer.run_name", f"final_seed{seed}")
        # The final model is the one we keep: save its attention weights + per-epoch logs.
        # HPO trials leave these OFF for speed (search-space fixed_overrides).
        _set_by_dotted_path(cfg, "trainer.save_attention_artifacts", True)
        _set_by_dotted_path(cfg, "trainer.log_per_epoch", True)
        _set_by_dotted_path(cfg, "trainer.enable_progress_bar", True)   # per-epoch bar for the final model
        cfg_path = output_dir / "configs" / f"final_seed{seed}.yaml"
        _write_yaml(cfg, cfg_path)

        rc = _run_and_tee(
            [sys.executable, "-m", "ppl.cli.entry_point", "-c", str(cfg_path),
             "--log-level", log_level],
            cwd=package_root, env=env, log_path=output_dir / f"final_seed{seed}.log",
        )
        metrics = _read_run_metrics(package_root / "results" / experiment_name) if rc == 0 else None
        runs.append({
            "seed": seed,
            "experiment_name": experiment_name,
            "returncode": rc,
            "metrics": metrics,
        })
        if rc != 0:
            LOGGER.warning("[FINAL] seed %d exited %d — see final_seed%d.log", seed, rc, seed)
        elif metrics is None:
            LOGGER.warning("[FINAL] seed %d produced no run_metrics.json", seed)
    return runs


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def col(key: str) -> list[float]:
        return [
            r["metrics"][key] for r in runs
            if r["metrics"] and isinstance(r["metrics"].get(key), (int, float))
        ]
    rmsd, rmse = col("val_rmsd_top1"), col("val_rmse")
    return {
        "n_runs": len(runs),
        "n_with_metrics": len(rmsd),
        "val_rmsd_top1_mean": _mean(rmsd), "val_rmsd_top1_std": _std(rmsd),
        "val_rmse_mean": _mean(rmse), "val_rmse_std": _std(rmse),
    }


# --------------------------------------------------------------------------- #
# runtime config prep (device/data overrides) + CLI
# --------------------------------------------------------------------------- #
def _prepare_runtime_configs(
    *, search_space_path: Path, base_config_path: Path, output_dir: Path, args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Bake CLI runtime overrides (device/data/etc.) into copies of the base config and
    search space so the HPO trials + final retrain all use them. Returns the copy paths."""
    base_cfg = _load_yaml(base_config_path)
    search_space = deepcopy(_load_yaml(search_space_path))

    overrides: dict[str, Any] = {}
    if args.data_csv is not None:
        overrides["data.csv_path"] = str(Path(args.data_csv).expanduser())
    for cli_val, dotted in [
        (args.device, "trainer.device"), (args.accelerator, "trainer.accelerator"),
        (args.devices, "trainer.devices"), (args.precision, "trainer.precision"),
        (args.num_workers, "data.num_workers"),
    ]:
        if cli_val is not None:
            overrides[dotted] = cli_val
    _apply_overrides(base_cfg, overrides)

    runtime_base_path = output_dir / "runtime_base_config.yaml"
    runtime_search_path = output_dir / "runtime_search_space.yaml"
    search_space["base_config"] = str(runtime_base_path)
    _write_yaml(base_cfg, runtime_base_path)
    _write_yaml(search_space, runtime_search_path)
    return runtime_base_path, runtime_search_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-objective Optuna HPO + best-rmsd_top1 selection + final retrain.",
    )
    p.add_argument("--search-space", required=True)
    p.add_argument("--base-config", default=None, help="Overrides base_config from the search space.")
    p.add_argument("--study-name", default="milk_hpo")
    p.add_argument("--storage", default=None, help="Optuna storage URL; default sqlite in trial-root.")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel HPO trials (each a training subprocess); on one GPU keep small (2-3).")
    p.add_argument("--trial-root", default=None, help="Where trials + pareto_front.json land.")
    p.add_argument("--output-dir", default=None, help="Where this pipeline writes its artifacts.")
    p.add_argument("--confirmation-seeds", type=int, nargs="+", default=[42],
                   help="Retrain the selected config once per seed (stability check).")
    p.add_argument("--skip-hpo", action="store_true", help="Reuse an existing pareto_front.json.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--dry-run", action="store_true", help="Dry-run the HPO (write configs, no training).")
    # runtime overrides baked into every run
    p.add_argument("--data-csv", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--accelerator", default=None)
    p.add_argument("--devices", default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args(argv)

    package_root, repo_root = _package_root(), _repo_root()
    env = _runtime_env(repo_root)
    bases = [Path.cwd(), package_root, repo_root]

    search_space_path = _resolve_path(args.search_space, bases=bases)
    base_config = args.base_config or _load_yaml(search_space_path).get("base_config")
    if not base_config:
        raise SystemExit("No base_config (pass --base-config or set it in the search space).")
    base_config_path = _resolve_path(base_config, bases=bases)

    # Resolve to ABSOLUTE paths: the runner + training run in subprocesses whose cwd is the
    # package root (<repo>/ppl), so a relative --output-dir/--trial-root would fork into two
    # different roots (pipeline writes results/, runner writes ppl/results/) and the pipeline
    # would then fail to find pareto_front.json. Absolute paths keep everything in one place.
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else package_root / "results" / f"{args.study_name}_final_pipeline"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_root = (Path(args.trial_root) if args.trial_root else output_dir / "optuna_trials").resolve()
    storage = args.storage or f"sqlite:///{trial_root / 'study.db'}"

    runtime_base_path, runtime_search_path = _prepare_runtime_configs(
        search_space_path=search_space_path, base_config_path=base_config_path,
        output_dir=output_dir, args=args,
    )

    # 1. HPO
    if not args.skip_hpo:
        rc = _run_hpo(
            search_space_path=runtime_search_path, base_config_path=runtime_base_path,
            study_name=args.study_name, storage=storage, n_trials=args.n_trials, n_jobs=args.n_jobs,
            trial_root=trial_root, log_level=args.log_level, env=env,
            package_root=package_root, log_path=output_dir / "optuna.log", dry_run=args.dry_run,
        )
        if rc != 0:
            raise SystemExit(f"HPO failed with exit code {rc}")
    if args.dry_run:
        LOGGER.info("[DRY-RUN] HPO dry-run complete; skipping selection + final retrain.")
        return 0

    # 2. select best trial from the Pareto front
    best = _select_best_from_pareto(trial_root / "pareto_front.json")
    LOGGER.info(
        "[SELECT] trial #%s: val_rmsd_top1=%.4f val_rmse=%.4f (highest rmsd_top1 on front)",
        best["number"], best["val_rmsd_top1"], best["val_rmse"],
    )
    selected_cfg = _load_yaml(_selected_trial_config(trial_root, args.study_name, best["number"]))

    # 3. retrain the selected config over the confirmation seeds. Final models get their
    # own subdir of the study's pipeline output (absolute, so create_results_directory
    # resolves straight there — same unified layout as the trials).
    final_base = str((output_dir / "final").resolve())
    runs = _retrain_over_seeds(
        selected_cfg=selected_cfg, seeds=args.confirmation_seeds, final_base_name=final_base,
        output_dir=output_dir, package_root=package_root, env=env, log_level=args.log_level,
    )
    stats = _summarize_runs(runs)

    summary = {
        "study_name": args.study_name,
        "pareto_front": str(trial_root / "pareto_front.json"),
        "selected_trial": best,
        "final_runs": runs,
        "final_stats": stats,
    }
    (output_dir / "final_pipeline_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOGGER.info(
        "[DONE] %d final run(s); val_rmsd_top1 %.4f±%.4f, val_rmse %.4f±%.4f -> %s",
        stats["n_runs"], stats["val_rmsd_top1_mean"], stats["val_rmsd_top1_std"],
        stats["val_rmse_mean"], stats["val_rmse_std"], output_dir / "final_pipeline_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
