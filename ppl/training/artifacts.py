"""Post-training artifact export: prediction CSVs, res.txt summary, test evaluation.

These operate on a ``ModelTrainer`` instance (``mt``) after ``trainer.fit`` and
reuse its checkpoint/metric helpers. Kept out of ``ModelTrainer`` to keep that
class focused on building and fitting.
"""
from __future__ import annotations

import logging

import pandas as pd

from ppl.pipeline.mlflow_utils import log_metrics
from ppl.pipeline.results_directory import create_results_directory
from ppl.training.predictions import predict_rows, regression_metrics

LOGGER = logging.getLogger(__name__)


def export_fit_artifacts(mt, dm, model, best_model, val_metrics) -> None:
    """Write train_fit.csv, val.csv, and res.txt for the best/in-memory model."""
    trainer = mt._last_trainer
    try:
        results_dir = create_results_directory(mt.experiment_name)

        plot_model = best_model if best_model is not None else model
        if hasattr(dm, "train_full_dataloader"):
            train_dl = dm.train_full_dataloader()
        else:
            train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()

        eval_epoch = getattr(
            plot_model,
            "_evaluation_epoch_override",
            getattr(plot_model, "current_epoch", None),
        )

        # Write train_fit.csv
        train_rows = []
        try:
            train_rows = predict_rows(
                plot_model, train_dl, mt.task, stage="val",
                eval_epoch=eval_epoch, use_series_labels=True,
            )
            if train_rows:
                train_df = pd.DataFrame(train_rows, columns=["mol_id", "true", "predicted"])
                train_df["true"] = train_df["true"].round(2)
                train_df["predicted"] = train_df["predicted"].round(2)
                train_df["abs_error"] = (train_df["true"] - train_df["predicted"]).abs().round(2)
                train_df.to_csv(results_dir / "train_fit.csv", index=False)
                LOGGER.info(f"[MODEL] Saved train predictions to {results_dir / 'train_fit.csv'}")
            else:
                LOGGER.info("[MODEL] No train rows collected for train_fit.csv in fit_validate")
        except Exception as e:
            LOGGER.warning(f"[MODEL] Failed to write train_fit.csv in fit_validate: {e}")

        # Write val.csv
        try:
            val_rows = predict_rows(
                plot_model, val_dl, mt.task, stage="val",
                eval_epoch=eval_epoch, use_series_labels=True,
            )
            if val_rows:
                val_df = pd.DataFrame(val_rows, columns=["mol_id", "true", "predicted"])
                val_df["true"] = val_df["true"].round(2)
                val_df["predicted"] = val_df["predicted"].round(2)
                val_df["abs_error"] = (val_df["true"] - val_df["predicted"]).abs().round(2)
                val_df.to_csv(results_dir / "val.csv", index=False)
                LOGGER.info(f"[MODEL] Saved validation predictions to {results_dir / 'val.csv'}")
            else:
                LOGGER.info("[MODEL] No validation rows collected for val.csv in fit_validate")
        except Exception as e:
            LOGGER.warning(f"[MODEL] Failed to write val.csv in fit_validate: {e}")

        # Train metrics from the same best-checkpoint train predictions.
        train_metrics = regression_metrics(train_rows, "train", mt.task)
        if train_metrics is None:
            LOGGER.warning(
                "[MODEL] Could not compute train metrics for res.txt; "
                "train_rows=%d task=%s", len(train_rows), mt.task,
            )

        # Early stopping epoch
        es_epoch = None
        try:
            from pytorch_lightning.callbacks.early_stopping import EarlyStopping as _ES
            for cb in trainer.callbacks:
                if isinstance(cb, _ES):
                    es_epoch = getattr(cb, "stopped_epoch", None)
                    if es_epoch in (None, 0):
                        es_epoch = getattr(cb, "early_stopped_epoch", None)
                    break
        except Exception:
            pass

        _write_res_txt(mt, results_dir, val_metrics, train_metrics, es_epoch)
    except Exception as e:
        LOGGER.debug(f"[MODEL] Artifact generation in fit_validate failed: {e}")


def _write_res_txt(mt, results_dir, val_metrics, train_metrics, es_epoch) -> None:
    trainer = mt._last_trainer
    try:
        lines = ["Run summary:\n", "Validation metrics (final/best):\n"]
        for k, v in val_metrics.items():
            try:
                val_str = f"{float(v):.6f}"
            except Exception:
                val_str = str(v)
            lines.append(f"  {k}: {val_str}\n")
        if train_metrics is not None:
            lines.append("\nTraining metrics:\n")
            for k, v in train_metrics.items():
                metric_key = str(k)
                if metric_key.startswith("val_"):
                    metric_key = f"train_{metric_key[4:]}"
                elif not metric_key.startswith("train_"):
                    metric_key = f"train_{metric_key}"
                try:
                    tr_str = f"{float(v):.6f}"
                except Exception:
                    tr_str = str(v)
                lines.append(f"  {metric_key}: {tr_str}\n")
        trained_epochs = int(getattr(trainer, "current_epoch", 0) or 0)
        lines.append("\nTraining schedule:\n")
        lines.append(f"  trained_epochs: {trained_epochs}\n")
        lines.append(f"  max_epochs: {int(mt.max_epochs)}\n")
        lines.append(f"  early_stopping_patience: {int(mt.model_cfg.optim.lr_patience)}\n")
        validation_monitor = mt._checkpoint_monitor_metric()
        best_epoch = mt._get_best_epoch_from_trainer(trainer)
        best_score = mt._get_best_checkpoint_score_from_trainer(trainer)
        lines.append("\nMetric source:\n")
        lines.append("  trial_metrics_source: best_checkpoint\n")
        lines.append(f"  checkpoint_monitor: {validation_monitor}\n")
        lines.append(f"  best_epoch: {best_epoch}\n" if best_epoch is not None else "  best_epoch: N/A\n")
        if best_score is not None:
            lines.append(f"  best_checkpoint_score: {best_score:.6f}\n")
        else:
            lines.append("  best_checkpoint_score: N/A\n")
        lines.append("\nEarly stopping:\n")
        if best_epoch is not None:
            lines.append(f"  stopped_epoch: {best_epoch}\n")
            if es_epoch is not None and es_epoch != 0 and es_epoch != best_epoch:
                lines.append(f"  early_stopped_epoch_internal: {es_epoch}\n")
        elif es_epoch is not None and es_epoch != 0:
            lines.append(f"  stopped_epoch: {es_epoch}\n")
        else:
            lines.append("  stopped_epoch: N/A (not triggered or unavailable)\n")
        with open(results_dir / "res.txt", "w") as f:
            f.writelines(lines)
        LOGGER.info(f"[MODEL] Wrote run summary to {results_dir / 'res.txt'}")
    except Exception as e:
        LOGGER.warning(f"[MODEL] Failed to write res.txt in fit_validate: {e}")


def evaluate_on_test(mt, dm, model) -> None:
    """Evaluate the best model on the test split and export test.csv + attention."""
    try:
        test_dl = None
        try:
            test_dl = dm.test_dataloader()
        except Exception:
            test_dl = None
        has_test = test_dl is not None and len(test_dl) > 0 if test_dl is not None else False

        if not has_test:
            LOGGER.info("[MODEL][TEST] No test split available – skipping test evaluation")
            return

        LOGGER.info("[MODEL][TEST] Evaluating best model on the test split")
        best_test_model = mt.get_best_model()
        test_model = best_test_model if best_test_model is not None else model
        test_best_epoch = mt._get_best_epoch_from_trainer(mt._last_trainer)
        if test_best_epoch is not None:
            setattr(test_model, "_evaluation_epoch_override", test_best_epoch)

        # Quiet temporary trainer to avoid extra progress bars/logging.
        from pytorch_lightning import Trainer as _PLTrainer
        temp_trainer = _PLTrainer(
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False,
        )
        test_metrics = temp_trainer.test(test_model, dataloaders=test_dl, verbose=False)[0]
        log_metrics(test_metrics, stage="test", prefix="test")

        if not mt.experiment_name:
            return
        try:
            results_dir = create_results_directory(mt.experiment_name)
            test_rows = predict_rows(test_model, test_dl, mt.task)
            if test_rows:
                test_df = pd.DataFrame(test_rows, columns=["mol_id", "true", "predicted"])
                test_df["predicted"] = test_df["predicted"].round(2)
                test_df["abs_error"] = (test_df["true"] - test_df["predicted"]).abs().round(2)
                test_df.to_csv(results_dir / "test.csv", index=False)
                LOGGER.info(f"[MODEL][TEST] Saved test predictions to {results_dir / 'test.csv'}")

            if bool(getattr(mt.trainer_cfg, "save_attention_artifacts", True)):
                _export_test_attention(mt, test_model, test_dl, results_dir)
        except Exception as e:
            LOGGER.warning(f"[MODEL][TEST] Failed to save test.csv: {e}")
    except Exception as e:
        LOGGER.debug(f"[MODEL][TEST] Test evaluation step failed: {e}")


def _export_test_attention(mt, test_model, test_dl, results_dir) -> None:
    try:
        from ppl.plotting.plot_attention_weights import plot_attention_weights_from_model
        attention_dir = results_dir / "test" / "attention_weights"
        attention_dir.mkdir(parents=True, exist_ok=True)

        conf_ids = mt._extract_conformer_ids()
        if conf_ids is None:
            LOGGER.info("[MODEL][TEST] Using indices as instance IDs for test attention weight plots")
        else:
            LOGGER.info(f"[MODEL][TEST] Using conformer IDs for test attention weight plots ({len(conf_ids)} bags)")

        plot_attention_weights_from_model(
            model=test_model, dataloader=test_dl,
            save_dir=str(attention_dir), max_bags=1000, conf_ids=conf_ids,
        )
        LOGGER.info(f"[MODEL][TEST] Saved test attention weights to {attention_dir}")
    except Exception as e_inner:
        LOGGER.warning(f"[MODEL][TEST] Failed to generate test attention weights: {e_inner}")
