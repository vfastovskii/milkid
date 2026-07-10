"""Attention-refinement + overfit-gap training curriculum for the MIL wrapper.

Split out of TrainingMethods (training.py) to keep that file navigable. These
methods were part of TrainingMethods and are mixed back in via _CurriculumMixin,
so bodies and ``self`` are unchanged. The logger name is pinned to
"ppl.models.lightning.training" so the per-epoch INFO summary line stays on the
narrative allowlist exactly as before.
"""
import logging

import torch

# Pinned (not __name__) so the epoch-summary INFO line keeps its narrative logger.
LOGGER = logging.getLogger("ppl.models.lightning.training")


class _CurriculumMixin:
    """Per-epoch metric summary + attention-refinement / overfit-gap scheduling."""

    def _attention_refinement_query_weight(self, epoch: int) -> float:
        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        if trigger_epoch is None:
            return 0.0

        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        start = float(
            getattr(self, "_attention_refinement_query_weight_start", 0.0) or 0.0
        )
        end = float(
            getattr(self, "_attention_refinement_query_weight_end", 1.0) or 1.0
        )
        phase_epoch = max(0, int(epoch) - int(trigger_epoch))
        progress = min(1.0, phase_epoch / float(query_epochs))
        weight = start + (end - start) * progress
        return float(max(0.0, min(1.0, weight)))

    def _attention_refinement_phase(self, epoch: int) -> str:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return "off"
        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        if trigger_epoch is None:
            return "watch"

        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        phase_epoch = int(epoch) - int(trigger_epoch)
        if phase_epoch <= 0:
            return "triggered"
        if phase_epoch <= query_epochs:
            return "query_ramp"
        return "done"

    def _attention_refinement_epoch_suffix(self) -> str:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return ""

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        phase = self._attention_refinement_phase(epoch)
        weight = self._attention_refinement_query_weight(epoch)
        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        if trigger_epoch is None:
            progress = "0/0"
        else:
            phase_epoch = max(0, min(query_epochs, epoch - int(trigger_epoch)))
            progress = f"{phase_epoch}/{query_epochs}"

        return (
            " attention_refinement="
            f"{phase} query_ramp={progress} forced_query_weight={weight:.3f} "
            f"lr_reduced={bool(getattr(self, '_attention_refinement_lr_reduced', False))}"
        )

    def _log_epoch_metric_summary(
        self,
        val_metrics: dict,
        val_loss,
    ) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not getattr(trainer, "is_global_zero", True):
            return

        train_metrics = getattr(self, "_last_train_metrics", {}) or {}
        mae_gap = self._metric_gap(
            train_metrics.get("train_mae"),
            val_metrics.get("mae"),
        )
        rmse_gap = self._metric_gap(
            train_metrics.get("train_rmse"),
            val_metrics.get("rmse"),
        )
        epoch = getattr(self, "current_epoch", "?")

        def _short(v):
            v = self._to_float_or_none(v)
            return f"{v:.3f}" if v is not None else "n/a"

        # Concise per-epoch line only for real training epochs; a train-less pass
        # (the post-fit best-model re-validation) would otherwise print nan.
        if train_metrics:
            LOGGER.info(
                "Epoch %s · train loss %s rmse %s mae %s · val loss %s rmse %s mae %s",
                epoch,
                _short(getattr(self, "_last_train_loss", None)),
                _short(train_metrics.get("train_rmse")),
                _short(train_metrics.get("train_mae")),
                _short(val_loss),
                _short(val_metrics.get("rmse")),
                _short(val_metrics.get("mae")),
            )
        # Full technical detail (gaps, refinement schedule, prototype state) at DEBUG.
        LOGGER.debug(
            "[EPOCH_METRICS] epoch=%s "
            "train_loss=%s train_mae=%s train_rmse=%s train_r2=%s "
            "val_loss=%s val_mae=%s val_rmse=%s val_r2=%s "
            "mae_gap=%s rmse_gap=%s%s%s",
            epoch,
            self._format_epoch_scalar(getattr(self, "_last_train_loss", None)),
            self._format_epoch_scalar(train_metrics.get("train_mae")),
            self._format_epoch_scalar(train_metrics.get("train_rmse")),
            self._format_epoch_scalar(train_metrics.get("train_r2")),
            self._format_epoch_scalar(val_loss),
            self._format_epoch_scalar(val_metrics.get("mae")),
            self._format_epoch_scalar(val_metrics.get("rmse")),
            self._format_epoch_scalar(val_metrics.get("r2")),
            self._format_epoch_scalar(mae_gap),
            self._format_epoch_scalar(rmse_gap),
            self._attention_refinement_epoch_suffix(),
            self._active_prototype_epoch_suffix(),
        )

    def _attention_refinement_metric_values(self, val_metrics: dict, val_loss):
        metric = str(
            getattr(self, "_attention_refinement_metric", "rmse") or "rmse"
        ).lower()
        train_metrics = getattr(self, "_last_train_metrics", {}) or {}
        if metric == "loss":
            train_value = getattr(self, "_last_train_loss", None)
            val_value = val_loss
        else:
            train_value = train_metrics.get(f"train_{metric}")
            val_value = val_metrics.get(metric)

        return metric, self._to_float_or_none(train_value), self._to_float_or_none(
            val_value
        )

    def _scale_attention_refinement_lrs(self) -> None:
        if bool(getattr(self, "_attention_refinement_lr_reduced", False)):
            return

        factor = float(getattr(self, "_attention_refinement_lr_factor", 0.1) or 0.1)
        factor = max(0.0, min(1.0, factor))
        trainer = getattr(self, "trainer", None)
        optimizers = getattr(trainer, "optimizers", None) if trainer is not None else None
        if not optimizers:
            LOGGER.warning(
                "[ATTN_REFINEMENT] Could not reduce embedder/predictor LR: "
                "trainer optimizers are not available"
            )
            return

        scaled_groups = []
        target_prefixes = ("Embedder.", "Predictor.")
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                name = str(group.get("name", ""))
                if not name.startswith(target_prefixes):
                    continue
                old_lr = float(group.get("lr", 0.0))
                new_lr = old_lr * factor
                group["lr"] = new_lr
                scaled_groups.append((name, old_lr, new_lr))

        self._attention_refinement_lr_reduced = True
        if not scaled_groups:
            LOGGER.warning(
                "[ATTN_REFINEMENT] No Embedder/Predictor optimizer groups were found"
            )
            return

        LOGGER.debug(
            "[ATTN_REFINEMENT] Reduced embedder/predictor LR by factor=%.4f: %s",
            factor,
            ", ".join(
                f"{name}:{old_lr:.3g}->{new_lr:.3g}"
                for name, old_lr, new_lr in scaled_groups
            ),
        )

    def _set_core_active_query_override(
        self,
        *,
        disabled: bool,
        forced_weight,
    ) -> None:
        core = getattr(self, "core", None)
        if core is None:
            return
        setattr(core, "active_query_force_disabled", bool(disabled))
        setattr(core, "active_query_forced_weight", forced_weight)

    def _apply_attention_refinement_phase(
        self,
        epoch: int,
        *,
        log: bool = False,
    ) -> None:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            self._set_core_active_query_override(disabled=False, forced_weight=None)
            return

        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        if trigger_epoch is None:
            self._set_core_active_query_override(disabled=True, forced_weight=0.0)
            return

        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        phase_epoch = int(epoch) - int(trigger_epoch)
        if phase_epoch <= 0:
            self._set_core_active_query_override(disabled=True, forced_weight=0.0)
            return

        if phase_epoch <= query_epochs:
            weight = self._attention_refinement_query_weight(epoch)
            self._set_core_active_query_override(
                disabled=False,
                forced_weight=weight,
            )
            if log:
                LOGGER.debug(
                    "[ATTN_REFINEMENT] epoch=%d query_ramp=%d/%d "
                    "forced_query_weight=%.3f",
                    epoch,
                    phase_epoch,
                    query_epochs,
                    weight,
                )
            return

        self._set_core_active_query_override(disabled=True, forced_weight=0.0)
        self._mark_attention_refinement_complete()

    def _mark_attention_refinement_complete(self) -> None:
        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        if bool(getattr(self, "_attention_refinement_stop_after_query_epochs", True)):
            trainer = getattr(self, "trainer", None)
            if trainer is not None:
                trainer.should_stop = True
            if not bool(getattr(self, "_attention_refinement_stop_logged", False)):
                self._attention_refinement_stop_logged = True
                LOGGER.debug(
                    "[ATTN_REFINEMENT] Completed %d query-ramp epochs after "
                    "trigger_epoch=%s; stopping training",
                    query_epochs,
                    trigger_epoch,
                )

    def _maybe_update_attention_refinement_schedule(
        self,
        val_metrics: dict,
        val_loss,
    ) -> None:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        if getattr(self, "_attention_refinement_trigger_epoch", None) is not None:
            self._apply_attention_refinement_phase(epoch, log=False)
            trigger_epoch = int(getattr(self, "_attention_refinement_trigger_epoch"))
            query_epochs = max(
                1,
                int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
            )
            if epoch >= trigger_epoch + query_epochs:
                self._mark_attention_refinement_complete()
            return

        metric, train_float, val_float = self._attention_refinement_metric_values(
            val_metrics,
            val_loss,
        )
        if train_float is None or val_float is None:
            LOGGER.debug(
                "[ATTN_REFINEMENT] Skipping trigger check at epoch=%d; "
                "missing train/val %s",
                epoch,
                metric,
            )
            return

        min_delta = float(
            getattr(self, "_attention_refinement_min_delta", 0.005) or 0.0
        )
        gap_threshold = float(
            getattr(self, "_attention_refinement_gap_threshold", 0.08) or 0.0
        )
        rel_threshold = float(
            getattr(self, "_attention_refinement_rel_gap_threshold", 0.15) or 0.0
        )
        patience = max(
            1,
            int(getattr(self, "_attention_refinement_patience", 1) or 1),
        )
        require_val_worse = bool(
            getattr(
                self,
                "_attention_refinement_require_val_worse_than_best",
                True,
            )
        )

        best_val = getattr(self, "_attention_refinement_best_val", None)
        if best_val is None or val_float < (best_val - min_delta):
            self._attention_refinement_best_val = val_float
            self._attention_refinement_bad_epochs = 0
            return

        gap = val_float - train_float
        rel_gap = gap / max(abs(train_float), 1e-8)
        gap_triggered = gap >= gap_threshold or rel_gap >= rel_threshold
        val_is_worse = val_float > (
            float(self._attention_refinement_best_val) + min_delta
        )

        if gap_triggered and (val_is_worse or not require_val_worse):
            self._attention_refinement_bad_epochs += 1
        else:
            self._attention_refinement_bad_epochs = 0

        if self._attention_refinement_bad_epochs < patience:
            return

        self._attention_refinement_trigger_epoch = epoch
        self._scale_attention_refinement_lrs()
        self._apply_attention_refinement_phase(epoch, log=False)

        query_epochs = max(
            1,
            int(getattr(self, "_attention_refinement_query_epochs", 25) or 25),
        )
        logging.getLogger("milk").info(
            "Attention refinement triggered at epoch %d (val %s worse than best, gap %.3f) — "
            "mixing the active-prototype query into attention over the next %d epochs "
            "(weight %.2f→%.2f), LRs scaled ×%.3g.",
            epoch, metric, gap, query_epochs,
            float(getattr(self, "_attention_refinement_query_weight_start", 0.0)),
            float(getattr(self, "_attention_refinement_query_weight_end", 1.0)),
            float(getattr(self, "_attention_refinement_lr_factor", 0.1) or 0.1),
        )
        LOGGER.debug(
            "[ATTN_REFINEMENT] Triggered at epoch=%d metric=%s train=%.6f "
            "val=%.6f best_val=%.6f gap=%.6f rel_gap=%.3f. Next %d epochs "
            "will ramp active-query weight from %.3f to %.3f.",
            epoch,
            metric,
            train_float,
            val_float,
            float(self._attention_refinement_best_val),
            gap,
            rel_gap,
            query_epochs,
            float(getattr(self, "_attention_refinement_query_weight_start", 0.0)),
            float(getattr(self, "_attention_refinement_query_weight_end", 1.0)),
        )

    def _maybe_stop_for_overfit_gap(self, val_metrics: dict, val_loss) -> None:
        if bool(getattr(self, "_attention_refinement_enabled", False)):
            return
        if not bool(getattr(self, "_overfit_gap_stop_enabled", False)):
            return

        epoch = int(getattr(self, "current_epoch", 0))
        metric = str(getattr(self, "_overfit_gap_metric", "rmse") or "rmse").lower()
        train_metrics = getattr(self, "_last_train_metrics", {}) or {}
        if metric == "loss":
            train_value = getattr(self, "_last_train_loss", None)
            val_value = val_loss
        else:
            train_value = train_metrics.get(f"train_{metric}")
            val_value = val_metrics.get(metric)

        train_float = self._to_float_or_none(train_value)
        val_float = self._to_float_or_none(val_value)
        if train_float is None or val_float is None:
            LOGGER.debug(
                "[OVERFIT_GAP] Skipping epoch=%s; missing train/val %s",
                epoch,
                metric,
            )
            return

        min_delta = float(getattr(self, "_overfit_gap_min_delta", 0.0) or 0.0)
        abs_threshold = float(
            getattr(self, "_overfit_gap_abs_threshold", 0.10) or 0.0
        )
        rel_threshold = float(
            getattr(self, "_overfit_gap_rel_threshold", 0.15) or 0.0
        )
        patience = max(1, int(getattr(self, "_overfit_gap_patience", 2) or 2))
        require_val_worse = bool(
            getattr(self, "_overfit_gap_require_val_worse_than_best", True)
        )

        best_val = getattr(self, "_overfit_gap_best_val", None)
        if best_val is None or val_float < (best_val - min_delta):
            self._overfit_gap_best_val = val_float
            self._overfit_gap_bad_epochs = 0
            return

        gap = val_float - train_float
        rel_gap = gap / max(abs(train_float), 1e-8)
        gap_triggered = gap >= abs_threshold or rel_gap >= rel_threshold
        val_is_worse = val_float > (float(self._overfit_gap_best_val) + min_delta)

        if gap_triggered and (val_is_worse or not require_val_worse):
            self._overfit_gap_bad_epochs += 1
        else:
            self._overfit_gap_bad_epochs = 0

        if self._overfit_gap_bad_epochs >= patience:
            trainer = getattr(self, "trainer", None)
            if trainer is not None:
                trainer.should_stop = True
            LOGGER.debug(
                "[OVERFIT_STOP] epoch=%d metric=%s train=%.6f val=%.6f "
                "best_val=%.6f gap=%.6f rel_gap=%.3f bad_epochs=%d/%d",
                epoch,
                metric,
                train_float,
                val_float,
                float(self._overfit_gap_best_val),
                gap,
                rel_gap,
                self._overfit_gap_bad_epochs,
                patience,
            )
