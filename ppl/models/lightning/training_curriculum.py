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
        """Query weight in the aggregator-focus phase: ramp 0 -> ``query_max_weight``
        over the smoothing window, then hold. Zero before the phase triggers."""
        trigger_epoch = getattr(self, "_attention_refinement_trigger_epoch", None)
        if trigger_epoch is None:
            return 0.0
        ramp = max(
            1, int(getattr(self, "_attention_refinement_query_ramp_epochs", 2) or 1)
        )
        max_weight = float(
            getattr(self, "_attention_refinement_query_max_weight", 0.8) or 0.0
        )
        phase_epoch = max(0, int(epoch) - int(trigger_epoch))
        progress = min(1.0, phase_epoch / float(ramp))
        return float(max(0.0, min(1.0, max_weight * progress)))

    def _attention_refinement_phase(self, epoch: int) -> str:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return "off"
        if getattr(self, "_attention_refinement_trigger_epoch", None) is None:
            return "watch"
        return "focus"

    def _attention_refinement_epoch_suffix(self) -> str:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return ""
        epoch = int(getattr(self, "current_epoch", 0) or 0)
        phase = self._attention_refinement_phase(epoch)
        weight = self._attention_refinement_query_weight(epoch)
        return (
            " aggregator_focus="
            f"{phase} forced_query_weight={weight:.3f} "
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
            getattr(self, "_attention_refinement_metric", "loss") or "loss"
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
        # Freeze the whole feature-extraction path — the per-instance embedder AND
        # its within-bag self-attention/contextualizer — plus the predictor, while
        # the aggregator and active-query builder keep learning during focus.
        target_prefixes = ("Embedder.", "Self-attention/contextualizer.", "Predictor.")
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
        if trigger_epoch is None or int(epoch) - int(trigger_epoch) <= 0:
            self._set_core_active_query_override(disabled=True, forced_weight=0.0)
            return

        weight = self._attention_refinement_query_weight(epoch)
        self._set_core_active_query_override(disabled=False, forced_weight=weight)
        if log:
            LOGGER.debug(
                "[AGG_FOCUS] epoch=%d focus_epoch=%d forced_query_weight=%.3f",
                epoch, int(epoch) - int(trigger_epoch), weight,
            )

    def _active_bank_ready(self) -> tuple[bool, int, int]:
        """Whether the active-prototype bank has enough prototypes for the query
        to inject — the same threshold the injection gate uses, so entering the
        focus phase guarantees the enrichment is not a silent no-op."""
        core = getattr(self, "core", None)
        bank = getattr(core, "active_prototype_bank", None) if core is not None else None
        if bank is None:
            return False, 0, 0
        min_active = int(getattr(core, "active_prototype_min_active", 0) or 0)
        n_active = int(bank.num_active())
        return n_active >= min_active, n_active, min_active

    def _maybe_update_attention_refinement_schedule(
        self,
        val_metrics: dict,
        val_loss,
    ) -> None:
        if not bool(getattr(self, "_attention_refinement_enabled", False)):
            return

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        metric, _train_float, val_float = self._attention_refinement_metric_values(
            val_metrics,
            val_loss,
        )
        if val_float is None:
            LOGGER.debug("[AGG_FOCUS] epoch=%d: missing val %s; skipping", epoch, metric)
            return

        min_delta = float(
            getattr(self, "_attention_refinement_min_delta", 0.005) or 0.0
        )
        patience = max(
            1, int(getattr(self, "_attention_refinement_patience", 3) or 3)
        )

        if getattr(self, "_attention_refinement_trigger_epoch", None) is None:
            # ---- Phase A (watch): enter focus on a val plateau, once bank ready ----
            self._apply_attention_refinement_phase(epoch)  # query stays off
            best = getattr(self, "_attention_refinement_best_val", None)
            if best is None or val_float < best - min_delta:
                self._attention_refinement_best_val = val_float
                self._attention_refinement_bad_epochs = 0
                return
            self._attention_refinement_bad_epochs = (
                int(getattr(self, "_attention_refinement_bad_epochs", 0)) + 1
            )
            if self._attention_refinement_bad_epochs < patience:
                return
            ready, n_active, min_active = self._active_bank_ready()
            if not ready:
                LOGGER.warning(
                    "[AGG_FOCUS] val %s plateaued at epoch %d but the active-prototype "
                    "bank is not ready (%d/%d active); deferring the aggregator-focus "
                    "phase until it fills.",
                    metric, epoch, n_active, min_active,
                )
                return  # stay in watch; trigger as soon as the bank is ready
            self._attention_refinement_trigger_epoch = epoch
            self._scale_attention_refinement_lrs()
            self._focus_best_val = val_float
            self._focus_bad_epochs = 0
            self._apply_attention_refinement_phase(epoch)
            factor = float(getattr(self, "_attention_refinement_lr_factor", 0.1) or 0.1)
            ramp = max(
                1, int(getattr(self, "_attention_refinement_query_ramp_epochs", 2) or 1)
            )
            max_w = float(
                getattr(self, "_attention_refinement_query_max_weight", 0.8) or 0.0
            )
            logging.getLogger("milk").info(
                "Aggregator-focus phase at epoch %d: val %s plateaued (%d epochs no gain) "
                "and %d active prototypes are ready — cutting embedder/predictor LR ×%.3g "
                "(the aggregator keeps its LR), ramping the active-prototype query to "
                "weight %.2f over %d epochs; training continues until val plateaus again.",
                epoch, metric, patience, n_active, factor, max_w, ramp,
            )
            return

        # ---- Phase B (focus): hold the ramp; stop on a second val plateau ----
        self._apply_attention_refinement_phase(epoch)
        best = getattr(self, "_focus_best_val", None)
        if best is None or val_float < best - min_delta:
            self._focus_best_val = val_float
            self._focus_bad_epochs = 0
            return
        self._focus_bad_epochs = int(getattr(self, "_focus_bad_epochs", 0)) + 1
        if self._focus_bad_epochs < patience:
            return
        # Floor: keep the active-prototype phase running for at least
        # min_focus_epochs so the aggregator can adapt to the injected prototypes
        # before the loss-plateau stop is allowed to fire. Loss still controls the
        # stop — it just cannot fire before the floor.
        min_focus = int(getattr(self, "_attention_refinement_min_focus_epochs", 0) or 0)
        focus_epoch = epoch - int(self._attention_refinement_trigger_epoch)
        if focus_epoch < min_focus:
            return
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            trainer.should_stop = True
        if not bool(getattr(self, "_attention_refinement_stop_logged", False)):
            self._attention_refinement_stop_logged = True
            logging.getLogger("milk").info(
                "Aggregator-focus phase converged at epoch %d (%d focus epochs, "
                "val %s plateaued again for %d); stopping training.",
                epoch, focus_epoch, metric, patience,
            )
