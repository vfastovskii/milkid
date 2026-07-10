"""Training utilities for MIL model."""

import logging
import traceback
import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)

class TrainingMethods(nn.Module):
    """Training methods mixin for MIL model."""

    @staticmethod
    def _extra_scalar(extras: dict, key: str, default=0.0):
        value = extras.get(key, default)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return default
            return value.detach().flatten()[0].item()
        return value

    def _reset_epoch_loss_accumulator(self, stage: str) -> None:
        if not hasattr(self, "_epoch_loss_sums"):
            self._epoch_loss_sums = {}
        if not hasattr(self, "_epoch_loss_counts"):
            self._epoch_loss_counts = {}
        self._epoch_loss_sums[stage] = 0.0
        self._epoch_loss_counts[stage] = 0

    def _update_epoch_loss_accumulator(
        self,
        stage: str,
        loss: torch.Tensor,
        batch_size: int,
    ) -> None:
        if not hasattr(self, "_epoch_loss_sums"):
            self._epoch_loss_sums = {}
        if not hasattr(self, "_epoch_loss_counts"):
            self._epoch_loss_counts = {}

        try:
            loss_value = float(loss.detach().mean().item())
        except Exception:
            return
        if not bool(torch.isfinite(torch.as_tensor(loss_value)).item()):
            return

        weight = max(int(batch_size), 1)
        self._epoch_loss_sums[stage] = self._epoch_loss_sums.get(stage, 0.0) + (
            loss_value * weight
        )
        self._epoch_loss_counts[stage] = (
            self._epoch_loss_counts.get(stage, 0) + weight
        )

    def _compute_epoch_loss(self, stage: str):
        if not hasattr(self, "_epoch_loss_sums") or not hasattr(
            self,
            "_epoch_loss_counts",
        ):
            return None
        count = self._epoch_loss_counts.get(stage, 0)
        if count <= 0:
            return None
        return self._epoch_loss_sums.get(stage, 0.0) / count

    def _consume_current_train_epoch_metrics(self) -> tuple:
        """Compute and reset train metrics for the current epoch once."""
        epoch = getattr(self, "current_epoch", None)
        if getattr(self, "_train_epoch_metrics_consumed_epoch", None) == epoch:
            return getattr(self, "_last_train_loss", None), getattr(
                self,
                "_last_train_metrics",
                {},
            )

        train_loss = self._compute_epoch_loss("train")
        if train_loss is None:
            # Standalone validation/test runs have no train updates. Do not call
            # torchmetrics.compute() on an empty train collection; R2 needs at
            # least two samples and will correctly raise otherwise.
            self._last_train_loss = None
            self._last_train_metrics = {}
            return self._last_train_loss, self._last_train_metrics

        self._last_train_loss = (
            torch.as_tensor(train_loss, device=self.device, dtype=torch.float32)
        )
        computed = self.train_metrics.compute()
        self._last_train_metrics = {
            f"train_{k}": v.detach().clone() for k, v in computed.items()
        }
        self.train_metrics.reset()
        self._reset_epoch_loss_accumulator("train")
        self._train_epoch_metrics_consumed_epoch = epoch
        return self._last_train_loss, self._last_train_metrics

    @staticmethod
    def _format_epoch_scalar(value) -> str:
        if value is None:
            return "nan"
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return "nan"
            value = value.detach().flatten()[0].item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "nan"
        if not bool(torch.isfinite(torch.as_tensor(value)).item()):
            return "nan"
        return f"{value:.6f}"

    @staticmethod
    def _to_float_or_none(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            value = value.detach().flatten()[0].item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not bool(torch.isfinite(torch.as_tensor(value)).item()):
            return None
        return value

    def _metric_gap(self, train_value, val_value):
        train_float = self._to_float_or_none(train_value)
        val_float = self._to_float_or_none(val_value)
        if train_float is None or val_float is None:
            return None
        return val_float - train_float

    def _active_prototype_epoch_suffix(self) -> str:
        bank = getattr(self.core, "active_prototype_bank", None)
        if bank is None:
            return ""

        epoch = getattr(self, "current_epoch", "unknown")
        num_active = bank.num_active()
        total_support = float(bank.counts[bank.active_mask].sum().item())
        max_prototypes = getattr(bank, "max_prototypes", 0)
        min_active = getattr(self.core, "active_prototype_min_active", 0)
        ready = num_active >= min_active
        query_weight = (
            self.core._effective_active_query_weight(epoch)
            if isinstance(epoch, int)
            and hasattr(self.core, "_effective_active_query_weight")
            else 0.0
        )
        num_updates = int(
            getattr(self.core, "active_prototype_num_updates", torch.zeros(())).item()
        )
        return (
            " active_proto="
            f"{num_active}/{max_prototypes} active_support={total_support:.0f} "
            f"active_ready={ready} active_query_weight={query_weight:.3f} "
            f"active_updates={num_updates}"
        )

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

    def _eval_dataset_has_noexp_duplicates(self, stage: str) -> bool:
        """Return whether the eval dataset contains canonical __noexp bags."""
        if stage not in {"val", "test"}:
            return False

        cache_attr = f"_cached_{stage}_has_noexp_duplicates"
        if hasattr(self, cache_attr):
            return bool(getattr(self, cache_attr))

        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None)
        dataset_attr = "_val" if stage == "val" else "_test"
        dataset = getattr(datamodule, dataset_attr, None)
        bag_ids = getattr(dataset, "_bag_ids", None)
        if bag_ids is None:
            return False

        has_noexp = bool(bag_ids) and any(
            str(bag_id).endswith("__noexp") for bag_id in bag_ids
        )
        setattr(self, cache_attr, has_noexp)
        return has_noexp

    def _filter_eval_batch_to_noexp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        bag_ids,
        key_padding_mask,
        cluster_ids,
        series_labels,
        stage: str,
    ):
        """Use only canonical non-experimental duplicates for eval metrics/loss.

        Validation/test datasets may contain paired bags:
        original full bag and a "__noexp" duplicate. Filtering after the forward
        pass makes metrics depend on batch boundaries. This filters before the
        model sees the batch, so changing batch_size cannot make full bags enter
        evaluation metrics.
        """
        if not self._eval_dataset_has_noexp_duplicates(stage):
            return x, y, bag_ids, key_padding_mask, cluster_ids, series_labels, False

        keep_list = [str(bag_id).endswith("__noexp") for bag_id in bag_ids]
        if not any(keep_list):
            return None, None, [], None, None, None, True

        keep = torch.as_tensor(keep_list, device=x.device, dtype=torch.bool)
        x = x[keep]
        y = y[keep]
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[keep]
        if cluster_ids is not None:
            cluster_ids = cluster_ids[keep]
        bag_ids = [bag_id for bag_id, keep_item in zip(bag_ids, keep_list) if keep_item]
        if series_labels is not None:
            series_labels = [
                series
                for series, keep_item in zip(series_labels, keep_list)
                if keep_item
            ]
        return x, y, bag_ids, key_padding_mask, cluster_ids, series_labels, False

    def _log_active_prototype_metrics(
        self,
        extras: dict,
        stage: str,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Log active prototype bank size and usage from core extras."""
        if "active_prototype_num_active" not in extras:
            return

        num_active = float(
            self._extra_scalar(extras, "active_prototype_num_active", 0.0)
        )
        total_support = float(
            self._extra_scalar(extras, "active_prototype_total_support", 0.0)
        )
        update_candidates = float(
            self._extra_scalar(extras, "active_prototype_update_candidates", 0.0)
        )
        active_query_used = float(
            bool(self._extra_scalar(extras, "active_query_used", False))
        )
        active_query_weight = float(
            self._extra_scalar(extras, "active_query_weight", 0.0)
        )
        ready = float(bool(self._extra_scalar(extras, "active_prototype_ready", False)))

        def as_metric(value: float) -> torch.Tensor:
            return torch.as_tensor(value, device=device, dtype=torch.float32)

        common_kwargs = {
            "batch_size": batch_size,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }
        on_step = False

        self.log(
            f"{stage}_active_prototypes",
            as_metric(num_active),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            **common_kwargs,
        )
        self.log(
            f"{stage}_active_proto_support",
            as_metric(total_support),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            **common_kwargs,
        )
        self.log(
            f"{stage}_active_query_used",
            as_metric(active_query_used),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            **common_kwargs,
        )
        self.log(
            f"{stage}_active_query_weight",
            as_metric(active_query_weight),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            **common_kwargs,
        )
        self.log(
            f"{stage}_active_proto_ready",
            as_metric(ready),
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            **common_kwargs,
        )

        if stage == "train":
            self.log(
                "train_active_proto_candidates",
                as_metric(update_candidates),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                **common_kwargs,
            )

    def _log_active_prototype_epoch_summary(self) -> None:
        bank = getattr(self.core, "active_prototype_bank", None)
        if bank is None:
            return

        num_active = bank.num_active()
        total_support = float(bank.counts[bank.active_mask].sum().item())
        max_prototypes = getattr(bank, "max_prototypes", 0)
        min_active = getattr(self.core, "active_prototype_min_active", 0)
        ready = num_active >= min_active
        epoch = getattr(self, "current_epoch", "unknown")
        warmup_epochs = getattr(self.core, "active_prototype_warmup_epochs", 0)
        query_start_epoch = getattr(
            self.core,
            "active_prototype_query_start_epoch",
            warmup_epochs,
        )
        query_ramp_epochs = getattr(
            self.core,
            "active_prototype_query_ramp_epochs",
            1,
        )
        query_weight = (
            self.core._effective_active_query_weight(epoch)
            if isinstance(epoch, int)
            and hasattr(self.core, "_effective_active_query_weight")
            else 0.0
        )
        warmup_done = False
        if isinstance(epoch, int):
            warmup_done = epoch >= warmup_epochs
        use_on_eval = getattr(self.core, "active_prototype_use_on_eval", False)
        alpha_key = getattr(self.core, "active_prototype_candidate_alpha_key", "alpha")
        num_updates = int(
            getattr(self.core, "active_prototype_num_updates", torch.zeros(())).item()
        )
        series_hist = (
            bank.series_histogram()
            if hasattr(bank, "series_histogram")
            else {}
        )
        series_hist_txt = (
            ", ".join(
                f"{series}:{count}"
                for series, count in sorted(series_hist.items())
            )
            if series_hist
            else "none"
        )
        missing_series = (
            bank.missing_seen_series()
            if hasattr(bank, "missing_seen_series")
            else []
        )
        missing_series_txt = (
            ",".join(str(series) for series in missing_series[:10])
            if missing_series
            else "none"
        )

        LOGGER.debug(
            "[ACTIVE_PROTO] epoch=%s identified_prototypes=%d/%d "
            "total_support=%.0f ready=%s min_required=%d warmup_done=%s "
            "warmup_epochs=%s query_start_epoch=%s query_ramp_epochs=%s "
            "query_weight=%.3f updates=%d alpha_key=%s use_on_eval=%s "
            "series_hist={%s} missing_seen_series={%s}",
            epoch,
            num_active,
            max_prototypes,
            total_support,
            ready,
            min_active,
            warmup_done,
            warmup_epochs,
            query_start_epoch,
            query_ramp_epochs,
            query_weight,
            num_updates,
            alpha_key,
            use_on_eval,
            series_hist_txt,
            missing_series_txt,
        )

    def forward(self, x):
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Model output
        """
        y_hat, _ = self.core(x)
        return y_hat

    def _shared_step(self, batch, stage: str):
        """Shared logic for training, validation, and test steps.

        Parameters
        ----------
        batch : tuple
            Batch of data (bags, labels, bag_ids)
        stage : str
            Current stage (train, val, test)

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Track memory usage periodically but much less frequently during training
        # to avoid interfering with progress bars
        if stage == "train":
            # Only log every 100 steps or so (0.5% probability)
            if torch.rand(1).item() < 0.005:
                self._log_memory_usage(f"{stage}_step", log_level="debug")
        else:
            # For validation and test, we can log more frequently
            if torch.rand(1).item() < 0.05:  # 5% probability
                self._log_memory_usage(f"{stage}_step", log_level="debug")

        # Unpack batch - expecting tuple format (bags, labels, bag_ids, padding_mask) from collate_mil
        try:
            # Log batch unpacking start
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Starting batch unpacking - batch type: {type(batch)}")
            
            # Standard tuple format:
            # (padded_bags, labels, bag_ids, padding_mask[, cluster_ids[, series_labels]])
            cluster_ids = None
            series_labels = None
            if len(batch) == 4:
                bags, y, bag_ids, key_padding_mask = batch
            elif len(batch) == 5:
                bags, y, bag_ids, key_padding_mask, cluster_ids = batch
            elif len(batch) == 6:
                (
                    bags,
                    y,
                    bag_ids,
                    key_padding_mask,
                    cluster_ids,
                    series_labels,
                ) = batch
            else:
                raise ValueError(
                    "Expected batch with 4, 5, or 6 elements, got "
                    f"{len(batch)}"
                )
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Unpacked padded batch format")
            
            # Log basic batch information
            bags_shape = bags.shape if hasattr(bags, 'shape') else 'no shape'
            labels_shape = y.shape if hasattr(y, 'shape') else 'no shape'
            num_bag_ids = len(bag_ids) if isinstance(bag_ids, (list, tuple)) else 'not list/tuple'
            mask_shape = key_padding_mask.shape if key_padding_mask is not None else 'None'
            
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Batch contents:")
            LOGGER.debug(f"  - bags shape: {bags_shape}")
            LOGGER.debug(f"  - labels shape: {labels_shape}")
            LOGGER.debug(f"  - bag_ids count: {num_bag_ids}")
            LOGGER.debug(f"  - padding_mask shape: {mask_shape}")
            LOGGER.debug(f"  - padding approach: before embedding (simpler processing)")
            
            # Process pre-padded tensor batch (simple processing)
            x = bags  # Pre-padded tensor [B, N_max, D_input]
            batch_size = x.shape[0] if hasattr(x, 'shape') else 1
            
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Processing pre-padded tensor batch")
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Tensor shape: {x.shape}")
            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Batch size: {batch_size}")

            y = y.float().flatten()
            if stage in {"val", "test"}:
                (
                    x,
                    y,
                    bag_ids,
                    key_padding_mask,
                    cluster_ids,
                    series_labels,
                    skip_batch,
                ) = (
                    self._filter_eval_batch_to_noexp(
                        x=x,
                        y=y,
                        bag_ids=bag_ids,
                        key_padding_mask=key_padding_mask,
                        cluster_ids=cluster_ids,
                        series_labels=series_labels,
                        stage=stage,
                    )
                )
                if skip_batch:
                    LOGGER.debug(
                        "[SHARED_STEP_%s] Skipping full-only eval batch; "
                        "__noexp duplicates are evaluated in another batch",
                        stage.upper(),
                    )
                    return None
                batch_size = x.shape[0] if hasattr(x, "shape") else 1

            LOGGER.debug(f"[SHARED_STEP_{stage.upper()}] Final preprocessing: "
                       f"x_shape={x.shape if hasattr(x, 'shape') else 'no shape'}, "
                       f"y_shape={y.shape}, batch_size={batch_size}")
        except Exception as e:
            LOGGER.error(f"Error unpacking batch in {stage}_step: {e}")
            LOGGER.error(f"Batch types: {[type(item) for item in batch]}")
            LOGGER.error(f"Batch shapes: {[item.shape if hasattr(item, 'shape') else None for item in batch]}")
            raise

        # Forward pass. Lightning controls precision through the Trainer.  Do
        # not open a nested autocast context here, otherwise HPO cannot safely
        # switch between fp32, fp16-mixed, and bf16-mixed.
        try:
            forward_epoch = getattr(self, "current_epoch", 0)
            if stage in {"val", "test"}:
                forward_epoch = getattr(
                    self,
                    "_evaluation_epoch_override",
                    forward_epoch,
                )

            def _forward_and_loss():
                logit, extras = self.core(
                    x,
                    key_padding_mask=key_padding_mask,
                    cluster_ids=cluster_ids,
                    series_labels=series_labels,
                    labels=y,
                    stage=stage,
                    current_epoch=forward_epoch,
                )
                logit = logit.flatten()
                if not torch.isfinite(logit).all():
                    def _finite_summary(name: str, tensor: torch.Tensor) -> str:
                        t = tensor.detach()
                        finite = torch.isfinite(t)
                        finite_count = int(finite.sum().item())
                        total = int(t.numel())
                        if finite_count == 0:
                            return f"{name}: finite=0/{total}"
                        values = t[finite]
                        return (
                            f"{name}: finite={finite_count}/{total} "
                            f"min={float(values.min().item()):.6g} "
                            f"max={float(values.max().item()):.6g}"
                        )

                    LOGGER.error(
                        "[NON_FINITE_FORWARD] stage=%s epoch=%s batch_size=%s "
                        "bag_ids_first=%s %s %s %s",
                        stage,
                        forward_epoch,
                        batch_size,
                        list(bag_ids[:5]) if isinstance(bag_ids, list) else bag_ids,
                        _finite_summary("x", x),
                        _finite_summary("y", y),
                        _finite_summary("logit", logit),
                    )
                loss = self.criterion(logit, y)
                reg_loss = extras.get("reg_loss")
                if reg_loss is not None:
                    if isinstance(reg_loss, torch.Tensor):
                        loss = loss + reg_loss.to(device=loss.device, dtype=loss.dtype)
                    else:
                        loss = loss + torch.as_tensor(
                            reg_loss,
                            device=loss.device,
                            dtype=loss.dtype,
                        )
                return logit, extras, loss

            logit, extras, loss = _forward_and_loss()

            self._log_active_prototype_metrics(
                extras=extras,
                stage=stage,
                batch_size=batch_size,
                device=logit.device,
            )

            metric_logit = logit
            metric_y = y
            loss_batch_size = batch_size

            # Evaluation batches were already filtered to canonical "__noexp"
            # bags when duplicate full/no-exp bags are present.
            if stage in ("val", "test"):
                try:
                    target = metric_y.to(
                        device=metric_logit.device,
                        dtype=metric_logit.dtype,
                    )
                    eval_loss = self.criterion.base(metric_logit, target)
                    if eval_loss.numel() > 1:
                        eval_loss = eval_loss.mean()
                    loss = eval_loss

                    if stage == "val":
                        self.log(
                            "val_loss",
                            eval_loss,
                            prog_bar=True,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                            add_dataloader_idx=False,
                            batch_size=loss_batch_size,
                        )
                except Exception as ve:
                    LOGGER.error(f"Error applying evaluation loss mask: {ve}")
                    raise

            # Check for NaN loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                raise FloatingPointError(
                    f"NaN or Inf loss detected in {stage}_step"
                )

            # Log attention weights and other extras for debugging
            if stage == "train" and torch.rand(1).item() < 0.01:  # 1% probability during training
                if hasattr(self.core.aggregator, 'last_attn') and self.core.aggregator.last_attn is not None:
                    attn = self.core.aggregator.last_attn
                    LOGGER.debug(f"Attention stats: min={attn.min().item():.4f}, max={attn.max().item():.4f}, "
                                f"mean={attn.mean().item():.4f}, std={attn.std().item():.4f}")

                alpha_key = "alpha_final" if "alpha_final" in extras else "alpha"
                if alpha_key in extras:
                    alpha = extras[alpha_key]
                    LOGGER.debug(f"Alpha stats: min={alpha.min().item():.4f}, max={alpha.max().item():.4f}, "
                                f"mean={alpha.mean().item():.4f}, std={alpha.std().item():.4f}")

                if 'entropy' in extras:
                    entropy = extras['entropy']
                    if hasattr(entropy, 'item'):
                        if entropy.numel() == 1:
                            # Single element tensor - use .item()
                            entropy_val = entropy.item()
                        else:
                            # Multi-element tensor - use mean
                            entropy_val = entropy.mean().item()
                    else:
                        # Already a scalar
                        entropy_val = entropy
                    LOGGER.debug(f"Entropy: {entropy_val:.4f}")

        except Exception as e:
            LOGGER.error(f"Error in forward pass or loss calculation in {stage}_step: {e}")
            LOGGER.error(traceback.format_exc())
            raise

        # Ensure loss is a scalar for PyTorch Lightning logging
        if loss.numel() > 1:
            loss = loss.mean()
        
        # Log loss:
        # - For validation, we don't log per-step val_loss; instead, we accumulate and log an epoch-weighted value in on_validation_epoch_end.
        # - For train/test, keep standard logging.
        self._update_epoch_loss_accumulator(stage, loss, loss_batch_size)
        if stage != "val":
            self.log(
                f"{stage}_loss",
                loss,
                prog_bar=(stage != "train"),
                batch_size=loss_batch_size,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
                reduce_fx="mean",
                add_dataloader_idx=False,
            )

        # Update metrics
        try:
            y_pred = torch.sigmoid(metric_logit) if self.task == "classification" else metric_logit
            metrics = getattr(self, f"{stage}_metrics")
            # Align dtype/device to avoid MPS mixed-precision issues in metric ops
            y_for_metrics = metric_y.to(device=y_pred.device, dtype=y_pred.dtype)
            metrics.update(y_pred, y_for_metrics)
        except Exception as e:
            LOGGER.error(f"Error updating metrics in {stage}_step: {e}")
            LOGGER.error(traceback.format_exc())
            raise

        return loss

    def on_save_checkpoint(self, checkpoint):
        checkpoint["attention_refinement_state"] = {
            "trigger_epoch": getattr(
                self,
                "_attention_refinement_trigger_epoch",
                None,
            ),
            "best_val": getattr(self, "_attention_refinement_best_val", None),
            "bad_epochs": getattr(self, "_attention_refinement_bad_epochs", 0),
            "lr_reduced": getattr(
                self,
                "_attention_refinement_lr_reduced",
                False,
            ),
        }

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("attention_refinement_state", {}) or {}
        self._attention_refinement_trigger_epoch = state.get("trigger_epoch")
        self._attention_refinement_best_val = state.get("best_val")
        self._attention_refinement_bad_epochs = int(state.get("bad_epochs", 0) or 0)
        self._attention_refinement_lr_reduced = bool(state.get("lr_reduced", False))

    def on_train_epoch_start(self):
        self._apply_attention_refinement_phase(
            int(getattr(self, "current_epoch", 0) or 0),
            log=True,
        )

    def training_step(self, batch, batch_idx):
        """Training step with memory management and error handling.

        Parameters
        ----------
        batch : tuple
            Batch of data
        batch_idx : int
            Batch index

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Periodically clear cache during training
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Reset gradient statistics at the beginning of each epoch
        if batch_idx == 0 and hasattr(self, 'gradient_tracker'):
            self.gradient_tracker.reset_stats()

        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Parameters
        ----------
        batch : tuple
            Batch of data
        batch_idx : int
            Batch index
        """
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step.

        Parameters
        ----------
        batch : tuple
            Batch of data
        batch_idx : int
            Batch index
        """
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        """Compute and log train metrics after each epoch.
        Stored copy is reused to log train vs val/test."""
        if (
            getattr(self, "_train_epoch_metrics_consumed_epoch", None)
            != getattr(self, "current_epoch", None)
        ):
            self._consume_current_train_epoch_metrics()
        self._log_active_prototype_epoch_summary()

        # Log gradient statistics if gradient tracker is available
        if hasattr(self, 'gradient_tracker'):
            # Get the logger
            if self.logger is not None:
                # If there are multiple loggers, find the MLflow logger
                if isinstance(self.logger, list):
                    for logger in self.logger:
                        if 'mlflow' in logger.__class__.__name__.lower():
                            self.gradient_tracker.log_stats_to_mlflow(logger)
                            break
                else:
                    # Single logger
                    self.gradient_tracker.log_stats_to_mlflow(self.logger)

    def on_validation_epoch_end(self):
        """Compute and log validation metrics (val_*), excluding val_loss which is logged per-step (on_epoch)."""
        self._consume_current_train_epoch_metrics()
        val_loss = self._compute_epoch_loss("val")
        computed_val = self.val_metrics.compute()
        log_dict = {f"val_{k}": v for k, v in computed_val.items()}
        if hasattr(self, "_last_train_metrics"):
            log_dict.update(self._last_train_metrics)
        self.log_dict(log_dict, sync_dist=True, add_dataloader_idx=False)
        self._log_epoch_metric_summary(computed_val, val_loss)
        self._maybe_update_attention_refinement_schedule(computed_val, val_loss)
        self._maybe_stop_for_overfit_gap(computed_val, val_loss)
        self.val_metrics.reset()
        self._reset_epoch_loss_accumulator("val")

    def on_test_epoch_end(self):
        """Compute and log test+fit metrics."""
        computed_test = self.test_metrics.compute()
        log_dict = {f"test_{k}": v for k, v in computed_test.items()}
        if hasattr(self, "_last_train_metrics"):
            log_dict.update(self._last_train_metrics)
        self.log_dict(log_dict, sync_dist=True, add_dataloader_idx=False)
        self.test_metrics.reset()
