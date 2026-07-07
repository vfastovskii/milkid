"""Checkpoint callback with epoch / attention-refinement gating."""
from __future__ import annotations

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MinEpochModelCheckpoint(ModelCheckpoint):
    """Select checkpoints only after configured epoch/attention-refinement gates."""

    def __init__(
        self,
        *args,
        min_epoch: int = 0,
        require_attention_refinement: bool = False,
        min_query_epochs: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_epoch = int(min_epoch)
        self.require_attention_refinement = bool(require_attention_refinement)
        self.min_query_epochs = max(1, int(min_query_epochs))

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        if trainer.current_epoch < self.min_epoch:
            return True
        if self.require_attention_refinement:
            module = getattr(trainer, "lightning_module", None)
            trigger_epoch = getattr(module, "_attention_refinement_trigger_epoch", None)
            if trigger_epoch is None:
                return True
            query_epochs_seen = trainer.current_epoch - int(trigger_epoch)
            if query_epochs_seen < self.min_query_epochs:
                return True
        return super()._should_skip_saving_checkpoint(trainer)
