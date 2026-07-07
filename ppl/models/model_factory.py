from __future__ import annotations

import logging

import pytorch_lightning as pl

from ppl.models.mil_core import MILCore
from ppl.models.mil_lightning_wrapper import MILModelLightningWrapper
from ppl.models.component_catalog import COMPONENT_ORDER, build_components

LOGGER = logging.getLogger(__name__)


class ModelFactory:
    """Build and validate the MIL model from a configuration."""

    @staticmethod
    def validate_model_architecture(components, task: str) -> None:
        """Warn on any dimension mismatch or unexpected predictor output size.

        Adjacent dimensions are already chained in :func:`build_components`, so
        this is a lightweight sanity check rather than a correction step.
        """
        names = list(COMPONENT_ORDER)
        for component, name in zip(components, names):
            in_dim = getattr(component, "input_dim", None)
            out_dim = getattr(component, "output_dim", None)
            LOGGER.info("[MODEL] %s: input_dim=%s output_dim=%s", name, in_dim, out_dim)

        for cur, nxt, cur_name, nxt_name in zip(
            components, components[1:], names, names[1:]
        ):
            cur_out = getattr(cur, "output_dim", None)
            nxt_in = getattr(nxt, "input_dim", None)
            if cur_out is not None and nxt_in is not None and cur_out != nxt_in:
                LOGGER.warning(
                    "[MODEL] Dimension mismatch: %s.output_dim=%s != %s.input_dim=%s",
                    cur_name, cur_out, nxt_name, nxt_in,
                )

        predictor_out = getattr(components[-1], "output_dim", None)
        if predictor_out not in (None, 1):
            LOGGER.warning(
                "[MODEL] Predictor output_dim=%s (expected 1 for %s)",
                predictor_out, task,
            )

    @staticmethod
    def build_model(cfg, input_dim: int, task: str) -> pl.LightningModule:
        """Build the Lightning module (embedder -> aggregator -> predictor).

        Parameters
        ----------
        cfg : ModelBuilderConfig
            Model configuration.
        input_dim : int
            Descriptor dimension fed to the embedder.
        task : str
            Task type (classification or regression).

        Returns
        -------
        pl.LightningModule
            Lightning module ready for training.
        """
        components = build_components(cfg, input_dim)
        ModelFactory.validate_model_architecture(components, task)

        core = MILCore(
            components,
            list(COMPONENT_ORDER),
            task=task,
            active_prototype_kwargs=getattr(cfg, "active_prototype_kwargs", {}),
        )
        model = MILModelLightningWrapper(core, task=task, optim_cfg=cfg.optim)

        n_params = sum(p.numel() for p in model.parameters())
        LOGGER.info(
            "[MODEL] Built %s (input_dim=%d, %s) with %d parameters",
            type(model).__name__, input_dim, task, n_params,
        )
        return model
