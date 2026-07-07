from __future__ import annotations

import dataclasses as dc
import logging
import pprint

import pytorch_lightning as pl

from ppl.utils.modelling_configs.model_builder_config import ModelBuilderConfig
from ppl.utils.model_builder.model_factory import ModelFactory

LOGGER = logging.getLogger(__name__)


class ModelBuilder:
    """Build a MIL model (embedder -> aggregator -> predictor) from a config."""

    def __init__(self, cfg: ModelBuilderConfig | None = None, **legacy_kwargs):
        """Initialize the model builder.

        Parameters
        ----------
        cfg : ModelBuilderConfig, optional
            Model configuration. If omitted, one is built from ``legacy_kwargs``.
        **legacy_kwargs
            Individual configuration overrides (used only when ``cfg`` is None).

        Raises
        ------
        ValueError
            If both ``cfg`` and ``legacy_kwargs`` are given, or the task is invalid.
        """
        if cfg is None:
            cfg = ModelBuilderConfig().update(**legacy_kwargs)
        elif legacy_kwargs:
            raise ValueError("Pass either cfg or individual kwargs, not both.")

        LOGGER.info("[MODEL] Building model from config:\n%s", pprint.pformat(dc.asdict(cfg)))

        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.task = cfg.task.lower()

        if self.task not in ("classification", "regression"):
            raise ValueError(
                f"Invalid task: {self.task}. Must be 'classification' or 'regression'"
            )

        LOGGER.info(
            "[MODEL] embedder=%s aggregator=%s predictor=%s",
            cfg.embedder_type,
            cfg.aggregator_type,
            cfg.predictor_type,
        )

    def build(self) -> pl.LightningModule:
        """Build the model and return a Lightning module ready for training.

        Raises
        ------
        ValueError
            If ``input_dim`` was not injected before building.
        """
        if self.input_dim is None:
            raise ValueError(
                "input_dim is None – ensure the number of descriptors is injected "
                "into ModelBuilderConfig before calling ModelBuilder.build()."
            )

        return ModelFactory.build_model(self.cfg, self.input_dim, self.task)
