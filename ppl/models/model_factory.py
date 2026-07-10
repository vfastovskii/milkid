from __future__ import annotations

import dataclasses as dc
import logging
import pprint

import pytorch_lightning as pl

from ppl.models.core import MILCore
from ppl.models.lightning import MILModelLightningWrapper
from ppl.models.component_catalog import COMPONENT_ORDER, build_components

LOGGER = logging.getLogger(__name__)


def _validate_model_architecture(components, task: str) -> None:
    """Log per-component dims and warn on any mismatch.

    Adjacent dimensions are already chained in :func:`build_components`, so this
    is a lightweight sanity check rather than a correction step.
    """
    names = list(COMPONENT_ORDER)
    for component, name in zip(components, names):
        LOGGER.info(
            "[MODEL] %s: input_dim=%s output_dim=%s",
            name,
            getattr(component, "input_dim", None),
            getattr(component, "output_dim", None),
        )

    for cur, nxt, cur_name, nxt_name in zip(components, components[1:], names, names[1:]):
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
            "[MODEL] Predictor output_dim=%s (expected 1 for %s)", predictor_out, task,
        )


def build_model(cfg, input_dim: int, task: str) -> pl.LightningModule:
    """Build the Lightning module (embedder -> aggregator -> predictor) from a config.

    Parameters
    ----------
    cfg : ModelBuilderConfig
        Model configuration.
    input_dim : int
        Descriptor dimension fed to the embedder; must be injected before building.
    task : str
        "classification" or "regression".

    Returns
    -------
    pl.LightningModule
        Lightning module ready for training.
    """
    task = task.lower()
    if task not in ("classification", "regression"):
        raise ValueError(f"Invalid task: {task}. Must be 'classification' or 'regression'")
    if input_dim is None:
        raise ValueError(
            "input_dim is None – inject the descriptor count into the config before building."
        )

    LOGGER.info("[MODEL] Building model from config:\n%s", pprint.pformat(dc.asdict(cfg)))
    LOGGER.info(
        "[MODEL] embedder=%s aggregator=%s predictor=%s",
        cfg.embedder_type, cfg.aggregator_type, cfg.predictor_type,
    )

    components = build_components(cfg, input_dim)
    _validate_model_architecture(components, task)

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
