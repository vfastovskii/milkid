"""Per-fingerprint-block feature scaling and normalization.

Descriptor columns look like ``GETAWAYFingerprint_42``; each fingerprint
block is standardized on train instances only, then (after clustering)
divided by sqrt(block size) before embedding.
"""
from __future__ import annotations

import logging
import math
import re
from collections import OrderedDict

import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _fingerprint_block_name(feature_name: str) -> str:
    """Infer a fingerprint block name from descriptor column basename.

    Expected columns look like ``GETAWAYFingerprint_42``. If a column does not
    end with a numeric suffix, the full name is treated as its own block.
    """
    match = re.match(r"^(?P<base>.+)_\d+$", str(feature_name))
    return match.group("base") if match else str(feature_name)


def _feature_blocks(feature_names: list[str]) -> "OrderedDict[str, list[int]]":
    blocks: "OrderedDict[str, list[int]]" = OrderedDict()
    for idx, name in enumerate(feature_names):
        blocks.setdefault(_fingerprint_block_name(name), []).append(idx)
    return blocks


def _format_block_sizes(feature_names: list[str]) -> str:
    blocks = _feature_blocks(feature_names)
    if not blocks:
        return "none"
    return ", ".join(f"{name}:{len(indices)}" for name, indices in blocks.items())


def _log_descriptor_blocks(stage: str, feature_names: list[str]) -> None:
    LOGGER.info(
        "[FEATURE_BLOCKS] %s total_features=%d blocks={%s}",
        stage,
        len(feature_names),
        _format_block_sizes(feature_names),
    )


def _transform_bags_by_fingerprint_blocks(
    *,
    train_bags: list[np.ndarray],
    val_bags: list[np.ndarray],
    test_bags: list[np.ndarray],
    feature_names: list[str],
) -> dict:
    """Fit one StandardScaler per fingerprint block on train instances only.

    This stage intentionally does not apply block-size normalization. Clustering
    uses this scaled-only descriptor space.
    """
    if not train_bags:
        raise ValueError("Cannot fit feature block scalers without training bags")

    blocks = _feature_blocks(feature_names)
    train_fit_data = np.vstack(train_bags).astype(np.float32, copy=False)
    scalers: dict[str, StandardScaler] = {}
    block_sizes = {name: len(indices) for name, indices in blocks.items()}

    LOGGER.info(
        "[FEATURE_SCALE] fit=train_only scaler=StandardScaler "
        "normalization=none stage=scale_for_clustering train_bags=%d "
        "train_instances=%d blocks=%d features=%d",
        len(train_bags),
        train_fit_data.shape[0],
        len(blocks),
        train_fit_data.shape[1],
    )
    LOGGER.info(
        "[FEATURE_SCALE] blocks_after_zero_var={%s}",
        ", ".join(
            f"{name}:n={size},scale=standardized"
            for name, size in block_sizes.items()
        ),
    )

    for block_name, indices in blocks.items():
        idx = np.asarray(indices, dtype=np.int64)
        scaler = StandardScaler().fit(train_fit_data[:, idx])
        scalers[block_name] = scaler
        LOGGER.info(
            "[FEATURE_SCALE] fitted block=%s features=%d train_mean_abs=%.6g "
            "train_scale_mean=%.6g",
            block_name,
            len(indices),
            float(np.abs(scaler.mean_).mean()),
            float(np.asarray(scaler.scale_).mean()),
        )

    def transform_split(split_name: str, bags: list[np.ndarray]) -> None:
        if not bags:
            LOGGER.info("[FEATURE_SCALE] transform split=%s skipped_empty", split_name)
            return
        n_instances = int(sum(len(bag) for bag in bags))
        LOGGER.info(
            "[FEATURE_SCALE] transform split=%s bags=%d instances=%d",
            split_name,
            len(bags),
            n_instances,
        )
        iterator = tqdm.tqdm(
            range(len(bags)),
            desc=f"Scaling {split_name} bags by fingerprint block",
            disable=len(bags) < 10,
        )
        for bag_idx in iterator:
            bag = bags[bag_idx].astype(np.float32, copy=False)
            out = np.empty_like(bag, dtype=np.float32)
            for block_name, indices in blocks.items():
                idx = np.asarray(indices, dtype=np.int64)
                block = scalers[block_name].transform(bag[:, idx])
                out[:, idx] = block.astype(np.float32)
            bags[bag_idx] = out
        transformed = np.vstack(bags).astype(np.float32, copy=False)
        summary = []
        for block_name, indices in blocks.items():
            idx = np.asarray(indices, dtype=np.int64)
            block_values = transformed[:, idx]
            summary.append(
                f"{block_name}:mean_abs={float(np.abs(block_values.mean(axis=0)).mean()):.4g},"
                f"var_sum={float(block_values.var(axis=0).sum()):.4g}"
            )
        LOGGER.info(
            "[FEATURE_SCALE] transformed split=%s stats={%s}",
            split_name,
            "; ".join(summary),
        )

    transform_split("train", train_bags)
    transform_split("validation", val_bags)
    transform_split("test", test_bags)

    return {
        "mode": "per_fingerprint_standard_scaler",
        "scalers": scalers,
        "block_sizes": block_sizes,
        "feature_blocks": {name: list(indices) for name, indices in blocks.items()},
    }


def _normalize_scaled_bags_by_fingerprint_blocks(
    *,
    train_bags: list[np.ndarray],
    val_bags: list[np.ndarray],
    test_bags: list[np.ndarray],
    feature_blocks: dict[str, list[int]],
) -> None:
    """Apply sqrt block-size normalization after clustering, before embedding."""
    LOGGER.info(
        "[FEATURE_NORM] normalization=divide_by_sqrt_block_size "
        "stage=embedder_input blocks=%d",
        len(feature_blocks),
    )
    LOGGER.info(
        "[FEATURE_NORM] blocks={%s}",
        ", ".join(
            f"{name}:n={len(indices)},scale=1/sqrt({len(indices)})="
            f"{1.0 / math.sqrt(max(1, len(indices))):.6f}"
            for name, indices in feature_blocks.items()
        ),
    )

    def normalize_split(split_name: str, bags: list[np.ndarray]) -> None:
        if not bags:
            LOGGER.info("[FEATURE_NORM] transform split=%s skipped_empty", split_name)
            return
        for bag_idx, bag in enumerate(bags):
            out = bag.astype(np.float32, copy=True)
            for indices in feature_blocks.values():
                idx = np.asarray(indices, dtype=np.int64)
                out[:, idx] = out[:, idx] / math.sqrt(max(1, len(indices)))
            bags[bag_idx] = out
        transformed = np.vstack(bags).astype(np.float32, copy=False)
        summary = []
        for block_name, indices in feature_blocks.items():
            idx = np.asarray(indices, dtype=np.int64)
            block_values = transformed[:, idx]
            summary.append(
                f"{block_name}:mean_abs={float(np.abs(block_values.mean(axis=0)).mean()):.4g},"
                f"var_sum={float(block_values.var(axis=0).sum()):.4g}"
            )
        LOGGER.info(
            "[FEATURE_NORM] transformed split=%s stats={%s}",
            split_name,
            "; ".join(summary),
        )

    normalize_split("train", train_bags)
    normalize_split("validation", val_bags)
    normalize_split("test", test_bags)
