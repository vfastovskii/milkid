"""Instance-importance utilities for KID interpretation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InstanceImportanceResult:
    """Instance-importance output for one dataset split."""

    selection_by_bag_id: Dict[str, list[int]]
    summary: Dict[str, float]
    csv_path: Path
    json_path: Path
    importance_dir: Path


def _as_device_tensor(value: Any, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _normalise_sum(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def _safe_filename(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "bag"


def _write_instance_importance_exports(
    df: pd.DataFrame,
    *,
    save_dir: Path,
) -> Path:
    """Write per-bag instance-importance CSVs next to ablation artifacts."""
    importance_dir = save_dir / "instance_importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "bag_id",
        "instance_idx",
        "instance_id",
        "label",
        "pred_full",
        "pred_ablated",
        "delta_pred",
        "impact_abs",
        "impact_positive",
        "attention",
        "attention_norm",
        "impact_norm",
        "attention_component",
        "impact_component",
        "selection_score",
        "instance_importance_weight",
        "importance_rank",
        "selected",
    ]
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(
            importance_dir / "all_instance_importance.csv",
            index=False,
        )
        return importance_dir

    export_df = df.loc[:, [col for col in columns if col in df.columns]].copy()
    export_df = export_df.sort_values(
        ["bag_id", "importance_rank", "instance_idx"],
        ascending=[True, True, True],
    )
    export_df.to_csv(importance_dir / "all_instance_importance.csv", index=False)

    for bag_id, bag_df in export_df.groupby("bag_id", sort=False):
        path = importance_dir / f"{_safe_filename(bag_id)}.csv"
        bag_df.to_csv(path, index=False)

    return importance_dir


def _core_forward(
    model,
    x: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    cluster_ids: Optional[torch.Tensor],
    *,
    stage: Optional[str] = None,
    current_epoch: Optional[int] = None,
):
    kwargs = {}
    if key_padding_mask is not None:
        kwargs["key_padding_mask"] = key_padding_mask
    if cluster_ids is not None:
        kwargs["cluster_ids"] = cluster_ids
    if stage is not None:
        kwargs["stage"] = stage
    if current_epoch is not None:
        kwargs["current_epoch"] = current_epoch
    pred, extras = model.core(x, **kwargs)
    return pred.flatten(), extras


def _stage_from_split_name(split_name: str) -> str:
    split = str(split_name).lower()
    if split.startswith("val"):
        return "val"
    if split.startswith("test"):
        return "test"
    return "train"


def _parse_batch(batch, device: torch.device):
    if isinstance(batch, dict):
        bags = batch.get("bags") if "bags" in batch else batch.get("x")
        labels = batch.get("labels") if "labels" in batch else batch.get("y")
        bag_ids = batch.get("bag_ids") if "bag_ids" in batch else batch.get("ids")
        key_padding_mask = batch.get("padding_mask")
        cluster_ids = batch.get("cluster_ids")
    else:
        bags = batch[0]
        labels = batch[1]
        bag_ids = batch[2]
        key_padding_mask = batch[3] if len(batch) > 3 else None
        cluster_ids = batch[4] if len(batch) > 4 else None

    x = bags[0] if isinstance(bags, list) else bags
    x = _as_device_tensor(x, device).float()
    labels = _as_device_tensor(labels, device)
    key_padding_mask = _as_device_tensor(key_padding_mask, device)
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.bool()
    cluster_ids = _as_device_tensor(cluster_ids, device)
    if cluster_ids is not None:
        cluster_ids = cluster_ids.long()
    return x, labels, list(bag_ids), key_padding_mask, cluster_ids


def run_instance_importance_selection(
    *,
    model,
    dataloader,
    save_dir: Path,
    min_keep: int,
    max_keep: int,
    attention_weight: float,
    impact_weight: float,
    conf_ids: Optional[dict[str, list[str]]] = None,
    expected_bag_ids: Optional[Sequence[str]] = None,
    split_name: str = "train",
) -> InstanceImportanceResult:
    """Ablate conformers and select a compact key-instance subset per bag.

    The score is a compromise between learned attention and prediction effect:

    score_i = w_attn * norm(alpha_i) + w_impact * norm(abs(y_full - y_without_i))
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    min_keep = max(1, int(min_keep))
    max_keep = max(min_keep, int(max_keep))
    attention_weight = float(attention_weight)
    impact_weight = float(impact_weight)

    rows: list[dict[str, Any]] = []
    selection_by_bag_id: Dict[str, list[int]] = {}
    original_counts_by_bag_id: Dict[str, int] = {}
    model.eval()
    device = next(model.parameters()).device
    eval_stage = _stage_from_split_name(split_name)
    eval_epoch = getattr(
        model,
        "_instance_importance_epoch",
        getattr(model, "current_epoch", None),
    )
    expected_keys = None
    expected_key_set = None
    if expected_bag_ids is not None:
        expected_keys = [str(bag_id) for bag_id in expected_bag_ids]
        expected_key_set = set(expected_keys)
        if len(expected_key_set) != len(expected_keys):
            raise ValueError("Expected train bag IDs contain duplicates")

    LOGGER.info(
        "[INSTANCE_IMPORTANCE] Running %s-set instance ablation: min_keep=%d "
        "max_keep=%d attention_weight=%.3f impact_weight=%.3f "
        "eval_stage=%s eval_epoch=%s expected_bags=%s",
        split_name,
        min_keep,
        max_keep,
        attention_weight,
        impact_weight,
        eval_stage,
        eval_epoch,
        len(expected_keys) if expected_keys is not None else "unknown",
    )

    with torch.no_grad():
        for batch in dataloader:
            x, labels, bag_ids, key_padding_mask, cluster_ids = _parse_batch(
                batch,
                device,
            )
            full_pred, extras = _core_forward(
                model,
                x,
                key_padding_mask,
                cluster_ids,
                stage=eval_stage,
                current_epoch=eval_epoch,
            )
            alpha = extras.get("alpha_final", extras.get("alpha"))
            if alpha is None:
                raise ValueError(
                    "Ablation selection requires extras['alpha_final'] or extras['alpha']"
                )
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(0)
            cluster_alpha_query = extras.get("cluster_alpha")
            cluster_alpha_final = extras.get("cluster_alpha_final")

            if key_padding_mask is None:
                valid_mask = torch.ones(
                    x.size(0),
                    x.size(1),
                    device=x.device,
                    dtype=torch.bool,
                )
            else:
                valid_mask = ~key_padding_mask.bool()

            for b, bag_id in enumerate(bag_ids):
                bag_key = str(bag_id)
                if bag_key in selection_by_bag_id:
                    raise ValueError(
                        "Ablation loader produced duplicate bag ID "
                        f"{bag_key!r}; use a sequential full-{split_name} loader"
                    )
                if expected_key_set is not None and bag_key not in expected_key_set:
                    raise ValueError(
                        "Ablation loader produced a bag not present in the "
                        f"expected train set: {bag_key!r}"
                    )

                valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    raise ValueError(f"Ablation bag {bag_key!r} has no valid instances")
                original_counts_by_bag_id[bag_key] = int(valid_idx.numel())

                alpha_b = alpha[b, valid_idx].detach().float().cpu().numpy()
                cluster_ids_np = None
                cluster_query_mass_b = None
                cluster_final_mass_b = None
                if cluster_ids is not None:
                    cluster_ids_np = (
                        cluster_ids[b, valid_idx].detach().long().cpu().numpy()
                    )
                    if cluster_alpha_query is not None:
                        c = cluster_ids[b, valid_idx].long().clamp(
                            0,
                            cluster_alpha_query.size(1) - 1,
                        )
                        cluster_query_mass_b = (
                            cluster_alpha_query[b, c].detach().float().cpu().numpy()
                        )
                    if cluster_alpha_final is not None:
                        c = cluster_ids[b, valid_idx].long().clamp(
                            0,
                            cluster_alpha_final.size(1) - 1,
                        )
                        cluster_final_mass_b = (
                            cluster_alpha_final[b, c].detach().float().cpu().numpy()
                        )
                full_value = float(full_pred[b].detach().cpu().item())
                label_value = (
                    float(labels[b].detach().cpu().item())
                    if labels is not None
                    else np.nan
                )

                deltas = []
                ablated_values = []
                for idx_t in valid_idx:
                    idx = int(idx_t.item())
                    if valid_idx.numel() <= 1:
                        ablated = full_value
                    else:
                        mask_b = (
                            torch.zeros(
                                1,
                                x.size(1),
                                dtype=torch.bool,
                                device=x.device,
                            )
                            if key_padding_mask is None
                            else key_padding_mask[b : b + 1].clone()
                        )
                        mask_b[0, idx] = True
                        cluster_ids_one = (
                            None
                            if cluster_ids is None
                            else cluster_ids[b : b + 1]
                        )
                        pred_ablated, _ = _core_forward(
                            model,
                            x[b : b + 1],
                            mask_b,
                            cluster_ids_one,
                            stage=eval_stage,
                            current_epoch=eval_epoch,
                        )
                        ablated = float(pred_ablated[0].detach().cpu().item())
                    ablated_values.append(ablated)
                    deltas.append(full_value - ablated)

                deltas_np = np.asarray(deltas, dtype=np.float64)
                impact_abs = np.abs(deltas_np)
                attention_norm = _normalise_sum(alpha_b)
                impact_norm = _normalise_sum(impact_abs)
                score = (
                    attention_weight * attention_norm
                    + impact_weight * impact_norm
                )
                importance_weight = _normalise_sum(score)

                order = np.argsort(-score)
                keep_n = min(max_keep, len(order))
                keep_n = max(min_keep, keep_n)
                keep_n = min(keep_n, len(order))
                selected_positions = set(order[:keep_n].tolist())
                ranks = np.empty_like(order)
                ranks[order] = np.arange(1, len(order) + 1)
                selected_indices = [
                    int(valid_idx[pos].item())
                    for pos in sorted(selected_positions)
                ]
                selection_by_bag_id[bag_key] = selected_indices

                instance_ids = conf_ids.get(bag_key) if conf_ids else None
                for pos, idx_t in enumerate(valid_idx):
                    idx = int(idx_t.item())
                    instance_id = (
                        instance_ids[idx]
                        if instance_ids is not None and idx < len(instance_ids)
                        else str(idx)
                    )
                    rows.append(
                        {
                            "bag_id": bag_key,
                            "instance_idx": idx,
                            "instance_id": instance_id,
                            "label": label_value,
                            "pred_full": full_value,
                            "pred_ablated": float(ablated_values[pos]),
                            "delta_pred": float(deltas_np[pos]),
                            "impact_abs": float(impact_abs[pos]),
                            "impact_positive": float(max(deltas_np[pos], 0.0)),
                            "attention": float(alpha_b[pos]),
                            "attention_norm": float(attention_norm[pos]),
                            "cluster_id": (
                                int(cluster_ids_np[pos])
                                if cluster_ids_np is not None
                                else -1
                            ),
                            "cluster_attention_query_mass": (
                                float(cluster_query_mass_b[pos])
                                if cluster_query_mass_b is not None
                                else np.nan
                            ),
                            "cluster_attention_final_mass": (
                                float(cluster_final_mass_b[pos])
                                if cluster_final_mass_b is not None
                                else np.nan
                            ),
                            "impact_norm": float(impact_norm[pos]),
                            "attention_component": float(
                                attention_weight * attention_norm[pos]
                            ),
                            "impact_component": float(
                                impact_weight * impact_norm[pos]
                            ),
                            "selection_score": float(score[pos]),
                            "instance_importance_weight": float(
                                importance_weight[pos]
                            ),
                            "importance_rank": int(ranks[pos]),
                            "selected": pos in selected_positions,
                        }
                    )

    if expected_keys is not None:
        missing = [
            bag_id
            for bag_id in expected_keys
            if bag_id not in selection_by_bag_id
        ]
        extra = sorted(set(selection_by_bag_id) - expected_key_set)
        if missing or extra:
            details = []
            if missing:
                details.append(
                    f"missing {len(missing)} expected train bags; first={missing[:10]}"
                )
            if extra:
                details.append(
                    f"unexpected {len(extra)} train bags; first={extra[:10]}"
                )
            raise ValueError(
                "Ablation selection did not cover the train set exactly: "
                + "; ".join(details)
            )

    for bag_id, selected_indices in selection_by_bag_id.items():
        original_count = original_counts_by_bag_id[bag_id]
        min_expected = min(min_keep, original_count)
        max_expected = min(max_keep, original_count)
        selected_count = len(selected_indices)
        if selected_count < min_expected or selected_count > max_expected:
            raise ValueError(
                f"Ablation selected {selected_count} instances for bag {bag_id!r}, "
                f"expected range [{min_expected}, {max_expected}]"
            )

    df = pd.DataFrame(rows)
    csv_path = save_dir / f"{split_name}_instance_importance_instances.csv"
    df.to_csv(csv_path, index=False)
    importance_dir = _write_instance_importance_exports(df, save_dir=save_dir)

    json_path = save_dir / "selected_instance_indices.json"
    with open(json_path, "w") as f:
        json.dump(selection_by_bag_id, f, indent=2, sort_keys=True)

    selected_counts = [len(v) for v in selection_by_bag_id.values()]
    single_conformer_bags = sum(
        1 for count in original_counts_by_bag_id.values() if count == 1
    )
    total_instances = int(len(df))
    selected_instances = int(df["selected"].sum()) if not df.empty else 0
    summary = {
        "ablation_bags": float(len(selection_by_bag_id)),
        "ablation_expected_bags": (
            float(len(expected_keys)) if expected_keys is not None else 0.0
        ),
        "ablation_missing_bags": 0.0,
        "ablation_single_conformer_bags": float(single_conformer_bags),
        "ablation_instances": float(total_instances),
        "ablation_selected_instances": float(selected_instances),
        "ablation_kept_fraction": (
            float(selected_instances / total_instances)
            if total_instances > 0
            else 0.0
        ),
        "ablation_mean_selected_per_bag": (
            float(np.mean(selected_counts)) if selected_counts else 0.0
        ),
        "ablation_max_selected_per_bag": (
            float(np.max(selected_counts)) if selected_counts else 0.0
        ),
    }
    LOGGER.info(
        "[INSTANCE_IMPORTANCE] Saved %d instance rows to %s; selected %d/%d instances "
        "(mean %.2f per bag, max %.0f, single-conformer bags=%d); "
        "instance-importance CSVs in %s",
        total_instances,
        csv_path,
        selected_instances,
        total_instances,
        summary["ablation_mean_selected_per_bag"],
        summary["ablation_max_selected_per_bag"],
        single_conformer_bags,
        importance_dir,
    )
    return InstanceImportanceResult(
        selection_by_bag_id=selection_by_bag_id,
        summary=summary,
        csv_path=csv_path,
        json_path=json_path,
        importance_dir=importance_dir,
    )
