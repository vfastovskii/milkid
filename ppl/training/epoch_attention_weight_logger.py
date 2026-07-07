from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)


class EpochAttentionWeightLogger(pl.Callback):
    """
    Logs attention weights for each bag at the end of each epoch.
    
    This callback logs attention weights for both training and validation bags
    at the end of each epoch, allowing for tracking the evolution of attention
    weights over the course of training.
    
    Parameters
    ----------
    save_dir : str
        Directory to save the attention weight plots.
    max_bags : int, default 1000
        Maximum number of bags to plot per epoch.
    conf_ids : Dict[str, list], optional
        Dictionary mapping bag IDs to lists of conformer IDs. If provided, these will be used
        as instance IDs in the plots instead of simple indices.
    """
    
    def __init__(
        self, 
        save_dir: str, 
        max_bags: int = 1500,
        conf_ids: Optional[Dict[str, List[str]]] = None,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_dir = save_dir
        self.max_bags = max_bags
        self.conf_ids = conf_ids
        self.max_epochs = max_epochs
    
    @staticmethod
    def _normalize_attention_1d(alpha: np.ndarray) -> np.ndarray:
        """Normalize a 1D attention vector to be non-negative and sum to 1.

        - Replaces NaN/Inf with 0.
        - If any negatives or non-positive sum: applies softmax for stability.
        - Else divides by sum.
        - Clips to [0, 1] to avoid tiny numerical drifts.
        """
        if alpha is None:
            return alpha
        a = np.asarray(alpha, dtype=float).flatten()
        if a.size == 0:
            return a
        # Sanitize invalid values
        a[~np.isfinite(a)] = 0.0
        # Early exit if all zeros
        if np.allclose(a, 0.0):
            return a
        s = float(a.sum())
        if np.any(a < 0.0) or s <= 0.0:
            m = float(np.max(a))
            exp_a = np.exp(a - m)
            denom = float(exp_a.sum())
            a = exp_a / denom if denom > 0.0 else np.zeros_like(a)
        else:
            a = a / s
        # Numerical safety
        a = np.clip(a, 0.0, 1.0)
        # Renormalize after clipping if needed
        total = a.sum()
        if total > 0:
            a = a / total
        return a
    
    def _plot_alpha(
        self,
        alpha: np.ndarray,
        bag_id: str,
        inst_ids: List[str],
        save_path: str,
        epoch: int = None,
    ) -> None:
        """
        Plot and save the distribution of attention weights inside a bag.
        
        Parameters
        ----------
        alpha : np.ndarray
            Attention weights (length = number of instances in the bag).
        bag_id : str
            Identifier of the bag; becomes part of the file name.
        inst_ids : List[str]
            Per-instance identifiers – shown on the y axis.
        save_path : str
            Full path where the PNG is written.
        epoch : int, optional
            The current epoch number, to be included in the plot title.
        """
        # Ensure alpha is a numpy array and normalize
        alpha = self._normalize_attention_1d(np.asarray(alpha, dtype=float))
        
        # Ensure inst_ids are strings
        inst_ids = [str(i) for i in inst_ids]
        
        # Check if lengths match
        if alpha.shape[0] != len(inst_ids):
            LOGGER.warning(
                f"alpha length ({alpha.shape[0]}) and inst_ids length ({len(inst_ids)}) mismatch for bag {bag_id}. "
                "Using indices instead."
            )
            inst_ids = [str(i) for i in range(alpha.shape[0])]
        
        # Safety: ensure numerical invariants before plotting
        s = float(np.sum(alpha))
        if s > 0.0 and not np.isclose(s, 1.0, atol=1e-6):
            alpha = alpha / s
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = np.round(alpha.astype(np.float64), 12)
        s = float(np.sum(alpha))
        if s > 0.0 and not np.isclose(s, 1.0, atol=1e-6):
            LOGGER.warning(f"[ATTN-PLOT] Re-normalized alpha for bag {bag_id} (sum={s:.6f}) before plotting")
        if float(np.max(alpha)) > 1.0 + 1e-9:
            LOGGER.warning(f"[ATTN-PLOT] Clipped alpha > 1 for bag {bag_id} (max={float(np.max(alpha)):.6f}) before plotting")

        # Create the figure
        fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(alpha))))
        
        y_pos = np.arange(len(alpha))
        ax.barh(y_pos, alpha, align="center", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(inst_ids)
        ax.invert_yaxis()  # highest weight on top
        ax.set_xlabel("Attention weight")
        
        # Include epoch number in the title if provided
        if epoch is not None:
            ax.set_title(f"Bag {bag_id} – Attention Weights (Epoch {epoch})")
        else:
            ax.set_title(f"Bag {bag_id} – Attention Weights")
        
        # Annotate with exact values
        for y, w in zip(y_pos, alpha):
            ax.text(
                w + 0.01,  # a tiny offset to the right of the bar
                y,
                f"{w:.3f}",
                va="center",
                fontsize=8,
            )
        
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        
        plt.close(fig)
        
    def _safe_extract_value(self, values, index):
        """
        Safely extract a value from a tensor or list, handling all edge cases.
        
        Parameters
        ----------
        values : torch.Tensor or list
            The tensor or list to extract a value from.
        index : int
            The index to extract.
            
        Returns
        -------
        float
            The extracted value as a Python number.
        """
        # Handle the case where values is None
        if values is None:
            return 0.0
            
        # Handle the case where values is a tensor
        if isinstance(values, torch.Tensor):
            # Handle 0-dim tensor
            if values.dim() == 0:
                return values.item()
            # Handle n-dim tensor
            return values[index].item()
            
        # Handle the case where values is a list
        if isinstance(values, list):
            # Handle the case where the element is a tensor
            if isinstance(values[index], torch.Tensor):
                return values[index].item()
            # Handle the case where the element is a number
            return values[index]
            
        # Default case
        return values
    
    def _save_csv(
        self,
        alpha: np.ndarray,
        true_value: float,
        pred_value: float,
        bag_id: str,
        inst_ids: List[str],
        save_path: str,
        cluster_ids: Optional[np.ndarray] = None,
        extra_scores: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Save attention weights, true value, and predicted value to a CSV file.
        
        Parameters
        ----------
        alpha : np.ndarray
            Attention weights (length = number of instances in the bag).
        true_value : float
            True value of the bag.
        pred_value : float
            Predicted value of the bag.
        bag_id : str
            Identifier of the bag; becomes part of the file name.
        inst_ids : List[str]
            Per-instance identifiers.
        save_path : str
            Full path where the CSV is written.
        """
        # Ensure alpha is a numpy array and normalize
        alpha = self._normalize_attention_1d(np.asarray(alpha, dtype=float))
        
        # Ensure inst_ids are strings
        inst_ids = [str(i) for i in inst_ids]
        
        # Check if lengths match
        if alpha.shape[0] != len(inst_ids):
            LOGGER.warning(
                f"alpha length ({alpha.shape[0]}) and inst_ids length ({len(inst_ids)}) mismatch for bag {bag_id}. "
                "Using indices instead."
            )
            inst_ids = [str(i) for i in range(alpha.shape[0])]
        if cluster_ids is not None:
            cluster_ids = np.asarray(cluster_ids).flatten()
            if cluster_ids.shape[0] != alpha.shape[0]:
                LOGGER.warning(
                    "[ATTN-CSV] Skipping cluster IDs for bag %s because length %d != alpha length %d",
                    bag_id,
                    cluster_ids.shape[0],
                    alpha.shape[0],
                )
                cluster_ids = None
        
        # Convert tensor values to Python numbers if needed
        if isinstance(true_value, torch.Tensor):
            true_value = true_value.item()
        if isinstance(pred_value, torch.Tensor):
            pred_value = pred_value.item()
        
        # Final safety before writing: enforce invariants and round
        s = float(np.sum(alpha))
        if s > 0.0 and not np.isclose(s, 1.0, atol=1e-8):
            LOGGER.debug(f"[ATTN-CSV] Re-normalizing alpha for bag {bag_id} at write time (sum={s:.12f})")
            alpha = alpha / s
        alpha = np.clip(alpha.astype(np.float64), 0.0, 1.0)
        s = float(np.sum(alpha))
        if s > 0.0:
            alpha = alpha / s
        maxv = float(np.max(alpha)) if alpha.size else 0.0
        if maxv > 1.0 + 1e-12:
            LOGGER.warning(f"[ATTN-CSV] Clipped alpha > 1 for bag {bag_id} at write time (max={maxv:.12f})")
            alpha = np.clip(alpha, 0.0, 1.0)
            s = float(np.sum(alpha))
            if s > 0.0:
                alpha = alpha / s
        # Round to stable decimals to avoid CSV textual artifacts
        alpha = np.round(alpha, 12)
        extra_scores = extra_scores or {}
        clean_extra_scores: Dict[str, np.ndarray] = {}
        for score_name, score_values in extra_scores.items():
            score = self._normalize_attention_1d(np.asarray(score_values, dtype=float))
            if score.shape[0] != alpha.shape[0]:
                LOGGER.warning(
                    "[ATTN-CSV] Skipping %s for bag %s because length %d != alpha length %d",
                    score_name,
                    bag_id,
                    score.shape[0],
                    alpha.shape[0],
                )
                continue
            clean_extra_scores[score_name] = np.round(score.astype(np.float64), 12)

        consensus = None
        alpha_score = clean_extra_scores.get("Instance Score Attention")
        if alpha_score is not None:
            consensus = self._normalize_attention_1d(alpha * alpha_score)
            consensus = np.round(consensus.astype(np.float64), 12)

        # Create the CSV file
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            extra_headers = list(clean_extra_scores)
            if consensus is not None:
                extra_headers.append("Consensus KID Score")
            writer.writerow(
                [
                    'Instance ID',
                    'Cluster ID',
                    'True Value',
                    'Predicted Value',
                    'Attention Weight',
                    *extra_headers,
                ]
            )
            # Write data for each instance
            for row_idx, (inst_id, weight) in enumerate(zip(inst_ids, alpha)):
                cluster_id = (
                    "" if cluster_ids is None else int(cluster_ids[row_idx])
                )
                extra_values = [
                    float(clean_extra_scores[name][row_idx])
                    for name in clean_extra_scores
                ]
                if consensus is not None:
                    extra_values.append(float(consensus[row_idx]))
                writer.writerow(
                    [
                        inst_id,
                        cluster_id,
                        true_value,
                        pred_value,
                        float(weight),
                        *extra_values,
                    ]
                )
    
    def _process_batch_attention(
        self,
        model: pl.LightningModule,
        batch: Any,
        stage: str,
        epoch: int,
    ) -> None:
        """
        Process a batch and log attention weights.
        
        Parameters
        ----------
        model : pl.LightningModule
            The model to extract attention weights from.
        batch : Any
            The batch of data.
        stage : str
            The stage (train or val).
        epoch : int
            The current epoch.
        """
        # Ensure epoch is 0-indexed to match user expectations
        # This prevents having extra epochs beyond what was configured in max_epochs
        if epoch > self.max_epochs - 1:  # If we're seeing an epoch beyond what was configured
            LOGGER.warning(f"Skipping unexpected epoch {epoch} (should be 0-{self.max_epochs-1} for a {self.max_epochs}-epoch run)")
            return
        
        # Unpack the batch - standard tuple format:
        # (bags, labels, bag_ids, padding_mask[, cluster_ids[, series_labels]])
        cluster_ids = None
        series_labels = None
        if isinstance(batch, dict):
            bags = batch.get("bags") if "bags" in batch else batch.get("x")
            true_values = batch.get("labels") if "labels" in batch else batch.get("y")
            bag_ids = batch.get("bag_ids") if "bag_ids" in batch else batch.get("ids")
            key_padding_mask = batch.get("padding_mask")
            cluster_ids = batch.get("cluster_ids")
            series_labels = batch.get("series_labels")
        else:
            if len(batch) == 4:
                bags, true_values, bag_ids, key_padding_mask = batch
            elif len(batch) == 5:
                bags, true_values, bag_ids, key_padding_mask, cluster_ids = batch
            elif len(batch) == 6:
                (
                    bags,
                    true_values,
                    bag_ids,
                    key_padding_mask,
                    cluster_ids,
                    series_labels,
                ) = batch
            else:
                raise ValueError(
                    "Expected attention logging batch with 4, 5, or 6 elements, "
                    f"got {len(batch)}"
                )
        
        # Get the device
        device = next(model.parameters()).device
        
        # Process pre-padded tensor batch
        x = bags.to(device)
        true_values = true_values.to(device) if isinstance(true_values, torch.Tensor) else true_values
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        if cluster_ids is not None:
            cluster_ids = cluster_ids.to(device)
        
        # Forward pass with masking
        with torch.no_grad():
            core_kwargs = {}
            if key_padding_mask is not None:
                core_kwargs["key_padding_mask"] = key_padding_mask
            if cluster_ids is not None:
                core_kwargs["cluster_ids"] = cluster_ids
            if true_values is not None:
                core_kwargs["labels"] = true_values
            if series_labels is not None:
                core_kwargs["series_labels"] = series_labels
            core_kwargs["stage"] = stage
            core_kwargs["current_epoch"] = int(epoch)
            pred_values, extras = model.core(x, **core_kwargs)
        
        # Extract final instance weights from aggregator output. For modern
        # hierarchical aggregators, alpha_final is the exact KID distribution
        # used for the prediction representation; alpha is kept as the same
        # tensor for backward compatibility.
        alpha_key = "alpha_final" if "alpha_final" in extras else "alpha"
        if alpha_key not in extras:
            LOGGER.warning(
                "No alpha/alpha_final found in aggregator extras for batched processing"
            )
            return
        
        # Alpha should be [B, N] where B is batch size, N is max instances (padded)
        alpha_batch = extras[alpha_key]  # [B, N]
        
        if len(alpha_batch.shape) != 2:
            LOGGER.warning(f"Expected alpha shape [B, N] for batched processing, got {alpha_batch.shape}")
            return

        def extract_optional_score(key: str, bag_index: int) -> Optional[np.ndarray]:
            score_batch = extras.get(key)
            if score_batch is None:
                return None
            if not isinstance(score_batch, torch.Tensor) or score_batch.dim() != 2:
                LOGGER.debug(
                    "Skipping optional score %s with unsupported shape/type %s",
                    key,
                    getattr(score_batch, "shape", type(score_batch)),
                )
                return None
            score_bag = score_batch[bag_index]
            if key_padding_mask is not None:
                return score_bag[~key_padding_mask[bag_index]].cpu().numpy()
            return score_bag.cpu().numpy()
        
        # Create the directory for this epoch and stage
        epoch_dir = Path(self.save_dir) / stage / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each bag in the batch
        for i, bag_id in enumerate(bag_ids):
            # Extract attention weights for this specific bag from pre-padded tensor format
            alpha_bag = alpha_batch[i]  # [N] padded
            
            # Remove padding using the mask
            if key_padding_mask is not None:
                # key_padding_mask[i] is True for padded positions
                mask_bag = ~key_padding_mask[i]  # True for real data
                current_alpha_np = alpha_bag[mask_bag].cpu().numpy()
                current_cluster_np = (
                    cluster_ids[i][mask_bag].cpu().numpy()
                    if cluster_ids is not None
                    else None
                )
            else:
                # No mask available, take all weights (may include padding)
                LOGGER.warning(f"No padding mask available for bag {bag_ids[i]}, using all attention weights")
                current_alpha_np = alpha_bag.cpu().numpy()
                current_cluster_np = (
                    cluster_ids[i].cpu().numpy() if cluster_ids is not None else None
                )
            
            # Generate instance IDs
            if self.conf_ids is not None and bag_id in self.conf_ids:
                # Use provided conformer IDs
                inst_ids = self.conf_ids[bag_id]
                # Ensure we have the right number of IDs
                if len(inst_ids) != len(current_alpha_np):
                    LOGGER.warning(
                        f"Number of conformer IDs ({len(inst_ids)}) doesn't match number of attention weights "
                        f"({len(current_alpha_np)}) for bag {bag_id}. Using indices instead."
                    )
                    inst_ids = list(range(len(current_alpha_np)))
            else:
                # Use indices as fallback
                inst_ids = list(range(len(current_alpha_np)))
            
            try:
                # Save plot
                plot_save_path = epoch_dir / f"{bag_id}.png"
                self._plot_alpha(
                    alpha=current_alpha_np,
                    bag_id=bag_id,
                    inst_ids=inst_ids,
                    save_path=plot_save_path,
                    epoch=epoch,
                )
                
                # Save CSV
                true_val = self._safe_extract_value(true_values, i)
                pred_val = self._safe_extract_value(pred_values, i)
                
                csv_save_path = epoch_dir / f"{bag_id}.csv"
                extra_scores = {
                    "Base Attention": extract_optional_score("alpha_base", i),
                    "Active-Conditioned Attention": extract_optional_score(
                        "alpha_active",
                        i,
                    ),
                    "Step1 Attention": extract_optional_score("alpha_step1", i),
                    "Step2 Attention": extract_optional_score("alpha_attn_step2", i),
                    "Instance Score Attention": extract_optional_score("alpha_score", i),
                }
                extra_scores = {
                    name: values
                    for name, values in extra_scores.items()
                    if values is not None
                }
                self._save_csv(
                    alpha=current_alpha_np,
                    true_value=true_val,
                    pred_value=pred_val,
                    bag_id=bag_id,
                    inst_ids=inst_ids,
                    save_path=csv_save_path,
                    cluster_ids=current_cluster_np,
                    extra_scores=extra_scores,
                )
            except Exception as e:
                LOGGER.error(f"Error processing bag {bag_id}: {e}")
                LOGGER.error(f"Alpha shape: {current_alpha_np.shape}, inst_ids length: {len(inst_ids)}")
                import traceback
                LOGGER.debug(traceback.format_exc())
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log attention weights for training bags at the end of each epoch.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The PyTorch Lightning module.
        """
        # Skip if we've already processed too many bags
        if hasattr(self, '_train_bags_processed') and self._train_bags_processed >= self.max_bags:
            return
        
        # Set the model to evaluation mode
        pl_module.eval()
        
        # Reset the counter for the number of bags processed
        self._train_bags_processed = 0
        
        # Process each batch in the training dataloader
        # Use trainer.datamodule.train_dataloader() instead of trainer.train_dataloader
        # to ensure we get a fresh dataloader for each epoch
        try:
            train_dataloader = trainer.datamodule.train_dataloader()
        except (AttributeError, TypeError):
            # Fallback to trainer.train_dataloader if datamodule is not available
            train_dataloader = trainer.train_dataloader
            
        if train_dataloader is None:
            LOGGER.warning("No training dataloader available")
            return
        
        # Use a subset of the training dataloader to avoid processing too many bags
        with torch.no_grad():
            for batch in train_dataloader:
                # PyTorch Lightning's current_epoch is 0-indexed, but we want to display it as 0-indexed
                # to match user expectations
                self._process_batch_attention(pl_module, batch, "train", trainer.current_epoch)
                
                # Increment the counter - tuple format: (bags, labels, bag_ids, padding_mask)
                bag_ids = batch[2]
                self._train_bags_processed += len(bag_ids)
                
                # Stop if we've processed enough bags
                if self._train_bags_processed >= self.max_bags:
                    break
        
        # Set the model back to training mode
        pl_module.train()
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log attention weights for validation bags at the end of each epoch.
        
        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer.
        pl_module : pl.LightningModule
            The PyTorch Lightning module.
        """
        # Skip if we've already processed too many bags
        if hasattr(self, '_val_bags_processed') and self._val_bags_processed >= self.max_bags:
            return
        
        # Set the model to evaluation mode
        pl_module.eval()
        
        # Reset the counter for the number of bags processed
        self._val_bags_processed = 0
        
        # Process each batch in the validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            LOGGER.warning("No validation dataloader available")
            return
        
        # Handle the case where val_dataloaders is a list
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]
        
        # Use a subset of the validation dataloader to avoid processing too many bags
        with torch.no_grad():
            for batch in val_dataloader:
                # PyTorch Lightning's current_epoch is 0-indexed, but we want to display it as 0-indexed
                # to match user expectations
                self._process_batch_attention(pl_module, batch, "val", trainer.current_epoch)
                
                # Increment the counter - tuple format: (bags, labels, bag_ids, padding_mask)
                bag_ids = batch[2]
                self._val_bags_processed += len(bag_ids)
                
                # Stop if we've processed enough bags
                if self._val_bags_processed >= self.max_bags:
                    break
        
        # Set the model back to training mode
        pl_module.train()
