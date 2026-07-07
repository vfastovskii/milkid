from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

LOGGER = logging.getLogger(__name__)


class EpochEmbeddingLogger(pl.Callback):
    """
    Logs embeddings from both embedder and aggregator (attention net) for validation bags at each epoch.
    
    This callback extracts and saves embeddings from the embedder component and attention-based
    aggregator to CSV files after each validation epoch during stage 1 training.
    
    Parameters
    ----------
    save_dir : str
        Directory to save the embedding CSV files.
    max_bags : int, default 1000
        Maximum number of bags to process per epoch.
    conf_ids : Dict[str, list], optional
        Dictionary mapping bag IDs to lists of conformer IDs. If provided, these will be used
        as instance IDs in the CSV files instead of simple indices.
    max_epochs : int, default 100
        Maximum number of epochs to log (to avoid unexpected extra epochs).
    """
    
    def __init__(
        self, 
        save_dir: str, 
        max_bags: int = 1000,
        conf_ids: Optional[Dict[str, List[str]]] = None,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_dir = save_dir
        self.max_bags = max_bags
        self.conf_ids = conf_ids
        self.max_epochs = max_epochs
    
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
    
    def _extract_embeddings(self, model: pl.LightningModule, x):
        """
        Extract embeddings from both embedder and aggregator components.
        
        Parameters
        ----------
        model : pl.LightningModule
            The model to extract embeddings from.
        x : torch.Tensor or dict
            The input tensor (old format) or dictionary (new format).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing embeddings from different components.
        """
        embeddings = {}
        
        # Handle different input formats
        if isinstance(x, dict):
            # New dictionary format - pass directly to model.core
            # MILCore knows how to handle dictionary format
            with torch.no_grad():
                pred_values, extras = model.core(x)
            
            # For dictionary format, we need to extract embeddings differently
            # The embedder output is part of the forward pass, we can't call it separately
            # We'll reconstruct embeddings from the forward pass information
            
            # Extract embedder outputs by processing each bag individually
            if 'bags' in x and hasattr(model.core, 'embedder'):
                try:
                    embedded_bags = []
                    for bag in x['bags']:
                        with torch.no_grad():
                            bag_embedding = model.core.embedder(bag)  # [N_i, D_embed]
                            embedded_bags.append(bag_embedding)
                    
                    # Pad embeddings to create batch format [B, N_max, D_embed]
                    if embedded_bags:
                        max_instances = x.get('max_instances', max(bag.shape[0] for bag in embedded_bags))
                        embed_dim = embedded_bags[0].shape[-1]
                        batch_size = len(embedded_bags)
                        
                        # Create padded tensor
                        padded_embeddings = torch.zeros(batch_size, max_instances, embed_dim, 
                                                       dtype=embedded_bags[0].dtype, 
                                                       device=embedded_bags[0].device)
                        
                        # Fill with actual embeddings
                        for i, emb in enumerate(embedded_bags):
                            seq_len = emb.shape[0]
                            padded_embeddings[i, :seq_len, :] = emb
                        
                        embeddings['embedder'] = padded_embeddings
                except Exception as e:
                    LOGGER.debug(f"Could not extract embedder outputs for dictionary format: {e}")
                    embeddings['embedder'] = None
        else:
            # Old tensor format - use existing logic
            # Extract embedder embeddings
            if hasattr(model.core, 'embedder'):
                with torch.no_grad():
                    embedder_output = model.core.embedder(x)
                    embeddings['embedder'] = embedder_output
            
            # Extract full forward pass to get aggregator embeddings
            with torch.no_grad():
                pred_values, extras = model.core(x)
            
        # If no attention info in extras, try to get it directly from aggregator
        if 'attn' not in extras and 'alpha' not in extras:
            if hasattr(model.core, 'aggregator'):
                # Handle different input formats for aggregator
                if isinstance(x, dict):
                    # For dictionary format, we can't easily extract intermediate embeddings
                    # The attention info should be available from the forward pass extras
                    # If not available, skip aggregator-specific extraction
                    LOGGER.debug("No attention info available from dictionary format forward pass")
                else:
                    # Old tensor format - get embedder output first
                    if hasattr(model.core, 'embedder'):
                        h = model.core.embedder(x)
                    else:
                        h = x
                    # Call aggregator with return_attn=True
                    with torch.no_grad():
                        _, agg_extras = model.core.aggregator(h, return_attn=True)
                        extras.update(agg_extras)
            
        # Extract final instance weights as embeddings. Prefer alpha_final/alpha
        # over raw attention maps so saved values match the prediction
        # representation.
        alpha_key = "alpha_final" if "alpha_final" in extras else "alpha"
        if alpha_key in extras:
            alpha = extras[alpha_key]
            
            # Handle different alpha shapes appropriately
            if len(alpha.shape) > 1:
                # For batched alpha [B, N], keep it as is for proper per-bag processing
                # The _process_batch_embeddings method will handle extraction per bag
                LOGGER.debug(f"Alpha from extras has shape {alpha.shape}, keeping batch dimension")
                embeddings['aggregator_attention'] = alpha
            else:
                # Single bag alpha [N], use as is
                embeddings['aggregator_attention'] = alpha
            
        elif 'attn' in extras:
            attn = extras['attn']
            
            # Handle VIT aggregator attention output
            # VIT aggregator returns either [B, H, 1, N] for CLS->instances or [B, H, N, N] for full attention
            if len(attn.shape) == 4:
                # Remove batch dimension for single bag
                if attn.shape[0] == 1:
                    attn = attn.squeeze(0)  # [H, 1, N] or [H, N, N]
                
                # For CLS->instances case [H, 1, N], remove singleton dimension
                if attn.shape[1] == 1:
                    attn = attn.squeeze(1)  # [H, N]
                    # Average over heads to get final attention weights
                    alpha = attn.mean(dim=0)  # [N]
                else:
                    # For full attention [H, N, N], average over heads then queries
                    attn = attn.mean(dim=0)  # [N, N]
                    alpha = attn.mean(dim=0)  # [N] - average attention received by each instance
            elif len(attn.shape) == 3:
                # [H, N] or [H, N, N]
                if attn.shape[1] == attn.shape[2]:
                    # [H, N, N] case
                    attn = attn.mean(dim=0)  # [N, N]
                    alpha = attn.mean(dim=0)  # [N]
                else:
                    # [H, N] case
                    alpha = attn.mean(dim=0)  # [N]
            elif len(attn.shape) == 2:
                # [H, N] case - average over heads
                alpha = attn.mean(dim=0)  # [N]
            else:
                LOGGER.warning(f"Unexpected attention shape: {attn.shape}, using as is")
                alpha = attn.flatten()
            
            embeddings['aggregator_attention'] = alpha
            
        return embeddings, pred_values
    
    def _save_embeddings_csv(
        self,
        embeddings: Dict[str, torch.Tensor],
        true_value: float,
        pred_value: float,
        bag_id: str,
        inst_ids: List[str],
        save_dir: Path,
        epoch: int,
    ) -> None:
        """
        Save embeddings to CSV files with proper naming and organization.
        
        Parameters
        ----------
        embeddings : Dict[str, torch.Tensor]
            Dictionary of embeddings from different components.
        true_value : float
            True value of the bag.
        pred_value : float
            Predicted value of the bag.
        bag_id : str
            Identifier of the bag.
        inst_ids : List[str]
            Per-instance identifiers.
        save_dir : Path
            Directory to save the CSV files.
        epoch : int
            Current epoch number.
        """
        # Convert tensor values to Python numbers if needed
        if isinstance(true_value, torch.Tensor):
            true_value = true_value.item()
        if isinstance(pred_value, torch.Tensor):
            pred_value = pred_value.item()
        
        # Save embedder embeddings
        if 'embedder' in embeddings:
            embedder_emb = embeddings['embedder'].cpu().numpy()
            
            # Handle different embedder output shapes
            if len(embedder_emb.shape) == 3:  # [B, N, D]
                embedder_emb = embedder_emb[0]  # Take first batch element [N, D]
            elif len(embedder_emb.shape) == 2:  # [N, D]
                pass  # Already correct shape
            else:
                LOGGER.warning(f"Unexpected embedder output shape: {embedder_emb.shape}")
                return
            
            # Ensure inst_ids match embedding dimensions
            if len(inst_ids) != embedder_emb.shape[0]:
                LOGGER.warning(
                    f"Instance IDs length ({len(inst_ids)}) doesn't match embedder output "
                    f"({embedder_emb.shape[0]}) for bag {bag_id}. Using indices instead."
                )
                inst_ids_to_use = [str(i) for i in range(embedder_emb.shape[0])]
            else:
                inst_ids_to_use = [str(i) for i in inst_ids]
            
            # Create DataFrame for embedder embeddings
            embedder_data = {
                'instance_id': inst_ids_to_use,
                'bag_id': [bag_id] * len(inst_ids_to_use),
                'true_value': [true_value] * len(inst_ids_to_use),
                'predicted_value': [pred_value] * len(inst_ids_to_use),
                'epoch': [epoch] * len(inst_ids_to_use)
            }
            
            # Add embedding dimensions as columns
            for dim_idx in range(embedder_emb.shape[1]):
                embedder_data[f'embedding_dim_{dim_idx}'] = embedder_emb[:, dim_idx]
            
            # Save embedder CSV
            embedder_csv_path = save_dir / f"{bag_id}_embedder.csv"
            pd.DataFrame(embedder_data).to_csv(embedder_csv_path, index=False)
            LOGGER.debug(f"Saved embedder data to {embedder_csv_path}")
        
        # Save aggregator attention embeddings
        if 'aggregator_attention' in embeddings:
            attention_emb = embeddings['aggregator_attention'].cpu().numpy()

            # Normalize attention to ensure non-negative and sum to 1
            att = attention_emb.astype(float).flatten()
            if att.size > 0:
                att[~np.isfinite(att)] = 0.0
                if not np.allclose(att, 0.0):
                    s = float(att.sum())
                    if np.any(att < 0.0) or s <= 0.0:
                        m = float(np.max(att))
                        exp_a = np.exp(att - m)
                        denom = float(exp_a.sum())
                        att = exp_a / denom if denom > 0.0 else np.zeros_like(att)
                    else:
                        att = att / s
                    att = np.clip(att, 0.0, 1.0)
                    tot = att.sum()
                    if tot > 0:
                        att = att / tot
            # Final safety: round and re-normalize to eliminate tiny drifts
            att = np.round(att.astype(np.float64), 12)
            att = np.clip(att, 0.0, 1.0)
            tot = float(att.sum())
            if tot > 0.0:
                att = att / tot
            attention_emb = att
            
            # Ensure inst_ids match attention dimensions
            if len(inst_ids) != len(attention_emb):
                LOGGER.warning(
                    f"Instance IDs length ({len(inst_ids)}) doesn't match attention output "
                    f"({len(attention_emb)}) for bag {bag_id}. Using indices instead."
                )
                inst_ids_to_use = [str(i) for i in range(len(attention_emb))]
            else:
                inst_ids_to_use = [str(i) for i in inst_ids]
            
            # Create DataFrame for aggregator attention embeddings
            attention_data = {
                'instance_id': inst_ids_to_use,
                'bag_id': [bag_id] * len(inst_ids_to_use),
                'true_value': [true_value] * len(inst_ids_to_use),
                'predicted_value': [pred_value] * len(inst_ids_to_use),
                'epoch': [epoch] * len(inst_ids_to_use),
                'attention_weight': attention_emb
            }
            
            # Save aggregator CSV
            aggregator_csv_path = save_dir / f"{bag_id}_aggregator.csv"
            pd.DataFrame(attention_data).to_csv(aggregator_csv_path, index=False)
            LOGGER.debug(f"Saved aggregator data to {aggregator_csv_path}")
    
    def _process_batch_embeddings(
        self,
        model: pl.LightningModule,
        batch: Any,
        epoch: int,
    ) -> None:
        """
        Process a batch and log embeddings.
        
        Parameters
        ----------
        model : pl.LightningModule
            The model to extract embeddings from.
        batch : Any
            The batch of data.
        epoch : int
            The current epoch.
        """
        # Ensure epoch is within expected range
        if epoch > self.max_epochs - 1:
            LOGGER.warning(f"Skipping unexpected epoch {epoch} (should be 0-{self.max_epochs-1} for a {self.max_epochs}-epoch run)")
            return
        
        # Unpack the batch. Extra fields such as cluster IDs and series labels
        # are ignored here because this logger only extracts embeddings.
        if not isinstance(batch, (list, tuple)) or len(batch) < 4:
            raise ValueError(
                "Expected embedding logging batch with at least 4 elements, "
                f"got {len(batch) if hasattr(batch, '__len__') else 'NA'}"
            )
        bags, true_values, bag_ids, key_padding_mask = batch[:4]
        
        # Get the device
        device = next(model.parameters()).device
        true_values = true_values.to(device) if isinstance(true_values, torch.Tensor) else true_values
        
        # Create the directory for this epoch
        epoch_dir = Path(self.save_dir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        
        # Process pre-padded tensor batch [B, N, D]
        x = bags.to(device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device)
        
        # Extract embeddings and predictions for the entire batch
        batch_embeddings, batch_pred_values = self._extract_embeddings(model, x)
        
        if not batch_embeddings:
            LOGGER.warning("No embeddings found in the batch")
            return
        
        # Process each bag in the batch
        for i, bag_id in enumerate(bag_ids):
            try:
                # Extract per-bag embeddings and handle masking
                bag_embeddings = {}
                
                if 'embedder' in batch_embeddings:
                    embedder_emb = batch_embeddings['embedder']
                    if len(embedder_emb.shape) == 3:  # [B, N, D]
                        bag_emb = embedder_emb[i]  # [N, D] for this bag
                        
                        # Apply mask if available to remove padding
                        if key_padding_mask is not None:
                            mask = ~key_padding_mask[i]  # True for real data
                            bag_emb = bag_emb[mask]  # [actual_N, D]
                        
                        bag_embeddings['embedder'] = bag_emb
                    elif len(embedder_emb.shape) == 2:  # [N, D] - single bag
                        bag_embeddings['embedder'] = embedder_emb
                
                if 'aggregator_attention' in batch_embeddings:
                    attention_emb = batch_embeddings['aggregator_attention']
                    if len(attention_emb.shape) == 2:  # [B, N]
                        bag_attention = attention_emb[i]  # [N] for this bag
                        
                        # Apply mask if available to remove padding
                        if key_padding_mask is not None:
                            mask = ~key_padding_mask[i]  # True for real data
                            bag_attention = bag_attention[mask]  # [actual_N]
                        
                        bag_embeddings['aggregator_attention'] = bag_attention
                    elif len(attention_emb.shape) == 1:  # [N] - single bag
                        bag_embeddings['aggregator_attention'] = attention_emb
                
                # Generate instance IDs for this bag
                if self.conf_ids is not None and bag_id in self.conf_ids:
                    inst_ids = self.conf_ids[bag_id]
                    # For padded batches, we need to handle the case where conf_ids 
                    # might not match the actual (unpadded) length
                    if 'embedder' in bag_embeddings:
                        actual_length = bag_embeddings['embedder'].shape[0]
                        if len(inst_ids) != actual_length:
                            LOGGER.warning(
                                f"Instance IDs length ({len(inst_ids)}) doesn't match actual bag length "
                                f"({actual_length}) for bag {bag_id}. Using indices instead."
                            )
                            inst_ids = list(range(actual_length))
                        else:
                            # Take only the first actual_length IDs (remove padding)
                            inst_ids = inst_ids[:actual_length]
                else:
                    # Determine number of instances from embeddings
                    if 'embedder' in bag_embeddings:
                        num_instances = bag_embeddings['embedder'].shape[0]
                    elif 'aggregator_attention' in bag_embeddings:
                        num_instances = bag_embeddings['aggregator_attention'].shape[0]
                    else:
                        LOGGER.warning(f"Could not determine number of instances for bag {bag_id}")
                        continue
                    
                    inst_ids = list(range(num_instances))
                
                # Get true and predicted values for this bag
                true_val = self._safe_extract_value(true_values, i)
                pred_val = self._safe_extract_value(batch_pred_values, i)
                
                # Save embeddings to CSV files
                self._save_embeddings_csv(
                    embeddings=bag_embeddings,
                    true_value=true_val,
                    pred_value=pred_val,
                    bag_id=bag_id,
                    inst_ids=inst_ids,
                    save_dir=epoch_dir,
                    epoch=epoch,
                )
            except Exception as e:
                LOGGER.error(f"Error processing embeddings for bag {bag_id}: {e}")
                import traceback
                LOGGER.debug(traceback.format_exc())
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log embeddings for validation bags at the end of each epoch.
        
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
        
        LOGGER.info(f"Extracting embeddings at epoch {trainer.current_epoch} for up to {self.max_bags} bags")
        
        # Use a subset of the validation dataloader to avoid processing too many bags
        with torch.no_grad():
            for batch in val_dataloader:
                self._process_batch_embeddings(pl_module, batch, trainer.current_epoch)
                
                # Increment the counter - tuple format: (bags, labels, bag_ids, padding_mask)
                bag_ids = batch[2]
                self._val_bags_processed += len(bag_ids)
                
                # Stop if we've processed enough bags
                if self._val_bags_processed >= self.max_bags:
                    break
        
        LOGGER.info(f"Finished extracting embeddings at epoch {trainer.current_epoch}, processed {self._val_bags_processed} bags")
        
        # Set the model back to training mode
        pl_module.train()
