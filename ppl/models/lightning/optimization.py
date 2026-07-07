"""Optimization utilities for MIL model."""

import logging
import traceback
import torch.nn as nn
import torch.optim as optim
from ppl.config.trainer_optim_config import TrainerOptimConfig

LOGGER = logging.getLogger(__name__)

class OptimizationMethods(nn.Module):
    """Optimization methods mixin for MIL model."""

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        This method supports multiple optimizers and schedulers:
        - AdamW (default)
        - SGD with momentum
        - RMSprop

        And multiple schedulers:
        - None (constant learning rate)
        - Cosine annealing
        - Reduce on plateau (default)
        - One-cycle policy
        - Step decay

        Returns
        -------
        dict or Optimizer
            Optimizer configuration for PyTorch Lightning
        """
        self._log_memory_usage("configure_optimizers")

        try:
            # Extract optimizer config from hparams
            cfg = TrainerOptimConfig(
                **{k: v for k, v in self.hparams.items() if k in TrainerOptimConfig.__annotations__}
            )

            # Get optimizer type from config or use default
            optimizer_type = getattr(cfg, "optimizer_type", "adamw").lower()

            # Always use different learning rates for embedder, aggregator, and predictor
            # Get the base learning rate
            base_lr = cfg.lr

            # Create parameter groups with different learning rates
            # Separate parameters into weights and biases/norms for selective weight decay
            embedder_module = self.core.embedder
            self_attention_module = getattr(embedder_module, "context_blocks", None)
            local_embedder_module = getattr(embedder_module, "local_embedder", None)
            if self_attention_module is not None and local_embedder_module is not None:
                embedder_module = local_embedder_module
            else:
                self_attention_module = None

            embedder_weights, embedder_biases_norms = self._split_params_for_weight_decay(embedder_module)
            if self_attention_module is not None:
                self_attention_weights, self_attention_biases_norms = self._split_params_for_weight_decay(
                    self_attention_module
                )
            else:
                self_attention_weights, self_attention_biases_norms = [], []
            aggregator_weights, aggregator_biases_norms = self._split_params_for_weight_decay(self.core.aggregator)
            predictor_weights, predictor_biases_norms = self._split_params_for_weight_decay(self.core.predictor)
            active_query_builder = getattr(self.core, "active_query_builder", None)
            if active_query_builder is not None:
                active_query_weights, active_query_biases_norms = self._split_params_for_weight_decay(
                    active_query_builder
                )
            else:
                active_query_weights, active_query_biases_norms = [], []

            # Get learning rate factors. These remain as a backward-compatible
            # fallback when absolute per-component LRs are not configured.
            embedder_lr_factor = getattr(cfg, "embedder_lr_factor", 0.1)  # Default to 10x smaller
            self_attention_lr_factor = getattr(cfg, "self_attention_lr_factor", embedder_lr_factor)
            aggregator_lr_factor = getattr(cfg, "aggregator_lr_factor", 1.0)
            predictor_lr_factor = getattr(cfg, "predictor_lr_factor", 1.0)
            active_query_lr_factor = getattr(cfg, "active_query_lr_factor", aggregator_lr_factor)

            def component_lr(absolute_lr, lr_factor: float) -> float:
                if absolute_lr is not None:
                    return float(absolute_lr)
                return float(base_lr * lr_factor)

            embedder_lr = component_lr(getattr(cfg, "embedder_lr", None), embedder_lr_factor)
            self_attention_lr = component_lr(
                getattr(cfg, "self_attention_lr", None),
                self_attention_lr_factor,
            )
            aggregator_lr = component_lr(
                getattr(cfg, "aggregator_lr", None),
                aggregator_lr_factor,
            )
            predictor_lr = component_lr(getattr(cfg, "predictor_lr", None), predictor_lr_factor)
            active_query_lr = component_lr(
                getattr(cfg, "active_query_lr", None),
                active_query_lr_factor,
            )

            param_groups = []

            def add_param_groups(
                name: str,
                weights,
                biases_norms,
                lr: float,
            ) -> None:
                if weights:
                    param_groups.append(
                        {
                            "params": list(weights),
                            "lr": lr,
                            "weight_decay": cfg.weight_decay,
                            "name": f"{name}.decay",
                        }
                    )
                if biases_norms:
                    param_groups.append(
                        {
                            "params": list(biases_norms),
                            "lr": lr,
                            "weight_decay": 0.0,
                            "name": f"{name}.no_decay",
                        }
                    )
                LOGGER.info(
                    "%s optimizer groups: decay_params=%d no_decay_params=%d lr=%s",
                    name,
                    sum(p.numel() for p in weights),
                    sum(p.numel() for p in biases_norms),
                    lr,
                )

            # Create component-level parameter groups with independent learning-rate
            # factors while keeping biases and normalization parameters out of
            # weight decay.
            add_param_groups(
                "Embedder",
                embedder_weights,
                embedder_biases_norms,
                embedder_lr,
            )
            add_param_groups(
                "Aggregator",
                aggregator_weights,
                aggregator_biases_norms,
                aggregator_lr,
            )
            add_param_groups(
                "Predictor",
                predictor_weights,
                predictor_biases_norms,
                predictor_lr,
            )
            if self_attention_weights or self_attention_biases_norms:
                add_param_groups(
                    "Self-attention/contextualizer",
                    self_attention_weights,
                    self_attention_biases_norms,
                    self_attention_lr,
                )
            if active_query_weights or active_query_biases_norms:
                add_param_groups(
                    "Active query builder",
                    active_query_weights,
                    active_query_biases_norms,
                    active_query_lr,
                )

            LOGGER.info(f"Using component-specific learning rates with base LR={base_lr}")
            LOGGER.info(f"Embedder LR: {embedder_lr}")
            if self_attention_weights or self_attention_biases_norms:
                LOGGER.info(
                    "Self-attention/contextualizer LR: %s (%d trainable parameters)",
                    self_attention_lr,
                    sum(p.numel() for p in self_attention_weights)
                    + sum(p.numel() for p in self_attention_biases_norms),
                )
            LOGGER.info(f"Aggregator LR: {aggregator_lr}")
            LOGGER.info(f"Predictor LR: {predictor_lr}")
            if active_query_weights or active_query_biases_norms:
                LOGGER.info(
                    "Active query builder LR: %s (%d trainable parameters)",
                    active_query_lr,
                    sum(p.numel() for p in active_query_weights)
                    + sum(p.numel() for p in active_query_biases_norms),
                )
            LOGGER.info(f"Weight decay: {cfg.weight_decay}")

            # Create optimizer based on type
            if optimizer_type == "sgd":
                momentum = getattr(cfg, "momentum", 0.9)
                nesterov = getattr(cfg, "nesterov", True)
                opt = optim.SGD(
                    param_groups, 
                    lr=base_lr,  # This will be overridden by param_groups
                    momentum=momentum,
                    nesterov=nesterov
                )
                LOGGER.info(f"Using SGD optimizer with base_lr={base_lr}, momentum={momentum}, nesterov={nesterov}")
            elif optimizer_type == "rmsprop":
                alpha = getattr(cfg, "alpha", 0.99)
                opt = optim.RMSprop(
                    param_groups, 
                    lr=base_lr,  # This will be overridden by param_groups
                    alpha=alpha
                )
                LOGGER.info(f"Using RMSprop optimizer with base_lr={base_lr}, alpha={alpha}")
            else:  # default to AdamW
                beta1 = getattr(cfg, "beta1", 0.9)
                beta2 = getattr(cfg, "beta2", 0.999)
                eps = getattr(cfg, "eps", 1e-8)
                opt = optim.AdamW(
                    param_groups, 
                    lr=base_lr,  # This will be overridden by param_groups
                    betas=(beta1, beta2),
                    eps=eps  # Add epsilon parameter for numerical stability
                )
                LOGGER.info(f"Using AdamW optimizer with base_lr={base_lr}, betas=({beta1}, {beta2}), eps={eps}")

            # Return optimizer if no scheduler is requested
            if cfg.scheduler == "none":
                LOGGER.info("Using constant learning rate (no scheduler)")
                return opt

            # Configure scheduler based on type
            if cfg.scheduler == "cosine":
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    opt, 
                    T_max=cfg.lr_t_max, 
                    eta_min=float(cfg.eta_min)
                )
                LOGGER.info(f"Using cosine annealing scheduler with T_max={cfg.lr_t_max}, eta_min={cfg.eta_min}")
                return {"optimizer": opt, "lr_scheduler": sched}

            elif cfg.scheduler == "one_cycle":
                # One-cycle policy for super-convergence
                max_lr = getattr(cfg, "max_lr", cfg.lr * 10)
                steps_per_epoch = getattr(cfg, "steps_per_epoch", 100)
                epochs = getattr(cfg, "max_epochs", 100)
                total_steps = steps_per_epoch * epochs

                sched = optim.lr_scheduler.OneCycleLR(
                    opt,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    pct_start=0.3,
                    div_factor=25.0,
                    final_div_factor=10000.0
                )
                LOGGER.info(f"Using one-cycle scheduler with max_lr={max_lr}, total_steps={total_steps}")
                return {"optimizer": opt, "lr_scheduler": sched}

            elif cfg.scheduler == "step":
                # Step decay
                step_size = getattr(cfg, "step_size", 30)
                gamma = getattr(cfg, "gamma", 0.1)

                sched = optim.lr_scheduler.StepLR(
                    opt,
                    step_size=step_size,
                    gamma=gamma
                )
                LOGGER.info(f"Using step decay scheduler with step_size={step_size}, gamma={gamma}")
                return {"optimizer": opt, "lr_scheduler": sched}

            else:  # default to plateau
                # Reduce on plateau
                patience = getattr(cfg, "lr_patience", 10)
                factor = getattr(cfg, "factor", 0.01)
                min_lr = getattr(cfg, "min_lr", 1e-6)

                sched = optim.lr_scheduler.ReduceLROnPlateau(
                    opt, 
                    mode="min", 
                    patience=patience,
                    factor=factor,
                    min_lr=min_lr
                )
                
                # Determine which metric to monitor based on validation data availability
                # Check if validation data is available at runtime through the trainer
                has_val_data = False
                try:
                    if hasattr(self, 'trainer') and self.trainer is not None:
                        # Check if trainer has a datamodule with validation data
                        if hasattr(self.trainer, 'datamodule') and self.trainer.datamodule is not None:
                            val_dataloader = self.trainer.datamodule.val_dataloader()
                            has_val_data = val_dataloader is not None and len(val_dataloader) > 0
                except:
                    # Fallback to configuration-based detection
                    val_partition = self.hparams.get("val_partition", getattr(cfg, "val_partition", True))
                    has_val_data = val_partition
                
                if has_val_data:
                    monitor_metric = (
                        "val_rmse"
                        if str(getattr(self, "task", "")).lower() == "regression"
                        else "val_loss"
                    )
                    LOGGER.info(
                        "Using reduce on plateau scheduler monitoring %s "
                        "(validation data available)",
                        monitor_metric,
                    )
                else:
                    monitor_metric = "train_loss_epoch" 
                    LOGGER.info(f"Using reduce on plateau scheduler monitoring train_loss_epoch (no validation data)")
                
                LOGGER.info(f"Scheduler config: patience={patience}, factor={factor}, monitor={monitor_metric}")
                return {
                    "optimizer": opt, 
                    "lr_scheduler": {
                        "scheduler": sched, 
                        "monitor": monitor_metric,
                        "interval": "epoch",
                        "frequency": 1
                    }
                }

        except Exception as e:
            LOGGER.error(f"Error configuring optimizer: {e}")
            LOGGER.error(traceback.format_exc())

            # Fallback to a simple AdamW optimizer with default settings
            LOGGER.warning("Using fallback AdamW optimizer with default settings")
            return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
