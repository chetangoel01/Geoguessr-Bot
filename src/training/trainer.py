"""
Training loop for GeoGuessr model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path
from typing import Dict, Optional, Callable
from tqdm import tqdm
import json
import time

from .losses import CombinedLoss
from .metrics import MetricTracker, compute_classification_metrics, compute_gps_metrics
from ..models.geoguessr_model import GeoGuessrModel
from ..config import Config


class Trainer:
    """
    Trainer for GeoGuessr model.
    
    Handles:
    - Training loop with gradient accumulation
    - Validation
    - Checkpointing
    - Learning rate scheduling
    - Mixed precision training
    """
    
    def __init__(
        self,
        model: GeoGuessrModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: CombinedLoss,
        config: Config,
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.device = device
        
        # Move loss function buffers to device
        if hasattr(self.loss_fn.classification_loss, 'soft_label_matrix'):
            self.loss_fn.classification_loss.soft_label_matrix = \
                self.loss_fn.classification_loss.soft_label_matrix.to(device)
        
        # Optimizer
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler('cuda') if config.training.use_amp else None
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_score': [],
            'learning_rates': []
        }
        
        # Metric tracker
        self.metric_tracker = MetricTracker()
    
    def setup_optimizer(self, phase: str = "frozen"):
        """
        Setup optimizer and scheduler for training phase.
        
        Args:
            phase: "frozen" (train heads only) or "finetune" (train everything)
        """
        if phase == "frozen":
            # Only train heads
            self.model.freeze_backbone()
            param_groups = [
                {'params': self.model.fusion.parameters(), 'lr': self.config.training.head_lr},
                {'params': self.model.classification_head.parameters(), 'lr': self.config.training.head_lr},
                {'params': self.model.gps_head.parameters(), 'lr': self.config.training.head_lr}
            ]
            num_epochs = self.config.training.num_epochs_frozen
        else:
            # Train everything with differential learning rates
            self.model.unfreeze_backbone()
            param_groups = self.model.get_optimizer_param_groups(
                backbone_lr=self.config.training.backbone_lr,
                head_lr=self.config.training.head_lr,
                weight_decay=self.config.training.weight_decay
            )
            num_epochs = self.config.training.num_epochs_finetune
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler
        # Account for gradient accumulation: scheduler steps once per accumulation cycle
        num_training_steps = (len(self.train_loader) * num_epochs) // self.config.training.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=1e-7
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps]
        )
        
        print(f"Optimizer setup for {phase} phase:")
        print(f"  Total steps: {num_training_steps}")
        print(f"  Warmup steps: {num_warmup_steps}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_gps_loss = 0.0
        num_batches = 0
        
        print(f"Starting epoch {self.current_epoch} - loading first batch...")
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        self.optimizer.zero_grad()
        
        first_batch_start = time.time()
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 0:
                first_batch_time = time.time() - first_batch_start
                print(f"First batch loaded in {first_batch_time:.1f} seconds ({first_batch_time/60:.1f} minutes)")
                print("Starting forward pass...")
            # Move to device with non_blocking for faster transfers
            images = batch['images'].to(self.device, non_blocking=True)
            state_labels = batch['state_label'].to(self.device, non_blocking=True)
            gps_targets = batch['gps'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision (bfloat16 is more stable on A100)
            if self.config.training.use_amp:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(images)
                    losses = self.loss_fn(
                        outputs['class_logits'],
                        outputs['gps_coords'],
                        state_labels,
                        gps_targets
                    )
                    loss = losses['loss'] / self.config.training.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                losses = self.loss_fn(
                    outputs['class_logits'],
                    outputs['gps_coords'],
                    state_labels,
                    gps_targets
                )
                loss = losses['loss'] / self.config.training.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                if self.config.training.use_amp:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.config.training.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track losses
            total_loss += losses['loss'].item()
            total_cls_loss += losses['cls_loss'].item()
            total_gps_loss += losses['gps_loss'].item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'cls': f"{losses['cls_loss'].item():.4f}",
                'gps': f"{losses['gps_loss'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_cls_loss': total_cls_loss / num_batches,
            'train_gps_loss': total_gps_loss / num_batches,
            'learning_rate': current_lr
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.metric_tracker.reset()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            images = batch['images'].to(self.device, non_blocking=True)
            state_labels = batch['state_label'].to(self.device, non_blocking=True)
            gps_targets = batch['gps'].to(self.device, non_blocking=True)
            
            # Forward pass
            if self.config.training.use_amp:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(images)
                    losses = self.loss_fn(
                        outputs['class_logits'],
                        outputs['gps_coords'],
                        state_labels,
                        gps_targets
                    )
            else:
                outputs = self.model(images)
                losses = self.loss_fn(
                    outputs['class_logits'],
                    outputs['gps_coords'],
                    state_labels,
                    gps_targets
                )
            
            total_loss += losses['loss'].item()
            num_batches += 1
            
            # Update metric tracker
            self.metric_tracker.update(
                outputs['class_logits'],
                outputs['gps_coords'],
                state_labels,
                gps_targets
            )
        
        # Compute final metrics
        metrics = self.metric_tracker.compute()
        metrics['val_loss'] = total_loss / num_batches
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        phase: str = "frozen",
        save_dir: Optional[Path] = None
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            phase: "frozen" or "finetune"
            save_dir: Directory to save checkpoints
        
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Starting {phase} training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        self.setup_optimizer(phase)
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # SAVE CHECKPOINT IMMEDIATELY after validation (before any prints that could fail)
            # This ensures we don't lose progress if there's a bug in logging
            is_best = val_metrics['final_score'] > self.best_score
            if is_best:
                self.best_score = val_metrics['final_score']
                patience_counter = 0
                
                if save_dir is not None:
                    self.save_checkpoint(
                        save_dir / f"best_{phase}.pt",
                        val_metrics
                    )
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if save_dir is not None and epoch % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(
                    save_dir / f"epoch_{self.current_epoch}_{phase}.pt",
                    val_metrics
                )
            
            # Now log results (safe to fail after checkpoints are saved)
            print(f"\nEpoch {self.current_epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(cls: {train_metrics['train_cls_loss']:.4f}, "
                  f"gps: {train_metrics['train_gps_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Score: {val_metrics['final_score']:.4f} "
                  f"(cls: {val_metrics['classification_score']:.4f}, "
                  f"gps: {val_metrics['gps_score']:.4f})")
            print(f"  GPS Mean Distance: {val_metrics['gps_mean_distance_km']:.1f} km")
            
            if is_best:
                print(f"  New best score! Saved checkpoint.")
            
            # Update history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_score'].append(val_metrics['final_score'])
            self.history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Early stopping
            if patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping after {patience_counter} epochs without improvement")
                break
        
        return self.history
    
    def save_checkpoint(self, path: Path, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_score': self.best_score,
            'metrics': metrics,
            'config': {
                'model': vars(self.config.model),
                'training': vars(self.config.training)
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best score: {self.best_score:.4f}")
        
        return checkpoint.get('metrics', {})
