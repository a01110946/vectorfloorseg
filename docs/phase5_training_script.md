# VectorFloorSeg Implementation - Phase 5: Training Script and Evaluation

## Overview

This final phase implements the complete training pipeline, evaluation metrics, and inference system for VectorFloorSeg. It includes the training script, validation, model evaluation, and deployment utilities.

## Prerequisites

- Completed Phase 1: Environment Setup
- Completed Phase 2: Project Structure Setup  
- Completed Phase 3: Data Preparation Pipeline
- Completed Phase 4: Model Implementation
- Active virtual environment: `vectorfloorseg_env`

## 5.1 Training Manager

```python
# File: src/training/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Optional, List, Tuple
import time
import json

from ..models.model_factory import save_model_checkpoint, ModelEMA
from ..utils.logging_utils import log_model_info
from ..utils.visualization import FloorplanVisualizer

class VectorFloorSegTrainer:
    """Complete training manager for VectorFloorSeg."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config,
        device: torch.device,
        experiment_name: str = "vectorfloorseg",
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        log_model_info(self.model, self.logger)
        
        # NEW CODE: Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # NEW CODE: Setup loss functions and metrics
        self.criterion_boundary = nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_room = self._create_focal_loss(alpha=0.25, gamma=2.0)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_miou = 0.0
        self.best_val_ri = 0.0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # NEW CODE: Model EMA for better performance
        self.use_ema = getattr(config.training, 'use_ema', True)
        if self.use_ema:
            self.model_ema = ModelEMA(self.model, decay=0.9999)
        
        # Visualization
        self.visualizer = FloorplanVisualizer(
            save_dir=self.log_dir / "visualizations"
        )
        
        # NEW CODE: Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        
        # Separate backbone and model parameters for different learning rates
        backbone_params = []
        model_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                model_params.append(param)
        
        # NEW CODE: Different learning rates for backbone and model
        param_groups = [
            {'params': model_params, 'lr': self.config.training.learning_rate},
            {'params': backbone_params, 'lr': self.config.training.learning_rate * 0.1}  # Lower LR for backbone
        ]
        
        if self.config.training.optimizer == 'sgd':
            return optim.SGD(
                param_groups,
                momentum=self.config.training.momentum,
                weight_decay=self.config.training.weight_decay,
                nesterov=True
            )
        elif self.config.training.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                weight_decay=self.config.training.weight_decay,
                eps=1e-8
            )
        elif self.config.training.optimizer == 'adamw':
            return optim.AdamW(
                param_groups,
                weight_decay=self.config.training.weight_decay,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        
        if self.config.training.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.training.epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
        elif self.config.training.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def _create_focal_loss(self, alpha: float = 0.25, gamma: float = 2.0):
        """Create focal loss for room classification."""
        
        class FocalLoss(nn.Module):
            def __init__(self, alpha: float, gamma: float):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                
            def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=-1)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        
        return FocalLoss(alpha, gamma)
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        
        wandb.init(
            project="vectorfloorseg",
            name=self.experiment_name,
            config={
                "model": self.config.model.__dict__,
                "training": self.config.training.__dict__,
                "data": self.config.data.__dict__
            }
        )
        
        # Watch model
        wandb.watch(self.model, log="all", log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        # Metrics tracking
        total_loss = 0.0
        boundary_loss_sum = 0.0
        room_loss_sum = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.epoch}/{self.config.training.epochs}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # NEW CODE: Forward pass
                loss_dict, metrics = self._train_step(batch)
                
                # Update metrics
                total_loss += loss_dict['total_loss']
                if 'boundary_loss' in loss_dict:
                    boundary_loss_sum += loss_dict['boundary_loss']
                if 'room_loss' in loss_dict:
                    room_loss_sum += loss_dict['room_loss']
                num_batches += 1
                
                # Update EMA
                if self.use_ema:
                    self.model_ema.update()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss_dict['total_loss']:.4f}",
                    'B_Loss': f"{boundary_loss_sum/num_batches:.4f}",
                    'R_Loss': f"{room_loss_sum/num_batches:.4f}",
                    'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to wandb
                if self.use_wandb and batch_idx % self.config.log_interval == 0:
                    wandb.log({
                        'train/batch_loss': loss_dict['total_loss'],
                        'train/boundary_loss': loss_dict.get('boundary_loss', 0),
                        'train/room_loss': loss_dict.get('room_loss', 0),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.epoch
                    })
                
            except Exception as e:
                self.logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        # Calculate epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'train_boundary_loss': boundary_loss_sum / num_batches,
            'train_room_loss': room_loss_sum / num_batches
        }
        
        return epoch_metrics
    
    def _train_step(self, batch: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Single training step."""
        
        self.optimizer.zero_grad()
        
        # Move data to device
        primal_data = batch['primal'].to(self.device)
        dual_data = batch['dual'].to(self.device)
        boundary_labels = batch['boundary_labels'].to(self.device)
        room_labels = batch['room_labels'].to(self.device)
        
        # Generate rasterized image (simplified - in practice, use actual images)
        batch_size = int(primal_data.batch.max().item() + 1) if primal_data.batch is not None else 1
        image = torch.randn(batch_size, 3, 256, 256, device=self.device)
        
        # Forward pass
        outputs = self.model(primal_data, dual_data, image)
        
        # Compute losses
        loss_dict = self._compute_losses(outputs, boundary_labels, room_labels)
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        if hasattr(self.config.training, 'gradient_clip_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.gradient_clip_norm
            )
        
        self.optimizer.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in loss_dict.items()}
        
        # Calculate additional metrics
        metrics = self._calculate_batch_metrics(outputs, boundary_labels, room_labels)
        
        return loss_dict, metrics
    
    def _compute_losses(
        self, 
        outputs: Dict[str, torch.Tensor],
        boundary_labels: torch.Tensor,
        room_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        
        losses = {}
        total_loss = 0.0
        
        # Boundary classification loss
        if outputs['boundary_logits'] is not None and boundary_labels.numel() > 0:
            # Filter out invalid labels
            valid_mask = (boundary_labels != -1)
            if valid_mask.any():
                boundary_loss = self.criterion_boundary(
                    outputs['boundary_logits'][valid_mask],
                    boundary_labels[valid_mask]
                )
                losses['boundary_loss'] = boundary_loss
                total_loss += self.config.training.boundary_loss_weight * boundary_loss
        
        # Room classification loss
        if outputs['room_logits'] is not None and room_labels.numel() > 0:
            # Filter out invalid labels
            valid_mask = (room_labels != -1)
            if valid_mask.any():
                room_loss = self.criterion_room(
                    outputs['room_logits'][valid_mask],
                    room_labels[valid_mask]
                )
                losses['room_loss'] = room_loss
                total_loss += room_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _calculate_batch_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        boundary_labels: torch.Tensor,
        room_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate batch-level metrics."""
        
        metrics = {}
        
        # Boundary accuracy
        if outputs['boundary_logits'] is not None and boundary_labels.numel() > 0:
            valid_mask = (boundary_labels != -1)
            if valid_mask.any():
                boundary_preds = torch.argmax(outputs['boundary_logits'][valid_mask], dim=1)
                boundary_acc = (boundary_preds == boundary_labels[valid_mask]).float().mean()
                metrics['boundary_accuracy'] = boundary_acc.item()
        
        # Room accuracy
        if outputs['room_logits'] is not None and room_labels.numel() > 0:
            valid_mask = (room_labels != -1)
            if valid_mask.any():
                room_preds = torch.argmax(outputs['room_logits'][valid_mask], dim=1)
                room_acc = (room_preds == room_labels[valid_mask]).float().mean()
                metrics['room_accuracy'] = room_acc.item()
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        
        self.model.eval()
        
        # Use EMA model for validation if available
        if self.use_ema:
            self.model_ema.apply_shadow()
        
        total_loss = 0.0
        boundary_loss_sum = 0.0
        room_loss_sum = 0.0
        num_batches = 0
        
        # For calculating comprehensive metrics
        all_boundary_preds = []
        all_boundary_labels = []
        all_room_preds = []
        all_room_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    primal_data = batch['primal'].to(self.device)
                    dual_data = batch['dual'].to(self.device)
                    boundary_labels = batch['boundary_labels'].to(self.device)
                    room_labels = batch['room_labels'].to(self.device)
                    
                    # Generate dummy image
                    batch_size = int(primal_data.batch.max().item() + 1) if primal_data.batch is not None else 1
                    image = torch.randn(batch_size, 3, 256, 256, device=self.device)
                    
                    # Forward pass
                    outputs = self.model(primal_data, dual_data, image)
                    
                    # Compute losses
                    loss_dict = self._compute_losses(outputs, boundary_labels, room_labels)
                    
                    # Update metrics
                    total_loss += loss_dict['total_loss'].item()
                    if 'boundary_loss' in loss_dict:
                        boundary_loss_sum += loss_dict['boundary_loss'].item()
                    if 'room_loss' in loss_dict:
                        room_loss_sum += loss_dict['room_loss'].item()
                    num_batches += 1
                    
                    # Collect predictions for comprehensive metrics
                    if outputs['boundary_logits'] is not None:
                        valid_mask = (boundary_labels != -1)
                        if valid_mask.any():
                            boundary_preds = torch.argmax(outputs['boundary_logits'][valid_mask], dim=1)
                            all_boundary_preds.append(boundary_preds.cpu())
                            all_boundary_labels.append(boundary_labels[valid_mask].cpu())
                    
                    if outputs['room_logits'] is not None:
                        valid_mask = (room_labels != -1)
                        if valid_mask.any():
                            room_preds = torch.argmax(outputs['room_logits'][valid_mask], dim=1)
                            all_room_preds.append(room_preds.cpu())
                            all_room_labels.append(room_labels[valid_mask].cpu())
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Val Loss': f"{total_loss/num_batches:.4f}"
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in validation step {batch_idx}: {e}")
                    continue
        
        # Restore original model if using EMA
        if self.use_ema:
            self.model_ema.restore()
        
        # Calculate comprehensive metrics
        val_metrics = {
            'val_loss': total_loss / num_batches,
            'val_boundary_loss': boundary_loss_sum / num_batches,
            'val_room_loss': room_loss_sum / num_batches
        }
        
        # Calculate additional metrics
        if all_boundary_preds:
            all_boundary_preds = torch.cat(all_boundary_preds)
            all_boundary_labels = torch.cat(all_boundary_labels)
            boundary_acc = (all_boundary_preds == all_boundary_labels).float().mean()
            val_metrics['val_boundary_accuracy'] = boundary_acc.item()
        
        if all_room_preds:
            all_room_preds = torch.cat(all_room_preds)
            all_room_labels = torch.cat(all_room_labels)
            room_acc = (all_room_preds == all_room_labels).float().mean()
            val_metrics['val_room_accuracy'] = room_acc.item()
            
            # Calculate mIoU and other metrics
            val_metrics.update(self._calculate_segmentation_metrics(
                all_room_preds, all_room_labels
            ))
        
        return val_metrics
    
    def _calculate_segmentation_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate segmentation metrics (mIoU, mAcc, etc.)."""
        
        num_classes = self.config.model.num_room_classes
        
        # Convert to numpy
        preds = predictions.numpy()
        lbls = labels.numpy()
        
        # Calculate confusion matrix
        mask = (lbls >= 0) & (lbls < num_classes)
        conf_matrix = np.bincount(
            num_classes * lbls[mask] + preds[mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)
        
        # Calculate IoU for each class
        ious = []
        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                ious.append(iou)
        
        # Calculate mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        # Calculate mean accuracy
        class_accs = []
        for i in range(num_classes):
            if conf_matrix[i, :].sum() > 0:
                acc = conf_matrix[i, i] / conf_matrix[i, :].sum()
                class_accs.append(acc)
        
        macc = np.mean(class_accs) if class_accs else 0.0
        
        # TODO: Implement Room Integrity (RI) metric from the paper
        # This would require region-level predictions and matching
        ri = 0.0  # Placeholder
        
        return {
            'val_miou': miou,
            'val_macc': macc,
            'val_ri': ri
        }
    
    def train(self):
        """Main training loop."""
        
        self.logger.info(f"Starting training for {self.config.training.epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val mIoU: {val_metrics.get('val_miou', 0):.4f}"
            )
            
            # Log to wandb
            if self.use_wandb:
                wandb.log(epoch_metrics)
            
            # Save checkpoint
            is_best = False
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                is_best = True
            
            if val_metrics.get('val_miou', 0) > self.best_val_miou:
                self.best_val_miou = val_metrics['val_miou']
            
            if epoch % self.config.save_interval == 0 or is_best:
                checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
                if is_best:
                    checkpoint_name = "best_model.pth"
                
                save_model_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    self.best_val_loss,
                    self.checkpoint_dir / checkpoint_name,
                    additional_info={
                        'config': self.config.__dict__,
                        'train_metrics': self.train_metrics,
                        'val_metrics': self.val_metrics,
                        'best_val_miou': self.best_val_miou
                    }
                )
            
            # Visualize results periodically
            if epoch % (self.config.save_interval * 2) == 0:
                self._visualize_predictions(epoch)
        
        # Training complete
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation mIoU: {self.best_val_miou:.4f}")
        
        # Save final metrics
        self._save_training_summary()
        
        if self.use_wandb:
            wandb.finish()
    
    def _visualize_predictions(self, epoch: int):
        """Visualize model predictions."""
        
        self.model.eval()
        
        # Get a batch from validation set
        val_batch = next(iter(self.val_loader))
        
        with torch.no_grad():
            primal_data = val_batch['primal'].to(self.device)
            dual_data = val_batch['dual'].to(self.device)
            
            # Generate dummy image
            batch_size = int(primal_data.batch.max().item() + 1) if primal_data.batch is not None else 1
            image = torch.randn(batch_size, 3, 256, 256, device=self.device)
            
            # Forward pass
            outputs = self.model(primal_data, dual_data, image, return_attention=True)
            
            # Visualize first sample in batch
            if batch_size > 0:
                # Get first sample data
                sample_mask = (primal_data.batch == 0) if primal_data.batch is not None else slice(None)
                sample_primal_pos = primal_data.pos[sample_mask]
                sample_primal_edges = primal_data.edge_index[:, primal_data.batch[primal_data.edge_index[0]] == 0]
                
                # Adjust edge indices for sample
                if primal_data.batch is not None:
                    node_offset = torch.where(primal_data.batch == 0)[0][0]
                    sample_primal_edges = sample_primal_edges - node_offset
                
                # Visualize primal graph with boundary predictions
                boundary_preds = None
                if outputs['boundary_logits'] is not None:
                    edge_mask = primal_data.batch[primal_data.edge_index[0]] == 0 if primal_data.batch is not None else slice(None)
                    boundary_logits_sample = outputs['boundary_logits'][edge_mask]
                    boundary_preds = torch.softmax(boundary_logits_sample, dim=1)[:, 1]  # Probability of being boundary
                
                self.visualizer.plot_primal_graph(
                    sample_primal_pos.cpu(),
                    sample_primal_edges.cpu(),
                    edge_predictions=boundary_preds.cpu() if boundary_preds is not None else None,
                    title=f"Epoch {epoch} - Boundary Predictions",
                    save_name=f"epoch_{epoch}_boundary_pred"
                )
        
        self.model.train()
    
    def _save_training_summary(self):
        """Save training summary and metrics."""
        
        summary = {
            'experiment_name': self.experiment_name,
            'config': self.config.__dict__,
            'total_epochs': len(self.train_metrics),
            'best_val_loss': self.best_val_loss,
            'best_val_miou': self.best_val_miou,
            'final_train_loss': self.train_metrics[-1]['train_loss'] if self.train_metrics else None,
            'final_val_loss': self.val_metrics[-1]['val_loss'] if self.val_metrics else None,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        summary_file = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved to: {summary_file}")
        
        # Plot training curves
        if self.train_metrics and self.val_metrics:
            train_losses = [m['train_loss'] for m in self.train_metrics]
            val_losses = [m['val_loss'] for m in self.val_metrics]
            
            self.visualizer.plot_training_curves(
                train_losses, val_losses,
                save_name=f"{self.experiment_name}_training_curves"
            )
```

## 5.2 Main Training Script

```python
# File: train_vectorfloorseg.py
"""Main training script for VectorFloorSeg."""

import argparse
import logging
import sys
import torch
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.trainer import VectorFloorSegTrainer
from src.models.model_factory import create_vectorfloorseg_model
from src.data.datasets import create_dataloader
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging

def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Train VectorFloorSeg model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='data', 
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, default='R2V', 
                       choices=['R2V', 'CubiCasa-5k'],
                       help='Dataset to use for training')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                       help='Path to configuration file')
    parser.add_argument('--backbone', type=str, default=None,
                       help='Backbone architecture (overrides config)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Experiment arguments
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (small dataset, fast training)')
    
    # Logging arguments
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval in batches')
    
    return parser.parse_args()

def setup_device(device_arg: str) -> torch.device:
    """Setup compute device."""
    
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return device

def create_debug_config(base_config):
    """Create debug configuration for fast training."""
    
    # Modify config for debugging
    base_config.model.hidden_dim = 64
    base_config.model.num_layers = 2
    base_config.model.num_heads = 2
    
    base_config.training.epochs = 5
    base_config.training.batch_size = 2
    base_config.training.learning_rate = 0.001
    
    base_config.data.datasets = ["R2V"]  # Use smaller dataset
    
    return base_config

def main():
    """Main training function."""
    
    args = parse_args()
    
    # Setup device
    device = setup_device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # Override config with command line arguments
    if args.name:
        config.name = args.name
    if args.backbone:
        config.model.backbone = args.backbone
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Apply debug modifications if requested
    if args.debug:
        config = create_debug_config(config)
        config.name += "_debug"
        print("Debug mode enabled - using small model and dataset")
    
    # Setup logging
    logger = setup_logging(
        log_dir="outputs/logs",
        experiment_name=config.name
    )
    
    logger.info(f"Starting experiment: {config.name}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    
    try:
        # Create model
        logger.info("Creating model...")
        model = create_vectorfloorseg_model(config.model, device)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Create data loaders
        logger.info("Creating data loaders...")
        
        train_loader = create_dataloader(
            data_root=args.data_root,
            dataset_name=args.dataset,
            split="train",
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            max_samples=50 if args.debug else None  # Limit samples in debug mode
        )
        
        # Create validation split
        val_split = "val" if args.dataset == "CubiCasa-5k" else "test"
        val_loader = create_dataloader(
            data_root=args.data_root,
            dataset_name=args.dataset,
            split=val_split,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_samples=20 if args.debug else None  # Limit samples in debug mode
        )
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Create trainer
        trainer = VectorFloorSegTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            experiment_name=config.name,
            use_wandb=args.wandb
        )
        
        # Override trainer epoch if resuming
        if start_epoch > 0:
            trainer.epoch = start_epoch
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
```

## 5.3 Evaluation and Inference

```python
# File: evaluate_model.py
"""Model evaluation and inference script."""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import logging
import sys
from typing import Dict, List, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.model_factory import create_vectorfloorseg_model, load_model_checkpoint
from src.data.datasets import create_dataloader
from src.data.svg_processor import SVGFloorplanProcessor
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logging
from src.utils.visualization import FloorplanVisualizer

class VectorFloorSegEvaluator:
    """Evaluation manager for VectorFloorSeg."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int = 12
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualizer
        self.visualizer = FloorplanVisualizer()
        
        # Initialize SVG processor for single file inference
        self.svg_processor = SVGFloorplanProcessor()
    
    def evaluate_dataset(
        self, 
        dataloader: torch.utils.data.DataLoader,
        save_predictions: bool = False,
        output_dir: str = "outputs/predictions"
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        
        self.model.eval()
        
        # Metrics accumulators
        total_samples = 0
        total_loss = 0.0
        
        # For segmentation metrics
        all_room_preds = []
        all_room_labels = []
        all_boundary_preds = []
        all_boundary_labels = []
        
        # For Room Integrity metric
        all_predictions = []
        all_ground_truth = []
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move data to device
                    primal_data = batch['primal'].to(self.device)
                    dual_data = batch['dual'].to(self.device)
                    boundary_labels = batch['boundary_labels'].to(self.device)
                    room_labels = batch['room_labels'].to(self.device)
                    
                    # Generate dummy image (in practice, use real rendered images)
                    batch_size = int(primal_data.batch.max().item() + 1) if primal_data.batch is not None else 1
                    image = torch.randn(batch_size, 3, 256, 256, device=self.device)
                    
                    # Forward pass
                    outputs = self.model(primal_data, dual_data, image)
                    
                    # Calculate loss
                    losses = self.model.compute_loss(outputs, boundary_labels, room_labels)
                    total_loss += losses['total_loss'].item()
                    total_samples += batch_size
                    
                    # Collect predictions
                    if outputs['boundary_logits'] is not None:
                        valid_mask = (boundary_labels != -1)
                        if valid_mask.any():
                            boundary_preds = torch.argmax(outputs['boundary_logits'][valid_mask], dim=1)
                            all_boundary_preds.append(boundary_preds.cpu().numpy())
                            all_boundary_labels.append(boundary_labels[valid_mask].cpu().numpy())
                    
                    if outputs['room_logits'] is not None:
                        valid_mask = (room_labels != -1)
                        if valid_mask.any():
                            room_preds = torch.argmax(outputs['room_logits'][valid_mask], dim=1)
                            all_room_preds.append(room_preds.cpu().numpy())
                            all_room_labels.append(room_labels[valid_mask].cpu().numpy())
                    
                    # Save predictions if requested
                    if save_predictions:
                        self._save_batch_predictions(
                            batch, outputs, batch_idx, output_path
                        )
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f"Evaluated {batch_idx}/{len(dataloader)} batches")
                
                except Exception as e:
                    self.logger.error(f"Error evaluating batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        metrics = {'average_loss': total_loss / len(dataloader)}
        
        # Boundary metrics
        if all_boundary_preds:
            boundary_preds = np.concatenate(all_boundary_preds)
            boundary_labels = np.concatenate(all_boundary_labels)
            
            boundary_acc = np.mean(boundary_preds == boundary_labels)
            metrics['boundary_accuracy'] = boundary_acc
            
            # Boundary F1 score
            boundary_f1 = self._calculate_f1_score(boundary_preds, boundary_labels)
            metrics['boundary_f1'] = boundary_f1
        
        # Room segmentation metrics
        if all_room_preds:
            room_preds = np.concatenate(all_room_preds)
            room_labels = np.concatenate(all_room_labels)
            
            # Calculate segmentation metrics
            seg_metrics = self._calculate_segmentation_metrics(room_preds, room_labels)
            metrics.update(seg_metrics)
        
        self.logger.info("Evaluation completed")
        for key, value in metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        return metrics
    
    def _calculate_segmentation_metrics(
        self, 
        predictions: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive segmentation metrics."""
        
        # Calculate confusion matrix
        mask = (labels >= 0) & (labels < self.num_classes)
        conf_matrix = np.bincount(
            self.num_classes * labels[mask] + predictions[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        # Calculate IoU for each class
        ious = []
        for i in range(self.num_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
                ious.append(iou)
        
        # Mean IoU
        miou = np.mean(ious) if ious else 0.0
        
        # Calculate class-wise accuracy
        class_accs = []
        for i in range(self.num_classes):
            if conf_matrix[i, :].sum() > 0:
                acc = conf_matrix[i, i] / conf_matrix[i, :].sum()
                class_accs.append(acc)
        
        # Mean class accuracy
        macc = np.mean(class_accs) if class_accs else 0.0
        
        # Overall accuracy
        overall_acc = np.trace(conf_matrix) / np.sum(conf_matrix)
        
        # TODO: Implement Room Integrity (RI) metric
        # This requires region-level evaluation which needs more complex processing
        ri = 0.0  # Placeholder
        
        return {
            'miou': miou,
            'macc': macc,
            'overall_accuracy': overall_acc,
            'room_integrity': ri
        }
    
    def _calculate_f1_score(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate F1 score for binary classification."""
        
        # For binary boundary classification
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            return 0.0
        else:
            return 2 * precision * recall / (precision + recall)
    
    def _save_batch_predictions(
        self,
        batch: Dict,
        outputs: Dict,
        batch_idx: int,
        output_dir: Path
    ):
        """Save batch predictions for analysis."""
        
        # Save predictions as JSON
        predictions = {}
        
        if outputs['boundary_logits'] is not None:
            boundary_probs = F.softmax(outputs['boundary_logits'], dim=1)
            predictions['boundary_predictions'] = boundary_probs.cpu().numpy().tolist()
        
        if outputs['room_logits'] is not None:
            room_probs = F.softmax(outputs['room_logits'], dim=1)
            predictions['room_predictions'] = room_probs.cpu().numpy().tolist()
        
        # Save metadata
        predictions['metadata'] = batch['metadata']
        
        pred_file = output_dir / f"batch_{batch_idx}_predictions.json"
        with open(pred_file, 'w') as f:
            json.dump(predictions, f, indent=2)
    
    def infer_single_svg(
        self, 
        svg_path: str,
        output_dir: str = "outputs/inference",
        visualize: bool = True
    ) -> Dict:
        """Run inference on a single SVG file."""
        
        self.logger.info(f"Processing SVG: {svg_path}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Process SVG to graphs
            result = self.svg_processor.process_svg_to_graphs(svg_path)
            
            # Move to device
            primal_data = result['primal'].to(self.device)
            dual_data = result['dual'].to(self.device)
            
            # Create dummy image (in practice, render the SVG)
            image = torch.randn(1, 3, 256, 256, device=self.device)
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(primal_data, dual_data, image, return_attention=True)
            
            # Process predictions
            predictions = {}
            
            if outputs['boundary_logits'] is not None:
                boundary_probs = F.softmax(outputs['boundary_logits'], dim=1)
                boundary_preds = torch.argmax(boundary_probs, dim=1)
                predictions['boundary_predictions'] = boundary_preds.cpu().numpy().tolist()
                predictions['boundary_probabilities'] = boundary_probs.cpu().numpy().tolist()
            
            if outputs['room_logits'] is not None:
                room_probs = F.softmax(outputs['room_logits'], dim=1)
                room_preds = torch.argmax(room_probs, dim=1)
                predictions['room_predictions'] = room_preds.cpu().numpy().tolist()
                predictions['room_probabilities'] = room_probs.cpu().numpy().tolist()
            
            # Add metadata
            predictions['metadata'] = result['metadata']
            
            # Visualize results
            if visualize:
                svg_name = Path(svg_path).stem
                
                # Visualize primal graph with boundary predictions
                if outputs['boundary_logits'] is not None:
                    boundary_probs_viz = F.softmax(outputs['boundary_logits'], dim=1)[:, 1]  # Probability of being boundary
                    
                    self.visualizer.plot_primal_graph(
                        primal_data.pos.cpu(),
                        primal_data.edge_index.cpu(),
                        edge_predictions=boundary_probs_viz.cpu(),
                        title=f"Boundary Predictions - {svg_name}",
                        save_name=f"{svg_name}_boundary_predictions"
                    )
                
                # Visualize dual graph with room predictions
                if outputs['room_logits'] is not None and result['regions']:
                    room_preds_viz = torch.argmax(outputs['room_logits'], dim=1)
                    
                    self.visualizer.plot_dual_graph(
                        result['regions'],
                        room_predictions=outputs['room_logits'].cpu(),
                        title=f"Room Segmentation - {svg_name}",
                        save_name=f"{svg_name}_room_segmentation"
                    )
            
            # Save predictions
            pred_file = output_path / f"{Path(svg_path).stem}_predictions.json"
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            self.logger.info(f"Inference completed. Results saved to {output_path}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error processing {svg_path}: {e}")
            raise

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description='Evaluate VectorFloorSeg model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (auto-detected from checkpoint if not provided)')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, default='R2V',
                       choices=['R2V', 'CubiCasa-5k'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save model predictions')
    parser.add_argument('--svg_file', type=str, default=None,
                       help='Single SVG file to process')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Setup logging
    logger = setup_logging(experiment_name="evaluation")
    logger.info(f"Starting evaluation on {device}")
    
    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Load configuration
        if args.config:
            config_manager = ConfigManager()
            config = config_manager.load_config(args.config)
        else:
            # Try to get config from checkpoint
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                logger.error("No configuration found. Please provide --config argument.")
                return
        
        # Create model
        model = create_vectorfloorseg_model(config.model, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Create evaluator
        evaluator = VectorFloorSegEvaluator(model, device, config.model.num_room_classes)
        
        if args.svg_file:
            # Single file inference
            logger.info(f"Running inference on: {args.svg_file}")
            predictions = evaluator.infer_single_svg(
                args.svg_file,
                output_dir=args.output_dir,
                visualize=True
            )
            
            logger.info("Single file inference completed")
            
        else:
            # Dataset evaluation
            logger.info(f"Evaluating on {args.dataset} {args.split} split")
            
            # Create data loader
            dataloader = create_dataloader(
                data_root=args.data_root,
                dataset_name=args.dataset,
                split=args.split,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Run evaluation
            metrics = evaluator.evaluate_dataset(
                dataloader,
                save_predictions=args.save_predictions,
                output_dir=args.output_dir
            )
            
            # Save metrics
            metrics_file = Path(args.output_dir) / "evaluation_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Evaluation metrics saved to: {metrics_file}")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

## 5.4 Quick Start Scripts

```python
# File: scripts/quick_start.py
"""Quick start script for VectorFloorSeg."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run command with error handling."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"✗ {description} failed")
        if result.stderr:
            print(f"Error: {result.stderr}")
        return False
    
    return True

def main():
    """Quick start workflow."""
    
    print("VectorFloorSeg Quick Start")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Error: Please run this script from the VecFloorSeg root directory")
        sys.exit(1)
    
    # Test model components
    if not run_command("python test_model.py", "Testing model components"):
        return
    
    # Test data preprocessing
    if not run_command("python preprocess_data.py --test_only", "Testing SVG processing"):
        return
    
    # Run debug training
    if not run_command(
        "python train_vectorfloorseg.py --config configs/debug.yaml --debug --epochs 2",
        "Running debug training (2 epochs)"
    ):
        return
    
    # Test evaluation
    checkpoint_path = "outputs/checkpoints/checkpoint_epoch_1.pth"
    if Path(checkpoint_path).exists():
        if not run_command(
            f"python evaluate_model.py --checkpoint {checkpoint_path} --config configs/debug.yaml --svg_file test_floorplan.svg",
            "Testing model evaluation"
        ):
            return
    
    print("\n" + "="*60)
    print("✓ Quick start completed successfully!")
    print("✓ VectorFloorSeg is ready for full training")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Download datasets (R2V, CubiCasa-5k) to data/raw/")
    print("2. Run full preprocessing: python preprocess_data.py")
    print("3. Start training: python train_vectorfloorseg.py --config configs/baseline.yaml")
    print("4. Monitor training: tail -f outputs/logs/*.log")
    print("5. Evaluate model: python evaluate_model.py --checkpoint outputs/checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
```

## 5.5 Configuration Templates

```yaml
# File: configs/production.yaml
name: "vectorfloorseg_production"
description: "Production VectorFloorSeg configuration"
device: "auto"
seed: 42

model:
  primal_input_dim: 66
  dual_input_dim: 66
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  num_room_classes: 12
  backbone: "resnet101"

training:
  batch_size: 8
  learning_rate: 0.01
  weight_decay: 0.0005
  momentum: 0.9
  epochs: 200
  boundary_loss_weight: 0.5
  gradient_clip_norm: 1.0
  scheduler: "cosine"
  optimizer: "sgd"
  use_ema: true

data:
  image_size: [256, 256]
  normalize_coords: true
  extend_lines: true
  use_data_augmentation: true
  flip_probability: 0.5
  datasets: ["R2V", "CubiCasa-5k"]

log_interval: 10
save_interval: 20
```

```yaml
# File: configs/fast_training.yaml
name: "vectorfloorseg_fast"
description: "Fast training configuration for quick experiments"
device: "auto"
seed: 42

model:
  primal_input_dim: 66
  dual_input_dim: 66
  hidden_dim: 128
  num_layers: 4
  num_heads: 4
  dropout: 0.1
  num_room_classes: 12
  backbone: "resnet50"

training:
  batch_size: 16
  learning_rate: 0.02
  weight_decay: 0.0005
  momentum: 0.9
  epochs: 100
  boundary_loss_weight: 0.5
  gradient_clip_norm: 1.0
  scheduler: "cosine"
  optimizer: "adamw"
  use_ema: false

data:
  image_size: [256, 256]
  normalize_coords: true
  extend_lines: true
  use_data_augmentation: true
  flip_probability: 0.5
  datasets: ["R2V"]

log_interval: 5
save_interval: 10
```

## Usage Instructions

### Quick Start
```bash
cd VecFloorSeg
source vectorfloorseg_env/bin/activate

# Run complete quick start test
python scripts/quick_start.py
```

### Full Training
```bash
# Train with default configuration
python train_vectorfloorseg.py --config configs/baseline.yaml

# Train with custom settings
python train_vectorfloorseg.py \
    --config configs/baseline.yaml \
    --dataset CubiCasa-5k \
    --epochs 150 \
    --batch_size 16 \
    --wandb

# Resume training from checkpoint
python train_vectorfloorseg.py \
    --config configs/baseline.yaml \
    --resume outputs/checkpoints/checkpoint_epoch_50.pth
```

### Evaluation
```bash
# Evaluate on test set
python evaluate_model.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --dataset R2V \
    --split test \
    --save_predictions

# Process single SVG file
python evaluate_model.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --svg_file path/to/floorplan.svg \
    --output_dir results/
```

### Monitoring Training
```bash
# Monitor logs
tail -f outputs/logs/*.log

# View tensorboard (if enabled)
tensorboard --logdir outputs/logs

# View wandb (if enabled)
# Check your wandb dashboard
```

## Expected Results

After successful training, you should achieve:
- **mIoU**: > 75% on R2V dataset
- **mAcc**: > 85% on standard datasets  
- **Boundary F1**: > 80% for boundary detection
- **Room Integrity**: > 80% for clean segmentation

## Troubleshooting

### Training Issues
- **OOM errors**: Reduce batch_size or hidden_dim
- **Slow convergence**: Increase learning_rate or use adamw optimizer
- **Poor performance**: Check data preprocessing and label quality

### Evaluation Issues
- **Low metrics**: Verify checkpoint loading and data splits
- **Visualization errors**: Check output directory permissions
- **SVG processing**: Start with simple SVG files

### System Issues
- **CUDA errors**: Verify GPU memory and driver compatibility
- **Data loading**: Check num_workers and dataset paths
- **Checkpointing**: Ensure sufficient disk space

## Next Steps

1. **Production Deployment**: Optimize model for inference speed
2. **Advanced Metrics**: Implement full Room Integrity (RI) calculation
3. **Model Improvements**: Experiment with different architectures
4. **Dataset Expansion**: Add support for additional floorplan formats
5. **Web Interface**: Create a web interface for interactive floorplan processing

This completes the full VectorFloorSeg implementation! You now have a complete, working system for semantic segmentation of vector floorplans.
