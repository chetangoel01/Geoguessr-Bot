"""
Full GeoGuessr model combining backbone, fusion, and task heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .backbone import get_backbone, count_parameters
from .fusion import MultiViewFusion
from .heads import ClassificationHead, GPSRegressionHead
from ..config import Config


class GeoGuessrModel(nn.Module):
    """
    End-to-end GeoGuessr model.
    
    Architecture:
        4 images -> Backbone (shared) -> 4 embeddings -> Fusion -> 
        -> Classification Head -> State logits
        -> GPS Regression Head -> Coordinates
    """
    
    def __init__(self, config: Config, num_classes: int = 33):
        """
        Args:
            config: Configuration object
            num_classes: Number of states to classify
        """
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Load backbone
        self.backbone = get_backbone(config)
        embedding_dim = self.backbone.embedding_dim
        
        # Multi-view fusion
        self.fusion = MultiViewFusion(
            embedding_dim=embedding_dim,
            num_views=config.model.num_views,
            fusion_method=config.model.fusion_method,
            hidden_dim=config.model.hidden_dim
        )
        
        # Task heads
        self.classification_head = ClassificationHead(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout_rate
        )
        
        self.gps_head = GPSRegressionHead(
            input_dim=self.fusion.output_dim,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout_rate,
            normalize_output=True
        )
        
        # Print parameter counts
        self._print_param_counts()
    
    def _print_param_counts(self):
        """Print parameter counts for each component."""
        backbone_total, backbone_train = count_parameters(self.backbone)
        fusion_total, fusion_train = count_parameters(self.fusion)
        cls_total, cls_train = count_parameters(self.classification_head)
        gps_total, gps_train = count_parameters(self.gps_head)
        
        total = backbone_total + fusion_total + cls_total + gps_total
        trainable = backbone_train + fusion_train + cls_train + gps_train
        
        print(f"\n{'='*50}")
        print("Model Parameter Counts:")
        print(f"{'='*50}")
        print(f"Backbone:     {backbone_total:>12,} total, {backbone_train:>12,} trainable")
        print(f"Fusion:       {fusion_total:>12,} total, {fusion_train:>12,} trainable")
        print(f"Cls Head:     {cls_total:>12,} total, {cls_train:>12,} trainable")
        print(f"GPS Head:     {gps_total:>12,} total, {gps_train:>12,} trainable")
        print(f"{'-'*50}")
        print(f"Total:        {total:>12,} total, {trainable:>12,} trainable")
        print(f"{'='*50}\n")
    
    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode all views through the backbone.
        
        Args:
            images: Tensor of shape (batch, num_views, C, H, W)
        
        Returns:
            View embeddings of shape (batch, num_views, embedding_dim)
        """
        batch_size, num_views, C, H, W = images.shape
        
        # Reshape to process all views through backbone
        # (batch * num_views, C, H, W)
        images_flat = images.view(batch_size * num_views, C, H, W)
        
        # Get embeddings
        # (batch * num_views, embedding_dim)
        embeddings_flat = self.backbone(images_flat)
        
        # Reshape back
        # (batch, num_views, embedding_dim)
        embeddings = embeddings_flat.view(batch_size, num_views, -1)
        
        return embeddings
    
    def forward(
        self,
        images: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Tensor of shape (batch, num_views, C, H, W)
            return_embeddings: If True, also return intermediate embeddings
        
        Returns:
            Dictionary containing:
                - class_logits: (batch, num_classes)
                - gps_coords: (batch, 2)
                - fused_embedding: (batch, embedding_dim) if return_embeddings
                - view_embeddings: (batch, num_views, embedding_dim) if return_embeddings
        """
        # Encode all views
        view_embeddings = self.encode_views(images)
        
        # Fuse views
        fused = self.fusion(view_embeddings)
        
        # Get predictions
        class_logits = self.classification_head(fused)
        gps_coords = self.gps_head(fused)
        
        output = {
            'class_logits': class_logits,
            'gps_coords': gps_coords
        }
        
        if return_embeddings:
            output['fused_embedding'] = fused
            output['view_embeddings'] = view_embeddings
        
        return output
    
    def predict(
        self,
        images: torch.Tensor,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions for submission.
        
        Args:
            images: Tensor of shape (batch, num_views, C, H, W)
            top_k: Number of top state predictions
        
        Returns:
            Dictionary containing:
                - top_k_states: (batch, k) - state indices
                - top_k_probs: (batch, k) - probabilities
                - gps_coords: (batch, 2)
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(images)
            
            # Get top-k predictions
            probs = F.softmax(output['class_logits'], dim=-1)
            top_k_probs, top_k_states = torch.topk(probs, k=top_k, dim=-1)
            
            return {
                'top_k_states': top_k_states,
                'top_k_probs': top_k_probs,
                'gps_coords': output['gps_coords']
            }
    
    def freeze_backbone(self):
        """Freeze the backbone for initial training."""
        self.backbone.freeze()
    
    def unfreeze_backbone(self, unfreeze_layers: Optional[int] = None):
        """Unfreeze the backbone for fine-tuning."""
        self.backbone.unfreeze(unfreeze_layers)
        self._print_param_counts()
    
    def get_optimizer_param_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 1e-6
    ) -> list:
        """
        Get parameter groups with different learning rates.
        
        Args:
            backbone_lr: Learning rate for backbone
            head_lr: Learning rate for heads
            weight_decay: Weight decay
        
        Returns:
            List of parameter groups for optimizer
        """
        param_groups = [
            {
                'params': self.backbone.parameters(),
                'lr': backbone_lr,
                'weight_decay': weight_decay,
                'name': 'backbone'
            },
            {
                'params': self.fusion.parameters(),
                'lr': head_lr,
                'weight_decay': weight_decay,
                'name': 'fusion'
            },
            {
                'params': self.classification_head.parameters(),
                'lr': head_lr,
                'weight_decay': weight_decay,
                'name': 'classification_head'
            },
            {
                'params': self.gps_head.parameters(),
                'lr': head_lr,
                'weight_decay': weight_decay,
                'name': 'gps_head'
            }
        ]
        
        return param_groups


def load_model(
    config: Config,
    checkpoint_path: Optional[str] = None,
    num_classes: int = 33
) -> GeoGuessrModel:
    """
    Load model, optionally from checkpoint.
    
    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file
        num_classes: Number of states
    
    Returns:
        Loaded model
    """
    model = GeoGuessrModel(config, num_classes=num_classes)
    
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")
    
    return model
