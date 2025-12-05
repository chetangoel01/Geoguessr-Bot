"""
Loss functions for GeoGuessr model training.

Key innovation: Haversine-smoothed cross entropy gives partial credit
for predicting nearby states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class HaversineSmoothedCrossEntropy(nn.Module):
    """
    Cross entropy loss with haversine-smoothed soft labels.
    
    Instead of hard labels (0 or 1), this creates soft labels where
    neighboring states receive partial probability mass based on
    their geographic distance to the true state.
    
    This teaches the model that predicting Nevada when the answer is
    California is better than predicting Maine.
    """
    
    def __init__(
        self,
        distance_matrix: np.ndarray,
        temperature: float = 300.0,
        label_smoothing: float = 0.0
    ):
        """
        Args:
            distance_matrix: Pairwise haversine distances between state centroids
                            Shape: (num_states, num_states)
            temperature: Smoothing temperature in km. Higher = smoother distribution.
                        ~300-500 km works well for US states.
            label_smoothing: Additional uniform label smoothing (0 to 1)
        """
        super().__init__()
        
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.num_classes = distance_matrix.shape[0]
        
        # Precompute soft label matrix
        # soft_labels[i, j] = P(predict j | true state i)
        log_probs = -distance_matrix / temperature
        soft_labels = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
        
        # Apply additional label smoothing if requested
        if label_smoothing > 0:
            uniform = np.ones_like(soft_labels) / self.num_classes
            soft_labels = (1 - label_smoothing) * soft_labels + label_smoothing * uniform
        
        self.register_buffer(
            'soft_label_matrix',
            torch.tensor(soft_labels, dtype=torch.float32)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Model output of shape (batch, num_classes)
            targets: Target state indices of shape (batch,)
        
        Returns:
            Scalar loss
        """
        # Get soft labels for each target
        soft_targets = self.soft_label_matrix[targets]  # (batch, num_classes)
        
        # Compute cross entropy with soft labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)
        
        return loss.mean()


class StandardCrossEntropy(nn.Module):
    """
    Standard cross entropy with optional label smoothing.
    """
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross entropy loss."""
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )


class GPSLoss(nn.Module):
    """
    Loss for GPS coordinate regression.
    
    Supports multiple variants:
    - MSE: Mean squared error
    - MAE: Mean absolute error
    - Haversine: Actual geographic distance
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        normalize: bool = True
    ):
        """
        Args:
            loss_type: One of "mse", "mae", "smooth_l1", "haversine"
            normalize: If True, normalize coordinates to similar scale
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.normalize = normalize
        
        # Normalization factors (actual US ranges from dataset)
        # Latitude: ~25° to ~71° = 46° range
        # Longitude: ~-125° to ~-66° = 59° range
        self.register_buffer('lat_scale', torch.tensor(46.0))
        self.register_buffer('lon_scale', torch.tensor(59.0))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GPS loss.
        
        Args:
            pred: Predicted coordinates of shape (batch, 2)
            target: Target coordinates of shape (batch, 2)
        
        Returns:
            Scalar loss
        """
        if self.normalize:
            # Normalize to similar scales
            pred_norm = pred.clone()
            target_norm = target.clone()
            pred_norm[:, 0] = pred[:, 0] / self.lat_scale
            pred_norm[:, 1] = pred[:, 1] / self.lon_scale
            target_norm[:, 0] = target[:, 0] / self.lat_scale
            target_norm[:, 1] = target[:, 1] / self.lon_scale
        else:
            pred_norm = pred
            target_norm = target
        
        if self.loss_type == "mse":
            return F.mse_loss(pred_norm, target_norm)
        elif self.loss_type == "mae":
            return F.l1_loss(pred_norm, target_norm)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_norm, target_norm)
        elif self.loss_type == "haversine":
            return self._haversine_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _haversine_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute haversine distance loss in km."""
        # Convert to radians
        pred_rad = torch.deg2rad(pred)
        target_rad = torch.deg2rad(target)
        
        lat1, lon1 = pred_rad[:, 0], pred_rad[:, 1]
        lat2, lon2 = target_rad[:, 0], target_rad[:, 1]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        
        # Earth radius in km
        distance = 6371.0 * c
        
        # Normalize by max US distance (~5000 km)
        return (distance / 5000.0).mean()


class CombinedLoss(nn.Module):
    """
    Combined classification and GPS regression loss.
    """
    
    def __init__(
        self,
        classification_loss: nn.Module,
        gps_loss: nn.Module,
        classification_weight: float = 0.8,
        gps_weight: float = 0.2
    ):
        """
        Args:
            classification_loss: Loss for state classification
            gps_loss: Loss for GPS regression
            classification_weight: Weight for classification loss
            gps_weight: Weight for GPS loss
        """
        super().__init__()
        
        self.classification_loss = classification_loss
        self.gps_loss = gps_loss
        self.classification_weight = classification_weight
        self.gps_weight = gps_weight
    
    def forward(
        self,
        class_logits: torch.Tensor,
        gps_pred: torch.Tensor,
        class_targets: torch.Tensor,
        gps_targets: torch.Tensor
    ) -> dict:
        """
        Compute combined loss.
        
        Returns:
            Dictionary with 'loss' (total), 'cls_loss', and 'gps_loss'
        """
        cls_loss = self.classification_loss(class_logits, class_targets)
        gps_loss = self.gps_loss(gps_pred, gps_targets)
        
        total_loss = self.classification_weight * cls_loss + \
                     self.gps_weight * gps_loss
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'gps_loss': gps_loss
        }


def create_loss_function(
    config,
    distance_matrix: np.ndarray,
    class_weights: Optional[torch.Tensor] = None
) -> CombinedLoss:
    """
    Factory function to create the loss function based on config.
    
    Args:
        config: Configuration object
        distance_matrix: Haversine distance matrix between states
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        CombinedLoss module
    """
    if config.training.use_haversine_smoothing:
        cls_loss = HaversineSmoothedCrossEntropy(
            distance_matrix=distance_matrix,
            temperature=config.training.haversine_temperature
        )
    else:
        cls_loss = StandardCrossEntropy(
            num_classes=distance_matrix.shape[0],
            label_smoothing=0.1,
            class_weights=class_weights
        )
    
    gps_loss = GPSLoss(loss_type="smooth_l1", normalize=True)
    
    combined = CombinedLoss(
        classification_loss=cls_loss,
        gps_loss=gps_loss,
        classification_weight=config.training.classification_weight,
        gps_weight=config.training.gps_weight
    )
    
    return combined
