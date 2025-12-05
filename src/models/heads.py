"""
Task-specific heads for state classification and GPS regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ClassificationHead(nn.Module):
    """
    Classification head for predicting US states.
    
    Architecture:
        Linear -> GELU -> Dropout -> Linear -> Logits
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of states to classify
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
        
        # Initialize final layer with small weights
        nn.init.xavier_uniform_(self.classifier[-1].weight, gain=0.01)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.classifier(x)
    
    def get_top_k_predictions(
        self,
        x: torch.Tensor,
        k: int = 5,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k predictions with probabilities.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
            k: Number of top predictions
            return_probs: If True, return probabilities instead of logits
        
        Returns:
            top_k_indices: (batch, k)
            top_k_values: (batch, k) - probabilities or logits
        """
        logits = self.forward(x)
        
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            values, indices = torch.topk(probs, k=k, dim=-1)
        else:
            values, indices = torch.topk(logits, k=k, dim=-1)
        
        return indices, values


class GPSRegressionHead(nn.Module):
    """
    Regression head for predicting GPS coordinates.
    
    Two variants:
    1. Direct: Predict raw latitude/longitude
    2. Hierarchical: Predict offset from predicted state centroid
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        normalize_output: bool = True
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            normalize_output: If True, apply tanh and scale to valid GPS range
        """
        super().__init__()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # latitude, longitude
        )
        
        self.normalize_output = normalize_output
        
        # US bounding box (approximate)
        # Latitude: 24.5 to 49.5 (continental) or 18 to 71 (including AK, HI)
        # Longitude: -125 to -66
        self.register_buffer('lat_range', torch.tensor([18.0, 72.0]))
        self.register_buffer('lon_range', torch.tensor([-180.0, -65.0]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
        
        Returns:
            GPS coordinates of shape (batch, 2) - [latitude, longitude]
        """
        output = self.regressor(x)
        
        if self.normalize_output:
            # Apply tanh and scale to valid range
            output = torch.tanh(output)
            
            lat = output[:, 0:1] * (self.lat_range[1] - self.lat_range[0]) / 2 + \
                  (self.lat_range[0] + self.lat_range[1]) / 2
            lon = output[:, 1:2] * (self.lon_range[1] - self.lon_range[0]) / 2 + \
                  (self.lon_range[0] + self.lon_range[1]) / 2
            
            output = torch.cat([lat, lon], dim=1)
        
        return output


class HierarchicalGPSHead(nn.Module):
    """
    Hierarchical GPS prediction: predict offset from state centroid.
    
    This constrains predictions to be within a reasonable distance
    of the predicted state, reducing wild outliers.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        max_offset_degrees: float = 5.0
    ):
        """
        Args:
            input_dim: Input embedding dimension
            num_classes: Number of states (for conditioning)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            max_offset_degrees: Maximum offset from centroid in degrees
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.max_offset = max_offset_degrees
        
        # Offset predictor
        self.offset_regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # lat_offset, lon_offset
        )
        
        # State centroids will be set during training
        self.register_buffer(
            'state_centroids',
            torch.zeros(num_classes, 2)  # (num_states, 2) for lat, lon
        )
    
    def set_centroids(self, centroids: torch.Tensor):
        """Set state centroid coordinates."""
        self.state_centroids.copy_(centroids)
    
    def forward(
        self,
        x: torch.Tensor,
        state_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
            state_indices: Predicted state indices of shape (batch,)
        
        Returns:
            GPS coordinates of shape (batch, 2)
        """
        # Get centroids for predicted states
        centroids = self.state_centroids[state_indices]  # (batch, 2)
        
        # Predict offset
        offset = self.offset_regressor(x)
        offset = torch.tanh(offset) * self.max_offset  # Constrain offset
        
        # Add offset to centroid
        gps = centroids + offset
        
        return gps


class DualHead(nn.Module):
    """
    Combined classification and regression head.
    
    Shares some computation for efficiency.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.num_classes = num_classes
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input embeddings of shape (batch, input_dim)
        
        Returns:
            class_logits: (batch, num_classes)
            gps_coords: (batch, 2)
        """
        shared = self.shared(x)
        
        class_logits = self.classification_head(shared)
        gps_coords = self.regression_head(shared)
        
        return class_logits, gps_coords
