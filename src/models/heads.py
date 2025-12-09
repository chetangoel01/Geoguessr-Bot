"""
Task-specific heads for state classification and GPS regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..config import GPS_BOUNDS


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
        self.register_buffer('lat_range', torch.tensor(GPS_BOUNDS['lat']))
        self.register_buffer('lon_range', torch.tensor(GPS_BOUNDS['lon']))
    
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