"""
Multi-view fusion strategies for combining 4 directional embeddings.

Based on PIGEON's finding that simple averaging outperforms learned fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiViewFusion(nn.Module):
    """
    Fuses embeddings from multiple views (N, E, S, W) into a single representation.
    
    Supports multiple fusion strategies:
    - average: Element-wise mean (PIGEON's best approach)
    - concat: Concatenation followed by projection
    - attention: Learned attention weights per view
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_views: int = 4,
        fusion_method: str = "average",
        hidden_dim: Optional[int] = None
    ):
        """
        Args:
            embedding_dim: Dimension of each view's embedding
            num_views: Number of views to fuse (default 4 for N/E/S/W)
            fusion_method: One of "average", "concat", "attention"
            hidden_dim: Hidden dimension for projection layers
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_views = num_views
        self.fusion_method = fusion_method
        
        if fusion_method == "average":
            # No learnable parameters - just averaging
            self.output_dim = embedding_dim
            
        elif fusion_method == "concat":
            # Concatenate all views and project back to embedding_dim
            hidden = hidden_dim or embedding_dim
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim * num_views, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, embedding_dim)
            )
            self.output_dim = embedding_dim
            
        elif fusion_method == "attention":
            # Learn attention weights for each view
            self.attention = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )
            self.output_dim = embedding_dim
            
        elif fusion_method == "weighted":
            # Learnable per-view weights
            self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
            self.output_dim = embedding_dim
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        print(f"MultiViewFusion initialized with method: {fusion_method}")
    
    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse multiple view embeddings.
        
        Args:
            view_embeddings: Tensor of shape (batch, num_views, embedding_dim)
        
        Returns:
            Fused embedding of shape (batch, output_dim)
        """
        batch_size = view_embeddings.shape[0]
        
        if self.fusion_method == "average":
            # Simple element-wise mean
            fused = view_embeddings.mean(dim=1)
            
        elif self.fusion_method == "concat":
            # Flatten and project
            concat = view_embeddings.view(batch_size, -1)
            fused = self.projection(concat)
            
        elif self.fusion_method == "attention":
            # Compute attention weights
            # (batch, num_views, 1)
            attn_logits = self.attention(view_embeddings)
            attn_weights = F.softmax(attn_logits, dim=1)
            
            # Weighted sum
            fused = (view_embeddings * attn_weights).sum(dim=1)
            
        elif self.fusion_method == "weighted":
            # Learned fixed weights
            weights = F.softmax(self.view_weights, dim=0)
            fused = (view_embeddings * weights.view(1, -1, 1)).sum(dim=1)
        
        return fused
    
    def get_attention_weights(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability (only for attention fusion).
        
        Args:
            view_embeddings: Tensor of shape (batch, num_views, embedding_dim)
        
        Returns:
            Attention weights of shape (batch, num_views)
        """
        if self.fusion_method != "attention":
            return torch.ones(view_embeddings.shape[0], self.num_views) / self.num_views
        
        attn_logits = self.attention(view_embeddings)
        attn_weights = F.softmax(attn_logits, dim=1).squeeze(-1)
        return attn_weights


class DirectionalEmbedding(nn.Module):
    """
    Optional: Add learned directional embeddings to each view.
    
    This can help the model learn direction-specific patterns
    (e.g., "north-facing views often show X").
    """
    
    def __init__(self, embedding_dim: int, num_views: int = 4):
        super().__init__()
        
        self.direction_embeddings = nn.Parameter(
            torch.randn(num_views, embedding_dim) * 0.02
        )
    
    def forward(self, view_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add directional embeddings to view embeddings.
        
        Args:
            view_embeddings: (batch, num_views, embedding_dim)
        
        Returns:
            Embeddings with directional bias: (batch, num_views, embedding_dim)
        """
        return view_embeddings + self.direction_embeddings.unsqueeze(0)
