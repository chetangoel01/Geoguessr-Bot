"""
Backbone model loading for StreetCLIP and GeoCLIP.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import CLIPModel, CLIPProcessor
from ..config import Config


class StreetCLIPBackbone(nn.Module):
    """
    StreetCLIP vision encoder wrapper.
    
    StreetCLIP is a CLIP model fine-tuned on street-level imagery
    with geographic captions.
    """
    
    def __init__(
        self,
        model_name: str = "geolocal/StreetCLIP",
        freeze: bool = True
    ):
        super().__init__()
        
        # Load the full CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = self.clip_model.vision_model
        
        # Get embedding dimension
        self.embedding_dim = self.clip_model.config.vision_config.hidden_size
        
        # Freeze if requested
        if freeze:
            self.freeze()
        
        print(f"Loaded StreetCLIP backbone with embedding dim: {self.embedding_dim}")
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        print("StreetCLIP backbone frozen")
    
    def unfreeze(self, unfreeze_layers: Optional[int] = None):
        """
        Unfreeze backbone parameters.
        
        Args:
            unfreeze_layers: If specified, only unfreeze the last N encoder layers.
                           If None, unfreeze everything.
        """
        if unfreeze_layers is None:
            for param in self.vision_model.parameters():
                param.requires_grad = True
            print("StreetCLIP backbone fully unfrozen")
        else:
            # First freeze everything
            for param in self.vision_model.parameters():
                param.requires_grad = False
            
            # Unfreeze last N layers
            encoder_layers = self.vision_model.encoder.layers
            num_layers = len(encoder_layers)
            
            for i in range(max(0, num_layers - unfreeze_layers), num_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
            
            # Also unfreeze the final layer norm and pooler
            if hasattr(self.vision_model, 'post_layernorm'):
                for param in self.vision_model.post_layernorm.parameters():
                    param.requires_grad = True
            
            print(f"Unfroze last {unfreeze_layers} layers of StreetCLIP")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            pixel_values: Image tensor of shape (batch, channels, height, width)
        
        Returns:
            Pooled embeddings of shape (batch, embedding_dim)
        """
        outputs = self.vision_model(pixel_values=pixel_values)
        
        # Use the pooler output (CLS token after projection)
        # Shape: (batch, embedding_dim)
        return outputs.pooler_output


class GeoCLIPBackbone(nn.Module):
    """
    GeoCLIP vision encoder wrapper.
    
    GeoCLIP is trained to align images with GPS coordinates
    using contrastive learning.
    """
    
    def __init__(
        self,
        freeze: bool = True
    ):
        super().__init__()
        
        # Import geoclip
        try:
            from geoclip import GeoCLIP
        except ImportError:
            raise ImportError("Please install geoclip: pip install geoclip")
        
        # Load GeoCLIP model
        self.geoclip = GeoCLIP()
        self.image_encoder = self.geoclip.image_encoder
        
        # GeoCLIP uses CLIP ViT-L/14 with 768-dim embeddings
        self.embedding_dim = 512  # GeoCLIP projects to 512
        
        if freeze:
            self.freeze()
        
        print(f"Loaded GeoCLIP backbone with embedding dim: {self.embedding_dim}")
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("GeoCLIP backbone frozen")
    
    def unfreeze(self, unfreeze_layers: Optional[int] = None):
        """Unfreeze backbone parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        print("GeoCLIP backbone unfrozen")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the vision encoder.
        
        Args:
            pixel_values: Image tensor of shape (batch, channels, height, width)
        
        Returns:
            Image embeddings of shape (batch, embedding_dim)
        """
        # GeoCLIP's image encoder returns normalized embeddings
        embeddings = self.image_encoder(pixel_values)
        return embeddings


def load_streetclip(
    model_name: str = "geolocal/StreetCLIP",
    freeze: bool = True
) -> StreetCLIPBackbone:
    """Load StreetCLIP backbone."""
    return StreetCLIPBackbone(model_name=model_name, freeze=freeze)


def load_geoclip(freeze: bool = True) -> GeoCLIPBackbone:
    """Load GeoCLIP backbone."""
    return GeoCLIPBackbone(freeze=freeze)


def get_backbone(config: Config) -> nn.Module:
    """
    Get the appropriate backbone based on config.
    
    Args:
        config: Configuration object
    
    Returns:
        Backbone model
    """
    if config.model.backbone_type == "streetclip":
        return load_streetclip(
            model_name=config.model.backbone_name,
            freeze=config.model.freeze_backbone
        )
    elif config.model.backbone_type == "geoclip":
        return load_geoclip(freeze=config.model.freeze_backbone)
    else:
        raise ValueError(f"Unknown backbone type: {config.model.backbone_type}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
