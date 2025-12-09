"""
Image preprocessing and augmentation transforms.

Uses transforms compatible with CLIP models (224x224 or 336x336 input).
"""

from torchvision import transforms
from typing import Tuple


# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# StreetCLIP uses 336x336 input (ViT-L/14-336)
STREETCLIP_IMAGE_SIZE = 336

# Standard CLIP uses 224x224
CLIP_IMAGE_SIZE = 224


def get_train_transforms(
    image_size: int = STREETCLIP_IMAGE_SIZE,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get training transforms with optional augmentation.
    
    Args:
        image_size: Target image size (336 for StreetCLIP, 224 for base CLIP)
        use_augmentation: Whether to apply data augmentation
    
    Returns:
        Composed transforms
    """
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # Light augmentations that don't destroy geographic information
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05  # Small hue shift to preserve colors
                )
            ], p=0.5),
            transforms.ToTensor(),
            # Random erasing BEFORE normalization so it operates on [0,1] range
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), value=0),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])


def get_val_transforms(
    image_size: int = STREETCLIP_IMAGE_SIZE
) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size
    
    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_tta_transforms(
    image_size: int = STREETCLIP_IMAGE_SIZE
) -> list:
    """
    Get test-time augmentation transforms.
    
    Returns multiple transform variants for TTA.
    Note: We avoid horizontal flips as they would swap E/W directions
    and corrupt geographic information.
    
    Args:
        image_size: Target image size
    
    Returns:
        List of transform compositions
    """
    base_transform = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
    
    tta_variants = [
        # Original
        transforms.Compose(base_transform),
        
        # Slight brightness variations
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=-0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
        
        # Slight contrast variations
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]),
    ]
    
    return tta_variants


def denormalize(
    tensor,
    mean: Tuple[float, ...] = CLIP_MEAN,
    std: Tuple[float, ...] = CLIP_STD
):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std
    
    Returns:
        Denormalized tensor
    """
    import torch
    
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean
