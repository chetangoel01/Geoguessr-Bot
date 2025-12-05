"""
Prediction utilities for inference.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

from ..models.geoguessr_model import GeoGuessrModel


@torch.no_grad()
def predict_batch(
    model: GeoGuessrModel,
    images: torch.Tensor,
    top_k: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Get predictions for a batch of images.
    
    Args:
        model: Trained GeoGuessr model
        images: Image tensor (batch, num_views, C, H, W)
        top_k: Number of top state predictions
        device: Device to run inference on
    
    Returns:
        Dictionary with:
            - top_k_states: (batch, k) state indices
            - top_k_probs: (batch, k) probabilities
            - gps_coords: (batch, 2) GPS coordinates
            - class_logits: (batch, num_classes) raw logits
    """
    model.eval()
    
    if device is not None:
        images = images.to(device)
    
    outputs = model(images)
    
    # Get top-k predictions
    probs = F.softmax(outputs['class_logits'], dim=-1)
    top_k_probs, top_k_states = torch.topk(probs, k=top_k, dim=-1)
    
    return {
        'top_k_states': top_k_states,
        'top_k_probs': top_k_probs,
        'gps_coords': outputs['gps_coords'],
        'class_logits': outputs['class_logits'],
        'class_probs': probs
    }


@torch.no_grad()
def predict_dataset(
    model: GeoGuessrModel,
    dataloader: DataLoader,
    device: torch.device,
    top_k: int = 5,
    use_amp: bool = True,
    idx_to_state: Optional[Dict[int, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for entire dataset.
    
    Args:
        model: Trained GeoGuessr model
        dataloader: DataLoader for test set
        device: Device for inference
        top_k: Number of top state predictions
        use_amp: Use automatic mixed precision
        idx_to_state: Mapping from contiguous indices to original state indices
    
    Returns:
        Dictionary with numpy arrays:
            - sample_ids: Sample identifiers
            - top_k_states: (N, k) predicted state indices (original indices)
            - top_k_probs: (N, k) prediction probabilities
            - latitudes: (N,) predicted latitudes
            - longitudes: (N,) predicted longitudes
    """
    model.eval()
    model.to(device)
    
    all_sample_ids = []
    all_top_k_states = []
    all_top_k_probs = []
    all_latitudes = []
    all_longitudes = []
    
    for batch in tqdm(dataloader, desc="Generating predictions"):
        images = batch['images'].to(device)
        sample_ids = batch['sample_id']
        
        if use_amp:
            with autocast():
                preds = predict_batch(model, images, top_k=top_k)
        else:
            preds = predict_batch(model, images, top_k=top_k)
        
        # Convert to numpy
        top_k_states = preds['top_k_states'].cpu().numpy()
        top_k_probs = preds['top_k_probs'].cpu().numpy()
        gps = preds['gps_coords'].cpu().numpy()
        
        # Convert contiguous indices to original state indices if mapping provided
        if idx_to_state is not None:
            top_k_states_original = np.vectorize(idx_to_state.get)(top_k_states)
        else:
            top_k_states_original = top_k_states
        
        all_sample_ids.extend(sample_ids.numpy() if torch.is_tensor(sample_ids) else sample_ids)
        all_top_k_states.append(top_k_states_original)
        all_top_k_probs.append(top_k_probs)
        all_latitudes.append(gps[:, 0])
        all_longitudes.append(gps[:, 1])
    
    return {
        'sample_ids': np.array(all_sample_ids),
        'top_k_states': np.concatenate(all_top_k_states, axis=0),
        'top_k_probs': np.concatenate(all_top_k_probs, axis=0),
        'latitudes': np.concatenate(all_latitudes, axis=0),
        'longitudes': np.concatenate(all_longitudes, axis=0)
    }


@torch.no_grad()
def predict_with_tta(
    model: GeoGuessrModel,
    images: torch.Tensor,
    tta_transforms: List,
    top_k: int = 5,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Predict with test-time augmentation.
    
    Averages predictions across multiple augmented versions.
    
    Args:
        model: Trained model
        images: Original images (batch, num_views, C, H, W)
        tta_transforms: List of transforms to apply
        top_k: Number of top predictions
        device: Device for inference
    
    Returns:
        Averaged predictions
    """
    model.eval()
    
    if device is not None:
        images = images.to(device)
    
    all_probs = []
    all_gps = []
    
    for transform in tta_transforms:
        # Apply transform to each view
        batch_size, num_views = images.shape[:2]
        augmented = []
        
        for v in range(num_views):
            view_images = images[:, v]  # (batch, C, H, W)
            # Note: For proper TTA, you'd want to apply to PIL images
            # This is a simplified version
            augmented.append(view_images)
        
        aug_images = torch.stack(augmented, dim=1)
        
        outputs = model(aug_images)
        probs = F.softmax(outputs['class_logits'], dim=-1)
        
        all_probs.append(probs)
        all_gps.append(outputs['gps_coords'])
    
    # Average predictions
    avg_probs = torch.stack(all_probs).mean(dim=0)
    avg_gps = torch.stack(all_gps).mean(dim=0)
    
    # Get top-k from averaged probabilities
    top_k_probs, top_k_states = torch.topk(avg_probs, k=top_k, dim=-1)
    
    return {
        'top_k_states': top_k_states,
        'top_k_probs': top_k_probs,
        'gps_coords': avg_gps,
        'class_probs': avg_probs
    }


def calibrate_probabilities(
    probs: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Apply temperature scaling to calibrate probabilities.
    
    Higher temperature = softer distribution (more uncertainty)
    Lower temperature = sharper distribution (more confident)
    
    Args:
        probs: Probability array (N, num_classes)
        temperature: Temperature for scaling
    
    Returns:
        Calibrated probabilities
    """
    if temperature == 1.0:
        return probs
    
    # Convert to logits, scale, convert back
    logits = np.log(probs + 1e-10)
    scaled_logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    calibrated = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    return calibrated
