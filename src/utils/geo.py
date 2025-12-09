"""
Geographic utilities and calculations.
"""

import numpy as np
import torch
from typing import Union, Tuple

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculate the haversine distance between two GPS coordinates in kilometers.
    
    Args:
        lat1, lon1: First coordinate (degrees)
        lat2, lon2: Second coordinate (degrees)
    
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def haversine_distance_batch(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> np.ndarray:
    """
    Vectorized haversine distance calculation (NumPy).
    
    Args:
        coords1: Array of shape (N, 2) with [lat, lon]
        coords2: Array of shape (M, 2) with [lat, lon]
    
    Returns:
        Distance matrix of shape (N, M) in kilometers
    """
    lat1 = np.radians(coords1[:, 0:1])
    lon1 = np.radians(coords1[:, 1:2])
    lat2 = np.radians(coords2[:, 0:1]).T
    lon2 = np.radians(coords2[:, 1:2]).T
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    return EARTH_RADIUS_KM * c


def haversine_distance_torch(
    pred: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Compute haversine distance in kilometers (PyTorch).
    
    Args:
        pred: Predicted GPS (batch, 2) - [lat, lon] in degrees
        target: Target GPS (batch, 2) - [lat, lon] in degrees
    
    Returns:
        Distances in km (batch,)
    """
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
    
    return EARTH_RADIUS_KM * c
