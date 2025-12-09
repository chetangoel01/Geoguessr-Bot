"""
State-related utilities for the GeoGuessr competition.

Includes:
- Haversine distance calculation
- State centroid computation
- Soft label generation for haversine-smoothed loss
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

from ..utils.geo import haversine_distance, haversine_distance_batch


def compute_state_centroids(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Compute the centroid (mean GPS) for each state in the dataset.
    
    Args:
        df: DataFrame with 'state_idx', 'latitude', 'longitude' columns
        save_path: Optional path to save centroids as JSON
    
    Returns:
        Dictionary mapping state_idx to (latitude, longitude)
    """
    centroids = {}
    
    for state_idx in df['state_idx'].unique():
        state_df = df[df['state_idx'] == state_idx]
        centroid_lat = state_df['latitude'].mean()
        centroid_lon = state_df['longitude'].mean()
        centroids[int(state_idx)] = (float(centroid_lat), float(centroid_lon))
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(centroids, f, indent=2)
    
    return centroids


def load_state_centroids(path: Path) -> Dict[int, Tuple[float, float]]:
    """Load state centroids from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    # Convert string keys back to int
    return {int(k): tuple(v) for k, v in data.items()}


def compute_haversine_matrix(
    centroids: Dict[int, Tuple[float, float]],
    state_to_idx: Dict[int, int],
    save_path: Optional[Path] = None
) -> np.ndarray:
    """
    Compute pairwise haversine distance matrix between state centroids.
    
    Args:
        centroids: Dict mapping state_idx to (lat, lon)
        state_to_idx: Dict mapping original state indices to contiguous indices
        save_path: Optional path to save matrix as .npy
    
    Returns:
        Distance matrix of shape (num_states, num_states)
    """
    num_states = len(state_to_idx)
    distance_matrix = np.zeros((num_states, num_states))
    
    # Get ordered list of original state indices
    idx_to_state = {v: k for k, v in state_to_idx.items()}
    
    for i in range(num_states):
        for j in range(num_states):
            state_i = idx_to_state[i]
            state_j = idx_to_state[j]
            
            lat1, lon1 = centroids[state_i]
            lat2, lon2 = centroids[state_j]
            
            distance_matrix[i, j] = haversine_distance(lat1, lon1, lat2, lon2)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, distance_matrix)
    
    return distance_matrix


def load_haversine_matrix(path: Path) -> np.ndarray:
    """Load haversine distance matrix from .npy file."""
    return np.load(path)


def get_soft_labels(
    target_state: int,
    distance_matrix: np.ndarray,
    temperature: float = 300.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate soft labels using haversine-smoothed distribution.
    
    The soft label for state i given true state j is:
        p(i|j) = exp(-distance(i,j) / temperature) / Z
    
    This gives partial credit to nearby states.
    
    Args:
        target_state: Contiguous index of the true state (0-32)
        distance_matrix: Pairwise distance matrix
        temperature: Smoothing temperature in km (higher = smoother)
        device: Torch device for output tensor
    
    Returns:
        Soft label tensor of shape (num_states,)
    """
    distances = distance_matrix[target_state]
    
    # Compute soft labels
    log_probs = -distances / temperature
    soft_labels = np.exp(log_probs - log_probs.max())  # Numerical stability
    soft_labels = soft_labels / soft_labels.sum()
    
    tensor = torch.tensor(soft_labels, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def get_soft_labels_batch(
    target_states: torch.Tensor,
    distance_matrix: np.ndarray,
    temperature: float = 300.0
) -> torch.Tensor:
    """
    Generate soft labels for a batch of targets.
    
    Args:
        target_states: Tensor of shape (batch_size,) with contiguous state indices
        distance_matrix: Pairwise distance matrix
        temperature: Smoothing temperature in km
    
    Returns:
        Soft labels tensor of shape (batch_size, num_states)
    """
    batch_size = target_states.shape[0]
    num_states = distance_matrix.shape[0]
    
    # Get distances for each target
    target_np = target_states.cpu().numpy()
    distances = distance_matrix[target_np]  # (batch_size, num_states)
    
    # Compute soft labels
    log_probs = -distances / temperature
    log_probs = log_probs - log_probs.max(axis=1, keepdims=True)  # Stability
    soft_labels = np.exp(log_probs)
    soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)
    
    return torch.tensor(soft_labels, dtype=torch.float32, device=target_states.device)


def get_neighboring_states(
    state_idx: int,
    distance_matrix: np.ndarray,
    threshold_km: float = 500.0
) -> List[int]:
    """
    Get indices of states within a distance threshold.
    
    Args:
        state_idx: Contiguous state index
        distance_matrix: Pairwise distance matrix
        threshold_km: Distance threshold in kilometers
    
    Returns:
        List of neighboring state indices
    """
    distances = distance_matrix[state_idx]
    neighbors = np.where(distances <= threshold_km)[0]
    return neighbors.tolist()


class StateLabelEncoder:
    """
    Handles conversion between original state indices and contiguous labels.
    Also provides soft label generation.
    """
    
    def __init__(
        self,
        state_to_idx: Dict[int, int],
        centroids: Optional[Dict[int, Tuple[float, float]]] = None,
        distance_matrix: Optional[np.ndarray] = None
    ):
        self.state_to_idx = state_to_idx
        self.idx_to_state = {v: k for k, v in state_to_idx.items()}
        self.num_classes = len(state_to_idx)
        self.centroids = centroids
        self.distance_matrix = distance_matrix
    
    def encode(self, state_idx: int) -> int:
        """Convert original state index to contiguous label."""
        return self.state_to_idx[state_idx]
    
    def decode(self, label: int) -> int:
        """Convert contiguous label back to original state index."""
        return self.idx_to_state[label]
    
    def get_soft_labels(
        self,
        target: int,
        temperature: float = 300.0
    ) -> torch.Tensor:
        """Get soft labels for a single target."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix required for soft labels")
        return get_soft_labels(target, self.distance_matrix, temperature)
    
    def get_soft_labels_batch(
        self,
        targets: torch.Tensor,
        temperature: float = 300.0
    ) -> torch.Tensor:
        """Get soft labels for a batch of targets."""
        if self.distance_matrix is None:
            raise ValueError("Distance matrix required for soft labels")
        return get_soft_labels_batch(targets, self.distance_matrix, temperature)
    
    def get_centroid(self, label: int) -> Tuple[float, float]:
        """Get the centroid GPS coordinates for a contiguous label."""
        if self.centroids is None:
            raise ValueError("Centroids required")
        state_idx = self.idx_to_state[label]
        return self.centroids[state_idx]
