"""
Ensemble utilities for combining predictions from multiple models.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import torch


def _average_gps(
    predictions_list: List[Dict[str, np.ndarray]],
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Helper to average GPS coordinates."""
    n_samples = len(predictions_list[0]['sample_ids'])
    avg_lat = np.zeros(n_samples)
    avg_lon = np.zeros(n_samples)
    
    if weights is None:
        n_models = len(predictions_list)
        weights = [1.0 / n_models] * n_models
    
    for i, preds in enumerate(predictions_list):
        avg_lat += weights[i] * preds['latitudes']
        avg_lon += weights[i] * preds['longitudes']
        
    return avg_lat, avg_lon


def ensemble_predictions(
    predictions_list: List[Dict[str, np.ndarray]],
    method: str = "average",
    weights: Optional[List[float]] = None,
    top_k: int = 5
) -> Dict[str, np.ndarray]:
    """
    Ensemble predictions from multiple models.
    
    Args:
        predictions_list: List of prediction dictionaries from predict_dataset
        method: Ensemble method ("average", "weighted", "voting")
        weights: Optional weights for weighted average
        top_k: Number of top predictions to return
    
    Returns:
        Ensembled predictions
    """
    n_models = len(predictions_list)
    
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Get sample IDs from first model (should be same for all)
    sample_ids = predictions_list[0]['sample_ids']
    n_samples = len(sample_ids)
    
    if method in ["average", "weighted"]:
        # Average probabilities (need to reconstruct from top-k)
        # This is approximate - ideally we'd have full probability vectors
        
        # For GPS, we can directly average
        avg_lat, avg_lon = _average_gps(predictions_list, weights)
        
        # For classification, use voting with probability weighting
        # Collect all predictions and their probabilities
        all_votes = np.zeros((n_samples, 50))  # Max state index is 49
        
        for i, preds in enumerate(predictions_list):
            for k in range(min(top_k, preds['top_k_states'].shape[1])):
                states = preds['top_k_states'][:, k]
                probs = preds['top_k_probs'][:, k]
                
                for sample_idx in range(n_samples):
                    state = states[sample_idx]
                    prob = probs[sample_idx]
                    all_votes[sample_idx, state] += weights[i] * prob
        
        # Get top-k from aggregated votes
        top_k_indices = np.argsort(-all_votes, axis=1)[:, :top_k]
        top_k_probs = np.take_along_axis(all_votes, top_k_indices, axis=1)
        
        # Normalize probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(axis=1, keepdims=True) + 1e-10)
        
        return {
            'sample_ids': sample_ids,
            'top_k_states': top_k_indices,
            'top_k_probs': top_k_probs,
            'latitudes': avg_lat,
            'longitudes': avg_lon
        }
    
    elif method == "voting":
        # Majority voting for classification
        all_votes = np.zeros((n_samples, 50))
        
        for preds in predictions_list:
            # Count first prediction as vote
            for sample_idx, state in enumerate(preds['top_k_states'][:, 0]):
                all_votes[sample_idx, state] += 1
        
        top_k_indices = np.argsort(-all_votes, axis=1)[:, :top_k]
        top_k_votes = np.take_along_axis(all_votes, top_k_indices, axis=1)
        top_k_probs = top_k_votes / n_models
        
        # Average GPS
        avg_lat, avg_lon = _average_gps(predictions_list)
        
        return {
            'sample_ids': sample_ids,
            'top_k_states': top_k_indices,
            'top_k_probs': top_k_probs,
            'latitudes': avg_lat,
            'longitudes': avg_lon
        }
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def weighted_ensemble(
    predictions_list: List[Dict[str, np.ndarray]],
    val_scores: List[float],
    top_k: int = 5
) -> Dict[str, np.ndarray]:
    """
    Ensemble with weights proportional to validation scores.
    
    Args:
        predictions_list: List of prediction dictionaries
        val_scores: Validation scores for each model
        top_k: Number of top predictions
    
    Returns:
        Ensembled predictions
    """
    # Use validation scores as weights
    weights = [score ** 2 for score in val_scores]  # Square to amplify differences
    
    return ensemble_predictions(
        predictions_list,
        method="weighted",
        weights=weights,
        top_k=top_k
    )


def rank_fusion(
    predictions_list: List[Dict[str, np.ndarray]],
    k: int = 60,
    top_k: int = 5
) -> Dict[str, np.ndarray]:
    """
    Reciprocal Rank Fusion for combining rankings.
    
    RRF score = sum(1 / (k + rank_i)) for each model i
    
    Args:
        predictions_list: List of prediction dictionaries
        k: RRF parameter (typically 60)
        top_k: Number of top predictions
    
    Returns:
        Ensembled predictions using RRF
    """
    n_samples = len(predictions_list[0]['sample_ids'])
    rrf_scores = np.zeros((n_samples, 50))  # Max state index
    
    for preds in predictions_list:
        for sample_idx in range(n_samples):
            for rank, state in enumerate(preds['top_k_states'][sample_idx]):
                rrf_scores[sample_idx, state] += 1.0 / (k + rank + 1)
    
    # Get top-k from RRF scores
    top_k_indices = np.argsort(-rrf_scores, axis=1)[:, :top_k]
    top_k_scores = np.take_along_axis(rrf_scores, top_k_indices, axis=1)
    
    # Normalize to probabilities
    top_k_probs = top_k_scores / (top_k_scores.sum(axis=1, keepdims=True) + 1e-10)
    
    # Average GPS
    avg_lat, avg_lon = _average_gps(predictions_list)
    
    return {
        'sample_ids': predictions_list[0]['sample_ids'],
        'top_k_states': top_k_indices,
        'top_k_probs': top_k_probs,
        'latitudes': avg_lat,
        'longitudes': avg_lon
    }


def stacking_blend(
    train_preds: List[Dict[str, np.ndarray]],
    train_targets: Dict[str, np.ndarray],
    test_preds: List[Dict[str, np.ndarray]],
    top_k: int = 5
) -> Dict[str, np.ndarray]:
    """
    Learn optimal blending weights from validation predictions.
    
    Args:
        train_preds: Validation predictions from each model
        train_targets: True validation targets
        test_preds: Test predictions to blend
        top_k: Number of top predictions
    
    Returns:
        Blended test predictions
    """
    from scipy.optimize import minimize
    
    n_models = len(train_preds)
    
    def score_blend(weights):
        """Negative score for minimization."""
        weights = np.abs(weights)  # Keep positive
        weights = weights / weights.sum()  # Normalize
        
        blended = ensemble_predictions(train_preds, method="weighted", weights=list(weights))
        
        # Compute approximate score (top-1 accuracy as proxy)
        correct = (blended['top_k_states'][:, 0] == train_targets['state_labels']).mean()
        
        return -correct
    
    # Optimize weights
    initial_weights = np.ones(n_models) / n_models
    result = minimize(score_blend, initial_weights, method='Nelder-Mead')
    
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    print(f"Optimal blend weights: {optimal_weights}")
    
    return ensemble_predictions(
        test_preds,
        method="weighted",
        weights=list(optimal_weights),
        top_k=top_k
    )