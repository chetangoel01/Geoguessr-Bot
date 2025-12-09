"""
Evaluation metrics for GeoGuessr competition.

Implements the official competition scoring:
- Weighted top-k classification (70%)
- GPS haversine distance (30%)
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from ..config import TOP_K_WEIGHTS, SCORING_MAX_DISTANCE
from ..utils.geo import haversine_distance_torch


def compute_weighted_topk_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted top-k classification score.
    
    Args:
        predictions: Top-k predictions (batch, k) - state indices
        targets: True state indices (batch,)
        k: Number of predictions to consider
    
    Returns:
        score: Mean weighted score (0 to 1)
        details: Dictionary with per-position match rates
    """
    batch_size = predictions.shape[0]
    
    # Weights for each position
    weights = torch.tensor([
        TOP_K_WEIGHTS.get(i + 1, 0) for i in range(k)
    ], device=predictions.device)
    
    scores = torch.zeros(batch_size, device=predictions.device)
    position_matches = {i + 1: 0 for i in range(k)}
    
    for i in range(batch_size):
        target = targets[i]
        preds = predictions[i]
        
        # Find first matching position
        matched = False
        for pos in range(k):
            if preds[pos] == target:
                scores[i] = weights[pos]
                position_matches[pos + 1] += 1
                matched = True
                break
        
        # If no match, score is 0
        if not matched:
            position_matches['no_match'] = position_matches.get('no_match', 0) + 1
    
    # Compute rates
    details = {f'pos_{k}_rate': v / batch_size for k, v in position_matches.items()}
    details['mean_score'] = scores.mean().item()
    
    return scores.mean().item(), details


def compute_gps_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_distance: float = SCORING_MAX_DISTANCE
) -> Tuple[float, Dict[str, float]]:
    """
    Compute GPS regression score.
    
    Score = max(0, 1 - mean_distance_km / max_distance)
    
    Args:
        pred: Predicted GPS (batch, 2)
        target: Target GPS (batch, 2)
        max_distance: Maximum distance threshold in km (default from config)
    
    Returns:
        score: GPS score (0 to 1)
        details: Dictionary with distance statistics
    """
    distances = haversine_distance_torch(pred, target)
    
    mean_dist = distances.mean().item()
    median_dist = distances.median().item()
    
    score = max(0, 1 - mean_dist / max_distance)
    
    details = {
        'mean_distance_km': mean_dist,
        'median_distance_km': median_dist,
        'min_distance_km': distances.min().item(),
        'max_distance_km': distances.max().item(),
        'std_distance_km': distances.std().item(),
        'gps_score': score
    }
    
    return score, details


def compute_competition_score(
    class_predictions: torch.Tensor,
    gps_predictions: torch.Tensor,
    class_targets: torch.Tensor,
    gps_targets: torch.Tensor,
    classification_weight: float = 0.70,
    gps_weight: float = 0.30
) -> Dict[str, float]:
    """
    Compute the full competition score.
    
    Final Score = 0.70 × Classification Score + 0.30 × GPS Score
    
    Args:
        class_predictions: Top-k state predictions (batch, k)
        gps_predictions: GPS predictions (batch, 2)
        class_targets: True state indices (batch,)
        gps_targets: True GPS coordinates (batch, 2)
        classification_weight: Weight for classification (default 0.70)
        gps_weight: Weight for GPS (default 0.30)
    
    Returns:
        Dictionary with all scores and metrics
    """
    # Classification score
    cls_score, cls_details = compute_weighted_topk_score(
        class_predictions, class_targets
    )
    
    # GPS score
    gps_score, gps_details = compute_gps_score(
        gps_predictions, gps_targets
    )
    
    # Combined score
    final_score = classification_weight * cls_score + gps_weight * gps_score
    
    results = {
        'final_score': final_score,
        'classification_score': cls_score,
        'gps_score': gps_score,
        **{f'cls_{k}': v for k, v in cls_details.items()},
        **{f'gps_{k}': v for k, v in gps_details.items()}
    }
    
    return results


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Compute classification metrics from logits.
    
    Args:
        logits: Model output (batch, num_classes)
        targets: True labels (batch,)
        top_k: Number of top predictions
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Get predictions
    probs = torch.softmax(logits, dim=-1)
    top_k_preds = torch.topk(probs, k=top_k, dim=-1).indices
    
    # Top-1 accuracy
    top1_preds = top_k_preds[:, 0]
    top1_acc = (top1_preds == targets).float().mean().item()
    
    # Top-k accuracy
    topk_correct = (top_k_preds == targets.unsqueeze(1)).any(dim=1)
    topk_acc = topk_correct.float().mean().item()
    
    # Weighted score
    weighted_score, _ = compute_weighted_topk_score(top_k_preds, targets, k=top_k)
    
    return {
        'top1_accuracy': top1_acc,
        f'top{top_k}_accuracy': topk_acc,
        'weighted_score': weighted_score
    }


def compute_gps_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute GPS regression metrics.
    
    Args:
        pred: Predicted GPS (batch, 2)
        target: Target GPS (batch, 2)
    
    Returns:
        Dictionary with distance metrics
    """
    distances = haversine_distance_torch(pred, target)
    
    # Percentile thresholds
    thresholds = [1, 25, 100, 200, 500, 1000, 2500]
    
    metrics = {
        'mean_km': distances.mean().item(),
        'median_km': distances.median().item(),
        'std_km': distances.std().item()
    }
    
    for thresh in thresholds:
        pct = (distances <= thresh).float().mean().item()
        metrics[f'within_{thresh}km'] = pct
    
    return metrics


class MetricTracker:
    """
    Track and accumulate metrics over batches.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked values."""
        self.class_preds = []
        self.gps_preds = []
        self.class_targets = []
        self.gps_targets = []
    
    def update(
        self,
        class_logits: torch.Tensor,
        gps_pred: torch.Tensor,
        class_targets: torch.Tensor,
        gps_targets: torch.Tensor,
        top_k: int = 5
    ):
        """Add batch results to tracker."""
        with torch.no_grad():
            probs = torch.softmax(class_logits, dim=-1)
            top_k_preds = torch.topk(probs, k=top_k, dim=-1).indices
            
            self.class_preds.append(top_k_preds.cpu())
            self.gps_preds.append(gps_pred.cpu())
            self.class_targets.append(class_targets.cpu())
            self.gps_targets.append(gps_targets.cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics from all accumulated batches."""
        class_preds = torch.cat(self.class_preds, dim=0)
        gps_preds = torch.cat(self.gps_preds, dim=0)
        class_targets = torch.cat(self.class_targets, dim=0)
        gps_targets = torch.cat(self.gps_targets, dim=0)
        
        return compute_competition_score(
            class_preds, gps_preds,
            class_targets, gps_targets
        )
