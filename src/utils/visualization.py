"""
Visualization utilities for debugging and analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def visualize_sample(
    images: torch.Tensor,
    predictions: Optional[Dict] = None,
    ground_truth: Optional[Dict] = None,
    state_names: Optional[Dict[int, str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a single sample with 4 directional views.
    
    Args:
        images: Tensor of shape (4, C, H, W) - the 4 views
        predictions: Dict with 'top_k_states', 'gps_coords', etc.
        ground_truth: Dict with 'state', 'latitude', 'longitude'
        state_names: Mapping from state index to name
        save_path: Path to save figure
    """
    from .seed import set_seed  # Avoid circular import
    from ..data.preprocessing import denormalize
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    directions = ['North', 'East', 'South', 'West']
    
    for i, (ax, direction) in enumerate(zip(axes, directions)):
        img = images[i]
        
        # Denormalize
        img = denormalize(img)
        
        # Convert to numpy for display
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        
        ax.imshow(img_np)
        ax.set_title(direction)
        ax.axis('off')
    
    # Add title with predictions
    title_parts = []
    
    if predictions is not None:
        pred_states = predictions.get('top_k_states', [])
        if len(pred_states) > 0:
            top_state = pred_states[0]
            if state_names:
                state_name = state_names.get(int(top_state), f"State {top_state}")
            else:
                state_name = f"State {top_state}"
            title_parts.append(f"Pred: {state_name}")
        
        gps = predictions.get('gps_coords')
        if gps is not None:
            title_parts.append(f"GPS: ({gps[0]:.2f}, {gps[1]:.2f})")
    
    if ground_truth is not None:
        true_state = ground_truth.get('state')
        if true_state is not None:
            if state_names:
                state_name = state_names.get(int(true_state), f"State {true_state}")
            else:
                state_name = f"State {true_state}"
            title_parts.append(f"True: {state_name}")
        
        lat = ground_truth.get('latitude')
        lon = ground_truth.get('longitude')
        if lat is not None and lon is not None:
            title_parts.append(f"True GPS: ({lat:.2f}, {lon:.2f})")
    
    if title_parts:
        fig.suptitle(' | '.join(title_parts), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_predictions(
    samples: List[Dict],
    predictions: Dict[str, np.ndarray],
    idx_to_state: Dict[int, int],
    state_names: Dict[int, str],
    n_samples: int = 8,
    save_dir: Optional[str] = None
):
    """
    Visualize multiple predictions.
    
    Args:
        samples: List of sample dicts from dataset
        predictions: Predictions from predict_dataset
        idx_to_state: Mapping from contiguous to original state indices
        state_names: Mapping from state index to name
        n_samples: Number of samples to visualize
        save_dir: Directory to save figures
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(n_samples, len(samples))):
        sample = samples[i]
        
        pred_dict = {
            'top_k_states': predictions['top_k_states'][i],
            'gps_coords': np.array([predictions['latitudes'][i], 
                                   predictions['longitudes'][i]])
        }
        
        gt_dict = None
        if 'state_idx' in sample:
            gt_dict = {
                'state': sample['state_idx'],
                'latitude': sample.get('latitude'),
                'longitude': sample.get('longitude')
            }
        
        save_path = save_dir / f"sample_{i}.png" if save_dir else None
        
        visualize_sample(
            sample['images'],
            predictions=pred_dict,
            ground_truth=gt_dict,
            state_names=state_names,
            save_path=save_path
        )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional class names
        normalize: Whether to normalize by row
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    if class_names:
        ax.set(xticks=np.arange(len(class_names)),
               yticks=np.arange(len(class_names)),
               xticklabels=class_names,
               yticklabels=class_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training curves.
    
    Args:
        history: Dict with 'train_loss', 'val_loss', 'val_score', etc.
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curves
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Validation score
    if 'val_score' in history:
        axes[1].plot(history['val_score'], label='Val Score', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Validation Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rates' in history:
        axes[2].plot(history['learning_rates'], label='LR', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gps_predictions(
    pred_lats: np.ndarray,
    pred_lons: np.ndarray,
    true_lats: Optional[np.ndarray] = None,
    true_lons: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot GPS predictions on a map.
    
    Args:
        pred_lats: Predicted latitudes
        pred_lons: Predicted longitudes
        true_lats: True latitudes (optional)
        true_lons: True longitudes (optional)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot predictions
    ax.scatter(pred_lons, pred_lats, c='blue', alpha=0.5, s=10, label='Predictions')
    
    # Plot ground truth if available
    if true_lats is not None and true_lons is not None:
        ax.scatter(true_lons, true_lats, c='red', alpha=0.3, s=10, label='Ground Truth')
    
    # Set US bounds
    ax.set_xlim(-130, -65)
    ax.set_ylim(20, 55)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('GPS Predictions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention(
    model,
    images: torch.Tensor,
    device: torch.device,
    save_path: Optional[str] = None
):
    """
    Visualize attention weights from the model.
    
    For ViT-based models, shows which image regions the model attends to.
    
    Args:
        model: The GeoGuessr model
        images: Input images (1, 4, C, H, W)
        device: Device for inference
        save_path: Path to save figure
    """
    from ..data.preprocessing import denormalize
    
    model.eval()
    images = images.to(device)
    
    # Get attention weights if using attention fusion
    if hasattr(model.fusion, 'get_attention_weights'):
        with torch.no_grad():
            view_embeddings = model.encode_views(images)
            attn_weights = model.fusion.get_attention_weights(view_embeddings)
        
        attn_weights = attn_weights.cpu().numpy()[0]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        directions = ['North', 'East', 'South', 'West']
        
        for i, (ax, direction) in enumerate(zip(axes, directions)):
            img = images[0, i]
            img = denormalize(img)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            
            ax.imshow(img_np)
            ax.set_title(f"{direction}\nWeight: {attn_weights[i]:.3f}")
            ax.axis('off')
            
            # Add border based on attention weight
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(attn_weights[i] * 10)
                spine.set_color('red')
        
        plt.suptitle('View Attention Weights')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print("Model doesn't use attention fusion - weights are equal")


def plot_state_distribution(
    predictions: np.ndarray,
    state_names: Dict[int, str],
    title: str = "State Prediction Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of predicted states.
    
    Args:
        predictions: Array of predicted state indices
        state_names: Mapping from index to state name
        title: Plot title
        save_path: Path to save figure
    """
    unique, counts = np.unique(predictions, return_counts=True)
    
    names = [state_names.get(int(idx), f"Unknown ({idx})") for idx in unique]
    
    # Sort by count
    sorted_idx = np.argsort(-counts)
    names = [names[i] for i in sorted_idx]
    counts = counts[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, counts, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Count')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
