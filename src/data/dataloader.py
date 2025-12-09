"""
DataLoader factory for GeoGuessr competition.
"""

import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split

from .dataset import GeoGuessrDataset, GeoGuessrSubset
from .preprocessing import get_train_transforms, get_val_transforms
from .state_utils import (
    compute_state_centroids,
    compute_haversine_matrix,
    StateLabelEncoder
)
from ..config import Config
from ..utils.seed import seed_worker, get_generator


def _get_image_size(config: Config) -> int:
    """Determine image size based on backbone."""
    return 336 if config.model.backbone_type == "streetclip" else 224


def create_dataloaders(
    config: Config,
    return_encoder: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[StateLabelEncoder]]:
    """
    Create train and validation DataLoaders.
    
    Args:
        config: Configuration object
        return_encoder: Whether to return the StateLabelEncoder
    
    Returns:
        train_loader, val_loader, and optionally label_encoder
    """
    # Load training CSV
    train_df = pd.read_csv(config.paths.train_csv)
    
    # Get transforms
    image_size = _get_image_size(config)
    train_transform = get_train_transforms(
        image_size=image_size,
        use_augmentation=True
    )
    val_transform = get_val_transforms(
        image_size=image_size
    )
    
    # Create full dataset to get state mapping
    full_dataset = GeoGuessrDataset(
        csv_path=str(config.paths.train_csv),
        image_dir=str(config.paths.train_images),
        transform=train_transform,
        is_test=False
    )
    
    state_to_idx, idx_to_state = full_dataset.get_state_mapping()
    num_classes = full_dataset.num_classes
    
    print(f"Found {num_classes} unique states in the dataset")
    print(f"Total samples: {len(full_dataset)}")
    
    # Compute or load state centroids
    if config.paths.state_centroids.exists():
        from .state_utils import load_state_centroids
        centroids = load_state_centroids(config.paths.state_centroids)
        print("Loaded existing state centroids")
    else:
        centroids = compute_state_centroids(train_df, config.paths.state_centroids)
        print("Computed and saved state centroids")
    
    # Compute or load haversine distance matrix
    if config.paths.haversine_matrix.exists():
        from .state_utils import load_haversine_matrix
        distance_matrix = load_haversine_matrix(config.paths.haversine_matrix)
        print("Loaded existing haversine matrix")
    else:
        distance_matrix = compute_haversine_matrix(
            centroids, state_to_idx, config.paths.haversine_matrix
        )
        print("Computed and saved haversine distance matrix")
    
    # Create stratified train/val split to ensure all states are represented
    # Get state labels for stratification
    state_labels = train_df['state_idx'].values

    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=config.training.val_split,
        stratify=state_labels,
        random_state=config.training.seed
    )
    train_indices = list(train_indices)
    val_indices = list(val_indices)
    
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Create separate datasets for train and val with appropriate transforms
    train_dataset_with_transform = GeoGuessrDataset(
        csv_path=str(config.paths.train_csv),
        image_dir=str(config.paths.train_images),
        transform=train_transform,
        state_to_idx=state_to_idx,
        is_test=False
    )
    
    val_dataset_with_transform = GeoGuessrDataset(
        csv_path=str(config.paths.train_csv),
        image_dir=str(config.paths.train_images),
        transform=val_transform,
        state_to_idx=state_to_idx,
        is_test=False
    )
    
    # Create subsets
    train_subset = GeoGuessrSubset(train_dataset_with_transform, train_indices)
    val_subset = GeoGuessrSubset(val_dataset_with_transform, val_indices)
    
    # Create DataLoaders with optimized settings for large batches
    train_loader = DataLoader(
        train_subset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=4 if config.training.num_workers > 0 else 2,
        worker_init_fn=seed_worker if config.training.num_workers > 0 else None,
        generator=get_generator(config.training.seed)
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=4 if config.training.num_workers > 0 else 2,
        worker_init_fn=seed_worker if config.training.num_workers > 0 else None
    )
    
    if return_encoder:
        encoder = StateLabelEncoder(
            state_to_idx=state_to_idx,
            centroids=centroids,
            distance_matrix=distance_matrix
        )
        return train_loader, val_loader, encoder
    
    return train_loader, val_loader, None


def create_test_dataloader(
    config: Config,
    state_to_idx: Dict[int, int]
) -> DataLoader:
    """
    Create test DataLoader for inference.
    
    Args:
        config: Configuration object
        state_to_idx: State index mapping from training
    
    Returns:
        test_loader
    """
    transform = get_val_transforms(
        image_size=_get_image_size(config)
    )
    
    test_dataset = GeoGuessrDataset(
        csv_path=str(config.paths.test_csv),
        image_dir=str(config.paths.test_images),
        transform=transform,
        state_to_idx=state_to_idx,
        is_test=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.inference.num_workers,
        pin_memory=True
    )
    
    return test_loader


def get_class_weights(
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Uses inverse frequency weighting.
    
    Args:
        train_loader: Training DataLoader
        num_classes: Number of classes
        device: Device for output tensor
    
    Returns:
        Class weight tensor
    """
    class_counts = torch.zeros(num_classes)
    
    for batch in train_loader:
        labels = batch['state_label']
        for label in labels:
            class_counts[label] += 1
    
    # Inverse frequency weighting
    total = class_counts.sum()
    weights = total / (num_classes * class_counts + 1e-6)
    
    # Normalize
    weights = weights / weights.sum() * num_classes
    
    return weights.to(device)
