"""
PyTorch Dataset for GeoGuessr 4-view street images.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple, List
import pandas as pd
import numpy as np


class GeoGuessrDataset(Dataset):
    """
    Dataset for GeoGuessr competition with 4 directional views per sample.
    
    Each sample contains:
        - 4 images (north, east, south, west)
        - State index (classification target)
        - GPS coordinates (regression target)
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        state_to_idx: Optional[Dict[int, int]] = None,
        is_test: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV file (train_ground_truth.csv or sample_submission.csv)
            image_dir: Directory containing images
            transform: Torchvision transforms to apply to images
            state_to_idx: Mapping from original state indices to contiguous indices (0-32)
            is_test: If True, don't try to load labels
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.is_test = is_test
        
        # Directions in the order they appear in the CSV
        self.directions = ['north', 'east', 'south', 'west']
        
        # Create state index mapping if not provided
        if state_to_idx is None and not is_test:
            unique_states = sorted(self.df['state_idx'].unique())
            self.state_to_idx = {s: i for i, s in enumerate(unique_states)}
            self.idx_to_state = {i: s for s, i in self.state_to_idx.items()}
        else:
            self.state_to_idx = state_to_idx
            if state_to_idx is not None:
                self.idx_to_state = {i: s for s, i in state_to_idx.items()}
            else:
                self.idx_to_state = None
                
        # Store original state indices for reference
        if not is_test:
            self.original_state_indices = sorted(self.df['state_idx'].unique())
            self.num_classes = len(self.original_state_indices)
        else:
            self.original_state_indices = None
            self.num_classes = None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
                - images: Tensor of shape (4, C, H, W) for 4 views
                - state_idx: Original state index (for submission)
                - state_label: Contiguous state label 0-32 (for training)
                - latitude: GPS latitude
                - longitude: GPS longitude
                - sample_id: Sample identifier
        """
        row = self.df.iloc[idx]
        
        # Load all 4 directional images
        images = []
        for direction in self.directions:
            img_filename = row[f'image_{direction}']
            img_path = self.image_dir / img_filename
            
            image = Image.open(img_path).convert('RGB')
            
            if self.transform is not None:
                image = self.transform(image)
            
            images.append(image)
        
        # Stack images: (4, C, H, W)
        images = torch.stack(images, dim=0)
        
        result = {
            'images': images,
            'sample_id': row['sample_id']
        }
        
        # Add labels if training
        if not self.is_test:
            original_state_idx = row['state_idx']
            result['state_idx'] = original_state_idx
            result['state_label'] = self.state_to_idx[original_state_idx]
            result['latitude'] = torch.tensor(row['latitude'], dtype=torch.float32)
            result['longitude'] = torch.tensor(row['longitude'], dtype=torch.float32)
            result['gps'] = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
        
        return result
    
    def get_state_mapping(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Returns the state index mappings."""
        return self.state_to_idx, self.idx_to_state
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Returns the number of samples per state."""
        if self.is_test:
            return {}
        return self.df['state_idx'].value_counts().to_dict()
    
    def get_gps_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Returns the min/max latitude and longitude in the dataset."""
        if self.is_test:
            return {}
        return {
            'latitude': (self.df['latitude'].min(), self.df['latitude'].max()),
            'longitude': (self.df['longitude'].min(), self.df['longitude'].max())
        }


class GeoGuessrSubset(Dataset):
    """
    Subset of GeoGuessrDataset for train/val splitting.
    """
    
    def __init__(self, dataset: GeoGuessrDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[self.indices[idx]]
    
    @property
    def state_to_idx(self):
        return self.dataset.state_to_idx
    
    @property
    def idx_to_state(self):
        return self.dataset.idx_to_state
    
    @property
    def num_classes(self):
        return self.dataset.num_classes
