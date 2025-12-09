"""
Configuration for GeoGuessr Competition Model

All hyperparameters, paths, and settings in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class PathConfig:
    """Paths configuration"""
    # Data paths
    data_root: Path = Path("data/raw")
    train_images: Path = Path("data/raw/train_images")
    test_images: Path = Path("data/raw/test_images")
    train_csv: Path = Path("data/raw/train_ground_truth.csv")
    test_csv: Path = Path("data/raw/sample_submission.csv")
    state_mapping_csv: Path = Path("data/raw/state_mapping.csv")
    
    # Processed data
    processed_dir: Path = Path("data/processed")
    state_centroids: Path = Path("data/processed/state_centroids.json")
    haversine_matrix: Path = Path("data/processed/haversine_matrix.npy")
    train_val_split: Path = Path("data/processed/train_val_split.csv")
    
    # Outputs
    checkpoints_dir: Path = Path("checkpoints")
    submissions_dir: Path = Path("submissions")
    logs_dir: Path = Path("outputs/training_logs")
    attention_maps_dir: Path = Path("outputs/attention_maps")


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Backbone
    backbone_name: str = "geolocal/StreetCLIP"  # HuggingFace model name
    backbone_type: str = "streetclip"  # "streetclip" or "geoclip"
    freeze_backbone: bool = True  # Freeze during initial training
    
    # Model dimensions
    embedding_dim: int = 768  # StreetCLIP ViT-L output dimension
    hidden_dim: int = 512  # Hidden layer in heads
    
    # Heads
    num_states: int = 33  # Number of US states in dataset
    num_gps_outputs: int = 2  # latitude, longitude
    dropout_rate: float = 0.1
    
    # Fusion
    fusion_method: str = "average"  # "average", "concat", "attention"
    num_views: int = 4  # N, E, S, W


@dataclass
class TrainingConfig:
    """Training configuration"""
    # General
    seed: int = 42
    num_epochs_frozen: int = 5  # Epochs with frozen backbone
    num_epochs_finetune: int = 10  # Epochs with unfrozen backbone
    
    # Batch size
    batch_size: int = 32  # Per GPU - adjust based on VRAM
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * accumulation
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = "adamw"
    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 1e-6
    
    # Scheduler
    scheduler: str = "cosine"  # "cosine", "linear", "step"
    warmup_ratio: float = 0.1
    
    # Loss weights
    classification_weight: float = 0.8
    gps_weight: float = 0.2
    
    # Haversine smoothing
    use_haversine_smoothing: bool = True
    haversine_temperature: float = 300.0  # km - tune for state-level granularity
    
    # Validation
    val_split: float = 0.1
    early_stopping_patience: int = 3
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_n_checkpoints: int = 3


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 64
    num_workers: int = 4
    use_tta: bool = False  # Test-time augmentation
    
    # Top-k predictions
    top_k: int = 5  # Number of state predictions to output
    
    # Ensembling
    ensemble_method: str = "average"  # "average", "weighted"
    ensemble_weights: Optional[List[float]] = None


@dataclass
class Config:
    """Master configuration"""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Experiment tracking
    experiment_name: str = "geoguessr_baseline"
    use_wandb: bool = False
    wandb_project: str = "geoguessr-competition"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path_attr in ['checkpoints_dir', 'submissions_dir', 'logs_dir', 
                          'attention_maps_dir', 'processed_dir']:
            path = getattr(self.paths, path_attr)
            path.mkdir(parents=True, exist_ok=True)


def get_config(**overrides) -> Config:
    """
    Get configuration with optional overrides.
    
    Example:
        config = get_config(
            training={"batch_size": 64, "num_epochs_frozen": 3}
        )
    """
    config = Config()
    
    for key, value in overrides.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                sub_config = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(sub_config, sub_key):
                        setattr(sub_config, sub_key, sub_value)
            else:
                setattr(config, key, value)
    
    return config


# Competition-specific constants
STATE_INDEX_TO_NAME = {
    0: "Alabama", 1: "Alaska", 3: "Arkansas", 4: "California",
    5: "Colorado", 6: "Connecticut", 7: "Delaware", 8: "Florida",
    9: "Georgia", 10: "Hawaii", 11: "Idaho", 12: "Illinois",
    13: "Indiana", 14: "Iowa", 15: "Kansas", 16: "Kentucky",
    17: "Louisiana", 18: "Maine", 19: "Maryland", 20: "Massachusetts",
    21: "Michigan", 22: "Minnesota", 23: "Mississippi", 24: "Missouri",
    25: "Montana", 26: "Nebraska", 27: "Nevada", 28: "New Hampshire",
    29: "New Jersey", 30: "New Mexico", 31: "New York", 32: "North Carolina",
    33: "North Dakota", 34: "Ohio", 35: "Oklahoma", 36: "Oregon",
    37: "Pennsylvania", 38: "Rhode Island", 39: "South Carolina",
    40: "South Dakota", 41: "Tennessee", 42: "Texas", 43: "Utah",
    44: "Vermont", 45: "Virginia", 46: "Washington", 47: "West Virginia",
    48: "Wisconsin", 49: "Wyoming"
}

# Note: The actual states in the dataset are a subset - will be determined from data
VALID_STATE_INDICES = list(STATE_INDEX_TO_NAME.keys())

# Scoring weights from competition
TOP_K_WEIGHTS = {
    1: 1.00,
    2: 0.60,
    3: 0.40,
    4: 0.25,
    5: 0.15
}

# Geographic Constants
# GPS Bounds (approximate US bounds including AK/HI)
GPS_BOUNDS = {
    'lat': (18.0, 72.0),
    'lon': (-180.0, -65.0)
}

# Normalization scales for loss (approximate range of US coordinates)
NORMALIZATION_SCALES = {
    'lat': 46.0,  # ~71 - ~25
    'lon': 59.0   # ~-66 - ~-125 (approx)
}

# Max distance for scoring (5000 km)
SCORING_MAX_DISTANCE = 5000.0
