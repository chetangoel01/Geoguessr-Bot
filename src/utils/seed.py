"""
Reproducibility utilities.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for some operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def seed_worker(worker_id: int):
    """
    Seed function for DataLoader workers.
    
    Use with DataLoader:
        DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int = 42) -> torch.Generator:
    """
    Get a seeded random generator for DataLoader.
    
    Use with DataLoader:
        DataLoader(..., generator=get_generator(42))
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
