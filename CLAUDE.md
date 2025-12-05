# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Visual geolocation model for predicting US states and GPS coordinates from 4-directional street view images (N/E/S/W). This is a Kaggle competition for **CS-GY 6643 Computer Vision** at NYU Tandon (Fall 2025), based on PIGEON (CVPR 2024) architecture.

**Competition scoring**: 70% weighted top-5 state classification + 30% GPS regression

---

## Commands

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run training notebook
jupyter notebook notebooks/main.ipynb
```

### Modal Cloud Training (GPU)
```bash
# Run training on Modal with A100 GPU
modal run modal_train.py::train

# Deploy to Modal
modal deploy modal_train.py
```

---

## Architecture

### Model Pipeline
```
4 images (N/E/S/W) → StreetCLIP backbone (shared) → 4 embeddings → Average fusion →
  → Classification Head → 33 state logits (top-5 predictions)
  → GPS Regression Head → (lat, lon)
```

### Key Design Decisions
1. **Backbone**: StreetCLIP (CLIP ViT-L/14-336 fine-tuned on street imagery) from HuggingFace `geolocal/StreetCLIP`
2. **Fusion**: Simple embedding averaging (PIGEON found this outperforms learned attention - do NOT use complex fusion)
3. **Loss**: Haversine-smoothed cross-entropy (τ=75) creates soft labels based on geographic distance between states
4. **Training**: Two-phase - frozen backbone first, then full fine-tuning with differential learning rates

### Module Organization
- `src/config.py`: All hyperparameters in dataclasses (`Config`, `ModelConfig`, `TrainingConfig`, etc.)
- `src/models/geoguessr_model.py`: Main `GeoGuessrModel` class combining backbone, fusion, and heads
- `src/training/trainer.py`: `Trainer` class with gradient accumulation, AMP, checkpointing
- `src/training/losses.py`: `CombinedLoss` with haversine-smoothed CE + GPS regression loss
- `src/data/state_utils.py`: Haversine distance calculations and soft label generation
- `src/inference/predict.py`: Prediction utilities for generating submissions

### Configuration Pattern
```python
from src.config import get_config
config = get_config(training={"batch_size": 64, "num_epochs_frozen": 3})
```

---

## Data Layout

### Expected Structure
Data in `data/raw/` (local) or `/data/data/kaggle_dataset/` (Modal):
- `train_images/` - 263,920 training images (65,980 samples × 4 directions)
- `test_images/` - 65,980 test images (16,495 samples × 4 directions)
- `train_ground_truth.csv` - sample_id, state, state_idx, latitude, longitude, image_north/east/south/west
- `sample_submission.csv` - template for predictions
- `state_mapping.csv` - state index mappings (non-consecutive 0-49, only 33 states present)

### Image Format
- Resolution: 256×256 RGB JPG
- Naming: `img_XXXXXX_direction.jpg` (e.g., `img_000042_north.jpg`)
- Directions: north (0°), east (90°), south (180°), west (270°)

---

## Scoring Details

### Weighted Top-K Classification (70%)
| Position | Weight | Required |
|----------|--------|----------|
| 1 | 1.00 | Yes |
| 2 | 0.60 | No |
| 3 | 0.40 | No |
| 4 | 0.25 | No |
| 5 | 0.15 | No |

Only the **first matching position** scores. Duplicates are ignored.

### GPS Score (30%)
```
gps_score = max(0, 1 - (mean_haversine_distance_km / 5000))
```

### Final Score
```
Final Score = 0.70 × Classification Score + 0.30 × GPS Score
```

---

## Submission Format

CSV must contain exactly **16,495 rows** with columns:
- `sample_id` (int)
- `image_north`, `image_east`, `image_south`, `image_west` (filenames)
- `predicted_state_idx_1` (int 0-49, **REQUIRED**)
- `predicted_state_idx_2` through `predicted_state_idx_5` (int 0-49 or -1, optional)
- `predicted_latitude` (float -90 to 90, **REQUIRED**)
- `predicted_longitude` (float -180 to 180, **REQUIRED**)

---

## Modal Configuration

When running on Modal:
- Data mounted at `/data/data/kaggle_dataset/`
- `modal_train.py` automatically overrides path configs
- Enables TF32 and cuDNN optimizations for A100
- Uses higher batch sizes (512 vs 32 local)

### Modal Rules
- Apps/Volumes/Secrets use kebab-case naming
- Use `import modal` with qualified names
- GPU options: `gpu="A100"`, `gpu="H100"`, `gpu="A100:2"`

---

## Code Quality Checklist

### Data Pipeline
- [ ] Images loaded correctly from all 4 directions
- [ ] CLIP-specific normalization (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
- [ ] Efficient loading (num_workers, prefetch, pin_memory)
- [ ] No data leakage between train/validation splits

### Model
- [ ] Multi-view fusion uses averaging (NOT complex attention)
- [ ] Backbone frozen initially, unfrozen for fine-tuning
- [ ] Correct output dimensions (33 states, 2 GPS coords)
- [ ] Haversine loss uses τ=75

### Training
- [ ] Mixed precision (AMP) enabled
- [ ] Gradient accumulation if batch size limited
- [ ] Learning rate: 1e-5 backbone, 1e-3 heads
- [ ] Checkpointing best validation model

### Inference/Submission
- [ ] All 16,495 test samples processed
- [ ] Top-5 predictions sorted by confidence (descending)
- [ ] GPS coordinates in valid US ranges (lat 25-71°N, lon negative)
- [ ] No NaN/null in required fields

### Common Bugs to Watch For
- Longitude must be negative for US locations
- State index mismatch (use actual indices from state_mapping.csv, not 0-32)
- Averaging embeddings on wrong dimension
- Haversine: degrees vs radians confusion
- Memory leaks from not detaching tensors during inference

---

## Useful Resources

### Models
- StreetCLIP: `geolocal/StreetCLIP` on HuggingFace
- GeoCLIP: `pip install geoclip`
- OSV-5M: `osv5m/baseline` on HuggingFace

### Repositories
- `VicenteVivan/geo-clip` - GeoCLIP implementation
- `LukasHaas/PIGEON` - Architecture reference (no weights)
- `TIBHannover/GeoEstimation` - Training scripts

### Documentation
- Modal: modal.com/docs
- CLIP: HuggingFace transformers docs

---

## Competition Timeline
- **Deadline**: ~16 days remaining
- **Leaderboard**: 50% public, 50% private (final ranking)

Focus on working baseline first, then iterate.