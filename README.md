# GeoGuessr Street View Competition

Visual geolocation model for predicting US states and GPS coordinates from 4-directional street view images.

## Competition Overview

- **Task**: Predict US state (33 classes) + GPS coordinates from N/E/S/W street view images
- **Scoring**: 70% weighted top-5 classification + 30% GPS regression
- **Dataset**: 65,980 training samples, 16,495 test samples

## Architecture

Based on PIGEON (CVPR 2024) findings:

1. **Backbone**: StreetCLIP (CLIP ViT-L/14-336 fine-tuned on street imagery)
2. **Fusion**: Simple embedding averaging (outperforms learned attention)
3. **Loss**: Haversine-smoothed cross-entropy for geographic-aware training
4. **Heads**: Classification (33 states) + GPS regression

## Project Structure

```
geoguessr-competition/
├── notebooks/
│   └── main.ipynb              # Main training notebook
├── src/
│   ├── config.py               # All hyperparameters
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── dataloader.py       # DataLoader factory
│   │   ├── preprocessing.py    # Image transforms
│   │   └── state_utils.py      # Haversine calculations, soft labels
│   ├── models/
│   │   ├── backbone.py         # StreetCLIP/GeoCLIP loading
│   │   ├── fusion.py           # Multi-view fusion
│   │   ├── heads.py            # Classification & GPS heads
│   │   └── geoguessr_model.py  # Full model
│   ├── training/
│   │   ├── losses.py           # Haversine-smoothed CE
│   │   ├── metrics.py          # Competition scoring
│   │   └── trainer.py          # Training loop
│   ├── inference/
│   │   ├── predict.py          # Prediction utilities
│   │   ├── ensemble.py         # Model ensembling
│   │   └── submission.py       # CSV generation
│   └── utils/
│       ├── seed.py             # Reproducibility
│       └── visualization.py    # Debugging plots
├── data/
│   ├── raw/                    # Competition data
│   └── processed/              # Computed centroids, matrices
├── checkpoints/                # Model weights
└── submissions/                # Output CSVs
```

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place competition data in `data/raw/`:
- `train_images/`
- `test_images/`
- `train_ground_truth.csv`
- `sample_submission.csv`
- `state_mapping.csv`

### 3. Download Pre-trained Weights (Optional)

Skip training and download our best fine-tuned model (2.2GB) directly from Hugging Face:

```bash
python scripts/download_model.py
```
*Model will be saved to `checkpoints/best_finetune.pt`*

Alternatively, download manually from [chetangoel01/GeoguessrModel](https://huggingface.co/chetangoel01/GeoguessrModel).

### 4. Training

Open `notebooks/main.ipynb` and run cells sequentially:

1. **Phase 1**: Train heads with frozen backbone (5 epochs)
2. **Phase 2**: Fine-tune full model (10 epochs)

### 5. Inference

Generate submission:
```python
from src.inference import predict_dataset, create_submission

predictions = predict_dataset(model, test_loader, device)
create_submission(predictions, 'data/raw/sample_submission.csv', 'submissions/submission.csv')
```

## Key Innovations

### Haversine-Smoothed Loss

Instead of hard labels, creates soft targets based on geographic distance:

```
P(state_i | true_state_j) ∝ exp(-distance(i,j) / temperature)
```

This teaches the model that predicting Nevada when the answer is California is better than predicting Maine.

### Multi-View Fusion

PIGEON found simple averaging beats learned attention:

```python
fused = (embed_north + embed_east + embed_south + embed_west) / 4
```

### Top-5 Weighted Scoring

Competition rewards calibrated uncertainty:
- Position 1: 100% credit
- Position 2: 60% credit
- Position 3: 40% credit
- Position 4: 25% credit
- Position 5: 15% credit

## Training Tips

1. **Start with frozen backbone** - Train heads first for stability
2. **Use haversine smoothing** - Temperature ~300km works well for states
3. **Monitor GPS distance** - Median distance more informative than mean
4. **Ensemble** - 3-5 models with different seeds typically adds 1-3%

## Expected Performance

- Baseline (random): ~0.17 final score
- Frozen backbone: ~0.50-0.60 final score  
- Fine-tuned: ~0.70-0.80 final score
- With ensembling: ~0.75-0.85 final score

## References

- [PIGEON](https://arxiv.org/abs/2307.05845) - Predicting Image Geolocations
- [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) - Street-level CLIP
- [GeoCLIP](https://github.com/VicenteVivan/geo-clip) - Contrastive geolocation
