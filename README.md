# GeoGuessr Street View Competition

Visual geolocation model for predicting US states and GPS coordinates from 4-directional street view images.

## Project Overview

This repository hosts a state-of-the-art visual geolocation system capable of determining location (US State and GPS coordinates) from a set of four cardinal street view images. Built upon the **StreetCLIP** vision transformer, our pipeline employs a multi-view fusion architecture that integrates visual cues from all directions.

Key features of our approach include:
- **Dual-Head Prediction**: simultaneously predicts coarse-grained state classification and fine-grained GPS regression.
- **Geographically Aware Loss**: utilizes a custom Haversine-smoothed loss function that penalizes predictions based on physical distance, teaching the model geographic topology.
- **Cloud-Native Training**: Fully integrated with **Modal** for seamless serverless training and data management on high-end GPUs.

## Architecture

Based on PIGEON (CVPR 2024) findings:

1. **Backbone**: StreetCLIP (CLIP ViT-L/14-336 fine-tuned on street imagery)
2. **Fusion**: Simple embedding averaging (outperforms learned attention)
3. **Loss**: Haversine-smoothed cross-entropy for geographic-aware training
4. **Heads**: Classification (33 states) + GPS regression

## Project Structure

```
geoguessr-competition/
├── modal_train.py              # Main training script (Modal)
├── scripts/
│   ├── download_model.py       # Download weights from Hugging Face
│   ├── upload_dataset_to_modal.py # Upload data to Modal volume
│   └── upload_to_hf.py         # Upload weights to Hugging Face
├── src/
│   ├── config.py               # All hyperparameters
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── dataloader.py       # DataLoader factory
│   │   ├── preprocessing.py    # Image transforms
│   │   └── state_utils.py      # State mappings and utilities
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
│       ├── geo.py              # Geographic calculations
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

**For Local Training:**
Place competition data in `data/raw/`:
- `train_images/`
- `test_images/`
- `train_ground_truth.csv`
- `sample_submission.csv`
- `state_mapping.csv`

**For Modal Training:**
Upload the dataset to a Modal volume using the Kaggle API:

```bash
modal run scripts/upload_dataset_to_modal.py
```

### 3. Download Pre-trained Weights (Optional)

Skip training and download our best fine-tuned model (2.2GB) directly from Hugging Face:

```bash
python scripts/download_model.py
```
*Model will be saved to `checkpoints/best_finetune.pt`*

Alternatively, download manually from [chetangoel01/GeoguessrModel](https://huggingface.co/chetangoel01/GeoguessrModel).

### 4. Training

Train on [Modal](https://modal.com) with A100 GPU acceleration:

```bash
modal run modal_train.py
```

This pipeline automatically handles:
1. **Phase 1**: Train heads with frozen backbone (5 epochs)
2. **Phase 2**: Fine-tune full model (10 epochs)
3. **Inference**: Generate predictions on test set

### 5. Inference

The training pipeline above automatically generates a submission file.

To run inference manually (locally or on Modal):
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


## References

- [PIGEON](https://arxiv.org/abs/2307.05845) - Predicting Image Geolocations
- [StreetCLIP](https://huggingface.co/geolocal/StreetCLIP) - Street-level CLIP
- [GeoCLIP](https://github.com/VicenteVivan/geo-clip) - Contrastive geolocation
