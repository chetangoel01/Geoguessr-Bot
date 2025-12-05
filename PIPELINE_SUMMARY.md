# Complete Project Pipeline Breakdown

## 1. The Big Picture

This is a visual geolocation competition where you're given 4 street view images (North, East, South, West) from a single location and must predict:
1. Which of 33 US states the image is from (70% of score)
2. The exact GPS coordinates (30% of score)

The architecture follows PIGEON (CVPR 2024), which achieved superhuman GeoGuessr performance.

---

## 2. Configuration System (`src/config.py`)

This is the central nervous system of the project. Every hyperparameter lives here.

**Imports:**
- `dataclasses`: Python's way to create structured configuration objects with type hints and defaults
- `pathlib.Path`: Object-oriented filesystem paths
- `typing`: Type hints for lists, optionals
- `torch`: Only used to check `torch.cuda.is_available()` for default device

**Structure:**

There are 5 dataclasses that nest together:

**`PathConfig`**: Every file path the project needs
- `data_root`, `train_images`, `test_images`: Where raw data lives
- `train_csv`, `test_csv`, `state_mapping_csv`: The CSV files from Kaggle
- `processed_dir`, `state_centroids`, `haversine_matrix`: Computed artifacts (centroids of each state, distance matrix between states)
- `checkpoints_dir`, `submissions_dir`, `logs_dir`: Output locations

**`ModelConfig`**: Architecture decisions
- `backbone_name = "geolocal/StreetCLIP"`: HuggingFace model identifier
- `backbone_type = "streetclip"`: Determines which loader to use
- `freeze_backbone = True`: Initially frozen to train heads only
- `embedding_dim = 768`: StreetCLIP outputs 768-dim vectors
- `hidden_dim = 512`: Size of hidden layers in classification/GPS heads
- `num_states = 33`: Output classes
- `fusion_method = "average"`: How to combine 4 view embeddings (PIGEON found averaging beats attention)
- `num_views = 4`: N/E/S/W

**`TrainingConfig`**: How to train
- `num_epochs_frozen = 5`: Train heads while backbone is frozen
- `num_epochs_finetune = 10`: Then unfreeze and train everything
- `batch_size = 32`, `gradient_accumulation_steps = 4`: Effective batch = 128
- `backbone_lr = 1e-5`, `head_lr = 1e-3`: Differential learning rates (backbone gets smaller LR)
- `classification_weight = 0.8`, `gps_weight = 0.2`: Loss weighting
- `use_haversine_smoothing = True`, `haversine_temperature = 300.0`: The key innovation—soft labels based on geographic distance
- `use_amp = True`: Mixed precision for speed

**`InferenceConfig`**: Prediction settings
- `top_k = 5`: Generate 5 state predictions for partial credit
- `ensemble_method`: How to combine multiple models

**`Config`**: Master container that holds all four sub-configs plus device and experiment tracking settings. The `__post_init__` method creates all directories automatically.

**`get_config()`**: Factory function that lets you override specific settings.

**Constants:**
- `STATE_INDEX_TO_NAME`: Maps state indices (0, 1, 3, 4... note: non-consecutive!) to names
- `TOP_K_WEIGHTS`: Competition scoring weights (position 1 = 100%, position 2 = 60%, etc.)

---

## 3. Data Pipeline

### 3.1 State Utilities (`src/data/state_utils.py`)

*Not shown in files but referenced.* This would contain:

**`compute_state_centroids()`**: Takes training DataFrame, computes the average lat/lon for each state (the geographic center). Saves to JSON.

**`compute_haversine_matrix()`**: Creates an NxN matrix where entry [i,j] is the distance in km between state i's centroid and state j's centroid. Uses the haversine formula (accounts for Earth's curvature).

**`haversine_distance()`**: The core formula:
```
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × arcsin(√a)
distance = 6371 × c  (Earth's radius in km)
```

**`get_soft_labels()`**: This is crucial. Instead of a hard label [0,0,1,0,0...] for state 2, it creates soft labels like [0.01, 0.02, 0.7, 0.15, 0.02...] where nearby states get higher probability. The formula:

```
P(state_i | true_state) ∝ exp(-distance(i, true_state) / temperature)
```

With temperature=300km, a state 300km away gets exp(-1) ≈ 37% of the true state's weight.

### 3.2 Preprocessing (`src/data/preprocessing.py`)

*Not shown but referenced.* Would contain:

**CLIP normalization**: StreetCLIP expects images normalized with specific mean and std:
- mean = [0.48145466, 0.4578275, 0.40821073]
- std = [0.26862954, 0.26130258, 0.27577711]

**`denormalize()`**: Reverses normalization for visualization.

**Image transforms**: Resize to 336×336 (StreetCLIP's expected size), convert to tensor, normalize.

### 3.3 Dataset (`src/data/dataset.py`)

*Not shown but referenced.* A PyTorch Dataset class that:

1. Reads the CSV file
2. For each sample, loads 4 images (north, east, south, west)
3. Applies transforms to each image
4. Returns a dictionary with:
   - `images`: Tensor of shape (4, 3, 336, 336)
   - `state_label`: Contiguous index (0-32)
   - `gps`: Tensor of [latitude, longitude]
   - `sample_id`: For submission

### 3.4 DataLoader (`src/data/dataloader.py`)

*Not shown but referenced.* Factory functions:

**`create_dataloaders()`**: 
1. Loads training CSV
2. Splits into train/validation (90/10)
3. Creates `LabelEncoder` that maps original state indices (non-consecutive 0-49) to contiguous (0-32)
4. Creates Dataset objects with appropriate transforms
5. Wraps in DataLoader with proper batch size, workers, pinning

**`create_test_dataloader()`**: Same but for test set, no labels.

---

## 4. Model Architecture

### 4.1 Backbone (`src/models/backbone.py`)

**Imports:**
- `torch`, `torch.nn`: PyTorch fundamentals
- `transformers.CLIPModel, CLIPProcessor`: HuggingFace's CLIP implementation
- `..config.Config`: For accessing configuration

**`StreetCLIPBackbone` class:**

`__init__`:
- Downloads StreetCLIP from HuggingFace (`CLIPModel.from_pretrained("geolocal/StreetCLIP")`)
- Extracts just the vision model (ignores text encoder)
- Sets `embedding_dim = 768` (from ViT-L)
- Optionally freezes all parameters

`freeze()`: Sets `requires_grad = False` for all parameters. The backbone becomes a fixed feature extractor.

`unfreeze()`: Either unfreezes everything, or just the last N transformer layers (gradual unfreezing for stability).

`forward()`: 
- Takes pixel values (batch, 3, H, W)
- Passes through vision transformer
- Returns `pooler_output`—the CLS token after projection (batch, 768)

**`GeoCLIPBackbone` class:** Alternative backbone using the GeoCLIP library. Similar structure but uses their image encoder.

**`get_backbone()`**: Factory that reads config and returns the right backbone.

**`count_parameters()`**: Utility to count total/trainable parameters.

### 4.2 Fusion (`src/models/fusion.py`)

**Imports:**
- `torch`, `torch.nn`, `torch.nn.functional`: PyTorch basics

**`MultiViewFusion` class:**

This combines the 4 directional embeddings into one. PIGEON's key finding: **simple averaging beats learned attention**.

`__init__`: Based on `fusion_method`:
- `"average"`: No learnable parameters, just averages
- `"concat"`: Concatenates all 4 (4×768=3072), projects back to 768
- `"attention"`: Learns attention weights per view
- `"weighted"`: Learnable scalar weights per view

`forward()`: Takes (batch, 4, 768), returns (batch, 768)

For `"average"`:
```python
fused = view_embeddings.mean(dim=1)  # Average across view dimension
```

**`DirectionalEmbedding` class:** Optional—adds learned per-direction biases. Not used in default config.

### 4.3 Task Heads (`src/models/heads.py`)

**Imports:**
- `torch`, `torch.nn`, `torch.nn.functional`
- `typing`: For type hints

**`ClassificationHead` class:**

Architecture: Linear(768→512) → GELU → Dropout → Linear(512→33)

`__init__`: Builds the sequential network. Initializes final layer with small weights (Xavier with gain=0.01) to prevent large initial logits.

`forward()`: Input (batch, 768) → Output (batch, 33) logits

`get_top_k_predictions()`: Applies softmax, returns top-k indices and probabilities.

**`GPSRegressionHead` class:**

Architecture: Linear(768→512) → GELU → Dropout → Linear(512→256) → GELU → Dropout → Linear(256→2)

Key feature: `normalize_output=True` applies tanh and scales to valid US coordinate ranges:
- Latitude: 18° to 72° (includes Alaska)
- Longitude: -180° to -65°

This prevents the model from predicting impossible coordinates.

**`HierarchicalGPSHead`**: Alternative that predicts offset from state centroid. Not used by default.

**`DualHead`**: Combined head with shared layers. Not used by default.

### 4.4 Full Model (`src/models/geoguessr_model.py`)

**Imports:**
- `torch`, `torch.nn`, `torch.nn.functional`
- `typing`: Type hints
- `.backbone`, `.fusion`, `.heads`: The components
- `..config.Config`: Configuration

**`GeoGuessrModel` class:**

`__init__`:
1. Loads backbone via `get_backbone(config)`
2. Creates `MultiViewFusion` with backbone's embedding dim
3. Creates `ClassificationHead` with fusion's output dim
4. Creates `GPSRegressionHead`
5. Prints parameter counts

`_print_param_counts()`: Shows breakdown of parameters (backbone ~428M, heads ~800K).

`encode_views()`: 
- Input: (batch, 4, 3, H, W)
- Reshapes to (batch×4, 3, H, W) to process all views at once
- Passes through backbone
- Reshapes back to (batch, 4, 768)

`forward()`:
1. `view_embeddings = self.encode_views(images)` → (batch, 4, 768)
2. `fused = self.fusion(view_embeddings)` → (batch, 768)
3. `class_logits = self.classification_head(fused)` → (batch, 33)
4. `gps_coords = self.gps_head(fused)` → (batch, 2)
5. Returns dict with all outputs

`predict()`: Inference mode. Returns top-k states with probabilities and GPS.

`freeze_backbone()`, `unfreeze_backbone()`: Delegate to backbone.

`get_optimizer_param_groups()`: Returns parameter groups with different learning rates—backbone gets `backbone_lr`, heads get `head_lr`.

**`load_model()`**: Factory that creates model and optionally loads checkpoint.

---

## 5. Training Pipeline

### 5.1 Loss Functions (`src/training/losses.py`)

**Imports:**
- `torch`, `torch.nn`, `torch.nn.functional`
- `numpy`
- `typing`

**`HaversineSmoothedCrossEntropy` class:**

This is the key innovation. Instead of one-hot labels, it creates soft labels based on geographic distance.

`__init__`:
1. Takes the precomputed distance matrix (33×33)
2. For each state i, computes: `soft_labels[i, j] = exp(-distance[i,j] / temperature)`
3. Normalizes so each row sums to 1
4. Stores as a buffer (not a parameter—doesn't need gradients)

With temperature=300km:
- The true state gets the highest probability
- A state 150km away gets exp(-0.5) ≈ 60% of that
- A state 1500km away gets exp(-5) ≈ 0.7% of that

`forward()`:
1. Look up soft targets for each sample: `soft_targets = self.soft_label_matrix[targets]`
2. Compute log softmax of predictions
3. Cross entropy: `loss = -(soft_targets * log_probs).sum(dim=-1).mean()`

**`StandardCrossEntropy`**: Wrapper around PyTorch's with optional class weights.

**`GPSLoss` class:**

Supports multiple loss types:
- `"mse"`: Mean squared error
- `"mae"`: Mean absolute error  
- `"smooth_l1"`: Huber loss
- `"haversine"`: Actual geographic distance

The `normalize` option scales lat/lon to similar ranges (lat spans 46°, lon spans 59° in US).

`_haversine_loss()`: Computes actual km distance, normalizes by 5000km (US coast-to-coast).

**`CombinedLoss` class:**

Combines classification and GPS losses:
```python
total = classification_weight × cls_loss + gps_weight × gps_loss
```

Default: 0.8 × classification + 0.2 × GPS.

Returns dict with all three losses for logging.

**`create_loss_function()`**: Factory that creates the appropriate loss based on config.

### 5.2 Metrics (`src/training/metrics.py`)

**Imports:**
- `torch`, `numpy`
- `typing`
- `..config.TOP_K_WEIGHTS`

**`haversine_distance_torch()`**: GPU-accelerated haversine formula.

**`compute_weighted_topk_score()`**: 

The competition's classification metric:
1. For each sample, check predictions in order (1st, 2nd, 3rd...)
2. First match determines score (1.0, 0.6, 0.4, 0.25, 0.15)
3. No match = 0

Returns mean score and per-position match rates.

**`compute_gps_score()`**:

```python
score = max(0, 1 - mean_distance_km / 5000)
```

Perfect prediction = 1.0, 5000km error = 0.0.

**`compute_competition_score()`**: 

```python
final = 0.70 × classification_score + 0.30 × gps_score
```

**`compute_classification_metrics()`**, **`compute_gps_metrics()`**: Additional metrics like top-1 accuracy, distance percentiles.

**`MetricTracker` class**: Accumulates predictions across batches, then computes final metrics.

### 5.3 Trainer (`src/training/trainer.py`)

**Imports:**
- `torch`, `torch.nn`, DataLoader
- `torch.amp`: Mixed precision (GradScaler, autocast)
- `torch.optim`: AdamW optimizer
- `torch.optim.lr_scheduler`: CosineAnnealingLR, LinearLR, SequentialLR
- `pathlib`, `typing`, `tqdm`, `json`, `time`
- `.losses`, `.metrics`: Our loss and metric modules
- `..models.geoguessr_model`, `..config`

**`Trainer` class:**

`__init__`:
1. Moves model and loss function to device
2. Sets up gradient scaler for AMP
3. Initializes tracking variables (epoch, step, best score, history)
4. Creates MetricTracker

`setup_optimizer()`:

For `"frozen"` phase:
- Freezes backbone
- Creates optimizer for heads only at `head_lr`
- Scheduler: linear warmup → cosine decay

For `"finetune"` phase:
- Unfreezes backbone
- Creates optimizer with param groups (backbone at `backbone_lr`, heads at `head_lr`)
- Same scheduler pattern

`train_epoch()`:

The training loop:
1. Set model to train mode
2. Zero gradients
3. For each batch:
   - Move data to GPU (non_blocking for speed)
   - Forward pass with autocast (bfloat16 on A100)
   - Scale loss by gradient accumulation steps
   - Backward pass through scaler
   - Every N steps: unscale, clip gradients (max_norm=1.0), step optimizer, step scheduler
   - Track losses
4. Return epoch metrics

`validate()`:

1. Set model to eval mode
2. Reset metric tracker
3. For each batch:
   - Forward pass (no gradients)
   - Accumulate predictions in tracker
4. Compute and return all metrics

`train()`:

The full training loop:
1. Setup optimizer for phase
2. For each epoch:
   - Train epoch
   - Validate
   - Log results
   - Save checkpoint if best score
   - Early stopping if no improvement for N epochs
3. Return history

`save_checkpoint()`: Saves model state, optimizer state, scheduler state, scaler state, metrics, and config.

`load_checkpoint()`: Restores everything.

---

## 6. Inference and Submission

### 6.1 Prediction (`src/inference/predict.py`)

**Imports:**
- `torch`, `torch.nn.functional`
- `DataLoader`, `autocast`
- `typing`, `tqdm`, `numpy`
- `..models.geoguessr_model`

**`predict_batch()`**: Single batch inference. Returns top-k states, probabilities, GPS, and logits.

**`predict_dataset()`**:

Full dataset inference:
1. Set model to eval mode
2. For each batch:
   - Forward pass with optional AMP
   - Convert contiguous indices back to original state indices (using `idx_to_state` mapping)
   - Accumulate results
3. Return numpy arrays of all predictions

**`predict_with_tta()`**: Test-time augmentation—averages predictions across augmented versions.

**`calibrate_probabilities()`**: Temperature scaling to adjust confidence (higher temp = less confident).

### 6.2 Ensemble (`src/inference/ensemble.py`)

**Imports:**
- `numpy`, `typing`, `torch`

**`ensemble_predictions()`**:

Combines predictions from multiple models:
- `"average"`: Weighted average of probabilities
- `"voting"`: Majority vote

For classification, aggregates all predictions with their probabilities into a vote matrix, then takes top-k.

For GPS, simple weighted average.

**`weighted_ensemble()`**: Uses validation scores as weights (squared to amplify differences).

**`rank_fusion()`**: Reciprocal Rank Fusion—a sophisticated method that weights by rank position.

**`stacking_blend()`**: Learns optimal weights from validation predictions using optimization.

### 6.3 Submission (`src/inference/submission.py`)

**Imports:**
- `pandas`, `numpy`, `pathlib`, `typing`

**`create_submission()`**:

1. Load template CSV
2. Sort predictions by sample_id
3. Fill in all columns (5 state predictions + lat/lon)
4. Validate
5. Save

**`validate_submission()`**:

Checks all competition rules:
- Correct row count (16,495)
- All required columns present
- No nulls in required fields
- State indices in [0, 49]
- Latitude in [-90, 90]
- Longitude in [-180, 180]
- Warning (not error) for coordinates outside US bounds

**`analyze_submission()`**: Statistics about predictions (state distribution, GPS stats, top-k usage).

**`compare_submissions()`**: Compares two submissions for debugging.

---

## 7. Utilities

### 7.1 Seed (`src/utils/seed.py`)

**Imports:**
- `random`, `numpy`, `torch`, `os`

**`set_seed()`**: Sets seeds for Python random, numpy, torch (CPU and all GPUs), and environment variable. Also sets cudnn to deterministic mode.

**`seed_worker()`**: For DataLoader worker processes—ensures they get different but reproducible seeds.

**`get_generator()`**: Returns a seeded torch.Generator for DataLoader shuffling.

### 7.2 Visualization (`src/utils/visualization.py`)

**Imports:**
- `torch`, `numpy`, `matplotlib`
- `typing`, `pathlib`

**`visualize_sample()`**: Shows 4 directional views with predictions and ground truth.

**`visualize_predictions()`**: Multiple samples in a grid.

**`plot_confusion_matrix()`**: Standard confusion matrix visualization.

**`plot_training_curves()`**: Loss, score, and learning rate over epochs.

**`plot_gps_predictions()`**: Scatter plot of predictions on US map.

**`visualize_attention()`**: For attention fusion, shows which views the model weights.

**`plot_state_distribution()`**: Bar chart of predicted state frequencies.

---

## 8. Cloud Execution (`modal_train.py`)

**Imports at top level:**
- `modal`: The Modal cloud platform SDK
- `pathlib.Path`

**App setup:**
- Creates Modal app named `"geoguessr-competition"`
- Defines Docker image with all requirements + source code
- Creates persistent volume `"geoguessr-data"` for dataset storage

**`@app.function` decorator:**
- `image=geo_image`: Use the defined Docker image
- `gpu="A100-80GB"`: Request an A100
- `volumes={"/data": data_volume}`: Mount data volume
- `timeout=60*60*8`: 8 hour timeout

**`train()` function:**

This is essentially the entire training notebook as a function:

1. **Setup:**
   - Add project to Python path
   - Enable CUDA optimizations (expandable segments, cuDNN benchmark, TF32)
   - Set seed

2. **Path overrides:**
   - Changes all paths to use `/data/data/kaggle_dataset/` instead of local paths

3. **Data exploration:**
   - Loads CSV, prints statistics
   - Creates visualizations (state distribution, GPS scatter)
   - Shows sample images

4. **Data preparation:**
   - Computes state centroids
   - Creates label encoder (maps non-consecutive to consecutive indices)
   - Computes haversine distance matrix
   - Visualizes soft labels at different temperatures

5. **DataLoaders:**
   - Creates train/val loaders
   - Modal config: batch_size=512, gradient_accumulation=2 (effective=1024)

6. **Model setup:**
   - Creates GeoGuessrModel
   - Optionally compiles with torch.compile() (disabled by default due to issues)
   - Warm-up forward pass

7. **Loss setup:**
   - Creates CombinedLoss with haversine smoothing

8. **Training Phase 1:**
   - Trainer.train() with frozen backbone

9. **Training Phase 2:**
   - Load best frozen checkpoint
   - Trainer.train() with unfrozen backbone

10. **Evaluation:**
    - Final validation
    - Generate predictions on validation set
    - Visualizations

11. **Submission:**
    - Create test dataloader
    - Generate predictions
    - Create and validate submission CSV
    - Analyze results

---

## 9. Package Structure (`src/__init__.py`)

Simply exposes `Config` and `get_config` at the package level for cleaner imports.

---

## 10. Supporting Files

**`requirements.txt`**: All dependencies with minimum versions:
- PyTorch, torchvision, transformers for ML
- numpy, pandas, scikit-learn, scipy for data
- Pillow, opencv-python for images
- matplotlib, seaborn for visualization
- tqdm, wandb for logging
- huggingface-hub for model downloads

**`.gitignore`**: Standard Python/ML ignores plus data directories, checkpoints, submissions.

**`README.md`**: Project overview, quick start, architecture explanation.

**`CLAUDE.md`**: AI assistant guidance with architecture summary and common pitfalls.

**`notebooks/main.ipynb`**: Empty notebook (actual training in modal_train.py).

---

## Data Flow Summary

1. **Input**: 4 JPG images (256×256) per sample
2. **Preprocessing**: Resize to 336×336, normalize with CLIP stats
3. **Backbone**: Each image → StreetCLIP → 768-dim embedding
4. **Fusion**: 4 embeddings → average → 1 embedding
5. **Classification head**: 768-dim → 512-dim → 33 logits
6. **GPS head**: 768-dim → 512-dim → 256-dim → 2 coords (lat, lon)
7. **Loss**: Haversine-smoothed CE + scaled MSE
8. **Output**: Top-5 state predictions + GPS coordinates
9. **Score**: 0.7 × weighted_topk + 0.3 × (1 - distance/5000)