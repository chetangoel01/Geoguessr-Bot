import modal
from pathlib import Path

app = modal.App("geoguessr-competition")

# Get the project root directory
project_root = Path(__file__).parent

# Define image with dependencies and include source code
geo_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements(str(project_root / "requirements.txt"))
    .add_local_dir(project_root / "src", remote_path="/root/geoguessr-competition/src")
)

# Create volume for data
data_volume = modal.Volume.from_name("geoguessr-data", create_if_missing=True)

@app.function(
    image=geo_image,
    gpu="A100-80GB",
    volumes={"/data": data_volume},
    timeout=60 * 60 * 8,
)
def train():
    import os
    import sys
    
    # Add the mounted project directory to Python path
    project_path = "/root/geoguessr-competition"
    sys.path.insert(0, project_path)
    
    # Change working directory to project root
    os.chdir(project_path)
    
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os

    # Optimize PyTorch memory allocation to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    from pathlib import Path as _Path
    print("Working directory:", _Path(".").resolve())
    
    # Check GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Clear cache at start
        torch.cuda.empty_cache()
        
        # Enable cuDNN optimizations for consistent input sizes
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
        print("cuDNN optimizations enabled")
        
        # Enable TensorFloat32 (TF32) for faster training on A100/H100
        # TF32 uses 19-bit precision instead of 32-bit, ~1.2-1.5x speedup with minimal accuracy impact
        torch.set_float32_matmul_precision('high')  # Enables TF32 for matmul operations
        print("TensorFloat32 (TF32) enabled for faster training")

    # Import our modules - should work now
    from src.config import get_config, STATE_INDEX_TO_NAME
    from src.utils.seed import set_seed

    # Get configuration
    config = get_config()

    from pathlib import Path

    # Root of the Kaggle dataset inside the Modal volume
    DATA_ROOT = Path("/data/data/kaggle_dataset")

    print("Using DATA_ROOT:", DATA_ROOT)
    if DATA_ROOT.exists():
        print("Contents of DATA_ROOT:")
        for p in DATA_ROOT.iterdir():
            print("  ", p)
    else:
        print("WARNING: DATA_ROOT does not exist!")

    # ---- Override paths to use the Modal volume ----
    # CSVs
    if hasattr(config.paths, "train_csv"):
        config.paths.train_csv = DATA_ROOT / "train_ground_truth.csv"
        print("train_csv ->", config.paths.train_csv)

    if hasattr(config.paths, "test_csv"):
        # This is the template for submission; in your case it's sample_submission.csv
        config.paths.test_csv = DATA_ROOT / "sample_submission.csv"
        print("test_csv ->", config.paths.test_csv)

    # Images
    if hasattr(config.paths, "train_images"):
        config.paths.train_images = DATA_ROOT / "train_images"
        print("train_images ->", config.paths.train_images)

    if hasattr(config.paths, "test_images"):
        config.paths.test_images = DATA_ROOT / "test_images"
        print("test_images ->", config.paths.test_images)

    # State mapping (name may differ; handle both common variants)
    if hasattr(config.paths, "state_mapping_csv"):
        config.paths.state_mapping_csv = DATA_ROOT / "state_mapping.csv"
        print("state_mapping_csv ->", config.paths.state_mapping_csv)
    elif hasattr(config.paths, "state_mapping"):
        config.paths.state_mapping = DATA_ROOT / "state_mapping.csv"
        print("state_mapping ->", config.paths.state_mapping)

    # Ensure output dirs exist
    for attr in ("logs_dir", "checkpoints_dir", "submissions_dir"):
        if hasattr(config.paths, attr):
            d = Path(getattr(config.paths, attr))
            d.mkdir(parents=True, exist_ok=True)
            print(f"{attr} -> {d} (created if missing)")


    # Set seed for reproducibility
    # Note: We use non-deterministic cudnn for speed (set earlier in GPU config)
    # This means exact reproducibility is not guaranteed, but random seeds are still set
    # for consistent initialization and data shuffling
    import random
    import numpy as np
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    print(f"Random seed set to {config.training.seed} (non-deterministic cudnn for speed)")

    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Adjust config for your hardware
    # For A100 80GB, maximize VRAM usage while avoiding OOM
    # With StreetCLIP ViT-L, 336x336 images, 4 views per sample, and mixed precision:
    # - batch_size=768 caused OOM (tried to allocate 27GB when only 20GB free)
    # - Starting conservatively and can increase if stable
    
    config.training.batch_size = 512  # Conservative: 2x increase from 128, safe starting point
    config.training.gradient_accumulation_steps = 2  # Effective batch = 1024
    config.training.num_workers = 8  # Reduced from 16 to prevent memory pressure from prefetch buffers
    # Effective batch size: 512 * 2 = 1024 (was 32 * 4 = 128)
    # Images per forward pass: 512 * 4 = 2048 images (was 32 * 4 = 128 images = 16x increase!)
    # Expected VRAM usage: ~20-30GB (2.5-3.75x increase from 8GB)
    # If stable, can gradually increase to 320, 384, or 512
    
    # Model size optimizations (reduce for faster training / less memory)
    # ====================================================================
    # Option 1: Reduce hidden_dim in heads (biggest impact on model size)
    # config.model.hidden_dim = 256  # Reduce from 512 to 256
    #   - ~2x smaller classification head: 768*256 + 256*33 vs 768*512 + 512*33
    #   - ~2x smaller GPS head: 768*256 + 256*128 + 128*2 vs 768*512 + 512*256 + 256*2
    #   - Total reduction: ~400K parameters → ~200K parameters in heads
    #   - Speedup: ~10-15% faster forward/backward passes
    #   - Memory: ~15-20% less VRAM usage
    #   - Accuracy: Usually minimal impact (<1% accuracy drop)
    #
    # Option 2: Use smaller hidden_dim (more aggressive)
    # config.model.hidden_dim = 384  # Middle ground between 256 and 512
    #
    # Option 3: Backbone optimization (if you want to experiment)
    # Note: StreetCLIP is ViT-L (Large). You could try:
    # - Standard CLIP ViT-B/16 (smaller, but not geo-tuned)
    # - This would require changing backbone_name and backbone_type
    # - Significant speedup but accuracy loss likely
    #
    # Option 4: Fusion method (already optimal)
    # config.model.fusion_method = "average"  # Already set - no learnable params (best!)
    # Other options: "concat" (adds params), "attention" (adds params)
    #
    # Current model size breakdown:
    # - Backbone (StreetCLIP ViT-L): ~428M parameters (frozen during initial training)
    # - Fusion: 0 parameters (average method)
    # - Classification head: ~400K parameters (768→512→33)
    # - GPS head: ~400K parameters (768→512→256→2)
    # - Total trainable (frozen phase): ~800K parameters
    # - Total trainable (finetune phase): ~428M + 800K parameters

    print("Current configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"  Backbone: {config.model.backbone_name}")
    print(f"  Fusion method: {config.model.fusion_method}")
    print(f"  Haversine smoothing: {config.training.use_haversine_smoothing}")
    print(f"  Haversine temperature: {config.training.haversine_temperature} km")

    # ## 2. Data Exploration

    # Load training data CSV
    train_df = pd.read_csv(config.paths.train_csv)
    print(f"Training samples: {len(train_df)}")
    print(f"\nColumns: {list(train_df.columns)}")
    train_df.head()

    # State distribution
    state_counts = train_df['state'].value_counts()
    print(f"Number of unique states: {len(state_counts)}")
    print(f"\nState distribution:")
    print(state_counts)

    # Visualize state distribution
    fig, ax = plt.subplots(figsize=(12, 8))
    state_counts.plot(kind='barh', ax=ax)
    ax.set_xlabel('Number of samples')
    ax.set_title('Training Data - State Distribution')
    plt.tight_layout()
    plt.savefig(config.paths.logs_dir / "state_distribution.png")
    plt.close(fig)

    # GPS coordinate distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(train_df['longitude'], train_df['latitude'], alpha=0.1, s=1)
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].set_title('GPS Distribution')

    # Histograms
    axes[1].hist2d(train_df['longitude'], train_df['latitude'], bins=50, cmap='YlOrRd')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('GPS Density')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Count')

    plt.tight_layout()
    plt.savefig(config.paths.logs_dir / "gps_distribution.png")
    plt.close(fig)

    print(f"Latitude range: {train_df['latitude'].min():.2f} to {train_df['latitude'].max():.2f}")
    print(f"Longitude range: {train_df['longitude'].min():.2f} to {train_df['longitude'].max():.2f}")

    # Visualize some sample images
    from PIL import Image

    sample_idx = 0
    sample = train_df.iloc[sample_idx]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    directions = ['north', 'east', 'south', 'west']

    for ax, direction in zip(axes, directions):
        img_path = config.paths.train_images / sample[f'image_{direction}']
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(direction.capitalize())
        ax.axis('off')

    plt.suptitle(f"Sample {sample_idx}: {sample['state']} ({sample['latitude']:.4f}, {sample['longitude']:.4f})")
    plt.tight_layout()
    plt.savefig(config.paths.logs_dir / "sample_images.png")
    plt.close(fig)

    # ## 3. Data Preparation

    from src.data.state_utils import (
        compute_state_centroids,
        compute_haversine_matrix,
        haversine_distance
    )

    # Compute state centroids
    centroids = compute_state_centroids(train_df, config.paths.state_centroids)
    print(f"Computed centroids for {len(centroids)} states")

    # Show some centroids
    for state_idx, (lat, lon) in list(centroids.items())[:5]:
        state_name = STATE_INDEX_TO_NAME.get(state_idx, f"Unknown ({state_idx})")
        print(f"  {state_name}: ({lat:.2f}, {lon:.2f})")

    # Create state index mapping (original indices to contiguous 0-32)
    unique_states = sorted(train_df['state_idx'].unique())
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    print(f"Number of states: {len(state_to_idx)}")
    print(f"\nMapping (first 10):")
    for orig, cont in list(state_to_idx.items())[:10]:
        name = STATE_INDEX_TO_NAME.get(orig, "Unknown")
        print(f"  {orig} ({name}) -> {cont}")

    # Compute haversine distance matrix between state centroids
    distance_matrix = compute_haversine_matrix(
        centroids, state_to_idx, config.paths.haversine_matrix
    )

    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"\nExample distances (km):")
    print(f"  Min non-zero: {distance_matrix[distance_matrix > 0].min():.0f} km")
    print(f"  Max: {distance_matrix.max():.0f} km")
    print(f"  Mean: {distance_matrix.mean():.0f} km")

    # Visualize distance matrix
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get state names for labels
    state_names_ordered = [STATE_INDEX_TO_NAME.get(idx_to_state[i], f"S{i}") for i in range(len(state_to_idx))]
    # Abbreviate names
    state_abbrev = [name[:4] for name in state_names_ordered]

    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_xticks(range(len(state_abbrev)))
    ax.set_yticks(range(len(state_abbrev)))
    ax.set_xticklabels(state_abbrev, rotation=90, fontsize=8)
    ax.set_yticklabels(state_abbrev, fontsize=8)
    ax.set_title('Haversine Distance Between State Centroids (km)')
    plt.colorbar(im, label='Distance (km)')
    plt.tight_layout()
    plt.savefig(config.paths.logs_dir / "haversine_matrix.png")
    plt.close(fig)

    # Visualize soft labels for a sample state
    from src.data.state_utils import get_soft_labels

    # Pick California as example (if in dataset)
    example_state_orig = 4  # California
    if example_state_orig in state_to_idx:
        example_state = state_to_idx[example_state_orig]
        example_name = "California"
    else:
        example_state = 0
        example_name = STATE_INDEX_TO_NAME.get(idx_to_state[0], "Unknown")

    # Get soft labels with different temperatures
    temps = [100, 300, 500, 1000]

    fig, axes = plt.subplots(1, len(temps), figsize=(16, 4))

    for ax, temp in zip(axes, temps):
        soft_labels = get_soft_labels(example_state, distance_matrix, temperature=temp)
        ax.bar(range(len(soft_labels)), soft_labels.numpy())
        ax.set_title(f'Temperature = {temp} km')
        ax.set_xlabel('State Index')
        ax.set_ylabel('Probability')

    plt.suptitle(f'Soft Labels for {example_name} (True State)')
    plt.tight_layout()
    plt.savefig(config.paths.logs_dir / "soft_labels.png")
    plt.close(fig)

    # ## 4. Create DataLoaders

    from src.data.dataloader import create_dataloaders

    # Create train and validation dataloaders
    train_loader, val_loader, label_encoder = create_dataloaders(config, return_encoder=True)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Number of classes: {label_encoder.num_classes}")

    # Test a batch
    batch = next(iter(train_loader))

    print("Batch contents:")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value)}")

    # ## 5. Model Setup

    from src.models.geoguessr_model import GeoGuessrModel

    # Create model
    model = GeoGuessrModel(config, num_classes=label_encoder.num_classes)
    model = model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    # This can provide 20-30% speedup on A100
    # NOTE: 'reduce-overhead' mode uses CUDA graphs which can cause tensor reuse errors
    # 'default' mode is safer and still provides good speedup
    # 
    # IMPORTANT: If you see CUDAGraphs errors, set this to False
    # The first batch after compilation can also be very slow (5-15 min)
    USE_TORCH_COMPILE = False  # Disabled by default due to CUDAGraphs issues
    
    if USE_TORCH_COMPILE:
        try:
            print("Compiling model with torch.compile()...")
            print("Using 'default' mode (safer than 'reduce-overhead' which uses CUDA graphs)")
            # Use 'default' mode instead of 'reduce-overhead' to avoid CUDAGraphs issues
            # 'default' still provides good speedup without the tensor reuse problems
            model = torch.compile(model, mode='default')
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile() not available or failed: {e}")
            print("Continuing without compilation...")
    else:
        print("torch.compile() disabled - using standard model")
    
    # Clear cache before test forward pass
    torch.cuda.empty_cache()

    # Warm-up forward pass to "burn in" the compiled model
    # This is critical - the first forward pass after torch.compile() can take 5-15 minutes
    print("\n" + "="*60)
    print("Warming up compiled model (this may take 5-15 minutes)...")
    print("="*60)
    import time
    warmup_start = time.time()
    with torch.no_grad():
        # Use a smaller subset for test forward pass
        test_batch_size = min(32, batch['images'].shape[0])
        test_images = batch['images'][:test_batch_size].to(device)
        print(f"Running warm-up forward pass with batch size {test_batch_size}...")
        outputs = model(test_images)
    warmup_time = time.time() - warmup_start
    print(f"Warm-up completed in {warmup_time:.1f} seconds ({warmup_time/60:.1f} minutes)")
    print("="*60 + "\n")

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # ## 6. Loss Function Setup

    from src.training.losses import create_loss_function

    # Create loss function with haversine smoothing
    loss_fn = create_loss_function(config, distance_matrix)
    loss_fn = loss_fn.to(device)

    print("Loss function created with:")
    print(f"  Classification weight: {config.training.classification_weight}")
    print(f"  GPS weight: {config.training.gps_weight}")
    print(f"  Haversine smoothing: {config.training.use_haversine_smoothing}")
    print(f"  Temperature: {config.training.haversine_temperature} km")

    # Test loss computation with smaller batch
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_batch_size = min(32, batch['images'].shape[0])
        test_images = batch['images'][:test_batch_size].to(device)
        test_labels = batch['state_label'][:test_batch_size].to(device)
        test_gps = batch['gps'][:test_batch_size].to(device)

        outputs = model(test_images)
        losses = loss_fn(
            outputs['class_logits'],
            outputs['gps_coords'],
            test_labels,
            test_gps
        )

    print("Test loss computation:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    # ## 7. Training Phase 1: Frozen Backbone

    from src.training.trainer import Trainer

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device
    )

    # Train with frozen backbone
    history_frozen = trainer.train(
        num_epochs=config.training.num_epochs_frozen,
        phase="frozen",
        save_dir=config.paths.checkpoints_dir
    )

    # Plot training curves
    from src.utils.visualization import plot_training_curves

    plot_training_curves(history_frozen, save_path=config.paths.logs_dir / "frozen_training.png")

    # ## 8. Training Phase 2: Fine-tuning

    # Load best frozen checkpoint if needed
    best_frozen_path = config.paths.checkpoints_dir / "best_frozen.pt"
    if best_frozen_path.exists():
        trainer.load_checkpoint(best_frozen_path)

    # Fine-tune with unfrozen backbone
    history_finetune = trainer.train(
        num_epochs=config.training.num_epochs_finetune,
        phase="finetune",
        save_dir=config.paths.checkpoints_dir
    )

    # Plot combined training curves
    combined_history = {
        'train_loss': history_frozen['train_loss'] + history_finetune['train_loss'],
        'val_loss': history_frozen['val_loss'] + history_finetune['val_loss'],
        'val_score': history_frozen['val_score'] + history_finetune['val_score'],
        'learning_rates': history_frozen['learning_rates'] + history_finetune['learning_rates']
    }

    plot_training_curves(combined_history, save_path=config.paths.logs_dir / "full_training.png")

    # ## 9. Evaluation & Analysis

    # Load best model
    best_path = config.paths.checkpoints_dir / "best_finetune.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path)

    # Final validation
    final_metrics = trainer.validate()

    print("\n" + "="*50)
    print("Final Validation Results")
    print("="*50)
    print(f"Competition Score: {final_metrics['final_score']:.4f}")
    print(f"  Classification Score: {final_metrics['classification_score']:.4f}")
    print(f"  GPS Score: {final_metrics['gps_score']:.4f}")
    print(f"\nGPS Metrics:")
    print(f"  Mean Distance: {final_metrics['gps_mean_distance_km']:.1f} km")
    print(f"  Median Distance: {final_metrics['gps_median_distance_km']:.1f} km")

    # Generate predictions on validation set for analysis
    from src.inference.predict import predict_dataset

    val_predictions = predict_dataset(
        model=model,
        dataloader=val_loader,
        device=device,
        top_k=5,
        use_amp=config.training.use_amp,
        idx_to_state=label_encoder.idx_to_state
    )

    print(f"Generated predictions for {len(val_predictions['sample_ids'])} samples")

    # Visualize GPS predictions
    from src.utils.visualization import plot_gps_predictions

    plot_gps_predictions(
        val_predictions['latitudes'],
        val_predictions['longitudes'],
        save_path=config.paths.logs_dir / "gps_predictions.png"
    )

    # State prediction distribution
    from src.utils.visualization import plot_state_distribution

    plot_state_distribution(
        val_predictions['top_k_states'][:, 0],  # Top-1 predictions
        STATE_INDEX_TO_NAME,
        title="Validation Set - Predicted State Distribution",
        save_path=config.paths.logs_dir / "state_predictions.png"
    )

    # ## 10. Generate Test Predictions & Submission

    from src.data.dataloader import create_test_dataloader

    # Create test dataloader
    test_loader = create_test_dataloader(config, label_encoder.state_to_idx)
    print(f"Test batches: {len(test_loader)}")

    # Generate predictions
    test_predictions = predict_dataset(
        model=model,
        dataloader=test_loader,
        device=device,
        top_k=5,
        use_amp=config.training.use_amp,
        idx_to_state=label_encoder.idx_to_state
    )

    print(f"Generated predictions for {len(test_predictions['sample_ids'])} test samples")

    from src.inference.submission import create_submission, analyze_submission

    # Create submission file
    submission_path = config.paths.submissions_dir / "submission.csv"

    submission = create_submission(
        predictions=test_predictions,
        template_path=str(config.paths.test_csv),
        output_path=str(submission_path),
        validate=True
    )

    # Analyze submission
    analysis = analyze_submission(submission)

    print("Submission Analysis:")
    print(f"  Unique states predicted: {analysis['num_unique_states']}")
    print(f"  Latitude range: {analysis['lat_range']}")
    print(f"  Longitude range: {analysis['lon_range']}")

    # Top 5 most predicted states
    print("\nTop 5 most predicted states:")
    for state_idx, count in sorted(analysis['state_distribution'].items(), key=lambda x: -x[1])[:5]:
        state_name = STATE_INDEX_TO_NAME.get(int(state_idx), f"Unknown ({state_idx})")
        print(f"  {state_name}: {count}")

    # View submission
    submission.head(10)


if __name__ == "__main__":
    # To run this function, use:
    #   modal run modal_train.py::train
    # Or deploy and invoke:
    #   modal deploy modal_train.py
    #   modal run modal_train.py::train
    with app.run():
        train.remote()
