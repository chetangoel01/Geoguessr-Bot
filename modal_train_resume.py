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

# Use existing volume for data (contains checkpoint from frozen phase)
data_volume = modal.Volume.from_name("geoguessr-data", create_if_missing=True)


@app.function(
    image=geo_image,
    gpu="A100-80GB",
    volumes={"/data": data_volume},
    timeout=60 * 60 * 8,
)
def train_resume():
    """Resume training from frozen checkpoint with finetuning phase only."""
    import os
    import sys
    import time
    
    # Add the mounted project directory to Python path
    project_path = "/root/geoguessr-competition"
    sys.path.insert(0, project_path)
    os.chdir(project_path)
    
    import torch
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # Optimize PyTorch memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    print("="*60)
    print("RESUMING TRAINING FROM FROZEN CHECKPOINT")
    print("="*60)
    
    # Check GPU
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')
        print("cuDNN and TF32 optimizations enabled")

    from src.config import get_config, STATE_INDEX_TO_NAME

    config = get_config()

    # ---- Override paths to use the Modal volume ----
    DATA_ROOT = Path("/data/data/kaggle_dataset")
    print(f"Using DATA_ROOT: {DATA_ROOT}")
    
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")
    
    print("Contents of DATA_ROOT:")
    for p in DATA_ROOT.iterdir():
        print(f"   {p}")

    config.paths.train_csv = DATA_ROOT / "train_ground_truth.csv"
    config.paths.test_csv = DATA_ROOT / "sample_submission.csv"
    config.paths.train_images = DATA_ROOT / "train_images"
    config.paths.test_images = DATA_ROOT / "test_images"
    config.paths.state_mapping_csv = DATA_ROOT / "state_mapping.csv"
    
    # Use /data for persistent outputs
    config.paths.checkpoints_dir = Path("/data/checkpoints")
    config.paths.submissions_dir = Path("/data/submissions")
    config.paths.logs_dir = Path("outputs/training_logs")
    config.paths.attention_maps_dir = Path("outputs/attention_maps")
    config.paths.processed_dir = Path("data/processed")
    config.paths.state_centroids = config.paths.processed_dir / "state_centroids.json"
    config.paths.haversine_matrix = config.paths.processed_dir / "haversine_matrix.npy"
    
    # Create all necessary directories
    for d in [
        config.paths.checkpoints_dir, 
        config.paths.submissions_dir, 
        config.paths.logs_dir,
        config.paths.attention_maps_dir,
        config.paths.processed_dir,
        config.paths.state_centroids.parent,
        config.paths.haversine_matrix.parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {d}")
    
    # ---- Detect best checkpoint to resume from ----
    # Priority: emergency_checkpoint > best_finetune > epoch_N_finetune > best_frozen
    checkpoint_dir = config.paths.checkpoints_dir
    
    emergency_ckpt = checkpoint_dir / "emergency_checkpoint.pt"
    best_finetune_ckpt = checkpoint_dir / "best_finetune.pt"
    best_frozen_ckpt = checkpoint_dir / "best_frozen.pt"
    
    # Find any epoch checkpoints
    epoch_ckpts = sorted(checkpoint_dir.glob("epoch_*_finetune.pt"), 
                         key=lambda p: int(p.stem.split('_')[1]), reverse=True)
    
    resume_from_finetune = False
    start_epoch = 0
    
    if emergency_ckpt.exists():
        checkpoint_path = emergency_ckpt
        resume_from_finetune = True
        print(f"\n*** Found emergency checkpoint - will resume from interrupted training ***")
    elif best_finetune_ckpt.exists():
        checkpoint_path = best_finetune_ckpt
        resume_from_finetune = True
        print(f"\n*** Found best_finetune checkpoint - will resume training ***")
    elif epoch_ckpts:
        checkpoint_path = epoch_ckpts[0]  # Most recent epoch
        resume_from_finetune = True
        print(f"\n*** Found epoch checkpoint - will resume from {checkpoint_path.name} ***")
    elif best_frozen_ckpt.exists():
        checkpoint_path = best_frozen_ckpt
        resume_from_finetune = False
        print(f"\n*** Starting fresh finetune from frozen checkpoint ***")
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}. "
            "Run the frozen training phase first."
        )
    
    print(f"Checkpoint path: {checkpoint_path}")

    # Set seeds
    import random
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # ============================================================
    # KEY CHANGE: Partial backbone unfreezing to avoid OOM
    # ============================================================
    # Full finetuning of 428M params requires storing activations for
    # all 24 transformer layers during backprop - too much for 80GB.
    # 
    # Solution: Only unfreeze the last N layers of the backbone.
    # Lower layers learn generic features that transfer well anyway.
    # ============================================================
    
    UNFREEZE_LAYERS = 6  # Only finetune last 6 of 24 transformer layers
    
    config.training.batch_size = 64  # Can use larger batch with partial unfreeze
    config.training.gradient_accumulation_steps = 16  # Effective batch = 1024
    # Effective batch size: 64 * 16 = 1024
    
    config.training.num_workers = 4
    
    print("\nFinetuning configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.training.batch_size * config.training.gradient_accumulation_steps}")

    # ---- Load data ----
    train_df = pd.read_csv(config.paths.train_csv)
    print(f"Training samples: {len(train_df)}")

    from src.data.state_utils import compute_state_centroids, compute_haversine_matrix

    centroids = compute_state_centroids(train_df, config.paths.state_centroids)
    
    unique_states = sorted(train_df['state_idx'].unique())
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    distance_matrix = compute_haversine_matrix(
        centroids, state_to_idx, config.paths.haversine_matrix
    )
    print(f"Distance matrix shape: {distance_matrix.shape}")

    # ---- Create DataLoaders ----
    from src.data.dataloader import create_dataloaders

    train_loader, val_loader, label_encoder = create_dataloaders(config, return_encoder=True)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ---- Create Model ----
    from src.models.geoguessr_model import GeoGuessrModel

    model = GeoGuessrModel(config, num_classes=label_encoder.num_classes)
    model = model.to(device)

    # ---- Load checkpoint ----
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Previous best score: {checkpoint['best_score']:.4f}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"Previous metrics:")
        print(f"  Classification: {metrics.get('classification_score', 'N/A')}")
        print(f"  GPS: {metrics.get('gps_score', 'N/A')}")
    
    # Determine starting epoch for training loop
    if resume_from_finetune:
        start_epoch = checkpoint['epoch']  # Resume from next epoch
        print(f"\nWill resume training from epoch {start_epoch + 1}")
    else:
        start_epoch = 0
        print(f"\nWill start finetuning from epoch 1")

    # ---- Create Loss Function ----
    from src.training.losses import create_loss_function

    loss_fn = create_loss_function(config, distance_matrix)
    loss_fn = loss_fn.to(device)
    
    print("\nLoss function created with:")
    print(f"  Classification weight: {config.training.classification_weight}")
    print(f"  GPS weight: {config.training.gps_weight}")
    print(f"  Haversine smoothing: {config.training.use_haversine_smoothing}")
    print(f"  Temperature: {config.training.haversine_temperature} km")

    # ---- Warm-up forward pass ----
    # Critical: Test that model works with reduced batch size before full training
    print("\n" + "="*60)
    print("Running warm-up forward pass to verify memory fits...")
    print("="*60)
    
    # Get a test batch
    test_batch = next(iter(train_loader))
    
    # Clear everything before testing
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    warmup_start = time.time()
    
    # First test with frozen backbone (should work)
    model.freeze_backbone()
    with torch.no_grad():
        test_images = test_batch['images'].to(device)
        outputs = model(test_images)
        print(f"Frozen forward pass OK - class_logits: {outputs['class_logits'].shape}, gps: {outputs['gps_coords'].shape}")
    
    # CRITICAL: Clear all memory before unfrozen test
    del outputs, test_images
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Report memory after clearing
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        print(f"Memory after clearing frozen pass: {allocated:.1f} GB")
    
    # Now test with PARTIALLY unfrozen backbone (only last N layers)
    # This is the key change - we don't unfreeze everything
    model.unfreeze_backbone(unfreeze_layers=UNFREEZE_LAYERS)
    torch.cuda.empty_cache()
    
    # Do a forward + backward pass to test memory with gradients
    test_images = test_batch['images'].to(device)
    test_labels = test_batch['state_label'].to(device)
    test_gps = test_batch['gps'].to(device)
    
    # Use AMP like actual training
    from torch.amp import autocast
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(test_images)
        losses = loss_fn(
            outputs['class_logits'],
            outputs['gps_coords'],
            test_labels,
            test_gps
        )
    
    # Test backward pass
    losses['loss'].backward()
    
    warmup_time = time.time() - warmup_start
    print(f"\nWarm-up completed in {warmup_time:.1f} seconds")
    print(f"Test loss: {losses['loss'].item():.4f} (cls: {losses['cls_loss'].item():.4f}, gps: {losses['gps_loss'].item():.4f})")
    
    # Report memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU memory - Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB, Peak: {peak:.1f} GB")
    
    # Clean up before training
    del outputs, test_images, test_labels, test_gps, losses, test_batch
    model.zero_grad()
    torch.cuda.empty_cache()
    
    print("="*60)
    print("Warm-up successful! Proceeding with training...")
    print("="*60 + "\n")

    # ---- Create Trainer ----
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        config=config,
        device=device
    )
    
    # Set the best score from checkpoint so we only save improvements
    trainer.best_score = checkpoint['best_score']
    print(f"Trainer initialized with best_score = {trainer.best_score:.4f}")

    # ---- Run Finetuning with PARTIAL backbone unfreezing ----
    print("\n" + "="*60)
    print(f"Starting FINETUNE phase (last {UNFREEZE_LAYERS} backbone layers unfrozen)")
    print(f"Will train for {config.training.num_epochs_finetune} epochs")
    print(f"Checkpoints saved to: {config.paths.checkpoints_dir}")
    print("="*60 + "\n")
    
    # IMPORTANT: We need to manually setup the optimizer with partial unfreezing
    # because trainer.train() with phase="finetune" would unfreeze everything
    
    # First, ensure model has partial unfreezing applied
    model.freeze_backbone()  # Reset to frozen
    model.unfreeze_backbone(unfreeze_layers=UNFREEZE_LAYERS)  # Partially unfreeze
    
    # Manually setup optimizer (mimicking what trainer.setup_optimizer does for finetune)
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    param_groups = model.get_optimizer_param_groups(
        backbone_lr=config.training.backbone_lr,
        head_lr=config.training.head_lr,
        weight_decay=config.training.weight_decay
    )
    
    trainer.optimizer = AdamW(
        param_groups,
        weight_decay=config.training.weight_decay
    )
    
    # Setup scheduler
    num_epochs = config.training.num_epochs_finetune
    num_training_steps = (len(train_loader) * num_epochs) // config.training.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
    
    warmup_scheduler = LinearLR(
        trainer.optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    
    main_scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=1e-7
    )
    
    trainer.scheduler = SequentialLR(
        trainer.optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps]
    )
    
    print(f"Optimizer setup for partial finetune:")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Backbone LR: {config.training.backbone_lr}")
    print(f"  Head LR: {config.training.head_lr}")
    
    # If resuming from finetune checkpoint, restore optimizer and scheduler state
    if resume_from_finetune and 'optimizer_state_dict' in checkpoint:
        print("\nRestoring optimizer state from checkpoint...")
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            print("Restoring scheduler state from checkpoint...")
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and trainer.scaler is not None:
            print("Restoring AMP scaler state from checkpoint...")
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore global step
        if 'global_step' in checkpoint:
            trainer.global_step = checkpoint['global_step']
            print(f"Restored global_step: {trainer.global_step}")
    
    # Calculate remaining epochs
    remaining_epochs = num_epochs - start_epoch
    print(f"\nWill train for {remaining_epochs} more epochs (epochs {start_epoch + 1} to {num_epochs})")
    
    # Now run training epochs manually (without calling trainer.train which would re-setup optimizer)
    try:
        for epoch in range(start_epoch, num_epochs):
            trainer.current_epoch = epoch + 1
            epoch_start = time.time()
            
            # Training
            train_metrics = trainer.train_epoch()
            
            # Validation
            val_metrics = trainer.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log results - use .get() with defaults for safety
            print(f"\nEpoch {trainer.current_epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f} "
                  f"(cls: {train_metrics['train_cls_loss']:.4f}, "
                  f"gps: {train_metrics['train_gps_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Val Score: {val_metrics['final_score']:.4f} "
                  f"(cls: {val_metrics['classification_score']:.4f}, "
                  f"gps: {val_metrics['gps_score']:.4f})")
            
            # Safely get GPS distance metric
            gps_mean_dist = val_metrics.get('gps_gps_mean_distance_km') or val_metrics.get('gps_mean_distance_km', 'N/A')
            if isinstance(gps_mean_dist, (int, float)):
                print(f"  GPS Mean Distance: {gps_mean_dist:.1f} km")
            else:
                print(f"  GPS Mean Distance: {gps_mean_dist}")
            
            # Update history
            trainer.history['train_loss'].append(train_metrics['train_loss'])
            trainer.history['val_loss'].append(val_metrics['val_loss'])
            trainer.history['val_score'].append(val_metrics['final_score'])
            trainer.history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Save checkpoint if best
            if val_metrics['final_score'] > trainer.best_score:
                trainer.best_score = val_metrics['final_score']
                trainer.save_checkpoint(
                    config.paths.checkpoints_dir / "best_finetune.pt",
                    val_metrics
                )
                print(f"  New best score! Saved checkpoint.")
            
            # Save periodic checkpoint every epoch
            trainer.save_checkpoint(
                config.paths.checkpoints_dir / f"epoch_{trainer.current_epoch}_finetune.pt",
                val_metrics
            )
        
        history_finetune = trainer.history
        
        # Clean up emergency checkpoint if training completed successfully
        if emergency_ckpt.exists():
            print(f"\nRemoving emergency checkpoint (training completed successfully)")
            emergency_ckpt.unlink()
        
    except Exception as e:
        # Save emergency checkpoint on failure
        emergency_path = config.paths.checkpoints_dir / "emergency_checkpoint.pt"
        print(f"\nTraining interrupted! Saving emergency checkpoint to {emergency_path}")
        trainer.save_checkpoint(emergency_path, trainer.metric_tracker.compute() if trainer.metric_tracker.class_preds else {})
        raise e

    # ---- Plot training curves ----
    from src.utils.visualization import plot_training_curves
    plot_training_curves(history_finetune, save_path=config.paths.logs_dir / "finetune_training.png")
    print(f"Training curves saved to {config.paths.logs_dir / 'finetune_training.png'}")

    # ---- Final Evaluation ----
    best_finetune_path = config.paths.checkpoints_dir / "best_finetune.pt"
    if best_finetune_path.exists():
        print(f"\nLoading best finetuned model from {best_finetune_path}")
        trainer.load_checkpoint(best_finetune_path)
    else:
        print("\nNo best_finetune.pt found, using current model state")

    final_metrics = trainer.validate()

    print("\n" + "="*50)
    print("Final Validation Results")
    print("="*50)
    print(f"Competition Score: {final_metrics['final_score']:.4f}")
    print(f"  Classification Score: {final_metrics['classification_score']:.4f}")
    print(f"  GPS Score: {final_metrics['gps_score']:.4f}")
    
    # Safely get GPS metrics
    gps_mean = final_metrics.get('gps_gps_mean_distance_km') or final_metrics.get('gps_mean_distance_km')
    gps_median = final_metrics.get('gps_gps_median_distance_km') or final_metrics.get('gps_median_distance_km')
    print(f"\nGPS Metrics:")
    if gps_mean is not None:
        print(f"  Mean Distance: {gps_mean:.1f} km")
    if gps_median is not None:
        print(f"  Median Distance: {gps_median:.1f} km")

    # ---- Generate Test Predictions ----
    print("\n" + "="*50)
    print("Generating Test Predictions")
    print("="*50)
    
    from src.data.dataloader import create_test_dataloader
    from src.inference.predict import predict_dataset
    from src.inference.submission import create_submission, analyze_submission

    # Use smaller batch size for inference to be safe
    config.inference.batch_size = 64
    test_loader = create_test_dataloader(config, label_encoder.state_to_idx)
    print(f"Test batches: {len(test_loader)}")

    test_predictions = predict_dataset(
        model=model,
        dataloader=test_loader,
        device=device,
        top_k=5,
        use_amp=config.training.use_amp,
        idx_to_state=label_encoder.idx_to_state
    )

    print(f"Generated predictions for {len(test_predictions['sample_ids'])} test samples")

    # Create submission
    submission_path = config.paths.submissions_dir / "submission_finetuned.csv"

    submission = create_submission(
        predictions=test_predictions,
        template_path=str(config.paths.test_csv),
        output_path=str(submission_path),
        validate=True
    )

    analysis = analyze_submission(submission)

    print("\nSubmission Analysis:")
    print(f"  Unique states predicted: {analysis['num_unique_states']}")
    print(f"  Latitude range: {analysis['lat_range']}")
    print(f"  Longitude range: {analysis['lon_range']}")

    print("\nTop 5 most predicted states:")
    for state_idx, count in sorted(analysis['state_distribution'].items(), key=lambda x: -x[1])[:5]:
        state_name = STATE_INDEX_TO_NAME.get(int(state_idx), f"Unknown ({state_idx})")
        print(f"  {state_name}: {count}")

    # ---- Save training history ----
    import json
    history_path = config.paths.logs_dir / "finetune_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_finetune, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best model saved at: {best_finetune_path}")
    print(f"Submission saved at: {submission_path}")
    print("="*60)


if __name__ == "__main__":
    with app.run():
        train_resume.remote()