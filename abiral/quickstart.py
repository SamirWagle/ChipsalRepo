"""
Quick Start Example - Run Hybrid Fusion Model
This is the simplest way to get started with the framework
"""

from config import Config
from train import main

# ============================================================================
# STEP 1: Update these paths to match your data
# ============================================================================

IMAGE_FOLDER = "path/to/your/images"  # Folder containing your meme images
CSV_FILE = "path/to/your/data.csv"    # CSV with Image, Text, Label columns

# ============================================================================
# STEP 2: Configure the experiment
# ============================================================================

config = Config()
config.update(
    # Data paths
    IMAGE_FOLDER=IMAGE_FOLDER,
    CSV_PATH=CSV_FILE,
    
    # Column names in your CSV
    IMAGE_COL='Image',    # Column with image filenames
    TEXT_COL='Text',      # Column with extracted text
    LABEL_COL='Label',    # Column with labels (0, 1 for binary; 0, 1, 2 for 3-class)
    
    # Experiment settings
    USE_MODALITY='both',          # Use both image and text
    FUSION_STRATEGY='hybrid',     # Hybrid fusion (priority)
    IMAGE_TYPE='original',        # Use original images
    
    # Model settings
    FREEZE_IMAGE_ENCODER=True,    # Freeze CLIP encoder (faster training)
    FREEZE_TEXT_ENCODER=True,     # Freeze text encoder (faster training)
    HIDDEN_DIM=256,               # Hidden layer dimension
    DROPOUT=0.3,                  # Dropout rate
    HYBRID_ATTENTION_HEADS=4,     # Attention heads for hybrid fusion
    
    # Training settings
    BATCH_SIZE=16,                # Batch size (reduce if GPU memory issues)
    LEARNING_RATE=1e-4,           # Learning rate
    NUM_EPOCHS=50,                # Maximum epochs
    K_FOLDS=5,                    # Number of cross-validation folds
    
    # Optimizer and scheduler
    OPTIMIZER='adamw',            # AdamW optimizer
    WEIGHT_DECAY=1e-5,           # L2 regularization
    USE_SCHEDULER=True,           # Use learning rate scheduler
    SCHEDULER_TYPE='cosine',      # Cosine annealing
    
    # Early stopping
    PATIENCE=10,                  # Stop if no improvement for 10 epochs
    
    # Device
    DEVICE='cuda',                # Use 'cpu' if no GPU available
    
    # Output
    SAVE_DIR='outputs',           # Directory to save results
    SAVE_BEST_MODEL=True          # Save best model for each fold
)

# ============================================================================
# STEP 3: Run the experiment
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("QUICK START: Hybrid Fusion Multimodal Classification")
    print("="*80)
    
    # Print configuration
    config.print_config()
    
    # Run training
    summary, save_dir = main(config)
    
    # Print final results
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {save_dir}")
    print(f"\nCross-validation results (mean ± std):")
    print(f"  Accuracy:  {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    print(f"  F1 Score:  {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
    print(f"  Precision: {summary['precision']['mean']:.4f} ± {summary['precision']['std']:.4f}")
    print(f"  Recall:    {summary['recall']['mean']:.4f} ± {summary['recall']['std']:.4f}")
    print("\n" + "="*80)
    
    # Instructions for inference
    print("\nTo make predictions on new images, run:")
    print(f"  python test.py --model_path {save_dir}/best_model_fold_0.pt \\")
    print(f"                 --mode single \\")
    print(f"                 --image_path your_image.jpg \\")
    print(f"                 --text 'extracted text' \\")
    print(f"                 --use_modality both \\")
    print(f"                 --fusion_strategy hybrid")
    print("="*80 + "\n")
