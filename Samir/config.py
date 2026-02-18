"""
Configuration file for multimodal classification experiments
All hyperparameters and model settings are hot-pluggable here
"""

import torch

class Config:
    # ==================== Data Configuration ====================
    IMAGE_FOLDER = "path/to/images"  # Path to image folder
    CSV_PATH = "path/to/data.csv"  # Path to CSV file
    IMAGE_COL = "index"  # Column name for image filenames
    TEXT_COL = "text"  # Column name for extracted text
    LABEL_COL = "label"  # Column name for labels
    
    # ==================== Experiment Configuration ====================
    # Modality selection: 'image', 'text', 'both'
    USE_MODALITY = 'both'  # Options: 'image', 'text', 'both'
    
    # Image type: 'original', 'text_removed'
    IMAGE_TYPE = 'original'  # Options: 'original', 'text_removed'
    
    # Fusion strategy (only used when USE_MODALITY='both')
    FUSION_STRATEGY = 'hybrid'  # Options: 'early', 'late_voting', 'late_bagging', 'hybrid'
    
    # Number of classes (auto-detected from data if set to None)
    NUM_CLASSES = None  # Will be set automatically, or manually set to 2 or 3
    
    # ==================== Model Configuration ====================
    # Image encoder
    IMAGE_ENCODER = 'clip'  # Options: 'clip'
    CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
    FREEZE_IMAGE_ENCODER = True
    
    # Text encoder
    TEXT_ENCODER = 'bge-m3'  # Options: 'bge-m3', 'e5-large'
    BGE_MODEL_NAME = 'BAAI/bge-m3'
    E5_MODEL_NAME = 'intfloat/e5-large-v2'
    FREEZE_TEXT_ENCODER = True
    
    # Feature dimensions (will be set based on model selection)
    IMAGE_FEATURE_DIM = 512  # CLIP ViT-B/32
    TEXT_FEATURE_DIM = 1024  # BGE-M3
    
    # Fusion model configuration
    HIDDEN_DIM = 256  # Hidden dimension for fusion layers
    DROPOUT = 0.3  # Dropout rate
    
    # Hybrid fusion specific
    HYBRID_ATTENTION_HEADS = 4  # Number of attention heads
    
    # ==================== Training Configuration ====================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5  # L2 regularization
    NUM_EPOCHS = 50
    
    # K-Fold cross-validation
    K_FOLDS = 5
    
    # Early stopping
    PATIENCE = 10  # Number of epochs to wait for improvement
    MIN_DELTA = 1e-4  # Minimum change to qualify as improvement
    
    # ==================== Optimizer Configuration ====================
    OPTIMIZER = 'adamw'  # Options: 'adam', 'adamw', 'sgd'
    SGD_MOMENTUM = 0.9  # Only used if OPTIMIZER='sgd'
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'  # Options: 'cosine', 'step', 'plateau'
    SCHEDULER_PATIENCE = 5  # For plateau scheduler
    SCHEDULER_FACTOR = 0.5  # For plateau and step scheduler
    SCHEDULER_STEP_SIZE = 10  # For step scheduler
    
    # ==================== Device Configuration ====================
    # Auto-detect available device (CUDA on Windows/Linux, CPU otherwise)
    # Note: MPS (Apple Silicon) is not used as it may not be available in all PyTorch builds
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================== Logging Configuration ====================
    SAVE_DIR = "outputs"  # Directory to save models and results
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_BEST_MODEL = True
    
    # ==================== Image Preprocessing ====================
    IMAGE_SIZE = 224  # Input image size
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # ==================== Text Preprocessing ====================
    MAX_TEXT_LENGTH = 77  # Maximum text sequence length (CLIP default)
    
    # ==================== Random Seed ====================
    SEED = 42
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                print(f"Warning: {key} is not a valid configuration parameter")
    
    @classmethod
    def get_config_dict(cls):
        """Return configuration as dictionary"""
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not callable(v) and not isinstance(v, classmethod)}
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("CURRENT CONFIGURATION")
        print("=" * 60)
        for key, value in cls.get_config_dict().items():
            print(f"{key}: {value}")
        print("=" * 60)
