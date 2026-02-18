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
    
    TEXT_MODEL_OPTIONS = {
    # === BEST FOR NEPALI ===
    "muril": "google/muril-base-cased",           # Google's Indian Languages (includes Nepali)
    "muril_large": "google/muril-large-cased",    # Larger version, better accuracy
    "nepberta": "Shushant/NepBERTa",              # Native Nepali BERT (0.8B Nepali words)
    "nepali_roberta": "IRIIS/Nepali-RoBERTa-110M", # IRIIS 2024 Nepali model
    
    # === MULTILINGUAL (Fallback) ===
    "xlm_roberta": "xlm-roberta-base",            # 100 languages including Nepali
    "xlm_roberta_large": "xlm-roberta-large",     # Larger multilingual
    "mbert": "bert-base-multilingual-cased",      # Original multilingual BERT
    }
    # Text encoder
    TEXT_ENCODER = 'bge-m3'  # Options: 'bge-m3', 'e5-large'
    BGE_MODEL_NAME = 'BAAI/bge-m3'
    E5_MODEL_NAME = 'intfloat/e5-large-v2'
    FREEZE_TEXT_ENCODER = True
    
    # Partial unfreezing (advanced: unfreeze only last N layers)
    # If set to a number, overrides FREEZE_* settings and unfreezes last N layers
    # CLIP ViT has 12 layers, BGE-M3 has 24 layers
    UNFREEZE_IMAGE_LAYERS = None  # e.g., 3 = unfreeze last 3 layers of CLIP
    UNFREEZE_TEXT_LAYERS = None   # e.g., 4 = unfreeze last 4 layers of BGE-M3
    
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
    WEIGHT_DECAY = 1e-4  # L2 regularization
    NUM_EPOCHS = 50
    
    # Class weight exponent: controls how aggressively minority class is boosted
    # Formula: weight = (max_count / class_count) ^ exponent
    # 0.0 = equal weights, 0.3 = mild boost, 0.5 = moderate (default sqrt), 1.0 = full inverse frequency
    CLASS_WEIGHT_EXPONENT = 0.5
    
    # Label smoothing: reduces overconfidence, improves generalization
    # 0.0 = no smoothing (hard labels), 0.1 = recommended for small datasets
    LABEL_SMOOTHING = 0.0
    
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
    SCHEDULER_TYPE = 'plateau'  # Options: 'cosine', 'step', 'plateau'
    SCHEDULER_PATIENCE = 5  # For plateau scheduler
    SCHEDULER_FACTOR = 0.5  # For plateau and step scheduler
    SCHEDULER_STEP_SIZE = 10  # For step scheduler
    
    # ==================== Device Configuration ====================
    # Auto-detect available device (CUDA on Windows/Linux, MPS on macOS, CPU otherwise)
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'
    
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
