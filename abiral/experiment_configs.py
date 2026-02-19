"""
Example configurations for all possible experiments
Copy and modify these for your specific use case
"""

from config import Config
from train import main

# ==============================================================================
# BINARY CLASSIFICATION (HATE/NON-HATE) EXPERIMENTS
# ==============================================================================

def binary_hate_detection_hybrid():
    """Binary hate detection using hybrid fusion"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/hate_detection/images",
        CSV_PATH="data/hate_detection/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='original',
        NUM_CLASSES=2,  # Binary classification
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# 3-CLASS SENTIMENT ANALYSIS EXPERIMENTS
# ==============================================================================

def sentiment_analysis_hybrid():
    """Sentiment analysis (positive/negative/neutral) using hybrid fusion"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/sentiment/images",
        CSV_PATH="data/sentiment/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='original',
        NUM_CLASSES=3,  # 3-class classification
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# ABLATION STUDY 1: MODALITY COMPARISON
# ==============================================================================

def ablation_image_only():
    """Use only image features"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='image',  # Image only
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def ablation_text_only():
    """Use only text features"""
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images",
        CSV_PATH="../train.csv",
        USE_MODALITY='text',  # Text only
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def ablation_multimodal():
    """Use both image and text"""
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images",
        CSV_PATH="../train.csv",
        USE_MODALITY='both',  # Both modalities
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# ABLATION STUDY 2: FUSION STRATEGY COMPARISON
# ==============================================================================

def fusion_early():
    """Early fusion (concatenation)"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='early',  # Early fusion
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def fusion_late_voting():
    """Late fusion with voting"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='late_voting',  # Late fusion - voting
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def fusion_late_bagging():
    """Late fusion with bagging"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='late_bagging',  # Late fusion - bagging
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def fusion_hybrid():
    """Hybrid fusion (dynamic attention)"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',  # Hybrid fusion - PRIORITY
        IMAGE_TYPE='original',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5,
        HYBRID_ATTENTION_HEADS=4
    )
    return main(config)


# ==============================================================================
# ABLATION STUDY 3: IMAGE TYPE COMPARISON
# ==============================================================================

def image_original():
    """Use original images"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='image',
        IMAGE_TYPE='original',  # Original images
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def image_text_removed():
    """Use text-removed images"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='image',
        IMAGE_TYPE='text_removed',  # Text removed images
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def image_text_removed_with_text():
    """Use text-removed images + extracted text"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='text_removed',  # Text removed + extracted text
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=30,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# ADVANCED: HYPERPARAMETER TUNING
# ==============================================================================

def hyperparameter_tuning_example():
    """Example with different hyperparameters"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        
        # Model architecture
        HIDDEN_DIM=512,              # Larger hidden dimension
        DROPOUT=0.4,                 # Higher dropout
        HYBRID_ATTENTION_HEADS=8,    # More attention heads
        
        # Training
        BATCH_SIZE=32,               # Larger batch
        LEARNING_RATE=5e-5,          # Lower learning rate
        NUM_EPOCHS=100,              # More epochs
        K_FOLDS=10,                  # More folds
        
        # Regularization
        WEIGHT_DECAY=1e-4,          # Stronger L2 regularization
        
        # Optimizer and scheduler
        OPTIMIZER='adamw',
        USE_SCHEDULER=True,
        SCHEDULER_TYPE='cosine',
        
        # Early stopping
        PATIENCE=20,                 # More patience
        MIN_DELTA=1e-5
    )
    return main(config)


# ==============================================================================
# ADVANCED: TEXT ENCODER COMPARISON
# ==============================================================================

def text_encoder_bge_m3():
    """Use BGE-M3 text encoder"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        TEXT_ENCODER='bge-m3',       # BGE-M3 encoder
        BGE_MODEL_NAME='BAAI/bge-m3',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


def text_encoder_e5_large():
    """Use E5-Large text encoder"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        TEXT_ENCODER='e5-large',     # E5-Large encoder
        E5_MODEL_NAME='intfloat/e5-large-v2',
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        NUM_EPOCHS=50,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# ADVANCED: FINE-TUNING ENCODERS
# ==============================================================================

def finetune_encoders():
    """Fine-tune both image and text encoders"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        
        # Unfreeze encoders for fine-tuning
        FREEZE_IMAGE_ENCODER=False,  # Fine-tune CLIP
        FREEZE_TEXT_ENCODER=False,   # Fine-tune text encoder
        
        # Lower learning rate for fine-tuning
        LEARNING_RATE=1e-5,
        
        # More regularization
        DROPOUT=0.5,
        WEIGHT_DECAY=1e-3,
        
        BATCH_SIZE=8,  # Smaller batch due to more parameters
        NUM_EPOCHS=30,
        K_FOLDS=5
    )
    return main(config)


# ==============================================================================
# CPU-FRIENDLY CONFIGURATION
# ==============================================================================

def cpu_friendly():
    """Configuration for training on CPU"""
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        
        DEVICE='cpu',               # Use CPU
        BATCH_SIZE=4,               # Small batch size
        NUM_EPOCHS=20,              # Fewer epochs
        K_FOLDS=3,                  # Fewer folds
        
        FREEZE_IMAGE_ENCODER=True,  # Keep frozen for speed
        FREEZE_TEXT_ENCODER=True
    )
    return main(config)


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == '__main__':
    import sys
    
    experiments = {
        # Binary classification
        'binary_hate': binary_hate_detection_hybrid,
        
        # Sentiment analysis
        'sentiment': sentiment_analysis_hybrid,
        
        # Modality ablation
        'image_only': ablation_image_only,
        'text_only': ablation_text_only,
        'multimodal': ablation_multimodal,
        
        # Fusion strategy ablation
        'early': fusion_early,
        'late_voting': fusion_late_voting,
        'late_bagging': fusion_late_bagging,
        'hybrid': fusion_hybrid,
        
        # Image type ablation
        'original': image_original,
        'text_removed': image_text_removed,
        'text_removed_multi': image_text_removed_with_text,
        
        # Advanced
        'hypertuning': hyperparameter_tuning_example,
        'bge_m3': text_encoder_bge_m3,
        'e5_large': text_encoder_e5_large,
        'finetune': finetune_encoders,
        'cpu': cpu_friendly
    }
    
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
        if exp_name in experiments:
            print(f"\nRunning experiment: {exp_name}\n")
            experiments[exp_name]()
        else:
            print(f"Unknown experiment: {exp_name}")
            print(f"\nAvailable experiments:")
            for name in experiments.keys():
                print(f"  - {name}")
    else:
        print("Usage: python experiment_configs.py <experiment_name>")
        print("\nAvailable experiments:")
        for name in experiments.keys():
            print(f"  - {name}")
