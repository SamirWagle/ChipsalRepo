"""
CHiPSAL 2026 - Subtask A: Hate Speech Detection Configuration
=============================================================
Centralized configuration for all experiments
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TRAIN_IMAGES_DIR = BASE_DIR / "train" / "train_images"
EVAL_IMAGES_DIR = BASE_DIR / "eval" / "eval_images"  # Download from competition
TEST_IMAGES_DIR = BASE_DIR / "test" / "test_images"  # Download from competition

# Output directories
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, SUBMISSION_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TASK CONFIGURATION
# =============================================================================
TASK = "subtask_a"  # hate speech detection
NUM_CLASSES = 2  # 0: Non-Hate, 1: Hate
CLASS_NAMES = ["Non-Hate", "Hate"]

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
TRAIN_LABELS_FILE = BASE_DIR / "index_label_train.csv"
TRAIN_CSV_FILE = BASE_DIR / "train.csv"  # Contains index, label, text
OCR_CSV_FILE = BASE_DIR / "OCR_Dataset_Image.csv"

# Image configuration
IMAGE_SIZE = 224  # For most vision models
MAX_TEXT_LENGTH = 512  # For text models

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Text Models (for OCR text)
# IMPORTANT: For Nepali, use MuRIL or NepBERTa for best results!
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
# MuRIL is recommended as it explicitly includes Nepali in training
DEFAULT_TEXT_MODEL = "muril_large"

# Vision Models
VISION_MODEL_OPTIONS = {
    "resnet50": "resnet50",
    "vit_base": "google/vit-base-patch16-224",
    "swin_base": "microsoft/swin-base-patch4-window7-224",
    "efficientnet": "efficientnet_b0",
}
DEFAULT_VISION_MODEL = "resnet50"

# Multimodal Models
MULTIMODAL_MODEL_OPTIONS = {
    "clip": "openai/clip-vit-base-patch32",
    "blip": "Salesforce/blip-image-captioning-base",
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
SEED = 42
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# Cross-validation
NUM_FOLDS = 5
STRATIFIED = True

# Early stopping
PATIENCE = 3
MIN_DELTA = 0.001

# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
AUGMENTATION_CONFIG = {
    "horizontal_flip_prob": 0.5,
    "rotation_limit": 15,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "use_mixup": True,
    "mixup_alpha": 0.2,
}

# =============================================================================
# OCR CONFIGURATION
# =============================================================================
OCR_LANGUAGES = ['ne', 'en']  # Nepali and English
OCR_GPU = True  # Use GPU for OCR if available

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
EVAL_METRIC = "f1_macro"  # Primary metric for ranking
EVAL_METRICS = ["accuracy", "f1_macro", "precision", "recall", "confusion_matrix"]

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
NUM_WORKERS = 4 if torch.cuda.is_available() else 2

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Task: {TASK}")
print(f"[CONFIG] Num Classes: {NUM_CLASSES}")
