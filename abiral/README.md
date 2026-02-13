# Multimodal Hate Speech and Sentiment Classification

A modular and dynamic PyTorch framework for binary (hate/non-hate) and multi-class (positive/negative/neutral) classification of meme images using multimodal fusion strategies.

## Features

- **Modular Architecture**: Hot-pluggable components for different modalities, fusion strategies, and hyperparameters
- **Multiple Fusion Strategies**:
  - Early Fusion (Concatenation)
  - Late Fusion (Voting and Bagging)
  - Hybrid Fusion (Dynamic attention-based weighting) - **PRIORITY**
- **Flexible Modalities**:
  - Image only (original or text-removed)
  - Text only (extracted from images)
  - Multimodal (image + text)
- **Pre-trained Encoders**:
  - CLIP for image encoding
  - BGE-M3 or E5-Large for text encoding
  - Frozen or fine-tunable encoders
- **K-Fold Cross-Validation**: Robust evaluation without separate test set
- **Comprehensive Metrics**: Accuracy, F1, Precision, Recall, Confusion Matrix
- **Ablation Study Support**: Easy experimentation with different configurations

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd multimodal-classification

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### CSV Structure
Your CSV file should have the following columns:
- `Image`: Image filename (e.g., `meme_001.jpg`)
- `Text`: Extracted text from the image
- `Label`: Class label (0, 1 for binary; 0, 1, 2 for 3-class)

Example:
```csv
Image,Text,Label
meme_001.jpg,This is sample text,0
meme_002.jpg,Another meme text,1
```

### Image Folder
Place all images in a single folder. For text-removed images, append `_text_removed` to the filename:
- Original: `meme_001.jpg`
- Text-removed: `meme_001_text_removed.jpg`

## Configuration

All hyperparameters and settings are in `config.py`. Key configurations:

```python
# Data paths
IMAGE_FOLDER = "path/to/images"
CSV_PATH = "path/to/data.csv"

# Modality selection
USE_MODALITY = 'both'  # 'image', 'text', or 'both'
IMAGE_TYPE = 'original'  # 'original' or 'text_removed'

# Fusion strategy (for multimodal)
FUSION_STRATEGY = 'hybrid'  # 'early', 'late_voting', 'late_bagging', 'hybrid'

# Model settings
FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
HIDDEN_DIM = 256
DROPOUT = 0.3

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
K_FOLDS = 5
```

## Usage

### 1. Training

#### Quick Start (Hybrid Fusion - Priority)
```python
from config import Config
from train import main

config = Config()
config.update(
    IMAGE_FOLDER="data/images",
    CSV_PATH="data/train.csv",
    USE_MODALITY='both',
    FUSION_STRATEGY='hybrid',
    NUM_EPOCHS=50
)

summary, save_dir = main(config)
```

#### Command Line Training
```bash
python train.py
```

### 2. Running Experiments

#### Single Experiment
```bash
# Hybrid fusion (priority)
python experiments.py hybrid

# Image only
python experiments.py image_only

# Text only
python experiments.py text_only

# Early fusion
python experiments.py early

# Late fusion (voting)
python experiments.py late_voting

# Late fusion (bagging)
python experiments.py late_bagging
```

#### Full Ablation Study
```bash
# Run all experiments
python experiments.py all
```

### 3. Generate Submission CSV (Quick & Easy)

#### **Step 1: Check Your Model Configuration** (Optional but Recommended)
```bash
python test.py --mode check --model_path outputs/both_hybrid_original_20240215_143022
```
This shows you exactly what modality/fusion was used during training.

#### **Step 2: Generate Submission**

**Recommended: Simple Script**
```bash
python generate_submission.py \
    --model outputs/best_model_fold_0.pt \
    --test_images test_images \
    --test_csv test.csv \
    --output submission.csv
```

**Match your training configuration:**

| Model Trained | Command |
|--------------|---------|
| **Image-only** | `--modality image` |
| **Text-only** | `--modality text` |
| **Hybrid fusion** | `--modality both --fusion hybrid` |
| **Early fusion** | `--modality both --fusion early` |
| **Late voting** | `--modality both --fusion late_voting` |
| **Late bagging** | `--modality both --fusion late_bagging` |

**Examples:**

Image-only model:
```bash
python generate_submission.py \
    --model outputs/best_model_fold_0.pt \
    --test_images test_images \
    --test_csv test.csv \
    --output submission.csv \
    --modality image
```

Text-only model:
```bash
python generate_submission.py \
    --model outputs/best_model_fold_0.pt \
    --test_images test_images \
    --test_csv test.csv \
    --output submission.csv \
    --modality text
```

Hybrid fusion (multimodal):
```bash
python generate_submission.py \
    --model outputs/best_model_fold_0.pt \
    --test_images test_images \
    --test_csv test.csv \
    --output submission.csv \
    --modality both \
    --fusion hybrid
```

#### Alternative: Using test.py
```bash
python test.py \
    --mode submission \
    --model_path outputs/best_model_fold_0.pt \
    --image_folder test_images \
    --csv_path test.csv \
    --output_path submission.csv
```

#### Single Image Prediction
```bash
python test.py \
    --mode single \
    --model_path outputs/best_model_fold_0.pt \
    --image_path test_image.jpg \
    --text "Extracted text from image"
```

#### Python API for Inference
```python
from test import get_prediction_csv
from config import Config

# Generate submission CSV
config = Config()
config.update(USE_MODALITY='both', FUSION_STRATEGY='hybrid')

submission_df = get_prediction_csv(
    test_image_folder='test_images',
    test_csv_path='test.csv',
    model_path='outputs/best_model_fold_0.pt',
    output_path='submission.csv',
    config=config
)
```

**Output Format:**
```csv
index,label
0,1
1,0
2,1
```
Ready to submit!

## Ablation Study Experiments

The framework supports the following ablation experiments:

1. **Image Only (Original)**: Baseline using only original images
2. **Text Only**: Baseline using only extracted text
3. **Image + Text (Early Fusion)**: Concatenate features before classification
4. **Image + Text (Late Fusion - Voting)**: Separate classifiers with voting
5. **Image + Text (Late Fusion - Bagging)**: Ensemble of classifiers
6. **Image + Text (Hybrid Fusion)**: Dynamic attention-based fusion ⭐ **PRIORITY**
7. **Text-Removed Image Only**: Using images with text removed
8. **Text-Removed Image + Text**: Multimodal with text-removed images

## Model Architecture

### Hybrid Fusion Model (Priority)

```
Image → CLIP Encoder → Image Features (512D)
                                           ↓
                                    [Projection to 256D]
                                           ↓
                                    Multi-Head Attention
                                           ↓
                                    Gate Network (Dynamic Weighting)
                                           ↓
                                    Fusion Layer → Classifier
                                           ↑
                                    [Projection to 256D]
                                           ↑
Text → BGE-M3 Encoder → Text Features (1024D)
```

Key components:
- **Cross-modal attention**: Allows features to attend to each other
- **Dynamic gating**: Learns importance weights for each modality
- **Residual connections**: Preserves modality-specific information

## Outputs

Each experiment creates a directory in `outputs/` with:

```
outputs/
└── both_hybrid_original_20240215_143022/
    ├── config.json                          # Experiment configuration
    ├── cross_validation_summary.json        # Aggregated results
    ├── best_model_fold_0.pt                # Best model weights (each fold)
    ├── best_model_fold_1.pt
    ├── ...
    ├── fold_0_results.json                 # Detailed fold results
    ├── fold_1_results.json
    ├── ...
    ├── training_history_fold_0.png         # Training curves
    ├── training_history_fold_1.png
    ├── ...
    ├── confusion_matrix_fold_0.png         # Confusion matrices
    └── confusion_matrix_fold_1.png
```

## Hyperparameter Tuning

All hyperparameters are hot-pluggable through the config:

```python
config.update(
    LEARNING_RATE=5e-5,           # Adjust learning rate
    DROPOUT=0.4,                  # Increase dropout for regularization
    WEIGHT_DECAY=1e-4,           # L2 regularization
    HIDDEN_DIM=512,              # Increase model capacity
    BATCH_SIZE=32,               # Larger batch size
    OPTIMIZER='adamw',           # Change optimizer
    USE_SCHEDULER=True,          # Enable LR scheduling
    SCHEDULER_TYPE='cosine',     # Cosine annealing
    PATIENCE=15,                 # Early stopping patience
)
```

## Performance Tips

1. **GPU Memory**: Reduce `BATCH_SIZE` if running out of memory
2. **Training Speed**: Set `FREEZE_IMAGE_ENCODER=True` and `FREEZE_TEXT_ENCODER=True`
3. **Overfitting**: Increase `DROPOUT` and `WEIGHT_DECAY`
4. **Underfitting**: Increase `HIDDEN_DIM` or set `FREEZE_*_ENCODER=False`
5. **Class Imbalance**: Automatically handled with class weights

## Evaluation Metrics

The framework computes:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **Confusion Matrix**: Visual representation of predictions

For binary classification: standard binary metrics
For multi-class: macro-averaged metrics (equal weight to all classes)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_classification,
  title={Multimodal Classification Framework for Hate Speech and Sentiment Analysis},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```python
config.update(BATCH_SIZE=8)  # Reduce batch size
```

**Issue**: Models not downloading
```python
# Pre-download models
from transformers import CLIPModel, AutoModel
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
AutoModel.from_pretrained('BAAI/bge-m3')
```

**Issue**: Slow training
```python
config.update(
    FREEZE_IMAGE_ENCODER=True,  # Freeze encoders
    FREEZE_TEXT_ENCODER=True
)
```

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com]
