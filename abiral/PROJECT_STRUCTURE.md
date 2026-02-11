# Project Structure

## Overview
This is a complete, modular PyTorch framework for multimodal classification of memes with support for hate speech detection and sentiment analysis.

## Directory Structure

```
multimodal-classification/
│
├── config.py                    # Configuration file (all hyperparameters)
├── dataset.py                   # Dataset and dataloader utilities
├── encoders.py                  # Image (CLIP) and text (BGE-M3/E5) encoders
├── fusion_models.py             # All fusion strategies
├── model.py                     # Complete model wrapper
├── trainer.py                   # Training loop with K-fold CV
├── inference.py                 # Inference engine for predictions
├── train.py                     # Main training script
├── test.py                      # Testing/inference script
├── experiments.py               # Pre-configured experiments
├── experiment_configs.py        # All possible experiment configurations
├── quickstart.py                # Quick start example
├── utils.py                     # Analysis and visualization utilities
├── requirements.txt             # Python dependencies
├── README.md                    # Comprehensive documentation
└── PROJECT_STRUCTURE.md         # This file
```

## File Descriptions

### Core Files

#### `config.py`
Central configuration file containing all hyperparameters and settings:
- Data paths (image folder, CSV)
- Modality selection (image/text/both)
- Fusion strategy selection
- Model architecture parameters
- Training hyperparameters
- Optimizer and scheduler settings
- Device configuration

**Key Features:**
- Hot-pluggable configuration
- `update()` method for easy modification
- `print_config()` for debugging

#### `dataset.py`
Dataset and dataloader implementation:
- `MultimodalDataset`: Main dataset class
- `create_dataloaders()`: K-fold cross-validation setup
- `get_class_distribution()`: Class balance analysis

**Supports:**
- Image-only, text-only, or multimodal data
- Original or text-removed images
- Automatic K-fold splitting

#### `encoders.py`
Pre-trained encoder implementations:
- `ImageEncoder`: CLIP vision encoder
- `TextEncoder`: BGE-M3 or E5-Large encoder
- `ImageTransform`: Image preprocessing
- `get_text_tokenizer()`: Text tokenization

**Features:**
- Frozen or fine-tunable encoders
- Automatic feature dimension detection

#### `fusion_models.py`
All fusion strategy implementations:
- `EarlyFusionModel`: Concatenation-based fusion
- `LateFusionVotingModel`: Separate classifiers with voting
- `LateFusionBaggingModel`: Ensemble of classifiers
- `HybridFusionModel`: **[PRIORITY]** Dynamic attention-based fusion
- `UnimodalModel`: For single modality experiments

**Hybrid Fusion Features:**
- Multi-head cross-attention
- Dynamic gate network for modality weighting
- Learns importance of each modality adaptively

#### `model.py`
Complete model wrapper:
- `MultimodalClassifier`: Combines encoders and fusion model
- Handles different modality configurations
- Provides unified interface for training and inference

#### `trainer.py`
Training utilities:
- `train_epoch()`: Single epoch training
- `validate()`: Validation with metrics
- `train_fold()`: Complete fold training with early stopping
- `EarlyStopping`: Early stopping implementation
- Optimizer and scheduler creation
- Training visualization

**Features:**
- K-fold cross-validation
- Early stopping
- Learning rate scheduling
- Comprehensive metrics (accuracy, F1, precision, recall)
- Automatic plotting of training curves and confusion matrices

#### `inference.py`
Inference engine for predictions:
- `InferenceEngine`: Main inference class
- `load_model_for_inference()`: Load trained models
- Single image and batch prediction
- CSV-based prediction

**Usage:**
```python
model, transform, tokenizer = load_model_for_inference(model_path, config)
engine = InferenceEngine(model, config, transform, tokenizer)
prediction, probs, info = engine.predict_single_image(image_path, text)
```

### Scripts

#### `train.py`
Main training script:
- Orchestrates complete training pipeline
- K-fold cross-validation
- Result aggregation and saving
- Configuration management

**Run:**
```bash
python train.py
```

#### `test.py`
Testing and inference script:
- Command-line interface for predictions
- Single image or batch mode
- Configurable modality and fusion strategy

**Examples:**
```bash
# Single image
python test.py --model_path model.pt --mode single --image_path img.jpg

# Batch from CSV
python test.py --model_path model.pt --mode batch --csv_path test.csv
```

#### `experiments.py`
Pre-configured ablation experiments:
- Image only
- Text only
- Early/Late/Hybrid fusion
- Text-removed images
- Complete ablation study

**Run:**
```bash
python experiments.py hybrid      # Run hybrid fusion
python experiments.py all         # Run all experiments
```

#### `experiment_configs.py`
Comprehensive experiment configurations:
- Binary classification examples
- Multi-class classification examples
- All fusion strategies
- Hyperparameter tuning examples
- Fine-tuning configurations
- CPU-friendly configurations

**Run:**
```bash
python experiment_configs.py binary_hate
python experiment_configs.py sentiment
python experiment_configs.py finetune
```

#### `quickstart.py`
Simplest way to get started:
- Minimal configuration required
- Uses hybrid fusion (priority)
- Step-by-step instructions

**Run:**
```bash
python quickstart.py
```

#### `utils.py`
Analysis and visualization utilities:
- `load_experiment_results()`: Load experiment outputs
- `compare_experiments()`: Compare multiple experiments
- `plot_experiment_comparison()`: Visualization
- `analyze_fold_variance()`: Fold stability analysis
- `find_best_experiment()`: Find best model

**Usage:**
```bash
python utils.py find_best
python utils.py analyze outputs/experiment_name/
python utils.py compare outputs/exp1/ outputs/exp2/
```

## Output Structure

After training, outputs are saved in:

```
outputs/
└── <experiment_name>/
    ├── config.json                       # Saved configuration
    ├── cross_validation_summary.json     # Aggregated results
    ├── best_model_fold_0.pt             # Best model (fold 0)
    ├── best_model_fold_1.pt             # Best model (fold 1)
    ├── ...
    ├── fold_0_results.json              # Fold 0 detailed results
    ├── fold_1_results.json              # Fold 1 detailed results
    ├── ...
    ├── training_history_fold_0.png      # Training curves (fold 0)
    ├── training_history_fold_1.png      # Training curves (fold 1)
    ├── ...
    ├── confusion_matrix_fold_0.png      # Confusion matrix (fold 0)
    └── confusion_matrix_fold_1.png      # Confusion matrix (fold 1)
```

## Workflow

### 1. Basic Training

```python
from config import Config
from train import main

config = Config()
config.update(
    IMAGE_FOLDER="data/images",
    CSV_PATH="data/train.csv",
    USE_MODALITY='both',
    FUSION_STRATEGY='hybrid'
)

summary, save_dir = main(config)
```

### 2. Inference

```python
from inference import load_model_for_inference, InferenceEngine

model, transform, tokenizer = load_model_for_inference(model_path, config)
engine = InferenceEngine(model, config, transform, tokenizer)
prediction, probs, info = engine.predict_single_image(image_path, text)
```

### 3. Analysis

```python
from utils import compare_experiments, find_best_experiment

# Find best model
best_exp, results = find_best_experiment('outputs/')

# Compare experiments
df = compare_experiments([
    'outputs/exp1/',
    'outputs/exp2/',
    'outputs/exp3/'
], names=['Image Only', 'Text Only', 'Hybrid Fusion'])
```

## Key Design Decisions

### 1. Modularity
- Each component (encoder, fusion, training) is independent
- Easy to swap implementations
- Configuration-driven design

### 2. Flexibility
- Support for any combination of modalities
- Multiple fusion strategies
- Hot-pluggable hyperparameters

### 3. Robustness
- K-fold cross-validation (no separate test set needed)
- Early stopping to prevent overfitting
- Class weighting for imbalanced data

### 4. Reproducibility
- Fixed random seeds
- Configuration saving
- Complete result logging

### 5. Ease of Use
- Simple configuration system
- Pre-configured experiments
- Comprehensive documentation

## Extension Points

### Adding New Fusion Strategy

1. Create new fusion model in `fusion_models.py`:
```python
class MyFusionModel(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes, ...):
        # Your implementation
        pass
    
    def forward(self, image_features, text_features):
        # Your implementation
        return logits
```

2. Update `create_fusion_model()` in `fusion_models.py`:
```python
elif config.FUSION_STRATEGY == 'my_fusion':
    return MyFusionModel(...)
```

3. Add to config options in `config.py`

### Adding New Encoder

1. Create encoder class in `encoders.py`:
```python
class MyEncoder(nn.Module):
    def __init__(self, ...):
        # Your implementation
        pass
    
    def forward(self, inputs):
        # Your implementation
        return features
```

2. Update `create_encoders()` in `encoders.py`

### Adding New Metrics

1. Add metric computation in `trainer.py` `validate()`:
```python
my_metric = compute_my_metric(all_labels, all_preds)
metrics['my_metric'] = my_metric
```

2. Update history tracking in `train_fold()`

## Best Practices

1. **Always start with frozen encoders** for faster experimentation
2. **Use hybrid fusion** as the default (best performance)
3. **Run full K-fold CV** for reliable evaluation
4. **Check class distribution** before training
5. **Monitor training curves** for overfitting
6. **Compare multiple fusion strategies** in ablation studies
7. **Save configuration** for reproducibility

## Troubleshooting

See README.md for common issues and solutions.

## Citation

If you use this framework, please cite:
```bibtex
@software{multimodal_classification,
  title={Multimodal Classification Framework},
  author={Your Name},
  year={2024}
}
```
