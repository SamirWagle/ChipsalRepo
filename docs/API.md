# API Documentation

## Table of Contents

- [Core Models](#core-models)
- [Data Loading](#data-loading)
- [Training](#training)
- [Inference](#inference)
- [Utilities](#utilities)
- [Configuration](#configuration)

---

## Core Models

### `abiral.model.MultimodalClassifier`

Main classification model supporting multiple fusion strategies.

```python
from abiral.model import MultimodalClassifier

model = MultimodalClassifier(
    model_type='hybrid',           # 'text_only', 'image_only', 'early', 'late', 'hybrid'
    num_classes=2,                 # Binary classification
    hidden_dim=256,                # Hidden dimension for fusion
    dropout=0.5,                   # Dropout rate
    attention_heads=4,             # Number of attention heads (hybrid only)
    freeze_encoders=False,         # Whether to freeze CLIP/BGE-M3
    unfreeze_clip_layers=2,        # Fine-tune last N CLIP layers
    unfreeze_bge_layers=4,         # Fine-tune last N BGE layers
)
```

**Parameters:**
- `model_type` (str): Architecture type
  - `'text_only'`: BGE-M3 text encoder only
  - `'image_only'`: CLIP visual encoder only
  - `'early'`: Concatenation fusion
  - `'late'`: Ensemble voting/bagging
  - `'hybrid'`: Cross-modal attention + dynamic gating (recommended)
- `num_classes` (int): Number of output classes (default: 2)
- `hidden_dim` (int): Dimension of fusion layers (default: 256)
- `dropout` (float): Dropout probability (default: 0.5)
- `attention_heads` (int): Multi-head attention heads for hybrid (default: 4)
- `freeze_encoders` (bool): Freeze all encoder weights (default: False)
- `unfreeze_clip_layers` (int): Number of CLIP layers to fine-tune (default: 2)
- `unfreeze_bge_layers` (int): Number of BGE layers to fine-tune (default: 4)

**Methods:**

#### `forward(images, input_ids, attention_mask)`

Forward pass through the model.

```python
outputs = model(
    images=batch['image'],              # Tensor[batch_size, 3, 224, 224]
    input_ids=batch['input_ids'],       # Tensor[batch_size, max_len]
    attention_mask=batch['attention_mask']  # Tensor[batch_size, max_len]
)
# Returns: logits Tensor[batch_size, num_classes]
```

#### `predict(images, input_ids, attention_mask)`

Generate predictions with confidence scores.

```python
predictions = model.predict(
    images=batch['image'],
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask']
)
# Returns: dict with 'labels', 'probabilities', 'confidence'
```

---

## Data Loading

### `abiral.dataset.MultimodalDataset`

PyTorch Dataset for loading meme images and text.

```python
from abiral.dataset import MultimodalDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

dataset = MultimodalDataset(
    csv_file="train/train.csv",
    image_dir="train/train_images/",
    tokenizer=tokenizer,
    max_length=77,
    transform=None,  # Optional image transforms
    text_column='text',
    label_column='label',
    image_column='index'
)
```

**Parameters:**
- `csv_file` (str): Path to CSV with labels and text
- `image_dir` (str): Directory containing images
- `tokenizer`: Hugging Face tokenizer (BGE-M3 recommended)
- `max_length` (int): Maximum text sequence length (default: 77)
- `transform` (callable): Optional image augmentations
- `text_column` (str): CSV column name for text (default: 'text')
- `label_column` (str): CSV column name for labels (default: 'label')
- `image_column` (str): CSV column name for image filenames (default: 'index')

**Returns:**
Dictionary with:
- `image`: Tensor[3, 224, 224]
- `input_ids`: Tensor[max_length]
- `attention_mask`: Tensor[max_length]
- `label`: int (0 or 1)

### Data Augmentation

```python
from albumentations import (
    Compose, RandomBrightnessContrast, HueSaturationValue,
    GaussNoise, MotionBlur, Normalize
)
from albumentations.pytorch import ToTensorV2

train_transform = Compose([
    RandomBrightnessContrast(p=0.5),
    HueSaturationValue(p=0.3),
    GaussNoise(p=0.2),
    MotionBlur(p=0.2),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711]),
    ToTensorV2(),
])
```

---

## Training

### `abiral.trainer.Trainer`

Handles training loop with early stopping, logging, and checkpointing.

```python
from abiral.trainer import Trainer
from abiral.model import MultimodalClassifier
import torch.optim as optim

model = MultimodalClassifier(model_type='hybrid')
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device='cuda',
    num_epochs=30,
    early_stopping_patience=10,
    checkpoint_dir='outputs/m7_hybrid',
    log_interval=10,
    use_amp=True,  # Mixed precision training
)

# Train
trainer.train()

# Load best model
trainer.load_best_model()
```

**Key Features:**
- Automatic mixed precision (AMP) for faster training
- Early stopping with patience
- Best model checkpointing
- Comprehensive logging (loss, metrics, learning rate)
- Learning rate scheduling
- Class-weighted loss for imbalanced data

### Training from Command Line

```bash
python abiral/train.py \
    --experiment_name "m7_hybrid_fusion" \
    --model_type "hybrid" \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 30 \
    --early_stopping_patience 10 \
    --output_dir "outputs/m7" \
    --seed 42
```

---

## Inference

### `abiral.inference.HateDetectionInference`

Simplified inference interface for trained models.

```python
from abiral.inference import HateDetectionInference

# Initialize
detector = HateDetectionInference(
    model_path="outputs/m7_hybrid/best_model.pt",
    device="cuda"
)

# Single prediction
result = detector.predict(
    image_path="test_images/12345.jpg",
    text="यो नेता भ्रष्ट छ"
)

print(result)
# {
#     'label': 1,
#     'confidence': 0.87,
#     'probabilities': [0.13, 0.87],
#     'predicted_class': 'Hate'
# }
```

#### Batch Prediction

```python
results = detector.predict_batch(
    image_paths=["img1.jpg", "img2.jpg"],
    texts=["text 1", "text 2"],
    batch_size=32
)
```

#### Generate Submission File

```python
from abiral.generate_submission import generate_submission

generate_submission(
    model_path="outputs/m7_hybrid/best_model.pt",
    test_csv="test_release/index_text_test.csv",
    test_images="test_release/test_images/",
    output_file="submission.csv"
)
```

---

## Utilities

### Metrics

```python
from abiral.utils import compute_metrics

metrics = compute_metrics(
    y_true=labels,
    y_pred=predictions,
    y_probs=probabilities
)

print(metrics)
# {
#     'accuracy': 0.7015,
#     'f1_macro': 0.6834,
#     'precision': 0.6817,
#     'recall': 0.7023,
#     'confusion_matrix': [[...], [...]],
#     'classification_report': '...'
# }
```

### Class Weights (Imbalanced Data)

```python
from abiral.utils import compute_class_weights

class_weights = compute_class_weights(
    labels=[0, 1, 1, 0, 1, ...],
    gamma=0.3  # Frequency-inverse weighting parameter
)
```

### Visualization

```python
from abiral.utils import plot_confusion_matrix, plot_training_curves

# Confusion matrix
plot_confusion_matrix(
    y_true=labels,
    y_pred=predictions,
    classes=['Non-Hate', 'Hate'],
    save_path='confusion_matrix.png'
)

# Training curves
plot_training_curves(
    train_losses=[...],
    val_losses=[...],
    val_f1_scores=[...],
    save_path='training_curves.png'
)
```

---

## Configuration

### Experiment Configs

All model configurations are defined in `abiral/experiment_configs.py`:

```python
from abiral.experiment_configs import get_config

# Load M7 Hybrid Fusion config
config = get_config('m7_hybrid_fusion')

print(config)
# {
#     'model_type': 'hybrid',
#     'learning_rate': 2e-5,
#     'batch_size': 32,
#     'num_epochs': 30,
#     'dropout': 0.5,
#     'hidden_dim': 256,
#     'attention_heads': 4,
#     ...
# }
```

### Available Configs

- `m1_text_only`: BGE-M3 text-only baseline
- `m2_image_only`: CLIP image-only baseline
- `m3_image_text_removed`: Image-only with text-removed preprocessing
- `m4_early_fusion`: Simple concatenation fusion
- `m5_late_voting`: Soft voting ensemble
- `m6_late_bagging`: Bagging ensemble (k=3)
- `m7_hybrid_fusion`: **Cross-modal attention + gating (Primary System)**
- `m8_hybrid_text_removed`: Hybrid with text-removed images

### Custom Configuration

```python
custom_config = {
    'model_type': 'hybrid',
    'learning_rate': 1e-5,
    'batch_size': 16,
    'num_epochs': 50,
    'dropout': 0.3,
    'hidden_dim': 512,
    'attention_heads': 8,
    'label_smoothing': 0.2,
    'class_weight_gamma': 0.5,
    'unfreeze_clip_layers': 4,
    'unfreeze_bge_layers': 6,
}
```

---

## Advanced Usage

### K-Fold Cross-Validation

```python
from abiral.experiments import run_kfold_experiment

results = run_kfold_experiment(
    config_name='m7_hybrid_fusion',
    k_folds=5,
    stratified=True,
    seed=42,
    output_dir='outputs/cv_results'
)

print(f"Mean F1: {results['mean_f1']:.4f} ± {results['std_f1']:.4f}")
```

### Ensemble Predictions

```python
from abiral.inference import EnsembleInference

ensemble = EnsembleInference(
    model_paths=[
        "outputs/fold_0/best_model.pt",
        "outputs/fold_1/best_model.pt",
        "outputs/fold_2/best_model.pt",
    ],
    voting='soft',  # 'soft' or 'hard'
    device='cuda'
)

result = ensemble.predict(image_path="test.jpg", text="text")
```

### Attention Visualization

```python
from abiral.utils import visualize_attention

attention_maps = visualize_attention(
    model=model,
    image=image_tensor,
    text=text_tensor,
    save_path='attention_viz.png'
)
```

---

## Error Handling

All functions raise descriptive exceptions:

```python
try:
    detector = HateDetectionInference(model_path="invalid_path.pt")
except FileNotFoundError as e:
    print(f"Model not found: {e}")

try:
    result = detector.predict(image_path="invalid.jpg", text="text")
except ValueError as e:
    print(f"Invalid input: {e}")
```

---

## Performance Tips

1. **Use Mixed Precision**: Set `use_amp=True` for 2x speedup
2. **Batch Inference**: Always use `predict_batch()` for multiple samples
3. **GPU Memory**: Reduce `batch_size` if CUDA out-of-memory errors occur
4. **Caching**: Pre-tokenize texts and cache to disk for faster data loading
5. **Multi-GPU**: Use `torch.nn.DataParallel` for multiple GPUs

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## Testing

Run unit tests:

```bash
pytest tests/ -v
pytest tests/test_model.py -v
pytest tests/test_inference.py::test_single_prediction -v
```

---

For more examples, see:
- [Full Paper](../PaperPortion/FINAL_paper_report.md)
- [Experiment Guide](EXPERIMENTS.md)
- [Troubleshooting](TROUBLESHOOTING.md)
