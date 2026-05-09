# Experiment Guide

Complete guide to reproducing all results from our CHiPSAL 2026 paper.

## Table of Contents

- [Quick Start](#quick-start)
- [Full Ablation Study](#full-ablation-study)
- [Individual Models](#individual-models)
- [K-Fold Cross-Validation](#k-fold-cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Error Analysis](#error-analysis)
- [Visualization](#visualization)

---

## Quick Start

### Train Primary System (M7)

```bash
python abiral/train.py \
    --experiment_name "m7_hybrid_fusion" \
    --model_type "hybrid" \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 30 \
    --early_stopping_patience 10 \
    --seed 42
```

**Expected Results:**
- F1-Macro: ~0.6834
- Accuracy: ~0.7015
- Training time: ~52 minutes (RTX 3090)

---

## Full Ablation Study

Reproduce Table 4 from the paper (all 8 models with 5-fold CV).

### Run All Experiments

```bash
python abiral/experiments.py \
    --experiment_suite "full_ablation" \
    --k_folds 5 \
    --output_dir "outputs/paper_results" \
    --seed 42
```

This will train all 8 models:

| Model | Expected F1 | Training Time | GPU Memory |
|-------|-------------|---------------|------------|
| M1: Text-Only | 0.6455 | ~90 min | ~8 GB |
| M2: Image-Only | 0.5952 | ~60 min | ~6 GB |
| M3: Image (Text-Removed) | 0.5535 | ~65 min | ~6 GB |
| M4: Early Fusion | 0.6140 | ~140 min | ~12 GB |
| M5: Late Voting | 0.6511 | ~150 min | ~14 GB |
| M6: Late Bagging | 0.6267 | ~160 min | ~14 GB |
| M7: Hybrid Fusion | **0.6834** | ~260 min | ~18 GB |
| M8: Hybrid (Text-Removed) | 0.6436 | ~265 min | ~18 GB |

**Total Time:** ~20 hours (single RTX 3090)

### Parallel Execution

If you have multiple GPUs, train models in parallel:

```bash
# GPU 0: Text-based models
CUDA_VISIBLE_DEVICES=0 python abiral/train.py --model_type "text_only" &

# GPU 1: Image-based models
CUDA_VISIBLE_DEVICES=1 python abiral/train.py --model_type "image_only" &

# GPU 2: Fusion models
CUDA_VISIBLE_DEVICES=2 python abiral/train.py --model_type "hybrid" &
```

---

## Individual Models

### M1: Text-Only Baseline (BGE-M3)

```bash
python abiral/train.py \
    --config "configs/m1_text_only.yaml" \
    --k_folds 5 \
    --seed 42
```

**Configuration:**
```yaml
model_type: text_only
encoder: "BAAI/bge-m3"
learning_rate: 3e-5
batch_size: 64
num_epochs: 25
dropout: 0.3
unfreeze_layers: 4
optimizer: AdamW
weight_decay: 1e-2
label_smoothing: 0.1
class_weight_gamma: 0.3
```

**Key Findings:**
- Second-best overall performance (F1: 0.6455)
- Highest accuracy (0.7239) but misleading in imbalanced setting
- Demonstrates text modality dominance in Nepali memes
- 16.6% better than image-only, revealing CLIP's Devanagari limitations

---

### M2: Image-Only Baseline (CLIP ViT-B/32)

```bash
python abiral/train.py \
    --config "configs/m2_image_only.yaml" \
    --k_folds 5 \
    --seed 42
```

**Configuration:**
```yaml
model_type: image_only
encoder: "openai/clip-vit-base-patch32"
learning_rate: 1e-5
batch_size: 48
num_epochs: 30
dropout: 0.4
unfreeze_layers: 2
optimizer: AdamW
weight_decay: 5e-3
```

**Key Findings:**
- F1: 0.5952 (7th place)
- Struggles with Devanagari script (CLIP not trained on Indic text)
- Better than text-removed baseline (M3), showing typography value
- Visual-only signals insufficient for hate detection

---

### M3: Image-Only with Text-Removed

Tests if CLIP can detect hate from pure visual cues (no text overlay).

```bash
python abiral/train.py \
    --config "configs/m3_image_text_removed.yaml" \
    --k_folds 5 \
    --seed 42 \
    --preprocess_remove_text
```

**Text Removal Pipeline:**
```python
from abiral.utils import remove_text_from_image

# Uses EAST text detection + inpainting
image_no_text = remove_text_from_image(
    image_path="train_images/12345.jpg",
    method="inpainting",  # 'blur', 'inpainting', 'mask'
    save_path="train_images_no_text/12345.jpg"
)
```

**Findings:**
- Worst performance (F1: 0.5535)
- 5.8% drop from M2 → M3 quantifies typography information value
- Pure visual hate (symbols, gestures) is rare in dataset

---

### M4: Early Fusion (Concatenation)

```bash
python abiral/train.py \
    --config "configs/m4_early_fusion.yaml" \
    --k_folds 5 \
    --seed 42
```

**Architecture:**
```python
# Simple concatenation: [v; t] ∈ R^(512+1024)
v = clip_encoder(image)        # R^512
t = bge_encoder(text)          # R^1024
fused = torch.cat([v, t], dim=1)  # R^1536
logits = mlp(fused)            # R^2
```

**Configuration:**
```yaml
model_type: early_fusion
fusion_method: concatenation
hidden_dims: [1536, 768, 256, 2]
dropout: 0.5
learning_rate: 2e-5
batch_size: 32
```

**Findings:**
- F1: 0.6140 (5th place)
- 11.3% worse than hybrid fusion (M7)
- Simple concatenation lacks cross-modal interaction
- No adaptive modality weighting

---

### M5: Late Fusion (Soft Voting)

```bash
python abiral/train.py \
    --config "configs/m5_late_voting.yaml" \
    --k_folds 5 \
    --seed 42
```

**Ensemble Strategy:**
```python
# Train separate text and image models
text_model = train_text_only(...)   # M1
image_model = train_image_only(...) # M2

# Soft voting at inference
p_text = text_model(text)
p_image = image_model(image)
p_final = (p_text + p_image) / 2
```

**Findings:**
- F1: 0.6511 (4th place)
- Better than early fusion but worse than hybrid
- Equal modality weighting (50-50) suboptimal
- No sample-specific adaptation

---

### M6: Late Fusion (Bagging)

```bash
python abiral/train.py \
    --config "configs/m6_late_bagging.yaml" \
    --k_folds 5 \
    --k_bagging 3 \
    --seed 42
```

**Bagging Configuration:**
```python
# Train k=3 models on resampled data
for i in range(3):
    train_subset = resample(train_data, n_samples=len(train_data))
    model_i = train_model(train_subset)
    models.append(model_i)

# Average predictions
predictions = torch.stack([model(x) for model in models]).mean(dim=0)
```

**Findings:**
- F1: 0.6267 (6th place)
- **Worse than single best model (M7)**
- Bootstrap sampling ineffective with N=854 << k²=36
- Ensemble diversity insufficient in small-data regime
- Confirms theoretical limit: bagging needs N >> k²

---

### M7: Hybrid Fusion ⭐ (Primary System)

**Best performing model** — Cross-modal attention + dynamic gating.

```bash
python abiral/train.py \
    --config "configs/m7_hybrid_fusion.yaml" \
    --k_folds 5 \
    --seed 42
```

**Full Configuration:**
```yaml
model_type: hybrid
encoder_vision: "openai/clip-vit-base-patch32"
encoder_text: "BAAI/bge-m3"

# Architecture
hidden_dim: 256
attention_heads: 4
dropout: 0.5
fusion_strategy: "attention_gating"

# Fine-tuning
unfreeze_clip_layers: 2
unfreeze_bge_layers: 4

# Training
learning_rate: 2e-5
batch_size: 32
num_epochs: 30
optimizer: AdamW
weight_decay: 1e-2

# Loss
label_smoothing: 0.1
class_weight_gamma: 0.3

# Regularization
early_stopping_patience: 10
scheduler: ReduceLROnPlateau
scheduler_patience: 5
scheduler_factor: 0.5

# Augmentation
image_aug_prob: 0.3
text_aug_prob: 0.1
```

**Architecture Details:**

```python
# 1. Project encoders to common dimension
v' = Linear(v, 256)  # CLIP 512 → 256
t' = Linear(t, 256)  # BGE 1024 → 256

# 2. Stack as sequence
S = stack([v', t'])  # [2, 256]

# 3. 4-head self-attention
Q, K, V = WqS, WkS, WvS
Attn = softmax(QK^T / sqrt(d_k)) × V

# 4. Extract attended features
v̂ = Attn[0, :]
t̂ = Attn[1, :]

# 5. Dynamic gating
g = softmax(Wg · [v̂; t̂] + bg)  # g ∈ R^2, sum=1

# 6. Gate features
v_gated = g[0] · v̂
t_gated = g[1] · t̂

# 7. Classify
f = [v_gated; t_gated]  # R^512
logits = MLP(f)          # 512→256→128→2
```

**Performance Gains:**
- **vs. Text-Only (M1):** +5.9% F1 (9.1% relative)
- **vs. Early Fusion (M4):** +11.3% F1 (18.4% relative)
- **vs. Late Voting (M5):** +5.0% F1 (7.7% relative)
- **vs. Image-Only (M2):** +23.7% F1 (40% relative)

**Gating Statistics:**
```
Modality Weight Distribution (5-fold mean):
- Text-dominant (g_text > 0.7):    68% of samples
- Balanced (0.4 ≤ g_text ≤ 0.7):   24% of samples
- Image-dominant (g_text < 0.4):    8% of samples

Mean gate values: g_text = 0.73, g_image = 0.27
```

---

### M8: Hybrid Fusion (Text-Removed Images)

Tests hybrid architecture with pure visual inputs.

```bash
python abiral/train.py \
    --config "configs/m8_hybrid_text_removed.yaml" \
    --k_folds 5 \
    --seed 42 \
    --preprocess_remove_text
```

**Findings:**
- F1: 0.6436 (3rd place)
- 5.8% worse than M7 (0.6834 → 0.6436)
- Quantifies **typography information contribution**
- Still better than M1 text-only due to visual context
- Gating shifts: g_image increases from 0.27 → 0.48

---

## K-Fold Cross-Validation

### Stratified K-Fold

Ensures balanced class distribution across folds.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Train model on train_idx
    # Evaluate on val_idx
```

### Variance Analysis

```python
from abiral.experiments import analyze_cv_variance

results = analyze_cv_variance(
    model_config='m7_hybrid_fusion',
    k_folds=5,
    n_runs=3  # Multiple random seeds
)

print(f"Inter-fold variance: {results['fold_variance']:.4f}")
print(f"Inter-run variance: {results['run_variance']:.4f}")
```

**Paper Results:**

| Model | Mean F1 | Std Dev | Min | Max |
|-------|---------|---------|-----|-----|
| M7 | 0.6834 | ±0.018 | 0.665 | 0.702 |
| M1 | 0.6455 | ±0.022 | 0.620 | 0.671 |
| M5 | 0.6511 | ±0.019 | 0.628 | 0.675 |

---

## Hyperparameter Tuning

### Grid Search

```python
from abiral.experiments import grid_search

param_grid = {
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'dropout': [0.3, 0.5, 0.7],
    'hidden_dim': [128, 256, 512],
    'attention_heads': [2, 4, 8],
}

best_config = grid_search(
    model_type='hybrid',
    param_grid=param_grid,
    k_folds=5,
    metric='f1_macro'
)
```

### Bayesian Optimization

```python
from abiral.experiments import bayesian_optimization
from skopt import BayesSearchCV

search_space = {
    'learning_rate': (1e-6, 1e-4, 'log-uniform'),
    'dropout': (0.1, 0.7, 'uniform'),
    'hidden_dim': (128, 512),
    'attention_heads': (2, 8),
}

best_params = bayesian_optimization(
    model_type='hybrid',
    search_space=search_space,
    n_iter=50,
    k_folds=5
)
```

### Random Search

Faster alternative to grid search.

```bash
python abiral/experiments.py \
    --mode "random_search" \
    --n_trials 100 \
    --k_folds 3 \
    --config_base "m7_hybrid_fusion"
```

---

## Error Analysis

### Confusion Matrix Analysis

```python
from abiral.utils import detailed_error_analysis

analysis = detailed_error_analysis(
    model_path="outputs/m7_hybrid/best_model.pt",
    val_loader=val_loader
)

print(f"False Positives: {analysis['fp_count']}")
print(f"False Negatives: {analysis['fn_count']}")
print(f"FP Rate: {analysis['fpr']:.3f}")
print(f"FN Rate: {analysis['fnr']:.3f}")
```

**Paper Findings:**

| Error Type | Count | % | Dominant Patterns |
|------------|-------|---|-------------------|
| False Positives (FP) | 37 | 17.3% | Satirical memes (52%), political criticism (28%) |
| False Negatives (FN) | 26 | 12.2% | Coded language (56%), implicit hate (31%) |

### Per-Class Performance

```python
from sklearn.metrics import classification_report

report = classification_report(
    y_true=val_labels,
    y_pred=predictions,
    target_names=['Non-Hate', 'Hate'],
    digits=4
)
```

**Output:**
```
              precision    recall  f1-score   support

    Non-Hate     0.7235    0.6716    0.6966       348
        Hate     0.6817    0.7330    0.7065       720

    accuracy                         0.7015      1068
   macro avg     0.7026    0.7023    0.7016      1068
weighted avg    0.6976    0.7015    0.6987      1068
```

### Failure Case Visualization

```python
from abiral.utils import visualize_failures

visualize_failures(
    model=model,
    dataset=val_dataset,
    error_type='false_positive',  # or 'false_negative'
    num_samples=10,
    save_dir='outputs/error_analysis/'
)
```

---

## Visualization

### Training Curves

```python
from abiral.utils import plot_training_curves

plot_training_curves(
    train_losses=trainer.train_losses,
    val_losses=trainer.val_losses,
    val_f1_scores=trainer.val_f1_scores,
    save_path='outputs/training_curves.png'
)
```

### Attention Heatmaps

```python
from abiral.utils import visualize_cross_modal_attention

visualize_cross_modal_attention(
    model=model,
    image=sample_image,
    text=sample_text,
    save_path='outputs/attention_heatmap.png'
)
```

### Embedding Space Visualization (t-SNE)

```python
from abiral.utils import visualize_embedding_space

visualize_embedding_space(
    model=model,
    dataloader=val_loader,
    method='tsne',  # or 'umap'
    save_path='outputs/embedding_tsne.png'
)
```

### Gating Distribution

```python
from abiral.utils import analyze_gating_weights

gating_analysis = analyze_gating_weights(
    model=model,
    dataloader=val_loader
)

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(gating_analysis['text_gates'], bins=50, alpha=0.7, label='Text Gate')
plt.hist(gating_analysis['image_gates'], bins=50, alpha=0.7, label='Image Gate')
plt.xlabel('Gate Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('outputs/gating_distribution.png')
```

---

## Reproducibility

### Set All Random Seeds

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Environment Export

```bash
# Export exact environment
pip freeze > requirements_exact.txt
conda env export > environment.yml
```

### Hardware Specifications

For exact reproducibility, use:
- **GPU:** NVIDIA RTX 3090 (24GB)
- **CPU:** AMD Ryzen 9 5950X (16 cores)
- **RAM:** 64GB DDR4
- **CUDA:** 11.8
- **cuDNN:** 8.6.0
- **PyTorch:** 2.0.1
- **Transformers:** 4.30.2

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

---

## Citation

```bibtex
@inproceedings{thapa2026nepai,
  title={TeamNepAI@CHiPSAL 2026: Cross-Modal Attention Fusion with Dynamic Gating for Devanagari Hate Speech Detection in Multimodal Memes},
  author={Thapa, Samir and others},
  booktitle={Proceedings of CHiPSAL 2026},
  year={2026}
}
```
