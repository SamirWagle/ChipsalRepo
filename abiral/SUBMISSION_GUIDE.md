# 🚀 SUBMISSION GENERATION GUIDE

## Quick Start - Generate Submission CSV

You have **3 ways** to generate submission CSV:

---

## 🔍 Step 0: Check Your Model Type (IMPORTANT!)

Before generating submission, check what type of model you trained:

```bash
python test.py --mode check --model_path outputs/both_hybrid_original_20240215_143022
```

**Output:**
```
================================================================================
MODEL CONFIGURATION
================================================================================
Directory: outputs/both_hybrid_original_20240215_143022
--------------------------------------------------------------------------------
Model Type: Multimodal (Hybrid)
Modality: both
Fusion Strategy: hybrid
--------------------------------------------------------------------------------

SUBMISSION COMMAND
================================================================================
Use this command to generate submission CSV:

python generate_submission.py \
  --model outputs/both_hybrid_original_20240215_143022/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion hybrid
```

This tells you **exactly** what command to use!

---

## ✅ Method 1: Simple Script (RECOMMENDED)

Use the dedicated `generate_submission.py` script:

**Default (assumes multimodal hybrid):**
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv
```

**Specify model type explicitly:**

### For Image-Only Models:
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality image
```

### For Text-Only Models:
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality text
```

### For Multimodal Models:

**Hybrid Fusion (default):**
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion hybrid
```

**Early Fusion:**
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion early
```

**Late Fusion (Voting):**
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion late_voting
```

**Late Fusion (Bagging):**
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion late_bagging
```

---

## 📋 Model Type Quick Reference

**Match your training to your submission:**

| Training Command | Model Type | Submission Parameters |
|-----------------|------------|----------------------|
| `python experiments.py image_only` | Image-only | `--modality image` |
| `python experiments.py text_only` | Text-only | `--modality text` |
| `python experiments.py hybrid` | Hybrid fusion | `--modality both --fusion hybrid` |
| `python experiments.py early` | Early fusion | `--modality both --fusion early` |
| `python experiments.py late_voting` | Late voting | `--modality both --fusion late_voting` |
| `python experiments.py late_bagging` | Late bagging | `--modality both --fusion late_bagging` |

---

## ✅ Method 2: Using test.py with --mode submission

```bash
python test.py \
  --mode submission \
  --model_path outputs/best_model_fold_0.pt \
  --image_folder test_images \
  --csv_path test.csv \
  --output_path submission.csv \
  --fusion_strategy hybrid \
  --use_modality both
```

---

## ✅ Method 3: Python Code (in script)

```python
from test import get_prediction_csv
from config import Config

# Setup config
config = Config()
config.update(
    USE_MODALITY='both',
    FUSION_STRATEGY='hybrid',
    DEVICE='cuda'
)

# Generate submission
submission_df = get_prediction_csv(
    test_image_folder='test_images',
    test_csv_path='test.csv',
    model_path='outputs/best_model_fold_0.pt',
    output_path='submission.csv',
    config=config
)

print(f"Submission saved! {len(submission_df)} predictions")
```

---

## 📋 Requirements

Your **test CSV** must have:
- ✅ `index` column (required)
- ✅ `text` column (optional, used if modality is 'both' or 'text')

Example test.csv:
```csv
index,text
0,some text here
1,another text
2,more text
```

Or minimal (image-only):
```csv
index
0
1
2
```

---

## 📁 File Structure

```
abiral/
├── test_images/          # Your test images folder
│   ├── 0.png
│   ├── 1.png
│   └── 2.png
├── test.csv              # Test data (must have 'index' column)
├── outputs/
│   └── best_model_fold_0.pt  # Trained model
└── submission.csv        # Generated (index, label)
```

---

## 🎯 Output Format

The generated `submission.csv` will have **exactly** this format:

```csv
index,label
0,1
1,0
2,1
3,0
...
```

- `index` - Same as your test CSV
- `label` - Predicted class (0 or 1)

---

## 🔧 Options Explained

### Fusion Strategies:
- `hybrid` - **BEST** (attention-based, 80-85% accuracy)
- `early` - Fast (75-80% accuracy)
- `late_voting` - Ensemble average (78-82%)
- `late_bagging` - Weighted ensemble (79-83%)

### Modalities:
- `both` - Use images + text (recommended)
- `image` - Use only images
- `text` - Use only text

### Device:
- `cuda` - Use GPU (fast)
- `cpu` - Use CPU (slow)

---

## 💡 Examples

### Example 1: Basic Usage
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv
```

### Example 2: Different Model
```bash
python generate_submission.py \
  --model outputs/best_model_fold_2.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission_fold2.csv
```

### Example 3: Image-Only
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality image
```

### Example 4: CPU Mode
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --device cpu
```

---

## 🐛 Troubleshooting

### Error: "Test CSV must have 'index' column"
**Solution:** Make sure your test.csv has a column named `index`

### Error: "FileNotFoundError: No such file"
**Solution:** Check paths are correct:
```bash
# Verify files exist
ls test.csv
ls test_images/
ls outputs/best_model_fold_0.pt
```

### Error: "CUDA out of memory"
**Solution:** Use CPU mode:
```bash
python generate_submission.py \
  ... \
  --device cpu
```

### Warning: "Error predicting X.png"
This means an image couldn't be loaded. Check:
- Image file exists in test_images/
- Image is valid (not corrupted)
- Image filename matches index in CSV

---

## ⚡ Quick Commands Cheat Sheet

**Generate submission (default settings):**
```bash
python generate_submission.py --model outputs/best_model_fold_0.pt --test_images test_images --test_csv test.csv -o submission.csv
```

**Check submission file:**
```bash
head submission.csv
```

**Count predictions:**
```bash
# On Linux/Mac:
cut -d',' -f2 submission.csv | tail -n +2 | sort | uniq -c

# Or just view in Python:
python -c "import pandas as pd; df=pd.read_csv('submission.csv'); print(df['label'].value_counts())"
```

---

## 🎊 You're Ready!

Just run:
```bash
python generate_submission.py \
  --model YOUR_MODEL.pt \
  --test_images YOUR_TEST_FOLDER \
  --test_csv YOUR_TEST.csv \
  --output submission.csv
```

And submit `submission.csv`! 🚀
