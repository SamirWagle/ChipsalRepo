# Quick Reference - Model Selection & Testing

## List Available Models

```powershell
python generate_submission.py --list-models
```

**What it does:**
- Shows all model folders in `Output/`
- Displays modality (image/text/both)
- Shows number of folds available
- Lists which fold numbers exist

---

## Interactive Mode (Easiest)

```powershell
python generate_submission.py --interactive
```
or
```powershell
python generate_submission.py -i
```

**What it does:**
- Lists all models
- Prompts you to select a model by number
- Prompts you to select fold(s)
- Walks you through the process step by step

---

## Manual Selection (Full Control)

### 1. Select Model Folder

```powershell
# Use specific model folder
python generate_submission.py --model_dir Output/image_original_20260212_130010

# Use text model
python generate_submission.py --model_dir Output/text_20260212_100018

# Use multimodal model
python generate_submission.py --model_dir Output/both_hybrid_20260212_150000
```

### 2. Select Specific Fold(s)

```powershell
# Use only fold 0
python generate_submission.py --fold 0

# Use only fold 2
python generate_submission.py --fold 2

# Use folds 0, 1, and 2 (ensemble)
python generate_submission.py --fold 0,1,2

# Use all available folds (default)
python generate_submission.py --fold all
```

### 3. Combine Folder + Fold Selection

```powershell
# Test fold 0 from specific model
python generate_submission.py --model_dir Output/image_original_20260212_130010 --fold 0

# Test folds 0,1,2 from multimodal model
python generate_submission.py --model_dir Output/both_hybrid_20260212_150000 --fold 0,1,2

# Test all folds with custom output name
python generate_submission.py --model_dir Output/text_20260212_100018 --output text_test.csv
```

---

## Single Model Mode

```powershell
# Use middle fold only (faster, less accurate)
python generate_submission.py --single

# Single fold from specific model
python generate_submission.py --model_dir Output/both_hybrid_20260212_150000 --single
```

---

## Custom Output Names

```powershell
# Save to specific file
python generate_submission.py --output my_submission.csv

# Combine with model selection
python generate_submission.py --model_dir Output/text_20260212_100018 --output text_submission.csv

# Test specific fold with custom name
python generate_submission.py --fold 0 --output fold0_test.csv
```

---

## Complete Examples

### Example 1: Quick Test
```powershell
# Interactive selection (easiest)
python generate_submission.py -i
```

### Example 2: Test Specific Model & Fold
```powershell
# Test only fold 3 from image model
python generate_submission.py --model_dir Output/image_original_20260212_132844 --fold 3 --output fold3_test.csv
```

### Example 3: Ensemble Specific Folds
```powershell
# Ensemble folds 0,1,2 from multimodal model
python generate_submission.py --model_dir Output/both_hybrid_20260212_150000 --fold 0,1,2 --output ensemble_012.csv
```

### Example 4: Compare All Models
```powershell
# List what's available
python generate_submission.py --list-models

# Test each model
python generate_submission.py --model_dir Output/image_original_20260212_130010 --output image_130010.csv
python generate_submission.py --model_dir Output/image_original_20260212_132844 --output image_132844.csv
python generate_submission.py --model_dir Output/text_20260212_100018 --output text.csv
```

### Example 5: Test Each Fold Individually
```powershell
# Test each fold separately to see which performs best
python generate_submission.py --fold 0 --output fold_0.csv
python generate_submission.py --fold 1 --output fold_1.csv
python generate_submission.py --fold 2 --output fold_2.csv
python generate_submission.py --fold 3 --output fold_3.csv
python generate_submission.py --fold 4 --output fold_4.csv
```

---

## Default Behavior (No Arguments)

```powershell
python generate_submission.py
```

**Uses:**
- Model: `Output/image_original_20260212_132844`
- Folds: All available
- Output: `submission.csv`
- Mode: Ensemble

---

## Advanced: Custom Data Paths

```powershell
# Use different test data
python generate_submission.py --test_csv data/my_test.csv --test_images data/my_images/

# Complete custom setup
python generate_submission.py \
  --model_dir Output/both_hybrid_20260212_150000 \
  --fold 0,1,2 \
  --test_csv custom_test.csv \
  --test_images custom_images/ \
  --output custom_output.csv
```

---

## Workflow Examples

### Workflow 1: Find Best Fold
```powershell
# 1. Test each fold
python generate_submission.py --fold 0 --output fold0.csv
python generate_submission.py --fold 1 --output fold1.csv
python generate_submission.py --fold 2 --output fold2.csv

# 2. Submit each to competition portal
# 3. Use the best performing fold
```

### Workflow 2: Find Best Model
```powershell
# 1. List available
python generate_submission.py --list-models

# 2. Test each
python generate_submission.py --model_dir Output/image_* --output image.csv
python generate_submission.py --model_dir Output/text_* --output text.csv
python generate_submission.py --model_dir Output/both_* --output both.csv

# 3. Submit best one
```

### Workflow 3: Progressive Ensemble
```powershell
# Start with single fold (fast)
python generate_submission.py --fold 0 --output quick_test.csv

# If good, try 3-fold ensemble
python generate_submission.py --fold 0,1,2 --output ensemble_3.csv

# If still good, use all folds
python generate_submission.py --fold all --output ensemble_all.csv
```

---

## Tips

✅ **Use `--list-models` first** to see what's available  
✅ **Use `--interactive` for guided selection**  
✅ **Use `--fold 0` for quick testing** (fastest)  
✅ **Use `--fold all` for best accuracy** (slower)  
✅ **Use `--single` when you only want the middle fold**  
✅ **Save outputs with descriptive names** for easy tracking  

---

## Troubleshooting

**"No models found"**
```powershell
python generate_submission.py --list-models
# Check if Output/ folder has .pt files
```

**"Fold not found"**
```powershell
# List available folds first
python generate_submission.py --list-models
# Then use existing fold numbers
```

**Want to re-test quickly**
```powershell
# Reuse same command with different output
python generate_submission.py --model_dir Output/image_original_20260212_132844 --fold 0 --output test2.csv
```
