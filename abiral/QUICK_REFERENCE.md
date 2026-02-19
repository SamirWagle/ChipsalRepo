# 🎯 QUICK REFERENCE - Model Types & Submission

## Step 1: Check Your Model Type
```bash
python test.py --mode check --model_path outputs/YOUR_MODEL_DIR
```

## Step 2: Use Correct Command

### Image-Only Models
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality image
```

### Text-Only Models
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality text
```

### Multimodal Models

#### Hybrid Fusion (Best - 80-85%)
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion hybrid
```

#### Early Fusion (Fast - 75-80%)
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion early
```

#### Late Voting (78-82%)
```bash
python generate_submission.py \
  --model outputs/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion late_voting
```

#### Late Bagging (79-83%)
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

## 🔍 Training → Submission Mapping

| You Trained With | Use This Command |
|-----------------|------------------|
| `python experiments.py image_only` | `--modality image` |
| `python experiments.py text_only` | `--modality text` |
| `python experiments.py hybrid` | `--modality both --fusion hybrid` ⭐ |
| `python experiments.py early` | `--modality both --fusion early` |
| `python experiments.py late_voting` | `--modality both --fusion late_voting` |
| `python experiments.py late_bagging` | `--modality both --fusion late_bagging` |

---

## ⚠️ Common Mistakes

❌ **Wrong:** Using `--fusion hybrid` with image-only model
```bash
# This will ERROR!
python generate_submission.py ... --modality image --fusion hybrid
```

✅ **Correct:** No fusion for unimodal
```bash
python generate_submission.py ... --modality image
```

---

❌ **Wrong:** Not matching training config
```bash
# Trained with: python experiments.py early
# But using: --fusion hybrid  ← WRONG!
```

✅ **Correct:** Match training exactly
```bash
# Trained with: python experiments.py early
# Use: --modality both --fusion early  ✓
```

---

## 💡 Pro Tips

1. **Always check config first:**
   ```bash
   python test.py --mode check --model_path outputs/YOUR_MODEL_DIR
   ```

2. **Use the exact command it shows you!** Copy-paste to avoid errors.

3. **Default is hybrid:** If you don't specify `--fusion`, it uses `hybrid`.

4. **Test CSV needs:**
   - `index` column (required)
   - `text` column (optional, only if using text modality)

5. **Output format:**
   ```csv
   index,label
   0,1
   1,0
   2,1
   ```

---

## 🚀 Most Common Use Cases

### Scenario 1: You trained a hybrid model
```bash
# Check config (optional)
python test.py --mode check --model_path outputs/both_hybrid_original_20240215_143022

# Generate submission
python generate_submission.py \
  --model outputs/both_hybrid_original_20240215_143022/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality both \
  --fusion hybrid
```

### Scenario 2: You trained image-only
```bash
python generate_submission.py \
  --model outputs/image_only_20240215_143022/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality image
```

### Scenario 3: You trained text-only
```bash
python generate_submission.py \
  --model outputs/text_only_20240215_143022/best_model_fold_0.pt \
  --test_images test_images \
  --test_csv test.csv \
  --output submission.csv \
  --modality text
```

---

**That's it! Just match your training config and you're good to go! 🎉**
