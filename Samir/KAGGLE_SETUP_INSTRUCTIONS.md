# 🚀 Kaggle Setup Instructions

## 📦 What You Need to Upload to Kaggle

### 1. Dataset Files (Upload as Kaggle Dataset)

Create a Kaggle dataset with these files from your local folder:

```
your-dataset/
├── train.csv              ← From: ChipsalRepo/train.csv
└── train_images/          ← From: ChipsalRepo/train/train_images/
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ... (all image files)
```

**Steps to Upload Dataset:**
1. Go to Kaggle.com → Datasets → New Dataset
2. Click "Upload" and select:
   - `train.csv`
   - `train_images` folder (all images inside)
3. Name it: `hate-meme-classification` (or any name)
4. Set it to **Private** or **Public**
5. Click "Create"

---

### 2. Notebook File (Use in Kaggle Notebook)

**Option A: Copy-Paste Method** (Easiest)
1. Go to Kaggle.com → Code → New Notebook
2. Change to **Python Script** (not Notebook)
3. Copy entire content from `kaggle_notebook.py`
4. Paste into Kaggle

**Option B: Upload Method**
1. Go to Kaggle.com → Code → New Notebook
2. File → Import Notebook
3. Upload `kaggle_notebook.py`

---

## ⚙️ Configuration Changes Needed

### Update Dataset Path in Kaggle Notebook

Find this section (around line 60):

```python
class Config:
    # DATASET - UPDATE THESE PATHS FOR YOUR KAGGLE DATASET
    IMAGE_FOLDER = "/kaggle/input/your-dataset-name/train_images"
    CSV_PATH = "/kaggle/input/your-dataset-name/train.csv"
```

**Replace `your-dataset-name` with your actual dataset name!**

Example: If you named it `hate-meme-classification`:
```python
IMAGE_FOLDER = "/kaggle/input/hate-meme-classification/train_images"
CSV_PATH = "/kaggle/input/hate-meme-classification/train.csv"
```

---

## 🎛️ Kaggle Notebook Settings

### 1. Add Your Dataset
- Click "Add Data" on the right sidebar
- Search for your uploaded dataset
- Click "Add"

### 2. Enable GPU
- Settings (right sidebar) → Accelerator
- Select: **GPU T4 x2** (free tier)

### 3. Enable Internet
- Settings → Internet → **ON**
- Required to download CLIP and BGE-M3 models

---

## 🚀 Running the Notebook

### Quick Start
1. Open the notebook
2. Click "Run All" or press `Shift + Enter` on each cell
3. Wait ~30-60 minutes for training

### What Happens:
1. **Cell 1-2**: Installs dependencies (transformers, etc.)
2. **Cell 3-8**: Loads code (Dataset, Models, Training)
3. **Cell 9**: Loads your CSV and images
4. **Cell 10**: Prepares CLIP and BGE-M3
5. **Cell 11**: Trains 5-fold cross-validation
6. **Cell 12**: Shows final results

---

## 📊 Expected Output

### During Training (Each Fold ~6-10 minutes):
```
FOLD 1/5
================================================================================
Train size: 854, Val size: 214

Training: 100%|██████████| 54/54 [01:24<00:00]
Train Loss: 0.4523, Train Acc: 0.7812

Validation: 100%|██████████| 14/14 [00:12<00:00]
Val Loss: 0.5234, Val Acc: 0.7430, Val F1: 0.7256

...
```

### Final Results:
```
CROSS-VALIDATION SUMMARY
================================================================================
   fold  accuracy        f1  precision    recall
0     1    0.7430    0.7256     0.7389    0.7156
1     2    0.7523    0.7401     0.7512    0.7298
2     3    0.7664    0.7589     0.7623    0.7545
3     4    0.7383    0.7198     0.7301    0.7089
4     5    0.7757    0.7634     0.7698    0.7601

================================================================================
AVERAGE METRICS ACROSS FOLDS
================================================================================
Accuracy: 0.7551 ± 0.0156
F1: 0.7416 ± 0.0184
Precision: 0.7505 ± 0.0166
Recall: 0.7338 ± 0.0211
```

---

## 📁 Output Files (in `/kaggle/working/outputs/`)

```
outputs/
└── both_20240215_143022/
    ├── summary.json                   # Overall results
    ├── best_model_fold_0.pt          # Trained model (Fold 1)
    ├── best_model_fold_1.pt          # Trained model (Fold 2)
    ├── ...
    ├── fold_0_results.json           # Detailed metrics (Fold 1)
    ├── confusion_matrix_fold_0.png   # Visualization (Fold 1)
    └── ...
```

**Download Results:**
- Click "Save Version" (top right)
- Go to "Output" tab
- Download the `outputs` folder

---

## 🎯 Experiment Variations

### Run Different Modes (Edit Config):

**Text-Only:**
```python
config.USE_MODALITY = 'text'
config.BATCH_SIZE = 32  # Can increase since no images
```

**Image-Only:**
```python
config.USE_MODALITY = 'image'
```

**Multimodal (Both):**
```python
config.USE_MODALITY = 'both'
config.FUSION_STRATEGY = 'hybrid'  # or 'early'
```

### Adjust Training:
```python
config.NUM_EPOCHS = 20      # Faster (less accuracy)
config.BATCH_SIZE = 8       # If GPU memory error
config.LEARNING_RATE = 5e-5 # More careful learning
config.K_FOLDS = 3          # Faster cross-validation
```

---

## 🐛 Common Issues & Fixes

### 1. "No such file or directory: train.csv"
**Fix:** Update dataset path in Config class
```python
CSV_PATH = "/kaggle/input/YOUR-ACTUAL-DATASET-NAME/train.csv"
```

### 2. "CUDA out of memory"
**Fix:** Reduce batch size
```python
config.BATCH_SIZE = 8  # or even 4
```

### 3. "Cannot download models (HTTP error)"
**Fix:** Enable internet in settings
- Settings → Internet → ON

### 4. "Image file not found"
**Fix:** Check image filenames match CSV
- CSV has: `0`, `1`, `2`
- Files should be: `0.png`, `1.png`, `2.png`

Or update dataset.py to add extension:
```python
# Already handled in kaggle_notebook.py line ~170
if not image_filename.endswith(('.png', '.jpg', '.jpeg')):
    image_filename = f"{image_filename}.png"
```

### 5. Training is slow
**Fix:** 
- Enable GPU (Settings → Accelerator → GPU T4 x2)
- Reduce epochs: `config.NUM_EPOCHS = 20`
- Reduce folds: `config.K_FOLDS = 3`

---

## 📈 Improving Accuracy (Target: 80-85%)

### 1. Use Hybrid Fusion (Best)
```python
config.USE_MODALITY = 'both'
config.FUSION_STRATEGY = 'hybrid'
```

### 2. Increase Model Capacity
```python
config.HIDDEN_DIM = 512  # Default: 256
config.DROPOUT = 0.4     # Default: 0.3
```

### 3. Fine-tune Encoders (Advanced)
```python
config.FREEZE_TEXT_ENCODER = False
config.LEARNING_RATE = 1e-5  # Lower LR when fine-tuning
```

### 4. More Training
```python
config.NUM_EPOCHS = 50
config.PATIENCE = 15  # More patience
```

---

## 💾 Saving & Downloading Results

### 1. Save Kaggle Notebook Version
- Click "Save Version" (top right)
- Choose "Quick Save" or "Save & Run All"
- Wait for completion

### 2. Download Trained Models
- Go to "Output" tab
- Download `outputs` folder
- Contains all `.pt` model files

### 3. View Results
- `summary.json` - Overall performance
- `confusion_matrix_fold_X.png` - Prediction heatmaps
- `fold_X_results.json` - Per-fold metrics

---

## 🎓 Summary Checklist

- [ ] Upload dataset to Kaggle (train.csv + train_images/)
- [ ] Create new Kaggle notebook
- [ ] Copy code from `kaggle_notebook.py`
- [ ] Update dataset path in Config class
- [ ] Add dataset to notebook (right sidebar)
- [ ] Enable GPU (Settings → Accelerator)
- [ ] Enable Internet (Settings → Internet)
- [ ] Run all cells
- [ ] Wait for training (~30-60 min)
- [ ] Check results in final cell
- [ ] Download outputs folder

---

## 🚀 Next Steps After Training

1. **Analyze Results**
   - Check confusion matrix for error patterns
   - Look at per-class precision/recall

2. **Experiment**
   - Try different fusion strategies
   - Adjust hyperparameters
   - Test image-only vs text-only vs both

3. **Deploy**
   - Use best model for predictions
   - Create inference script for new data

---

**Good luck! 🎯**

If you encounter issues, check:
1. Dataset path is correct
2. GPU is enabled
3. Internet is enabled
4. All dataset files uploaded correctly
