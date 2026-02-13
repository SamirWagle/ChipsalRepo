# 🎯 GENERATE SUBMISSION - QUICK GUIDE

## Your Setup:
- ✅ Models: `abiral/Output/image_original_20260212_132844/` (5 folds)
- ✅ Test Data: `abiral/test_release/` (135 samples)
- ✅ Output: `submission.csv` with columns: **index, label**

## 🚀 GENERATE SUBMISSION NOW:

### Option 1: Double-Click (Easiest)
```
Double-click: RUN_SUBMISSION.bat
```

### Option 2: Command Line
```powershell
cd abiral
python generate_submission.py
```

This will:
1. Load all 5 models from Output folder
2. Ensemble predictions (majority voting)
3. Generate `submission.csv` with index, label columns

## 📊 Output Format

**submission.csv:**
```csv
index,label
2237.jpg,0
1485.jpg,1
1096.jpg,0
...
```

- **index** = image filename
- **label** = 0 (no-hate) or 1 (hate)

## ⚙️ Options

**Use single model (faster, no ensemble):**
```powershell
python generate_submission.py --single
```

**Use different model folder:**
```powershell
python generate_submission.py --model_dir Output/image_original_20260212_130010
```

**Custom output file:**
```powershell
python generate_submission.py --output my_submission.csv
```

## 📁 Files Created

After running, you'll have:
- `submission.csv` - Ready to submit!

## ✅ Quick Check

After generation, the script shows:
- Number of samples
- Distribution (how many hate vs no-hate)
- File location

Example output:
```
SUBMISSION READY!
================================================================================
✓ Saved: submission.csv
  Samples: 135

Distribution:
  no-hate  =  102 (75.56%)
  hate     =   33 (24.44%)
================================================================================
```

## 🔧 Troubleshooting

**Error: "No models found"**
- Check that `Output/image_original_20260212_132844/` has `.pt` files

**Error: "Image not found"**
- Check that `test_release/test_images/` has all images

**Error: "Config  not found"**
- Make sure `config.json` exists in the model folder

## 🎉 That's It!

Just run and get your submission.csv!
