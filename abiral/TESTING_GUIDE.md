# Testing Guide

## 🎯 Your Trained Models

You have successfully trained MoE (Mixture of Experts) models located in:
```
outputs/novel_moe/
├── best_model_fold_0.pt
├── best_model_fold_1.pt
└── results.json
```

**Model Performance:**
- Fold 0: F1-Score = 0.403
- Fold 1: F1-Score = 0.554 ⭐ (Use this one!)
- Average F1: 0.478

## 🚀 Quick Start - 3 Ways to Test

### Method 1: Double-click Batch File (Easiest!)
Simply double-click `RUN_TEST.bat` to test on the demo image.

### Method 2: Quick Test Script (Recommended)
```bash
# Activate environment
chipsal_env\Scripts\activate

# Test single demo image
cd abiral
python quick_test.py

# Test specific image
python quick_test.py --image path/to/your/image.jpg

# Test with text
python quick_test.py --image path/to/image.jpg --text "मेम text here"
```

### Method 3: Full Test Script (Advanced)
```bash
cd abiral

# Check model configuration
python test.py --mode check --model_path ..\outputs\novel_moe

# Single image prediction
python test.py --mode single \
    --model_path ..\outputs\novel_moe\best_model_fold_1.pt \
    --image_path ..\demo_input\1005.jpg

# Batch prediction
python test.py --mode batch \
    --model_path ..\outputs\novel_moe\best_model_fold_1.pt \
    --csv_path data.csv \
    --image_folder images \
    --output_path predictions.csv
```

## 📊 Generate Submission (When You Have Test Data)

### If you have test.csv and test_images folder:

**Option 1: Batch File**
```bash
# Double-click GENERATE_SUBMISSION.bat
```

**Option 2: Command Line**
```bash
chipsal_env\Scripts\activate
cd abiral
python quick_test.py --mode submission --csv ..\test.csv --images ..\test_images
```

This will create `submission.csv` with format:
```
index,label
1001,0
1002,1
1003,0
...
```

## 📁 Test Data Format

Your test CSV should have at minimum:
```csv
index,text
1001,"some text here"
1002,"another text"
```

Test images should be in a folder:
```
test_images/
├── 1001.png
├── 1002.png
├── 1003.png
...
```

## 🔍 Understanding Results

**Predicted Labels:**
- `0` = no-hate (safe meme)
- `1` = hate (hateful meme)

**Output includes:**
- Predicted class
- Confidence probabilities for each class
- Modality weights (how much image vs text contributed)

## 💡 Pro Tips

1. **Use Fold 1 Model**: It has better F1-score (0.554 vs 0.403)
   
2. **Ensemble Both Folds** (Advanced): For best results, average predictions from both folds:
   ```bash
   python test.py --mode submission --model_path outputs/novel_moe/best_model_fold_0.pt --output_path sub_fold0.csv
   python test.py --mode submission --model_path outputs/novel_moe/best_model_fold_1.pt --output_path sub_fold1.csv
   # Then average the predictions
   ```

3. **Check Model Config**: Before testing, verify model settings:
   ```bash
   python test.py --mode check --model_path outputs/novel_moe
   ```

## ❓ Troubleshooting

**Error: Model not found**
- Make sure you're in the correct directory
- Check that outputs/novel_moe/best_model_fold_1.pt exists

**Error: Image not found**
- Verify image path is correct
- Check that image files have .png, .jpg, or .jpeg extension

**Error: CUDA out of memory**
- The scripts automatically use CPU if CUDA isn't available
- If CUDA errors occur, add `--device cpu` flag

**Low confidence predictions**
- This is normal for difficult samples
- Check modality weights to see if text or image dominates

## 📈 Next Steps

1. ✅ Test on demo image (verify model works)
2. ✅ Test on a few sample images
3. ✅ Generate submission when you have test data
4. ✅ Submit to competition!

Good luck! 🎉
