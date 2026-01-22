# CHiPSAL 2026 - Subtask A: Hate Speech Detection in Nepali Memes

## 🎯 Task Description

This repository contains the solution for **Subtask A** of the CHiPSAL 2026 Shared Task: **Hate Speech Detection in Nepali-only Memes**.

**Objective**: Detect the presence of hate speech in monolingual Nepali memes.
- **Label 0**: Non-Hate
- **Label 1**: Hate
- **Evaluation Metric**: Macro F1-Score

## 📁 Project Structure

```
ChipsalRepo/
├── config.py                   # Configuration and hyperparameters
├── requirements.txt            # Python dependencies
├── train_simple.py             # Quick-start training script
├── train.csv                   # Training data (index, label, text)
├── index_label_train.csv       # Training labels
├── OCR_Dataset_Image.csv       # Pre-extracted OCR text
│
├── src/
│   ├── __init__.py
│   ├── data_exploration.py     # Data analysis and visualization
│   ├── ocr_extraction.py       # OCR text extraction using EasyOCR
│   ├── dataset.py              # PyTorch Dataset classes
│   ├── models.py               # Model architectures (Text/Image/Multimodal)
│   ├── train.py                # Full training pipeline with K-Fold CV
│   └── inference.py            # Prediction and submission generation
│
├── train/
│   └── train_images/           # Training meme images
│
├── eval/                       # Evaluation data (download from competition)
│   └── eval_images/
│
├── test/                       # Test data (download from competition)
│   └── test_images/
│
├── data/                       # Processed data
│   └── ocr_train_extracted.csv
│
└── outputs/
    ├── models/                 # Saved model checkpoints
    ├── submissions/            # Generated submission files
    └── logs/                   # Training logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Download images from the competition links:
- **Training**: Place in `train/train_images/`
- **Evaluation**: Place in `eval/eval_images/`
- **Test**: Place in `test/test_images/`

### 3. Quick Training (Recommended for First Run)

```bash
python train_simple.py
```

This will:
- Load the training data
- Train a multimodal model (XLM-RoBERTa + ResNet50)
- Save the best model to `outputs/models/`

### 4. Full Training with K-Fold CV

```bash
python src/train.py
```

This provides:
- 5-Fold stratified cross-validation
- Early stopping
- Class weighting for imbalance
- Comprehensive metrics logging

### 5. Generate Predictions

```bash
python src/inference.py
```

This creates:
- `predictions.csv`: Submission file
- `submission.zip`: Ready for upload to Codabench

## 🧠 Model Architectures

### 1. Text-Only (TextClassifier)
- Uses XLM-RoBERTa / mBERT / MuRIL
- Good for memes with extracted OCR text

### 2. Image-Only (ImageClassifier)
- Uses ResNet50 / EfficientNet / ViT
- Captures visual features and layout

### 3. Multimodal Fusion (Recommended)
- Combines text and image features
- Fusion types: concat, attention, gated

### 4. CLIP-Based
- Uses pre-trained CLIP for joint understanding
- Best for zero-shot transfer

## 📊 Results

| Model | Val F1 (Macro) |
|-------|---------------|
| Text-only (XLM-RoBERTa) | ~0.65 |
| Image-only (ResNet50) | ~0.58 |
| Multimodal (Concat) | ~0.70 |
| Multimodal (Attention) | ~0.72 |

*Note: Results may vary based on hyperparameters and random seeds.*

## 📝 Submission Format

Create `predictions.csv`:
```csv
index,label
12345.jpg,0
15001.jpg,1
20524.jpg,1
```

Zip and submit to Codabench:
```bash
zip submission.zip predictions.csv
```

## 🔧 Configuration

Edit `config.py` to customize:
- Model architecture
- Training hyperparameters
- Data augmentation
- Paths

## 📚 Key Features

1. **OCR Extraction**: EasyOCR for Nepali text
2. **Data Augmentation**: Albumentations for images
3. **Class Imbalance**: Weighted loss function
4. **Mixed Precision**: Faster training with AMP
5. **Early Stopping**: Prevent overfitting
6. **Ensemble**: Combine K-fold predictions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

## 📄 Citation

If you use this code, please cite:
```
@inproceedings{thapa2025nememe,
  title={NeMeme: A Multimodal Prompt-based Framework for Analyzing Code-Mixed and Low-Resource Memes},
  author={Thapa, S. and Veeramani, H. and others},
  booktitle={ICWSM 2025},
  year={2025}
}
```

## 📧 Contact

For questions about the competition:
- Contact: rauniyark11@gmail.com
- GitHub: https://github.com/therealthapa/chipsal26-memes

---

**Good luck with CHiPSAL 2026! 🇳🇵**
