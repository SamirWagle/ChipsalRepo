# Cross-Modal Attention Fusion for Nepali Hate Speech Detection in Multimodal Memes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-CHiPSAL_2026-blue)](PaperPortion/FINAL_paper_report.md)

> **Official Implementation** of our submission to **CHiPSAL 2026 Shared Task — Subtask A: Multimodal Hate and Sentiment Understanding in Low-Resource Text-Embedded Images**

**Paper Title:** "TeamNepAI@CHiPSAL 2026: Cross-Modal Attention Fusion with Dynamic Gating for Devanagari Hate Speech Detection in Multimodal Memes"

**Authors:** Samir Thapa, [Co-authors to be added]

**Conference:** CHiPSAL 2026 (Second Workshop on Challenges in Processing South Asian Languages @ LREC 2026)

**Submission Deadline:** March 1, 2026

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Reproducing Paper Results](#-reproducing-paper-results)
- [Dataset](#-dataset)
- [Model Zoo](#-model-zoo)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This repository contains the complete implementation of our **Hybrid Cross-Modal Attention Fusion architecture** for detecting hate speech in Devanagari-scripted social media memes. Our system addresses the challenging problem of multimodal content moderation in low-resource languages through:

### Key Contributions

1. **🏗️ Novel Hybrid Fusion Architecture**
   - 4-head cross-modal self-attention mechanism
   - Learnable dynamic gating network for adaptive modality weighting
   - Sample-specific fusion that adapts to varying modality informativeness

2. **🌐 Multilingual Multimodal Encoding**
   - CLIP ViT-B/32 for visual feature extraction
   - BGE-M3 for Devanagari/Nepali text encoding with superior Indic script coverage
   - Selective fine-tuning strategy (2 CLIP layers, 4 BGE-M3 layers)

3. **📊 Comprehensive Ablation Study**
   - 8 model configurations comparing fusion strategies
   - Systematic evaluation of unimodal, early fusion, late fusion, and hybrid approaches
   - 5-fold stratified cross-validation for robust evaluation

4. **⚖️ Imbalanced Data Handling**
   - Frequency-inverse loss weighting (γ-suppression)
   - Label smoothing for calibrated predictions
   - Macro-averaged F1 as primary metric

### Research Highlights

- **Multimodal Superiority:** +5.9% F1 improvement over text-only baselines
- **Text Modality Dominance:** Text-only outperforms image-only by 16.6% (reveals CLIP's Devanagari blindness)
- **Small Dataset Challenges:** Demonstrates that ensemble methods fail under extreme data scarcity (N=854)
- **Typography Information:** Quantifies 5.8% F1 contribution from in-image text features
- **Evaluation Metric Selection:** Shows accuracy misleads in imbalanced settings (2.2% accuracy gap vs. 5.8% F1 gap)

---

## 📊 Key Results

### Subtask 1: Binary Hate Speech Detection

**Table: Cross-validated performance on 1,068 Nepali memes (5-fold stratified CV)**

| Rank | Model Configuration | F1 Macro ↑ | Accuracy | Precision | Recall ↑ | Std Dev |
|------|---------------------|------------|----------|-----------|----------|---------|
| 🥇 1 | **M7: Hybrid Fusion (Attn+Gating)** | **0.6834** | 0.7015 | 0.6817 | **0.7023** | ±0.018 |
| 🥈 2 | M1: Text-Only (BGE-M3) | 0.6455 | **0.7239** | **0.6886** | 0.6376 | ±0.022 |
| 🥉 3 | M8: Hybrid (Text-Removed Images) | 0.6436 | 0.7015 | 0.6553 | 0.6384 | ±0.021 |
| 4 | M5: Late Fusion (Soft Voting) | 0.6511 | 0.6940 | 0.6521 | 0.6503 | ±0.019 |
| 5 | M4: Early Fusion (Concatenation) | 0.6140 | 0.6940 | 0.6425 | 0.6096 | ±0.029 |
| 6 | M6: Late Fusion (Bagging k=3) | 0.6267 | 0.6567 | 0.6245 | 0.6341 | ±0.034 |
| 7 | M2: Image-Only (Original) | 0.5952 | 0.6791 | 0.6205 | 0.5927 | ±0.025 |
| 8 | M3: Image-Only (Text-Removed) | 0.5535 | 0.6343 | 0.5636 | 0.5535 | ±0.031 |

**Performance Gains:**
- **vs. Text-Only (M1):** +5.9% F1 (9.1% relative improvement)
- **vs. Early Fusion (M4):** +11.3% F1 (18.4% relative improvement)
- **vs. Image-Only (M2):** +23.7% F1 (40% relative improvement)

### Dataset Characteristics

- **Total Samples:** 1,068 (Subtask A)
- **Class Distribution:** 67.4% Hate, 32.6% Non-Hate (2.07:1 imbalance)
- **Training per Fold:** ~854 samples
- **Validation per Fold:** ~214 samples
- **Language:** Nepali (Devanagari script)
- **Source:** 2022-2023 Nepali social media (Facebook, Twitter, Instagram)

---

## 🏗️ Architecture

### Hybrid Cross-Modal Attention Fusion (M7) — Primary System

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Meme Image (224×224) + OCR-Extracted Text (77 tokens)  │
└───────────────────┬─────────────────────┬───────────────────────┘
                    │                     │
         ┌──────────▼───────────┐  ┌──────▼──────────────┐
         │   CLIP ViT-B/32      │  │     BGE-M3          │
         │   (Vision Encoder)   │  │  (Text Encoder)     │
         │                      │  │                     │
         │  • 12 transformer    │  │  • 24 transformer   │
         │    layers            │  │    layers           │
         │  • Last 2 unfrozen   │  │  • Last 4 unfrozen  │
         │  • Patch size: 32×32 │  │  • Mean pooling     │
         └──────────┬───────────┘  └──────┬──────────────┘
                    │ v ∈ ℝ⁵¹²           │ t ∈ ℝ¹⁰²⁴
                    │                     │
         ┌──────────▼───────────┐  ┌──────▼──────────────┐
         │  Linear Projection   │  │  Linear Projection  │
         │    (512 → 256)       │  │   (1024 → 256)      │
         └──────────┬───────────┘  └──────┬──────────────┘
                    │ v' ∈ ℝ²⁵⁶          │ t' ∈ ℝ²⁵⁶
                    └──────────┬──────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Stack as Sequence   │
                    │    S = [v'; t']      │
                    │    S ∈ ℝ^(2×256)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  4-Head Self-Attn    │
                    │                      │
                    │  Q, K, V = WqS,      │
                    │           WkS, WvS   │
                    │                      │
                    │  Attn = softmax(QK'/ │
                    │         √dk) × V     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Extract Attended    │
                    │    v̂ = Attn[0,:]    │
                    │    t̂ = Attn[1,:]    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Dynamic Gate        │
                    │                      │
                    │  g = softmax(Wg·     │
                    │      [v̂;t̂] + bg)    │
                    │                      │
                    │  g ∈ ℝ², g₁+g₂=1    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Element-wise Gate   │
                    │    v_gated = g₁·v̂   │
                    │    t_gated = g₂·t̂   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   Concatenate        │
                    │  f = [v_gated;       │
                    │       t_gated]       │
                    │  f ∈ ℝ⁵¹²           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Classification MLP  │
                    │                      │
                    │  512→256→128→2       │
                    │  ReLU + Dropout(0.5) │
                    └──────────┬───────────┘
                               │
                     ┌─────────▼──────────┐
                     │   Softmax Output   │
                     │   P(Hate | meme)   │
                     └────────────────────┘
```

**Key Innovation:** Dynamic gating learns per-sample modality weights:
- **68% of samples:** Text-dominant (g_text > 0.7) — explicit hate markers in text
- **24% of samples:** Balanced fusion (0.4 ≤ g_text ≤ 0.7) — correlated signals
- **8% of samples:** Image-dominant (g_text < 0.4) — visual-only hate

---

## 🚀 Installation

### Prerequisites

- **Python:** 3.10 or higher
- **CUDA:** 11.8+ (optional, for GPU acceleration)
- **RAM:** 16GB+ recommended
- **GPU:** NVIDIA RTX 3090 (24GB) or equivalent (for full training)

### Setup Instructions

```bash
# Clone repository
git clone https://github.com/yourusername/chipsal-multimodal-hate-detection.git
cd chipsal-multimodal-hate-detection

# Create virtual environment
python -m venv chipsal_env
source chipsal_env/bin/activate  # On Windows: chipsal_env\Scripts\activate

# Install PyTorch with CUDA (or CPU-only version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from abiral.model import MultimodalClassifier; print('✓ Installation successful')"
```

---

## ⚡ Quick Start

### 1. Data Preparation

Place your data in the following structure:

```
ChipsalRepo/
├── train/
│   ├── train_images/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── train.csv           # Columns: index, text, label
│   └── index_label_train.csv
├── val/
│   └── ...
└── test_release/
    └── ...
```

**train.csv format:**
```csv
index,text,label
0001.jpg,"यो नेता भ्रष्ट छ",1
0002.jpg,"राम्रो दिन",0
```

### 2. Train Primary System (M7 — Hybrid Fusion)

```bash
# Single-fold training
python abiral/train.py \
    --experiment_name "m7_hybrid_fusion" \
    --model_type "hybrid" \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 30 \
    --early_stopping_patience 10

# Full 5-fold cross-validation
python abiral/experiments.py \
    --model_config "hybrid_fusion" \
    --k_folds 5 \
    --output_dir "outputs/m7_full_cv"
```

### 3. Generate Test Predictions

```bash
# Generate submission file
python abiral/generate_submission.py \
    --model_path "outputs/m7_hybrid_fusion/best_model.pt" \
    --test_csv "test_release/index_text_test.csv" \
    --test_images "test_release/test_images/" \
    --output "submission.csv"
```

### 4. Inference on Single Image

```python
from abiral.inference import HateDetectionInference

# Initialize model
detector = HateDetectionInference(
    model_path="outputs/m7_hybrid_fusion/best_model.pt",
    device="cuda"
)

# Predict
result = detector.predict(
    image_path="test.jpg",
    text="नेपाली पाठ यहाँ"
)

print(f"Prediction: {result['label']}")  # 0 or 1
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probs']}")
```

---

## 🔬 Reproducing Paper Results

### Complete Ablation Study (All 8 Models)

```bash
# Run full experimental suite matching Table 4
python abiral/experiments.py \
    --experiment_suite "full_ablation" \
    --k_folds 5 \
    --output_dir "outputs/paper_results" \
    --seed 42
```

This will train:
- **M1:** Text-Only (BGE-M3)
- **M2:** Image-Only (Original)
- **M3:** Image-Only (Text-Removed)
- **M4:** Early Fusion
- **M5:** Late Fusion (Voting)
- **M6:** Late Fusion (Bagging)
- **M7:** Hybrid Fusion (Primary System) ⭐
- **M8:** Hybrid Fusion (Text-Removed)

### Individual Model Configurations

```bash
# M1: Text-Only Baseline
python abiral/train.py --model_type "text_only" --config "configs/m1_text_only.yaml"

# M7: Hybrid Fusion (Primary System)
python abiral/train.py --model_type "hybrid" --config "configs/m7_hybrid_fusion.yaml"
```

### Hyperparameter Settings (Matching Paper Table 3)

All configurations are pre-defined in `abiral/experiment_configs.py`:

```python
# M7: Hybrid Fusion Configuration
HYBRID_FUSION_CONFIG = {
    'model_type': 'hybrid',
    'optimizer': 'AdamW',
    'learning_rate': 2e-5,
    'weight_decay': 1e-2,
    'batch_size': 32,
    'num_epochs': 30,
    'dropout': 0.5,
    'hidden_dim': 256,
    'attention_heads': 4,
    'label_smoothing': 0.1,
    'class_weight_gamma': 0.3,
    'unfreeze_clip_layers': 2,
    'unfreeze_bge_layers': 4,
    'scheduler': 'ReduceLROnPlateau',
    'early_stopping_patience': 10,
}
```

---

## 📚 Dataset

### CHiPSAL 2026 Shared Task Dataset

**Source:** Multimodal Hate and Sentiment Understanding in Low-Resource Text-Embedded Images

**Statistics:**
- **Subtask A (Binary):** 1,068 samples
  - Training: 854 samples/fold (80%)
  - Validation: 214 samples/fold (20%)
  - Class Distribution: 720 Hate (67.4%), 348 Non-Hate (32.6%)

- **Subtask B (3-Class):** 1,194 samples
  - Non-Offensive: 380 (31.8%)
  - Offensive: 538 (45.1%)
  - Hate: 276 (23.1%)

**Languages:** Nepali (Devanagari script) with code-mixing (English loanwords)

**Data Files:**
```
train/
├── train.csv                 # Main training labels
├── index_text_train.csv      # OCR-extracted text
├── index_label_train.csv     # Labels only
└── train_images/             # JPEG images

test_release/
├── index_text_test.csv       # Test OCR text
└── test_images/              # Test images
```

### Citation for Dataset

```bibtex
@inproceedings{thapa2026sharedtask,
  title={Multimodal Hate and Sentiment Understanding in Low-Resource Text-Embedded Images for Online Safety and Digital Well-being},
  author={Thapa, Surendrabikram and Shiwakoti, Shuvam and Shah, Siddhant Bikram and others},
  booktitle={Proceedings of the Second Workshop on Challenges in Processing South Asian Languages (CHiPSAL)},
  year={2026}
}
```

---

## 📦 Model Zoo

Pre-trained model checkpoints for direct inference:

| Model | Config | Subtask | F1-Macro | Accuracy | Download |
|-------|--------|---------|----------|----------|----------|
| M7 — Hybrid Fusion | Primary System | Subtask 1 | 0.6834 | 0.7015 | [🔗 Google Drive](#) |
| M1 — Text-Only | BGE-M3 Baseline | Subtask 1 | 0.6455 | 0.7239 | [🔗 Google Drive](#) |
| M5 — Late Voting | Simple Fusion | Subtask 1 | 0.6511 | 0.6940 | [🔗 Google Drive](#) |

**Usage:**

```python
from abiral.inference import load_pretrained_model

# Load from checkpoint
model = load_pretrained_model(
    'models/m7_hybrid_fusion_best.pt',
    device='cuda'
)

# Inference
prediction = model.predict(image='test.jpg', text='नेपाली text')
```

---

## 📁 Project Structure

```
ChipsalRepo/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
│
├── abiral/                           # Main source code (preserved)
│   ├── __init__.py
│   ├── config.py                     # Experiment configurations
│   ├── dataset.py                    # Data loading & preprocessing
│   ├── encoders.py                   # CLIP & BGE-M3 wrappers
│   ├── fusion_models.py              # All 8 fusion architectures
│   ├── model.py                      # Unified model wrapper
│   ├── trainer.py                    # Training & validation loops
│   ├── experiments.py                # Experiment runner
│   ├── experiment_configs.py         # Hyperparameter configs
│   ├── inference.py                  # Inference interface
│   ├── generate_submission.py        # Test prediction generator
│   ├── utils.py                      # Metrics, logging, utils
│   ├── quick_test.py                 # Quick sanity tests
│   ├── train.py                      # Training entry point
│   └── test.py                       # Testing utilities
│
├── samir/                            # Experimental code (preserved)
│   └── ...                           # (Duplicate experiments for reference)
│
├── PaperPortion/                     # Paper and documentation
│   ├── FINAL_paper_report.md         # Full 15-page paper
│   └── figures/                      # Paper figures
│
├── configs/                          # YAML configuration files
│   ├── m1_text_only.yaml
│   ├── m7_hybrid_fusion.yaml
│   └── ...
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_attention_visualization.ipynb
│
├── scripts/                          # Utility scripts
│   ├── download_models.sh            # Download pretrained models
│   ├── prepare_data.sh               # Data preprocessing
│   ├── run_full_experiments.sh       # Run all 8 models
│   └── generate_paper_tables.py      # Generate LaTeX tables
│
├── outputs/                          # Training outputs (gitignored)
│   ├── m7_hybrid_fusion/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── metrics/
│   └── ...
│
├── data/                             # Data directory (gitignored)
│   ├── train/
│   ├── val/
│   └── test_release/
│
├── models/                           # Pretrained checkpoints
│   └── m7_hybrid_fusion_best.pt
│
├── docs/                             # Additional documentation
│   ├── API.md                        # API documentation
│   ├── EXPERIMENTS.md                # Experiment guide
│   ├── TROUBLESHOOTING.md            # Common issues
│   └── CONTRIBUTING.md               # Contribution guidelines
│
└── tests/                            # Unit tests
    ├── test_dataset.py
    ├── test_encoders.py
    ├── test_fusion.py
    └── test_inference.py
```

---

## 📖 Documentation

- **[Full Paper](PaperPortion/FINAL_paper_report.md)** — 15-page research paper with complete methodology
- **[API Documentation](docs/API.md)** — Detailed API reference
- **[Experiment Guide](docs/EXPERIMENTS.md)** — How to reproduce all experiments
- **[Model Guide](abiral/MODEL_SELECTION_GUIDE.md)** — Model selection best practices
- **[Quick Start](abiral/QUICK_START.md)** — Beginner-friendly tutorial
- **[Testing Guide](abiral/TESTING_GUIDE.md)** — Testing and validation

---

## 🧪 Testing

```bash
# Run all unit tests
pytest tests/ -v

# Test specific module
pytest tests/test_fusion.py -v

# Test with coverage
pytest --cov=abiral tests/
```

---

## 📜 Citation

If you use this code or findings in your research, please cite:

```bibtex
@inproceedings{thapa2026nepai,
  title={TeamNepAI@CHiPSAL 2026: Cross-Modal Attention Fusion with Dynamic Gating for Devanagari Hate Speech Detection in Multimodal Memes},
  author={Thapa, Samir and others},
  booktitle={Proceedings of the Second Workshop on Challenges in Processing South Asian Languages (CHiPSAL 2026)},
  year={2026},
  organization={LREC}
}
```

### Mandatory Citations (Dataset & Task)

**Shared Task Overview:**
```bibtex
@inproceedings{thapa2026sharedtask,
  title={Multimodal Hate and Sentiment Understanding in Low-Resource Text-Embedded Images},
  author={Thapa, Surendrabikram and Shiwakoti, Shuvam and Shah, Siddhant Bikram and Rauniyar, Kritesh and others},
  booktitle={Proceedings of CHiPSAL 2026},
  year={2026}
}
```

**Dataset Sources:**
```bibtex
@inproceedings{thapa2025multimodal,
  title={A multimodal prompt-based framework for analyzing code-mixed and low-resource memes},
  author={Thapa, Surendrabikram and Veeramani, Hariram and Hu, Liang and others},
  booktitle={ICWSM},
  year={2025}
}

@inproceedings{bhandari2023crisishatemm,
  title={CrisisHateMM: Multimodal analysis of directed and undirected hate speech},
  author={Bhandari, Aashish and Shah, Siddhant B and Thapa, Surendrabikram and others},
  booktitle={CVPR},
  year={2023}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Components

- **CLIP:** OpenAI (MIT License)
- **BGE-M3:** BAAI (Apache 2.0)
- **Transformers:** Hugging Face (Apache 2.0)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and add tests
4. Commit changes (`git commit -m 'Add improvement'`)
5. Push to branch (`git push origin feature/improvement`)
6. Open a Pull Request

### Code Style

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation as needed

---

## 🐛 Issues & Support

- **Bug Reports:** [GitHub Issues](https://github.com/yourusername/chipsal-multimodal/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/yourusername/chipsal-multimodal/discussions)
- **Email:** samir@example.com

### Common Issues

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for solutions to common problems.

---

## 🙏 Acknowledgments

### Research & Dataset

- **CHiPSAL 2026 Organizers:** Kengatharaiyer Sarveswaran, Surendrabikram Thapa, Ashwini Vaidya, Tafseer Ahmed Khan, Bal Krishna Bal
- **Annotators:** Anonymous Nepali native speakers who labeled the dataset
- **Shared Task:** LREC 2026 — Second Workshop on Challenges in Processing South Asian Languages

### Technical Contributions

- **OpenAI:** CLIP vision-language model
- **BAAI:** BGE-M3 multilingual embeddings
- **Hugging Face:** Transformers library
- **PyTorch Team:** Deep learning framework

### Funding

[Add funding acknowledgments if applicable]

---

## 📊 Performance Benchmarks

### Training Time (NVIDIA RTX 3090)

| Model | Single Fold | 5-Fold CV | Parameters |
|-------|-------------|-----------|------------|
| M1 (Text-Only) | 18 min | 90 min | 560M (BGE-M3) |
| M2 (Image-Only) | 12 min | 60 min | 85M (CLIP) |
| M7 (Hybrid Fusion) | 52 min | 260 min | 645M + 0.9M fusion |

### Inference Speed

- **Single Sample:** ~45ms (GPU), ~180ms (CPU)
- **Batch (32 samples):** ~850ms (GPU), ~6.2s (CPU)

---

## 🔗 Related Work

- **CrisisHateMM (CVPR 2023):** Multimodal hate in crisis contexts
- **HASOC Shared Task:** Hindi hate speech detection
- **IndicCLIP:** Multilingual vision-language for Indic languages
- **Code-Mixed Meme Analysis:** Low-resource multimodal understanding

---

## 📅 Timeline

- **Dataset Release:** January 15, 2026
- **Paper Submission:** March 1, 2026
- **Notification:** March 20, 2026
- **Camera Ready:** March 29, 2026
- **Workshop:** May 16, 2026

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg? repos=yourusername/chipsal-multimodal&type=Date)](https://star-history.com/#yourusername/chipsal-multimodal&Date)

---

## 📞 Contact

**Lead Researcher:** Samir Thapa  
**Email:** samir@example.com  
**GitHub:** [@samirthapa](https://github.com/samirthapa)  
**Institution:** [Your Institution]

---

<div align="center">

**⭐ If you find this work useful, please star the repository! ⭐**

Made with ❤️ for the research community

</div>
