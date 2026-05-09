# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-01

### Added - Initial Release

#### Core Features
- **Hybrid Cross-Modal Attention Fusion (M7)** - Primary system achieving 0.6834 F1-macro
- **8 Model Configurations** - Complete ablation study (M1-M8)
- **5-Fold Cross-Validation** - Stratified CV for robust evaluation
- **CLIP ViT-B/32** - Visual encoder with selective fine-tuning
- **BGE-M3** - Multilingual text encoder with Devanagari support
- **Dynamic Gating Network** - Sample-specific modality weighting
- **Class-Weighted Loss** - Frequency-inverse weighting for imbalanced data

####

 Models
- `M1: Text-Only (BGE-M3)` - Text-only baseline (F1: 0.6455)
- `M2: Image-Only (CLIP)` - Image-only baseline (F1: 0.5952)
- `M3: Image-Only (Text-Removed)` - Pure visual baseline (F1: 0.5535)
- `M4: Early Fusion` - Concatenation fusion (F1: 0.6140)
- `M5: Late Fusion (Voting)` - Soft voting ensemble (F1: 0.6511)
- `M6: Late Fusion (Bagging)` - Bagging ensemble k=3 (F1: 0.6267)
- `M7: Hybrid Fusion` - Cross-modal attention + gating (F1: 0.6834) ⭐
- `M8: Hybrid (Text-Removed)` - Hybrid with text-removed images (F1: 0.6436)

#### Documentation
- Professional README with architecture diagrams
- Complete API documentation (`docs/API.md`)
- Experiment reproduction guide (`docs/EXPERIMENTS.md`)
- Troubleshooting guide (`docs/TROUBLESHOOTING.md`)
- Contributing guidelines (`docs/CONTRIBUTING.md`)
- Code of Conduct with research ethics (`docs/CODE_OF_CONDUCT.md`)

#### Configuration
- YAML configs for all 8 models (`configs/m*.yaml`)
- Pip-installable package (`setup.py`)
- Citation file for GitHub/Zenodo (`CITATION.cff`)
- MIT License with ethics guidelines
- Comprehensive `.gitignore`

#### Scripts
- `scripts/run_all_experiments.py` - Run full ablation study
- `scripts/prepare_data.py` - Data validation and preprocessing
- `scripts/generate_paper_tables.py` - LaTeX table generation

#### Training
- Mixed precision training (AMP)
- Early stopping with patience
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping
- Label smoothing
- Data augmentation (Albumentations)

#### Evaluation
- Macro-averaged F1 (primary metric)
- Confusion matrices
- Per-class metrics
- Attention visualization
- Gating weight analysis

#### Data Support
- CHiPSAL 2026 Shared Task dataset
- Subtask A (binary classification)
- 1,068 Nepali memes
- Devanagari script support
- OCR-extracted text

### Paper
- **Title:** "TeamNepAI@CHiPSAL 2026: Cross-Modal Attention Fusion with Dynamic Gating for Devanagari Hate Speech Detection in Multimodal Memes"
- **Conference:** CHiPSAL 2026 @ LREC 2026
- **Submission Date:** March 1, 2026

### Known Limitations
- CLIP has limited Devanagari script understanding (16.6% gap vs text-only)
- Small dataset (N=854/fold) limits ensemble effectiveness
- Bagging underperforms single best model due to data scarcity
- Text modality dominates fusion (g_text ≈ 0.73 mean)

### Dependencies
- Python >= 3.10
- PyTorch 2.0.1
- Transformers 4.30.2
- CUDA 11.8 (optional, for GPU)
- See `requirements.txt` for complete list

---

## [Unreleased]

### TODO
- [ ] Support for Subtask B (3-class sentiment)
- [ ] Hindi/English code-mixed dataset support
- [ ] Additional fusion architectures (Transformer-based)
- [ ] ONNX export for production
- [ ] Web demo interface
- [ ] Pre-trained model weights on Hugging Face Hub
- [ ] Attention visualization tools
- [ ] Bias analysis utilities
- [ ] Multi-GPU training support
- [ ] Docker containerization

### Future Work
- Investigate IndicCLIP for better Devanagari understanding
- Explore prompt-based learning
- Test on other South Asian languages (Hindi, Bengali, Tamil)
- Integrate with content moderation platforms
- Fairness auditing across demographic groups

---

## Version History

- **v1.0.0** (2026-03-01) - Initial release for CHiPSAL 2026 submission

---

## Maintenance

This project is actively maintained. For bug reports and feature requests, please open an issue on [GitHub](https://github.com/yourusername/chipsal-multimodal-hate-detection/issues).

**Maintainer:** Samir Thapa (samir@example.com)

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{thapa2026nepai,
  title={TeamNepAI@CHiPSAL 2026: Cross-Modal Attention Fusion with Dynamic Gating for Devanagari Hate Speech Detection in Multimodal Memes},
  author={Thapa, Samir and others},
  booktitle={Proceedings of CHiPSAL 2026},
  year={2026}
}
```
