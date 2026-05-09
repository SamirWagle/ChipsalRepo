# System Architecture Diagrams

This folder contains 8 Mermaid diagrams visualizing the Cross-Modal Attention Fusion system for Nepali Hate Speech Detection.

## 📊 Diagram Files

### 1. Overall Architecture Flow
**Files:** `1_overall_architecture.mmd` | `1_overall_architecture.png`

Complete end-to-end pipeline showing all 6 layers from input to evaluation, including:
- Input layer (images, text, labels)
- Preprocessing (CLIP transforms, BGE-M3 tokenization)
- Feature encoders (CLIP ViT-B/32, BGE-M3)
- Fusion strategies (8 model configurations)
- Loss & optimization
- Evaluation & results

---

### 2. Model Comparison - Performance Ranking
**Files:** `2_model_comparison.mmd` | `2_model_comparison.png`

Visual comparison of all 8 model configurations with performance metrics and key insights:
- 🥇 M7: Hybrid Fusion (F1: 0.6834) - **BEST MODEL**
- 🥈 M1: Text-Only (F1: 0.6455)
- 🥉 M8: Hybrid Text-Removed (F1: 0.6436)
- And 5 more configurations with insights

---

### 3. M7 Hybrid Fusion - Detailed Architecture
**Files:** `3_m7_hybrid_fusion.mmd` | `3_m7_hybrid_fusion.png`

Deep dive into the best performing model's internal architecture:
- Feature projection (512D/1024D → 256D)
- 4-head cross-modal attention
- Dynamic gate network
- Weighted fusion
- Classification head

---

### 4. Training Pipeline - 5-Fold CV Workflow
**Files:** `4_training_pipeline.mmd` | `4_training_pipeline.png`

Complete training workflow showing:
- 5-fold stratified cross-validation
- Training configuration (AdamW, weighted CE loss, early stopping)
- Metrics aggregation
- Model checkpointing
- Test inference

---

### 5. Fusion Strategy Taxonomy
**Files:** `5_fusion_taxonomy.mmd` | `5_fusion_taxonomy.png`

Hierarchical organization of all fusion approaches:
- **Unimodal baselines:** M1 (text-only), M2 (image-only), M3 (image text-removed)
- **Multimodal fusion:**
  - Early fusion (M4)
  - Late fusion (M5, M6)
  - Hybrid fusion (M7, M8) ⭐

---

### 6. Data Flow - Preprocessing Pipeline
**Files:** `6_data_preprocessing.mmd` | `6_data_preprocessing.png`

Detailed data processing from raw input to model-ready batches:
- CSV and image folder loading
- Image preprocessing (resize, normalize, augmentation)
- Text tokenization (BGE-M3, padding, masking)
- DataLoader configuration

---

### 7. Key Research Findings
**Files:** `7_research_findings.mmd` | `7_research_findings.png`

Research questions mapped to findings and insights:
- ✅ Multimodal superiority (+5.9% F1)
- ✅ Text modality dominance (+16.6% vs image)
- ✅ Hybrid > Late > Early fusion
- ✅ Typography contribution quantified (+5.8%)
- 💡 Cross-modal attention enables adaptive fusion
- 💡 Ensemble fails in extreme low-resource settings

---

### 8. Technical Stack - Implementation
**Files:** `8_technical_stack.mmd` | `8_technical_stack.png`

Complete technology stack visualization:
- **Core framework:** PyTorch 2.0+, Transformers, Scikit-learn
- **Pre-trained models:** CLIP ViT-B/32, BGE-M3, E5-Large
- **Data tools:** Pandas, NumPy, PIL/Pillow
- **Visualization:** Matplotlib, Seaborn
- **Hardware:** NVIDIA GPU (8GB+ VRAM), training time, inference speed

---

## 🎨 File Formats

- **`.mmd` files:** Mermaid diagram source code (editable)
- **`.png` files:** High-resolution PNG exports (1920×1080, white background)

## 🔧 How to Regenerate PNGs

If you need to regenerate the PNG files from the `.mmd` source:

```bash
# Install Mermaid CLI (one-time)
npm install -g @mermaid-js/mermaid-cli

# Convert single diagram
mmdc -i 1_overall_architecture.mmd -o 1_overall_architecture.png -w 1920 -H 1080 -b white

# Convert all diagrams (PowerShell)
Get-ChildItem -Filter *.mmd | ForEach-Object { 
    $name = $_.BaseName; 
    mmdc -i $_.Name -o "$name.png" -w 1920 -H 1080 -b white 
}
```

## 📄 Usage

These diagrams are designed for:
- 📝 Research papers and publications
- 🎤 Conference presentations (CHiPSAL 2026)
- 📚 Documentation and reports
- 🎓 Educational materials
- 💻 Project README files

---

## 🏆 Key Metrics

**Best Model: M7 Hybrid Fusion**
- **F1 Score:** 0.6834
- **Accuracy:** 70.15%
- **Precision:** 0.6817
- **Recall:** 0.7023

**Improvements:**
- +5.9% F1 vs. Text-Only baseline
- +11.3% F1 vs. Early Fusion
- +16.6% F1 (Text vs. Image modality)

---

**Generated:** March 2, 2026  
**Project:** CHiPSAL 2026 Shared Task - TeamNepAI  
**Task:** Subtask A - Multimodal Hate and Sentiment Understanding
