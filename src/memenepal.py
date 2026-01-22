"""
CHiPSAL 2026 - MemeNePAL: The Baseline to Beat
==============================================
Implementation of MemeNePAL (NeMeme paper, ICWSM 2025)

This is the COMPETITION BASELINE that achieves:
- Hate Detection F1: 0.6785
- Accuracy: 0.7259

Architecture: Prompt-based multimodal framework
- Uses prompt-assisted learning
- Combines image captions with text analysis
- Specifically designed for Nepali memes

Reference:
Thapa et al., "A Multimodal Prompt-based Framework for Analyzing 
Code-Mixed and Low-Resource Memes", ICWSM 2025

IMPORTANT: To win the competition, you need to BEAT F1 > 0.6785
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BlipProcessor, BlipForConditionalGeneration,
)
from PIL import Image
import timm
from pathlib import Path
from typing import Optional, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import *


# =============================================================================
# COMPETITION BASELINE RESULTS (From the paper)
# =============================================================================
"""
HATE DETECTION RESULTS FROM MEMENEPAL PAPER:

| Model                      | Precision | Recall | F1-score | Accuracy |
|----------------------------|-----------|--------|----------|----------|
| mBERT (text only)          | 0.5734    | 0.5912 | 0.5714   | 0.6322   |
| Nepali-BERT (text only)    | 0.5887    | 0.6143 | 0.5854   | 0.7138   |
| VGG-16 (visual only)       | 0.5535    | 0.5618 | 0.5542   | 0.6367   |
| ResNet-50 (visual only)    | 0.5430    | 0.5415 | 0.5368   | 0.6017   |
| VGG-16 + NepaliBERT        | 0.5798    | 0.6154 | 0.5787   | 0.7064   |
| Pro-Cap                    | 0.5991    | 0.6198 | 0.6093   | 0.7113   |
| LLaVA                      | 0.5125    | 0.5491 | 0.4892   | 0.6157   |
| MemeNePAL (BASELINE)       | 0.6914    | 0.7257 | 0.6785   | 0.7259   |

TARGET: Beat F1 > 0.6785
"""


# =============================================================================
# MEMENEPAL ARCHITECTURE
# =============================================================================

class MemeNePAL(nn.Module):
    """
    MemeNePAL: Multimodal Prompt-based Framework for Nepali Memes
    
    Key Innovation: Uses prompt-based learning to combine:
    1. Image understanding via generated captions
    2. OCR text from the meme
    3. Prompt engineering for classification
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                        MEME INPUT                           │
    │  ┌─────────────┐              ┌─────────────────────────┐  │
    │  │   Image     │              │   OCR Text (Nepali)     │  │
    │  └──────┬──────┘              └───────────┬─────────────┘  │
    │         │                                 │                │
    │         ▼                                 │                │
    │  ┌─────────────────┐                      │                │
    │  │ Image Captioner │                      │                │
    │  │ (BLIP / ViT-GPT)│                      │                │
    │  └────────┬────────┘                      │                │
    │           │ English Caption               │                │
    │           ▼                               ▼                │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │              PROMPT CONSTRUCTION                     │  │
    │  │                                                      │  │
    │  │  "The image shows: {caption}.                       │  │
    │  │   The text in the meme says: {ocr_text}.            │  │
    │  │   Based on this, is the meme hateful?"              │  │
    │  └─────────────────────────────────────────────────────┘  │
    │                          │                                 │
    │                          ▼                                 │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │           TEXT ENCODER (Nepali-BERT/MuRIL)          │  │
    │  └─────────────────────────────────────────────────────┘  │
    │                          │                                 │
    │                          ▼                                 │
    │  ┌─────────────────────────────────────────────────────┐  │
    │  │                 CLASSIFIER                           │  │
    │  │            [Non-Hate: 0, Hate: 1]                   │  │
    │  └─────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        text_encoder="google/muril-base-cased",
        use_caption_model=True,
        caption_model="Salesforce/blip-image-captioning-base",
        num_classes=NUM_CLASSES,
        dropout=0.3,
        freeze_caption_model=True
    ):
        super().__init__()
        
        print(f"\n{'='*60}")
        print("MemeNePAL: Prompt-based Multimodal Framework")
        print(f"{'='*60}")
        
        self.use_caption_model = use_caption_model
        
        # =====================================================================
        # IMAGE CAPTIONING MODEL (Optional, can use pre-computed captions)
        # =====================================================================
        if use_caption_model:
            print(f"[Caption] Loading {caption_model}...")
            self.caption_processor = BlipProcessor.from_pretrained(caption_model)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                caption_model,
                torch_dtype=torch.float16
            )
            
            if freeze_caption_model:
                for param in self.caption_model.parameters():
                    param.requires_grad = False
            
            print("[Caption] Model loaded (frozen)")
        
        # =====================================================================
        # TEXT ENCODER (For prompt understanding)
        # =====================================================================
        print(f"[Text] Loading {text_encoder}...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze early layers
        if hasattr(self.text_encoder, 'encoder'):
            for layer in list(self.text_encoder.encoder.layer)[:-2]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print(f"[Text] Hidden dim: {text_dim}")
        
        # =====================================================================
        # VISION ENCODER (For additional visual features)
        # =====================================================================
        print("[Vision] Loading ResNet-50...")
        self.vision_encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
        vision_dim = self.vision_encoder.num_features
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # =====================================================================
        # FUSION AND CLASSIFICATION
        # =====================================================================
        self.text_proj = nn.Linear(text_dim, 512)
        self.vision_proj = nn.Linear(vision_dim, 512)
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
        # Prompt templates
        self.prompt_template = (
            "Image description: {caption}. "
            "Text in meme: {ocr_text}. "
            "Analyze if this content is hateful."
        )
        
        print(f"{'='*60}\n")
    
    @torch.no_grad()
    def generate_caption(self, pixel_values):
        """Generate caption for image using BLIP."""
        if not self.use_caption_model:
            return [""] * pixel_values.shape[0]
        
        # Generate captions
        generated_ids = self.caption_model.generate(
            pixel_values=pixel_values,
            max_length=50
        )
        
        captions = self.caption_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )
        
        return captions
    
    def create_prompt(self, caption: str, ocr_text: str) -> str:
        """Create prompt from caption and OCR text."""
        return self.prompt_template.format(
            caption=caption,
            ocr_text=ocr_text if ocr_text else "No text detected"
        )
    
    def forward(
        self, 
        pixel_values,
        input_ids=None,
        attention_mask=None,
        ocr_texts=None,
        captions=None,
        **kwargs
    ):
        """
        Forward pass.
        
        Args:
            pixel_values: Image tensor [B, 3, H, W]
            input_ids: Pre-tokenized prompt (optional)
            attention_mask: Attention mask (optional)
            ocr_texts: List of OCR texts (optional)
            captions: Pre-computed captions (optional)
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Get vision features
        vision_feat = self.vision_encoder(pixel_values)
        vision_proj = self.vision_proj(vision_feat)
        
        # If pre-tokenized input is not provided, create prompts
        if input_ids is None:
            # Generate captions if not provided
            if captions is None:
                if self.use_caption_model:
                    captions = self.generate_caption(pixel_values)
                else:
                    captions = [""] * batch_size
            
            # Create prompts
            if ocr_texts is None:
                ocr_texts = [""] * batch_size
            
            prompts = [
                self.create_prompt(cap, ocr) 
                for cap, ocr in zip(captions, ocr_texts)
            ]
            
            # Tokenize
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
        
        # Encode prompt
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_output.last_hidden_state[:, 0, :]  # [CLS]
        text_proj = self.text_proj(text_feat)
        
        # Fuse
        combined = torch.cat([vision_proj, text_proj], dim=-1)
        fused = self.fusion(combined)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


# =============================================================================
# ENHANCED MEMENEPAL (To beat the baseline)
# =============================================================================

class MemeNePALEnhanced(nn.Module):
    """
    Enhanced MemeNePAL with improvements to beat the baseline:
    
    Improvements over original:
    1. Cross-attention fusion (not just concatenation)
    2. Multi-scale vision features
    3. Prompt ensemble (multiple prompts)
    4. Hate keyword features
    
    Target: F1 > 0.6785
    """
    
    def __init__(
        self,
        text_encoder="google/muril-base-cased",
        vision_encoder="efficientnet_b3",
        num_classes=NUM_CLASSES,
        dropout=0.3,
        use_keyword_features=True
    ):
        super().__init__()
        
        print(f"\n{'='*60}")
        print("MemeNePAL ENHANCED - Designed to Beat Baseline")
        print(f"{'='*60}")
        
        # =====================================================================
        # TEXT ENCODER
        # =====================================================================
        print(f"[Text] Loading {text_encoder}...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        text_dim = self.text_encoder.config.hidden_size
        
        # Fine-tune last 3 layers
        if hasattr(self.text_encoder, 'encoder'):
            for layer in list(self.text_encoder.encoder.layer)[:-3]:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # =====================================================================
        # VISION ENCODER (Multi-scale)
        # =====================================================================
        print(f"[Vision] Loading {vision_encoder}...")
        self.vision_encoder = timm.create_model(
            vision_encoder, 
            pretrained=True, 
            num_classes=0,
            features_only=True  # Get multi-scale features
        )
        
        # Get feature dimensions at different scales
        dummy = torch.randn(1, 3, 224, 224)
        features = self.vision_encoder(dummy)
        self.vision_dims = [f.shape[1] for f in features]
        
        # Use last two scales
        self.vision_proj1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.vision_dims[-1], 256)
        )
        self.vision_proj2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.vision_dims[-2], 256)
        )
        
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # =====================================================================
        # PROJECTIONS
        # =====================================================================
        self.hidden_dim = 512
        self.text_proj = nn.Linear(text_dim, self.hidden_dim)
        
        # =====================================================================
        # CROSS-ATTENTION FUSION
        # =====================================================================
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # =====================================================================
        # KEYWORD FEATURES
        # =====================================================================
        self.use_keyword_features = use_keyword_features
        self.hate_keywords = [
            # Nepali hate speech indicators
            "रण्डी", "छाडा", "मूर्ख", "गधा", "कुकुर",
            "दलित", "देशद्रोही", "गद्दार", "भ्रष्ट",
            # Transliterated
            "randi", "sala", "muji", "chikni"
        ]
        
        if use_keyword_features:
            self.keyword_proj = nn.Linear(len(self.hate_keywords), 64)
            fusion_input_dim = self.hidden_dim + 512 + 64  # text + vision + keywords
        else:
            fusion_input_dim = self.hidden_dim + 512
        
        # =====================================================================
        # CLASSIFICATION HEAD
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Multiple prompt templates for ensemble
        self.prompts = [
            "मेम विश्लेषण: {text}. यो घृणित छ वा छैन?",  # Nepali
            "Meme text: {text}. Is this hateful content?",  # English
            "Text: {text}. Classify as hate or non-hate.",  # Simple
        ]
        
        print(f"[Fusion] Cross-attention + Keyword features")
        print(f"[Prompts] {len(self.prompts)} prompt templates")
        print(f"{'='*60}\n")
    
    def detect_keywords(self, texts):
        """Detect hate keywords in text."""
        batch_size = len(texts)
        features = torch.zeros(batch_size, len(self.hate_keywords))
        
        for i, text in enumerate(texts):
            text_lower = str(text).lower()
            for j, kw in enumerate(self.hate_keywords):
                if kw.lower() in text_lower:
                    features[i, j] = 1.0
        
        return features
    
    def forward(
        self,
        pixel_values,
        input_ids=None,
        attention_mask=None,
        texts=None,
        **kwargs
    ):
        """Forward pass with enhanced fusion."""
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # =====================================================================
        # VISION FEATURES (Multi-scale)
        # =====================================================================
        vision_features = self.vision_encoder(pixel_values)
        
        feat1 = self.vision_proj1(vision_features[-1])  # [B, 256]
        feat2 = self.vision_proj2(vision_features[-2])  # [B, 256]
        vision_combined = torch.cat([feat1, feat2], dim=-1)  # [B, 512]
        
        # =====================================================================
        # TEXT FEATURES
        # =====================================================================
        if input_ids is None and texts is not None:
            # Use first prompt template for now
            prompts = [self.prompts[0].format(text=t) for t in texts]
            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
        
        text_output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get all token embeddings for attention
        text_hidden = text_output.last_hidden_state  # [B, seq, dim]
        text_cls = text_hidden[:, 0, :]  # [CLS] token
        text_proj = self.text_proj(text_cls)  # [B, hidden_dim]
        text_tokens = self.text_proj(text_hidden)  # [B, seq, hidden_dim]
        
        # =====================================================================
        # CROSS-ATTENTION (Vision attends to Text)
        # =====================================================================
        vision_query = text_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        attn_output, attn_weights = self.cross_attn(
            query=vision_query,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        text_attended = attn_output.squeeze(1)  # [B, hidden_dim]
        
        # =====================================================================
        # KEYWORD FEATURES
        # =====================================================================
        if self.use_keyword_features and texts is not None:
            keyword_feat = self.detect_keywords(texts).to(device)
            keyword_proj = self.keyword_proj(keyword_feat)
            combined = torch.cat([text_attended, vision_combined, keyword_proj], dim=-1)
        else:
            combined = torch.cat([text_attended, vision_combined], dim=-1)
        
        # =====================================================================
        # CLASSIFICATION
        # =====================================================================
        logits = self.classifier(combined)
        
        return logits


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_memenepal_model(
    enhanced=True,
    text_encoder="google/muril-base-cased",
    **kwargs
):
    """
    Create MemeNePAL model.
    
    Args:
        enhanced: If True, use enhanced version designed to beat baseline
        text_encoder: Text encoder to use
    """
    if enhanced:
        model = MemeNePALEnhanced(
            text_encoder=text_encoder,
            **kwargs
        )
    else:
        model = MemeNePAL(
            text_encoder=text_encoder,
            use_caption_model=False,  # Simpler version without captioning
            **kwargs
        )
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")
    
    return model


# =============================================================================
# WHAT YOU NEED TO BEAT
# =============================================================================
"""
COMPETITION TARGET:

MemeNePAL Baseline Results (Hate Detection):
- Precision: 0.6914
- Recall: 0.7257  
- F1-score: 0.6785  ← BEAT THIS!
- Accuracy: 0.7259

STRATEGIES TO WIN:
1. Use MemeNePALEnhanced (this file)
2. Ensemble multiple models
3. Augment training data
4. Use external Nepali hate speech data
5. Fine-tune on domain-specific data
6. Hyperparameter optimization
"""


if __name__ == "__main__":
    print("Testing MemeNePAL Models...\n")
    
    # Test enhanced model
    model = create_memenepal_model(enhanced=True)
    
    # Dummy inputs
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_texts = ["यो मेम हो", "This is a test meme"]
    
    output = model(
        pixel_values=dummy_image,
        texts=dummy_texts
    )
    
    print(f"\n[TEST] Output shape: {output.shape}")
    print("[SUCCESS] MemeNePAL works!")
    print(f"\n[TARGET] Beat F1 > 0.6785 to win!")
