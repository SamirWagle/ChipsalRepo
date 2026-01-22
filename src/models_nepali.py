"""
CHiPSAL 2026 - Subtask A: Nepali-Optimized Models
================================================
Models specifically designed for LOW-RESOURCE Nepali language.

Key Insight: 
- Vision encoders (CLIP, SigLIP) are language-agnostic for images
- Text encoders MUST be Nepali-aware for best results

Best Text Encoders for Nepali (ranked):
1. NepBERTa - Native Nepali BERT (BEST)
2. MuRIL - Google's Indian Languages model (includes Nepali)
3. XLM-RoBERTa - Multilingual (has some Nepali)
4. Nepali-RoBERTa - IRIIS 2024 model

References:
- NepBERTa: https://aclanthology.org/2022.findings-aacl.41/
- MuRIL: https://huggingface.co/google/muril-base-cased
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification
)
import timm
from pathlib import Path
from typing import Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import *


# =============================================================================
# NEPALI-SPECIFIC TEXT ENCODERS
# =============================================================================

NEPALI_TEXT_MODELS = {
    # Best: Native Nepali models
    "nepberta": "Shushant/NepBERTa",
    "nepali_roberta": "IRIIS/Nepali-RoBERTa-110M",
    
    # Good: Indian language models (include Nepali)
    "muril": "google/muril-base-cased",
    "muril_large": "google/muril-large-cased",
    
    # Fallback: Multilingual models
    "xlm_roberta": "xlm-roberta-base",
    "xlm_roberta_large": "xlm-roberta-large",
    "mbert": "bert-base-multilingual-cased",
}


def get_nepali_tokenizer_and_model(model_key="muril"):
    """
    Load the best available Nepali text encoder.
    
    Recommendation order:
    1. NepBERTa (if available) - trained on 0.8B Nepali words
    2. MuRIL - explicitly supports Nepali
    3. XLM-RoBERTa - has Nepali in training data
    """
    model_name = NEPALI_TEXT_MODELS.get(model_key, model_key)
    
    print(f"[Nepali] Loading text encoder: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        print(f"[Nepali] Successfully loaded {model_key}")
        return tokenizer, model
    except Exception as e:
        print(f"[WARNING] Failed to load {model_key}: {e}")
        print("[Nepali] Falling back to XLM-RoBERTa...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModel.from_pretrained("xlm-roberta-base")
        return tokenizer, model


# =============================================================================
# NEPALI-OPTIMIZED MULTIMODAL MODEL
# =============================================================================

class NepaliMemeClassifier(nn.Module):
    """
    Multimodal classifier optimized for Nepali memes.
    
    Architecture:
    - Vision: SigLIP/EVA-CLIP (language-agnostic, works universally)
    - Text: MuRIL or NepBERTa (Nepali-specific)
    - Fusion: Cross-attention for image-text interaction
    
    This is the RECOMMENDED model for CHiPSAL 2026.
    """
    
    def __init__(
        self,
        vision_model="resnet50",
        text_model="muril",  # Use MuRIL for Nepali
        num_classes=NUM_CLASSES,
        fusion_type="cross_attention",
        dropout=0.3,
        freeze_vision=True,
        freeze_text_layers=10  # Freeze first N layers of text encoder
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        print(f"\n{'='*60}")
        print("NEPALI MEME CLASSIFIER")
        print(f"{'='*60}")
        
        # =====================================================================
        # VISION ENCODER (Language-agnostic)
        # =====================================================================
        print(f"[Vision] Loading {vision_model}...")
        self.vision_encoder = timm.create_model(
            vision_model,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        vision_dim = self.vision_encoder.num_features
        
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            # Unfreeze last layer for fine-tuning
            if hasattr(self.vision_encoder, 'fc'):
                for param in self.vision_encoder.fc.parameters():
                    param.requires_grad = True
        
        print(f"[Vision] Output dim: {vision_dim}")
        
        # =====================================================================
        # NEPALI TEXT ENCODER
        # =====================================================================
        self.tokenizer, self.text_encoder = get_nepali_tokenizer_and_model(text_model)
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze early layers, fine-tune later layers
        if hasattr(self.text_encoder, 'encoder'):
            layers = list(self.text_encoder.encoder.layer)
            for i, layer in enumerate(layers):
                if i < freeze_text_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        print(f"[Text] Output dim: {text_dim}")
        print(f"[Text] Frozen layers: {freeze_text_layers}")
        
        # =====================================================================
        # PROJECTION LAYERS
        # =====================================================================
        self.hidden_dim = 512
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # =====================================================================
        # FUSION LAYER
        # =====================================================================
        if fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.text_to_vision_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            fusion_dim = self.hidden_dim * 2
            
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
            fusion_dim = self.hidden_dim
            
        else:  # concat
            fusion_dim = self.hidden_dim * 2
        
        # =====================================================================
        # CLASSIFICATION HEAD
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        print(f"[Fusion] Type: {fusion_type}")
        print(f"[Fusion] Output dim: {fusion_dim}")
        print(f"{'='*60}\n")
    
    def encode_text(self, input_ids, attention_mask):
        """Encode Nepali text."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Also return all token embeddings for attention
        all_tokens = outputs.last_hidden_state
        
        return cls_output, all_tokens
    
    def encode_image(self, image):
        """Encode image."""
        features = self.vision_encoder(image)
        return features
    
    def forward(self, image, input_ids, attention_mask, **kwargs):
        """
        Forward pass.
        
        Args:
            image: [B, 3, H, W] - Meme image
            input_ids: [B, seq_len] - Tokenized Nepali text
            attention_mask: [B, seq_len] - Attention mask
        """
        # Encode modalities
        vision_feat = self.encode_image(image)  # [B, vision_dim]
        text_cls, text_tokens = self.encode_text(input_ids, attention_mask)
        
        # Project to shared dimension
        vision_proj = self.vision_proj(vision_feat)  # [B, hidden_dim]
        text_proj = self.text_proj(text_cls)  # [B, hidden_dim]
        text_tokens_proj = self.text_proj(text_tokens)  # [B, seq_len, hidden_dim]
        
        # Fusion
        if self.fusion_type == "cross_attention":
            # Vision attends to text tokens
            vision_expanded = vision_proj.unsqueeze(1)  # [B, 1, hidden_dim]
            
            # Cross-attention: vision queries text
            attn_v2t, _ = self.cross_attn(
                query=vision_expanded,
                key=text_tokens_proj,
                value=text_tokens_proj,
                key_padding_mask=~attention_mask.bool()
            )
            
            # Cross-attention: text queries vision
            text_expanded = text_proj.unsqueeze(1)  # [B, 1, hidden_dim]
            attn_t2v, _ = self.text_to_vision_attn(
                query=text_expanded,
                key=vision_expanded,
                value=vision_expanded
            )
            
            # Combine
            fused = torch.cat([
                attn_v2t.squeeze(1),
                attn_t2v.squeeze(1)
            ], dim=-1)
            
        elif self.fusion_type == "gated":
            concat = torch.cat([vision_proj, text_proj], dim=-1)
            gate = self.gate(concat)
            fused = gate * vision_proj + (1 - gate) * text_proj
            
        else:  # concat
            fused = torch.cat([vision_proj, text_proj], dim=-1)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits


# =============================================================================
# ENHANCED MODEL WITH NEPALI HATE SPEECH KEYWORDS
# =============================================================================

class NepaliHateSpeechClassifier(NepaliMemeClassifier):
    """
    Enhanced classifier with Nepali hate speech keyword detection.
    
    Adds explicit hate speech keyword matching as an auxiliary feature.
    Common Nepali hate speech patterns are detected and used as additional signal.
    """
    
    # Common hate speech keywords/patterns in Nepali (derogatory terms, slurs)
    # Note: These are for detection purposes only
    HATE_INDICATORS = [
        # Gender-based
        "रण्डी", "छाडा", "बेश्या",
        # Caste-based
        "दलित", "अछुत", "डोम",
        # Religious
        "म्लेच्छ", "काफिर",
        # Political
        "देशद्रोही", "गद्दार",
        # General slurs
        "मूर्ख", "गधा", "कुकुर",
    ]
    
    def __init__(self, *args, use_keyword_features=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_keyword_features = use_keyword_features
        
        if use_keyword_features:
            # Additional projection for keyword features
            self.keyword_proj = nn.Linear(len(self.HATE_INDICATORS), 64)
            
            # Modify classifier to include keyword features
            old_in_features = self.classifier[2].in_features
            self.classifier[2] = nn.Linear(old_in_features + 64, old_in_features // 2)
    
    def detect_keywords(self, texts):
        """
        Detect hate speech keywords in text.
        Returns binary vector indicating presence of each keyword.
        """
        batch_size = len(texts)
        keyword_features = torch.zeros(batch_size, len(self.HATE_INDICATORS))
        
        for i, text in enumerate(texts):
            for j, keyword in enumerate(self.HATE_INDICATORS):
                if keyword in str(text):
                    keyword_features[i, j] = 1.0
        
        return keyword_features
    
    def forward(self, image, input_ids, attention_mask, raw_texts=None, **kwargs):
        """Forward with optional keyword detection."""
        # Get base features
        vision_feat = self.encode_image(image)
        text_cls, text_tokens = self.encode_text(input_ids, attention_mask)
        
        vision_proj = self.vision_proj(vision_feat)
        text_proj = self.text_proj(text_cls)
        text_tokens_proj = self.text_proj(text_tokens)
        
        # Fusion (same as parent)
        if self.fusion_type == "cross_attention":
            vision_expanded = vision_proj.unsqueeze(1)
            attn_v2t, _ = self.cross_attn(
                query=vision_expanded,
                key=text_tokens_proj,
                value=text_tokens_proj,
                key_padding_mask=~attention_mask.bool()
            )
            text_expanded = text_proj.unsqueeze(1)
            attn_t2v, _ = self.text_to_vision_attn(
                query=text_expanded,
                key=vision_expanded,
                value=vision_expanded
            )
            fused = torch.cat([attn_v2t.squeeze(1), attn_t2v.squeeze(1)], dim=-1)
        else:
            fused = torch.cat([vision_proj, text_proj], dim=-1)
        
        # Add keyword features if enabled
        if self.use_keyword_features and raw_texts is not None:
            keyword_feat = self.detect_keywords(raw_texts).to(fused.device)
            keyword_proj = self.keyword_proj(keyword_feat)
            
            # Go through classifier layers manually
            x = self.classifier[0](fused)  # LayerNorm
            x = self.classifier[1](x)  # Dropout
            x = torch.cat([x, keyword_proj], dim=-1)
            x = self.classifier[2](x)  # Linear with extended input
            x = self.classifier[3](x)  # GELU
            x = self.classifier[4](x)  # Dropout
            logits = self.classifier[5](x)  # Final Linear
        else:
            logits = self.classifier(fused)
        
        return logits


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_nepali_model(
    model_type="nepali_multimodal",
    vision_model="resnet50",
    text_model="muril",
    fusion_type="cross_attention",
    num_classes=NUM_CLASSES,
    **kwargs
):
    """
    Create Nepali-optimized model.
    
    Args:
        model_type: 'nepali_multimodal' or 'nepali_hate_detector'
        vision_model: Vision encoder (resnet50, efficientnet_b3, vit_base_patch16_224)
        text_model: Nepali text encoder (muril, nepberta, xlm_roberta)
        fusion_type: 'cross_attention', 'gated', 'concat'
    """
    
    if model_type == "nepali_multimodal":
        model = NepaliMemeClassifier(
            vision_model=vision_model,
            text_model=text_model,
            fusion_type=fusion_type,
            num_classes=num_classes,
            **kwargs
        )
    
    elif model_type == "nepali_hate_detector":
        model = NepaliHateSpeechClassifier(
            vision_model=vision_model,
            text_model=text_model,
            fusion_type=fusion_type,
            num_classes=num_classes,
            use_keyword_features=True,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")
    
    return model


# =============================================================================
# RECOMMENDED CONFIGURATIONS
# =============================================================================

RECOMMENDED_CONFIGS = {
    "best_accuracy": {
        "model_type": "nepali_multimodal",
        "vision_model": "efficientnet_b3",
        "text_model": "muril",  # Best for Nepali
        "fusion_type": "cross_attention",
    },
    "balanced": {
        "model_type": "nepali_multimodal",
        "vision_model": "resnet50",
        "text_model": "muril",
        "fusion_type": "cross_attention",
    },
    "fast": {
        "model_type": "nepali_multimodal",
        "vision_model": "resnet34",
        "text_model": "xlm_roberta",
        "fusion_type": "concat",
    },
    "with_keywords": {
        "model_type": "nepali_hate_detector",
        "vision_model": "resnet50",
        "text_model": "muril",
        "fusion_type": "cross_attention",
    }
}


if __name__ == "__main__":
    print("Testing Nepali-Optimized Models...\n")
    
    # Test model creation
    model = create_nepali_model(**RECOMMENDED_CONFIGS["balanced"])
    
    # Test forward pass
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 128))
    dummy_attention_mask = torch.ones(batch_size, 128)
    
    output = model(
        image=dummy_image,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask
    )
    
    print(f"\n[TEST] Input image shape: {dummy_image.shape}")
    print(f"[TEST] Output shape: {output.shape}")
    print("[SUCCESS] Model works correctly!")
