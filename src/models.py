"""
CHiPSAL 2026 - Subtask A: Model Architectures
=============================================
Text, Vision, and Multimodal models for hate speech detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import timm
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


# =============================================================================
# TEXT-ONLY MODELS
# =============================================================================

class TextClassifier(nn.Module):
    """
    Text-only classifier using pre-trained transformer models.
    Suitable for XLM-RoBERTa, mBERT, MuRIL, etc.
    """
    
    def __init__(
        self,
        model_name="xlm-roberta-base",
        num_classes=NUM_CLASSES,
        dropout=0.3,
        freeze_encoder=False
    ):
        super().__init__()
        self.model_name = model_name
        
        # Load pre-trained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        hidden_size = self.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        print(f"[TextClassifier] Model: {model_name}")
        print(f"[TextClassifier] Hidden size: {hidden_size}")
        print(f"[TextClassifier] Num classes: {num_classes}")
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        logits = self.classifier(cls_output)
        return logits
    
    def get_embeddings(self, input_ids, attention_mask):
        """Get text embeddings for multimodal fusion."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]


# =============================================================================
# IMAGE-ONLY MODELS
# =============================================================================

class ImageClassifier(nn.Module):
    """
    Image-only classifier using pre-trained vision models.
    Supports ResNet, EfficientNet, ViT, Swin, etc.
    """
    
    def __init__(
        self,
        model_name="resnet50",
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.3,
        freeze_backbone=False
    ):
        super().__init__()
        self.model_name = model_name
        
        # Load pre-trained vision model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        print(f"[ImageClassifier] Model: {model_name}")
        print(f"[ImageClassifier] Feature dim: {feature_dim}")
        print(f"[ImageClassifier] Num classes: {num_classes}")
    
    def forward(self, image, **kwargs):
        """
        Args:
            image: Image tensor [batch, 3, H, W]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        features = self.backbone(image)
        logits = self.classifier(features)
        return logits
    
    def get_embeddings(self, image):
        """Get image embeddings for multimodal fusion."""
        return self.backbone(image)


# =============================================================================
# MULTIMODAL MODELS
# =============================================================================

class MultimodalFusion(nn.Module):
    """
    Multimodal model combining text and image features.
    Uses late fusion approach.
    """
    
    def __init__(
        self,
        text_model_name="xlm-roberta-base",
        image_model_name="resnet50",
        num_classes=NUM_CLASSES,
        fusion_type="concat",  # 'concat', 'attention', 'gated'
        dropout=0.3,
        freeze_text=False,
        freeze_image=False
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # Text encoder
        self.text_config = AutoConfig.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_config.hidden_size
        
        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Image encoder
        self.image_encoder = timm.create_model(
            image_model_name,
            pretrained=True,
            num_classes=0
        )
        image_dim = self.image_encoder.num_features
        
        if freeze_image:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # Project both to same dimension
        self.hidden_dim = 512
        self.text_proj = nn.Linear(text_dim, self.hidden_dim)
        self.image_proj = nn.Linear(image_dim, self.hidden_dim)
        
        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = self.hidden_dim * 2
        elif fusion_type == 'attention':
            self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=8, batch_first=True)
            fusion_dim = self.hidden_dim
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid()
            )
            fusion_dim = self.hidden_dim
        else:
            fusion_dim = self.hidden_dim * 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
        
        print(f"[MultimodalFusion] Text model: {text_model_name}")
        print(f"[MultimodalFusion] Image model: {image_model_name}")
        print(f"[MultimodalFusion] Fusion type: {fusion_type}")
        print(f"[MultimodalFusion] Fusion dim: {fusion_dim}")
    
    def forward(self, input_ids, attention_mask, image, **kwargs):
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            image: Image tensor [batch, 3, H, W]
            
        Returns:
            logits: Classification logits [batch, num_classes]
        """
        # Get text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        text_features = self.text_proj(text_features)
        
        # Get image features
        image_features = self.image_encoder(image)
        image_features = self.image_proj(image_features)
        
        # Fuse features
        if self.fusion_type == 'concat':
            fused = torch.cat([text_features, image_features], dim=-1)
        
        elif self.fusion_type == 'attention':
            # Cross-attention: query=text, key/value=image
            combined = torch.stack([text_features, image_features], dim=1)
            attn_output, _ = self.attention(combined, combined, combined)
            fused = attn_output.mean(dim=1)
        
        elif self.fusion_type == 'gated':
            concat = torch.cat([text_features, image_features], dim=-1)
            gate = self.gate(concat)
            fused = gate * text_features + (1 - gate) * image_features
        
        else:
            fused = torch.cat([text_features, image_features], dim=-1)
        
        logits = self.classifier(fused)
        return logits


# =============================================================================
# CLIP-BASED MODEL
# =============================================================================

class CLIPClassifier(nn.Module):
    """
    CLIP-based classifier for multimodal understanding.
    Uses pre-trained CLIP model for joint image-text representations.
    """
    
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        num_classes=NUM_CLASSES,
        dropout=0.3,
        freeze_clip=True
    ):
        super().__init__()
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except:
            print("[WARNING] CLIP not available, using fallback")
            raise ImportError("CLIP model not available")
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # CLIP has 512 or 768 dim embeddings
        clip_dim = self.clip.config.projection_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(clip_dim * 2, clip_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_dim, num_classes)
        )
        
        print(f"[CLIPClassifier] Model: {model_name}")
        print(f"[CLIPClassifier] CLIP dim: {clip_dim}")
    
    def forward(self, input_ids, attention_mask, image, **kwargs):
        """Forward pass using CLIP embeddings."""
        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=image
        )
        
        text_embeds = outputs.text_embeds
        image_embeds = outputs.image_embeds
        
        # Concatenate embeddings
        combined = torch.cat([text_embeds, image_embeds], dim=-1)
        logits = self.classifier(combined)
        
        return logits


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(
    model_type='multimodal',  # 'text', 'image', 'multimodal', 'clip'
    text_model_name=None,
    image_model_name=None,
    num_classes=NUM_CLASSES,
    **kwargs
):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model to create
        text_model_name: Name of text model (if applicable)
        image_model_name: Name of image model (if applicable)
        num_classes: Number of output classes
        **kwargs: Additional model arguments
        
    Returns:
        model: PyTorch model
    """
    if text_model_name is None:
        text_model_name = TEXT_MODEL_OPTIONS[DEFAULT_TEXT_MODEL]
    if image_model_name is None:
        image_model_name = VISION_MODEL_OPTIONS[DEFAULT_VISION_MODEL]
    
    if model_type == 'text':
        model = TextClassifier(
            model_name=text_model_name,
            num_classes=num_classes,
            **kwargs
        )
    
    elif model_type == 'image':
        model = ImageClassifier(
            model_name=image_model_name,
            num_classes=num_classes,
            **kwargs
        )
    
    elif model_type == 'multimodal':
        model = MultimodalFusion(
            text_model_name=text_model_name,
            image_model_name=image_model_name,
            num_classes=num_classes,
            **kwargs
        )
    
    elif model_type == 'clip':
        model = CLIPClassifier(
            num_classes=num_classes,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}")
    print(f"[Model] Trainable parameters: {trainable_params:,}")
    
    return model


# Test models
if __name__ == "__main__":
    print("Testing Models...")
    
    # Test text classifier
    print("\n" + "="*50)
    print("Testing TextClassifier")
    model = create_model(model_type='text')
    dummy_ids = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    output = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    print(f"Output shape: {output.shape}")
    
    # Test image classifier
    print("\n" + "="*50)
    print("Testing ImageClassifier")
    model = create_model(model_type='image')
    dummy_img = torch.randn(2, 3, 224, 224)
    output = model(image=dummy_img)
    print(f"Output shape: {output.shape}")
    
    # Test multimodal
    print("\n" + "="*50)
    print("Testing MultimodalFusion")
    model = create_model(model_type='multimodal', fusion_type='concat')
    output = model(input_ids=dummy_ids, attention_mask=dummy_mask, image=dummy_img)
    print(f"Output shape: {output.shape}")
    
    print("\n[SUCCESS] All models tested successfully!")
