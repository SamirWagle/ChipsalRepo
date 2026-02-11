"""
Encoders module for image and text feature extraction
Supports CLIP for images and BGE-M3/E5-Large for text
"""

import torch
import torch.nn as nn
from transformers import (
    CLIPModel, CLIPProcessor,
    AutoModel, AutoTokenizer
)


class ImageEncoder(nn.Module):
    """
    Image encoder using CLIP vision model
    """
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', freeze=True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        
        # Load CLIP model
        print(f"Loading CLIP model: {model_name}")
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = self.clip_model.vision_model
        
        # Get feature dimension
        self.feature_dim = self.clip_model.config.projection_dim
        
        # Freeze if specified
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            print("Image encoder frozen")
        
        self.vision_model.eval() if freeze else self.vision_model.train()
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: tensor of shape (batch_size, 3, 224, 224)
        Returns:
            image_features: tensor of shape (batch_size, feature_dim)
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=pixel_values)
                # Use pooled output (CLS token)
                image_features = outputs.pooler_output
        else:
            outputs = self.vision_model(pixel_values=pixel_values)
            image_features = outputs.pooler_output
        
        # Project to embedding space
        image_features = self.clip_model.visual_projection(image_features)
        
        return image_features


class TextEncoder(nn.Module):
    """
    Text encoder using BGE-M3 or E5-Large
    """
    
    def __init__(self, model_type='bge-m3', model_name=None, freeze=True):
        super().__init__()
        self.model_type = model_type
        self.freeze = freeze
        
        # Set model name
        if model_name is None:
            if model_type == 'bge-m3':
                model_name = 'BAAI/bge-m3'
            elif model_type == 'e5-large':
                model_name = 'intfloat/e5-large-v2'
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading text model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get feature dimension
        self.feature_dim = self.model.config.hidden_size
        
        # Freeze if specified
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Text encoder frozen")
        
        self.model.eval() if freeze else self.model.train()
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling for sentence embeddings
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: tensor of shape (batch_size, seq_len)
            attention_mask: tensor of shape (batch_size, seq_len)
        Returns:
            text_features: tensor of shape (batch_size, feature_dim)
        """
        if self.freeze:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                text_features = self.mean_pooling(outputs, attention_mask)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            text_features = self.mean_pooling(outputs, attention_mask)
        
        # Normalize features (common for embedding models)
        text_features = nn.functional.normalize(text_features, p=2, dim=1)
        
        return text_features


def get_image_processor(model_name='openai/clip-vit-base-patch32'):
    """
    Get CLIP image processor
    """
    processor = CLIPProcessor.from_pretrained(model_name)
    return processor.image_processor


class TextTokenizer:
    """
    Callable tokenizer class to support pickling for multiprocessing
    """
    
    def __init__(self, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, text):
        """Tokenize text with padding and truncation"""
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )


def get_text_tokenizer(model_type='bge-m3', model_name=None, max_length=77):
    """
    Get text tokenizer with preprocessing function
    """
    if model_name is None:
        if model_type == 'bge-m3':
            model_name = 'BAAI/bge-m3'
        elif model_type == 'e5-large':
            model_name = 'intfloat/e5-large-v2'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return TextTokenizer(tokenizer, max_length)


class ImageTransform:
    """
    Image transform for CLIP preprocessing
    """
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, image):
        """
        Transform PIL image to tensor
        """
        # Process image
        inputs = self.processor(images=image, return_tensors='pt')
        return inputs['pixel_values'].squeeze(0)


def create_encoders(config):
    """
    Factory function to create image and text encoders based on config
    
    Args:
        config: Configuration object
        
    Returns:
        image_encoder, text_encoder (or None if not used)
    """
    image_encoder = None
    text_encoder = None
    
    # Create image encoder if needed
    if config.USE_MODALITY in ['image', 'both']:
        if config.IMAGE_ENCODER == 'clip':
            image_encoder = ImageEncoder(
                model_name=config.CLIP_MODEL_NAME,
                freeze=config.FREEZE_IMAGE_ENCODER
            )
            config.IMAGE_FEATURE_DIM = image_encoder.feature_dim
        else:
            raise ValueError(f"Unknown image encoder: {config.IMAGE_ENCODER}")
    
    # Create text encoder if needed
    if config.USE_MODALITY in ['text', 'both']:
        text_encoder = TextEncoder(
            model_type=config.TEXT_ENCODER,
            model_name=config.BGE_MODEL_NAME if config.TEXT_ENCODER == 'bge-m3' else config.E5_MODEL_NAME,
            freeze=config.FREEZE_TEXT_ENCODER
        )
        config.TEXT_FEATURE_DIM = text_encoder.feature_dim
    
    return image_encoder, text_encoder
