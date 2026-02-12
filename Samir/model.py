"""
Complete multimodal model that combines encoders and fusion models
"""

import torch
import torch.nn as nn


class MultimodalClassifier(nn.Module):
    """
    Complete multimodal classifier combining encoders and fusion model
    """
    
    def __init__(self, image_encoder, text_encoder, fusion_model, use_modality='both'):
        super().__init__()
        
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion_model = fusion_model
        self.use_modality = use_modality
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Forward pass through the model
        
        Args:
            pixel_values: (batch_size, 3, 224, 224) or None
            input_ids: (batch_size, seq_len) or None
            attention_mask: (batch_size, seq_len) or None
            
        Returns:
            logits: (batch_size, num_classes)
            additional_outputs: dict with extra information (e.g., weights)
        """
        additional_outputs = {}
        
        # Image-only mode
        if self.use_modality == 'image':
            image_features = self.image_encoder(pixel_values)
            logits = self.fusion_model(image_features)
            return logits, additional_outputs
        
        # Text-only mode
        elif self.use_modality == 'text':
            text_features = self.text_encoder(input_ids, attention_mask)
            logits = self.fusion_model(text_features)
            return logits, additional_outputs
        
        # Multimodal mode
        elif self.use_modality == 'both':
            # Extract features
            image_features = self.image_encoder(pixel_values)
            text_features = self.text_encoder(input_ids, attention_mask)
            
            # Fusion
            fusion_output = self.fusion_model(image_features, text_features)
            
            # Handle hybrid fusion (returns weights)
            if isinstance(fusion_output, tuple):
                logits, modality_weights = fusion_output
                additional_outputs['modality_weights'] = modality_weights
            else:
                logits = fusion_output
            
            return logits, additional_outputs
        
        else:
            raise ValueError(f"Unknown modality: {self.use_modality}")
    
    def predict(self, pixel_values=None, input_ids=None, attention_mask=None):
        """
        Make predictions (inference mode)
        
        Returns:
            predictions: (batch_size,) predicted class indices
            probabilities: (batch_size, num_classes) class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits, additional_outputs = self.forward(pixel_values, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities, additional_outputs


def create_complete_model(image_encoder, text_encoder, fusion_model, use_modality='both'):
    """
    Create complete multimodal classifier
    
    Args:
        image_encoder: Image encoder model or None
        text_encoder: Text encoder model or None
        fusion_model: Fusion/classification model
        use_modality: 'image', 'text', or 'both'
        
    Returns:
        model: Complete MultimodalClassifier
    """
    return MultimodalClassifier(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        fusion_model=fusion_model,
        use_modality=use_modality
    )
