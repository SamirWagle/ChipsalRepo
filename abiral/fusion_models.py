"""
Fusion models for multimodal classification
Implements Early Fusion, Late Fusion (Voting, Bagging), and Hybrid Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    """
    Early Fusion: Concatenate image and text features before classification
    """
    
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.fusion_dim = image_dim + text_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, image_dim)
            text_features: (batch_size, text_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        # Concatenate features
        fused_features = torch.cat([image_features, text_features], dim=1)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits


class LateFusionVotingModel(nn.Module):
    """
    Late Fusion with Voting: Train separate classifiers and vote on predictions
    """
    
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Image classifier
        self.image_classifier = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Text classifier
        self.text_classifier = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, image_dim)
            text_features: (batch_size, text_dim)
        Returns:
            logits: (batch_size, num_classes) - averaged logits
        """
        # Get predictions from both modalities
        image_logits = self.image_classifier(image_features)
        text_logits = self.text_classifier(text_features)
        
        # Average logits (soft voting)
        logits = (image_logits + text_logits) / 2
        
        return logits
    
    def get_individual_predictions(self, image_features, text_features):
        """Get predictions from each modality separately"""
        image_logits = self.image_classifier(image_features)
        text_logits = self.text_classifier(text_features)
        return image_logits, text_logits


class LateFusionBaggingModel(nn.Module):
    """
    Late Fusion with Bagging: Multiple classifiers with ensemble
    """
    
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256, dropout=0.3, n_estimators=3):
        super().__init__()
        
        self.n_estimators = n_estimators
        
        # Create multiple image classifiers
        self.image_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(n_estimators)
        ])
        
        # Create multiple text classifiers
        self.text_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) for _ in range(n_estimators)
        ])
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, image_dim)
            text_features: (batch_size, text_dim)
        Returns:
            logits: (batch_size, num_classes) - averaged predictions
        """
        image_predictions = []
        text_predictions = []
        
        # Get predictions from all estimators
        for i in range(self.n_estimators):
            image_predictions.append(self.image_classifiers[i](image_features))
            text_predictions.append(self.text_classifiers[i](text_features))
        
        # Average all predictions
        image_logits = torch.stack(image_predictions).mean(dim=0)
        text_logits = torch.stack(text_predictions).mean(dim=0)
        
        # Combine image and text predictions
        logits = (image_logits + text_logits) / 2
        
        return logits


class HybridFusionModel(nn.Module):
    """
    Hybrid Fusion: Dynamic attention-based fusion with learnable weights
    """
    
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256, dropout=0.3, num_heads=4):
        super().__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Project features to same dimension
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention for cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gate mechanism for dynamic weighting
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 weights for image and text
            nn.Softmax(dim=1)
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, image_dim)
            text_features: (batch_size, text_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = image_features.size(0)
        
        # Project to common dimension
        image_proj = self.image_projection(image_features)  # (batch_size, hidden_dim)
        text_proj = self.text_projection(text_features)    # (batch_size, hidden_dim)
        
        # Prepare for attention: (batch_size, seq_len=2, hidden_dim)
        multimodal_features = torch.stack([image_proj, text_proj], dim=1)
        
        # Self-attention across modalities
        attended_features, attention_weights = self.cross_attention(
            multimodal_features,
            multimodal_features,
            multimodal_features
        )
        
        # Extract attended features
        attended_image = attended_features[:, 0, :]  # (batch_size, hidden_dim)
        attended_text = attended_features[:, 1, :]   # (batch_size, hidden_dim)
        
        # Compute dynamic weights using gate network
        gate_input = torch.cat([attended_image, attended_text], dim=1)
        modality_weights = self.gate_network(gate_input)  # (batch_size, 2)
        
        # Apply weights
        weighted_image = attended_image * modality_weights[:, 0:1]
        weighted_text = attended_text * modality_weights[:, 1:2]
        
        # Fuse weighted features
        fused_features = torch.cat([weighted_image, weighted_text], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits, modality_weights  # Return weights for analysis


class UnimodalModel(nn.Module):
    """
    Unimodal classifier for image-only or text-only experiments
    """
    
    def __init__(self, feature_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features):
        """
        Args:
            features: (batch_size, feature_dim)
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.classifier(features)


def create_fusion_model(config):
    """
    Factory function to create fusion model based on configuration
    
    Args:
        config: Configuration object
        
    Returns:
        fusion_model: PyTorch model
    """
    num_classes = config.NUM_CLASSES
    hidden_dim = config.HIDDEN_DIM
    dropout = config.DROPOUT
    
    # Unimodal models
    if config.USE_MODALITY == 'image':
        return UnimodalModel(
            feature_dim=config.IMAGE_FEATURE_DIM,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    elif config.USE_MODALITY == 'text':
        return UnimodalModel(
            feature_dim=config.TEXT_FEATURE_DIM,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    # Multimodal models
    elif config.USE_MODALITY == 'both':
        if config.FUSION_STRATEGY == 'early':
            return EarlyFusionModel(
                image_dim=config.IMAGE_FEATURE_DIM,
                text_dim=config.TEXT_FEATURE_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        elif config.FUSION_STRATEGY == 'late_voting':
            return LateFusionVotingModel(
                image_dim=config.IMAGE_FEATURE_DIM,
                text_dim=config.TEXT_FEATURE_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        
        elif config.FUSION_STRATEGY == 'late_bagging':
            return LateFusionBaggingModel(
                image_dim=config.IMAGE_FEATURE_DIM,
                text_dim=config.TEXT_FEATURE_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                n_estimators=3
            )
        
        elif config.FUSION_STRATEGY == 'hybrid':
            return HybridFusionModel(
                image_dim=config.IMAGE_FEATURE_DIM,
                text_dim=config.TEXT_FEATURE_DIM,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_heads=config.HYBRID_ATTENTION_HEADS
            )
        
        else:
            raise ValueError(f"Unknown fusion strategy: {config.FUSION_STRATEGY}")
    
    else:
        raise ValueError(f"Unknown modality: {config.USE_MODALITY}")
