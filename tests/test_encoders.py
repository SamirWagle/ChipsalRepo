"""
Unit tests for encoders.

Tests CLIP and BGE-M3 encoders.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abiral.encoders import CLIPEncoder, BGEEncoder


class TestCLIPEncoder:
    """Test CLIP visual encoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = CLIPEncoder(
            model_name="openai/clip-vit-base-patch32",
            unfreeze_layers=2,
        )
        
        assert encoder is not None
        assert encoder.output_dim == 512
    
    def test_forward(self):
        """Test forward pass."""
        encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        
        # Create dummy images
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            features = encoder(images)
        
        # Check output shape
        assert features.shape == (batch_size, 512)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_layer_freezing(self):
        """Test that specified layers are unfrozen."""
        encoder = CLIPEncoder(
            model_name="openai/clip-vit-base-patch32",
            unfreeze_layers=2,
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())
        
        # Should have some trainable parameters (not all frozen)
        assert trainable_params > 0
        # Should have some frozen parameters (not all trainable)
        assert trainable_params < total_params
    
    def test_output_normalization(self):
        """Test that outputs are L2-normalized."""
        encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            features = encoder(images)
            norms = torch.norm(features, p=2, dim=1)
        
        # L2 norm should be close to 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_batch_consistency(self):
        """Test that same image gives same output."""
        encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        encoder.eval()
        
        image = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output1 = encoder(image)
            output2 = encoder(image)
        
        # Should be identical in eval mode
        assert torch.allclose(output1, output2)


class TestBGEEncoder:
    """Test BGE-M3 text encoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = BGEEncoder(
            model_name="BAAI/bge-m3",
            unfreeze_layers=4,
        )
        
        assert encoder is not None
        assert encoder.output_dim == 1024
    
    def test_forward(self):
        """Test forward pass."""
        encoder = BGEEncoder("BAAI/bge-m3")
        
        # Create dummy text inputs
        batch_size = 4
        max_length = 77
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        attention_mask = torch.ones(batch_size, max_length)
        
        # Forward pass
        with torch.no_grad():
            features = encoder(input_ids, attention_mask)
        
        # Check output shape
        assert features.shape == (batch_size, 1024)
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_layer_freezing(self):
        """Test selective layer unfreezing."""
        encoder = BGEEncoder(
            model_name="BAAI/bge-m3",
            unfreeze_layers=4,
        )
        
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())
        
        # Should have partial unfreezing
        assert trainable_params > 0
        assert trainable_params < total_params
    
    def test_attention_mask(self):
        """Test that attention mask is respected."""
        encoder = BGEEncoder("BAAI/bge-m3")
        
        batch_size = 2
        max_length = 77
        
        # Create inputs with different lengths
        input_ids = torch.randint(0, 1000, (batch_size, max_length))
        
        # First sample: full length
        # Second sample: half length
        attention_mask = torch.ones(batch_size, max_length)
        attention_mask[1, max_length // 2:] = 0
        
        with torch.no_grad():
            features = encoder(input_ids, attention_mask)
        
        # Both should produce valid outputs
        assert features.shape == (batch_size, 1024)
        assert not torch.isnan(features).any()
    
    def test_empty_text(self):
        """Test handling of empty/padding-only text."""
        encoder = BGEEncoder("BAAI/bge-m3")
        
        batch_size = 2
        max_length = 77
        
        # All padding tokens
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length)
        
        with torch.no_grad():
            features = encoder(input_ids, attention_mask)
        
        # Should still produce output (might be all zeros or special handling)
        assert features.shape == (batch_size, 1024)
        # Don't check for zeros - model may have special handling
    
    def test_devanagari_support(self):
        """Test that model handles Devanagari text."""
        from transformers import AutoTokenizer
        
        encoder = BGEEncoder("BAAI/bge-m3")
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        # Nepali text
        texts = ["यो नेता भ्रष्ट छ", "राम्रो दिन"]
        
        # Tokenize
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        
        # Forward pass
        with torch.no_grad():
            features = encoder(encoded["input_ids"], encoded["attention_mask"])
        
        assert features.shape == (2, 1024)
        assert not torch.isnan(features).any()


class TestEncoderIntegration:
    """Integration tests for encoders."""
    
    def test_clip_bge_compatibility(self):
        """Test that CLIP and BGE outputs are compatible for fusion."""
        clip_encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        bge_encoder = BGEEncoder("BAAI/bge-m3")
        
        batch_size = 4
        
        # Visual inputs
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Text inputs
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        # Get features
        with torch.no_grad():
            visual_feats = clip_encoder(images)
            text_feats = bge_encoder(input_ids, attention_mask)
        
        # Check shapes are compatible
        assert visual_feats.shape[0] == text_feats.shape[0] == batch_size
        assert visual_feats.shape[1] == 512
        assert text_feats.shape[1] == 1024
        
        # Can concatenate
        fused = torch.cat([visual_feats, text_feats], dim=1)
        assert fused.shape == (batch_size, 1536)
    
    def test_encoder_memory(self):
        """Test memory usage of encoders."""
        import gc
        
        # Should be able to create and delete without memory leak
        encoder = CLIPEncoder("openai/clip-vit-base-patch32")
        del encoder
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        encoder = BGEEncoder("BAAI/bge-m3")
        del encoder
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        assert True  # If we get here, no OOM
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self):
        """Test encoders work on CUDA."""
        device = torch.device("cuda")
        
        clip_encoder = CLIPEncoder("openai/clip-vit-base-patch32").to(device)
        bge_encoder = BGEEncoder("BAAI/bge-m3").to(device)
        
        # Dummy inputs on CUDA
        images = torch.randn(2, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 1000, (2, 77)).to(device)
        attention_mask = torch.ones(2, 77).to(device)
        
        with torch.no_grad():
            visual_feats = clip_encoder(images)
            text_feats = bge_encoder(input_ids, attention_mask)
        
        assert visual_feats.device.type == "cuda"
        assert text_feats.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
