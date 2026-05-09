"""
Unit tests for inference pipeline.

Tests single prediction, batch prediction, and ensemble modes.
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abiral.inference import InferenceEngine


@pytest.fixture
def sample_image(tmp_path):
    """Create sample test image."""
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def mock_model(tmp_path):
    """Create mock model checkpoint."""
    # Create a simple model for testing
    from abiral.fusion_models import TextOnlyModel
    
    model = TextOnlyModel(
        text_encoder_name="bert-base-multilingual-cased",
        num_classes=2,
        hidden_dim=256,
    )
    
    checkpoint_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_type": "text_only",
            "text_encoder": "bert-base-multilingual-cased",
            "num_classes": 2,
        }
    }, checkpoint_path)
    
    return str(checkpoint_path)


class TestInferenceEngine:
    """Test InferenceEngine class."""
    
    def test_initialization(self, mock_model):
        """Test engine initialization."""
        engine = InferenceEngine(
            model_path=mock_model,
            device="cpu",
        )
        
        assert engine is not None
        assert engine.device.type == "cpu"
    
    def test_single_prediction(self, mock_model, sample_image):
        """Test single image+text prediction."""
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        # Predict
        result = engine.predict(
            image_path=sample_image,
            text="यो नेता भ्रष्ट छ",
        )
        
        # Check result structure
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        
        assert result["label"] in [0, 1]
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == 2
        assert abs(sum(result["probabilities"]) - 1.0) < 1e-5
    
    def test_batch_prediction(self, mock_model, tmp_path):
        """Test batch prediction."""
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        # Create multiple test images
        images = []
        texts = ["यो नेता भ्रष्ट छ", "राम्रो दिन", "आज मौसम राम्रो छ"]
        
        for i, text in enumerate(texts):
            img = Image.new("RGB", (224, 224), color=(i * 50, i * 50, i * 50))
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            images.append(str(img_path))
        
        # Batch predict
        results = engine.predict_batch(
            image_paths=images,
            texts=texts,
            batch_size=2,
        )
        
        # Check results
        assert len(results) == 3
        
        for result in results:
            assert "label" in result
            assert "confidence" in result
            assert result["label"] in [0, 1]
    
    def test_text_only_mode(self, mock_model):
        """Test text-only prediction."""
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        # Predict without image
        result = engine.predict(
            image_path=None,
            text="यो नेता भ्रष्ट छ",
        )
        
        assert result["label"] in [0, 1]
    
    def test_invalid_image(self, mock_model):
        """Test handling of invalid image path."""
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        with pytest.raises(FileNotFoundError):
            engine.predict(
                image_path="/nonexistent/image.jpg",
                text="test",
            )
    
    def test_empty_text(self, mock_model, sample_image):
        """Test handling of empty text."""
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        # Should handle gracefully or raise appropriate error
        # (depends on implementation)
        try:
            result = engine.predict(
                image_path=sample_image,
                text="",
            )
            assert "label" in result
        except ValueError:
            pass  # Acceptable to raise error for empty text


class TestEnsembleInference:
    """Test ensemble inference."""
    
    def test_ensemble_voting(self, tmp_path):
        """Test late fusion voting ensemble."""
        from abiral.inference import EnsembleInference
        
        # Create multiple mock models
        model_paths = []
        for i in range(3):
            from abiral.fusion_models import TextOnlyModel
            
            model = TextOnlyModel(
                text_encoder_name="bert-base-multilingual-cased",
                num_classes=2,
            )
            
            path = tmp_path / f"model_{i}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"model_type": "text_only"},
            }, path)
            model_paths.append(str(path))
        
        # Create ensemble
        ensemble = EnsembleInference(
            model_paths=model_paths,
            mode="voting",
            device="cpu",
        )
        
        # Predict
        img = Image.new("RGB", (224, 224))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = ensemble.predict(
            image_path=str(img_path),
            text="test",
        )
        
        assert result["label"] in [0, 1]
    
    def test_ensemble_bagging(self, tmp_path):
        """Test bagging ensemble."""
        from abiral.inference import EnsembleInference
        
        # Similar to voting test
        model_paths = []
        for i in range(3):
            from abiral.fusion_models import TextOnlyModel
            
            model = TextOnlyModel(
                text_encoder_name="bert-base-multilingual-cased",
                num_classes=2,
            )
            
            path = tmp_path / f"model_{i}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"model_type": "text_only"},
            }, path)
            model_paths.append(str(path))
        
        ensemble = EnsembleInference(
            model_paths=model_paths,
            mode="bagging",
            device="cpu",
        )
        
        img = Image.new("RGB", (224, 224))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        result = ensemble.predict(
            image_path=str(img_path),
            text="test",
        )
        
        assert result["label"] in [0, 1]


class TestInferencePerformance:
    """Test inference performance."""
    
    def test_throughput(self, mock_model, tmp_path):
        """Test inference throughput."""
        import time
        
        engine = InferenceEngine(model_path=mock_model, device="cpu")
        
        # Create test images
        images = []
        for i in range(10):
            img = Image.new("RGB", (224, 224))
            img_path = tmp_path / f"test_{i}.jpg"
            img.save(img_path)
            images.append(str(img_path))
        
        texts = ["test"] * 10
        
        # Measure time
        start = time.time()
        results = engine.predict_batch(images, texts, batch_size=5)
        elapsed = time.time() - start
        
        # Should process all samples
        assert len(results) == 10
        
        # Log throughput (samples per second)
        throughput = 10 / elapsed
        print(f"\nThroughput: {throughput:.2f} samples/sec")
        
        # Sanity check: should be at least 1 sample/sec on CPU
        assert throughput > 1.0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_inference(self, mock_model, sample_image):
        """Test inference on GPU."""
        engine = InferenceEngine(model_path=mock_model, device="cuda")
        
        result = engine.predict(
            image_path=sample_image,
            text="test",
        )
        
        assert result["label"] in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
