"""
Unit tests for dataset module.

Tests data loading, preprocessing, and augmentation.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abiral.dataset import MultimodalDataset


@pytest.fixture
def sample_data(tmp_path):
    """Create sample dataset for testing."""
    
    # Create directories
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    
    # Create sample CSV
    import pandas as pd
    
    data = {
        "index": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "text": ["यो नेता भ्रष्ट छ", "राम्रो दिन", "आज मौसम राम्रो छ"],
        "label": [1, 0, 0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    # Create sample images
    for img_name in data["index"]:
        img = Image.new("RGB", (224, 224), color=(127, 127, 127))
        img.save(image_dir / img_name)
    
    return {
        "csv_path": str(csv_path),
        "image_dir": str(image_dir),
        "data": data,
    }


@pytest.fixture
def tokenizer():
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


class TestMultimodalDataset:
    """Test MultimodalDataset class."""
    
    def test_init(self, sample_data, tokenizer):
        """Test dataset initialization."""
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
            max_length=77,
        )
        
        assert len(dataset) == 3
        assert dataset.max_length == 77
    
    def test_getitem(self, sample_data, tokenizer):
        """Test getting single item."""
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
            max_length=77,
        )
        
        item = dataset[0]
        
        # Check keys
        assert "image" in item
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item
        
        # Check types
        assert isinstance(item["image"], torch.Tensor)
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["label"], int)
        
        # Check shapes
        assert item["image"].shape == (3, 224, 224)
        assert item["input_ids"].shape == (77,)
        assert item["attention_mask"].shape == (77,)
        
        # Check label
        assert item["label"] in [0, 1]
    
    def test_text_encoding(self, sample_data, tokenizer):
        """Test text encoding."""
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
            max_length=77,
        )
        
        item = dataset[0]
        
        # Decode back
        decoded = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
        
        # Should contain original text (may be truncated)
        assert len(decoded) > 0
    
    def test_image_loading(self, sample_data, tokenizer):
        """Test image loading and preprocessing."""
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
        )
        
        item = dataset[0]
        
        # Check image is normalized (values should be roughly in [-1, 1] range)
        assert item["image"].min() >= -3.0
        assert item["image"].max() <= 3.0
    
    def test_missing_image(self, sample_data, tokenizer):
        """Test handling of missing image."""
        # Remove one image
        os.remove(os.path.join(sample_data["image_dir"], "img2.jpg"))
        
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
        )
        
        # Should raise error or return None
        with pytest.raises(FileNotFoundError):
            _ = dataset[1]
    
    def test_dataloader(self, sample_data, tokenizer):
        """Test with DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
        )
        
        batch = next(iter(dataloader))
        
        # Check batch shapes
        assert batch["image"].shape == (2, 3, 224, 224)
        assert batch["input_ids"].shape == (2, 77)
        assert batch["attention_mask"].shape == (2, 77)
        assert batch["label"].shape == (2,)


class TestDataAugmentation:
    """Test data augmentation."""
    
    def test_with_augmentation(self, sample_data, tokenizer):
        """Test dataset with augmentation."""
        from albumentations import Compose, RandomBrightnessContrast
        from albumentations.pytorch import ToTensorV2
        
        transform = Compose([
            RandomBrightnessContrast(p=1.0),  # Always apply
            ToTensorV2(),
        ])
        
        dataset = MultimodalDataset(
            csv_file=sample_data["csv_path"],
            image_dir=sample_data["image_dir"],
            tokenizer=tokenizer,
            transform=transform,
        )
        
        item1 = dataset[0]
        item2 = dataset[0]  # Get same item twice
        
        # With augmentation, images should be different
        assert not torch.allclose(item1["image"], item2["image"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
