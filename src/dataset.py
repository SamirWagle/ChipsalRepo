"""
CHiPSAL 2026 - Subtask A: Dataset Classes
=========================================
PyTorch Dataset implementations for multimodal hate speech detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
import sys

# Add parent directory to path  
sys.path.append(str(Path(__file__).parent.parent))
from config import *


class MemeDataset(Dataset):
    """
    Multimodal Dataset for Nepali Meme Classification.
    Supports text-only, image-only, or multimodal inputs.
    """
    
    def __init__(
        self,
        data_df,
        image_dir,
        tokenizer=None,
        text_column='text',
        label_column='label',
        image_size=224,
        mode='multimodal',  # 'text', 'image', 'multimodal'
        is_training=True,
        augment=True
    ):
        """
        Args:
            data_df: DataFrame with columns [index, label, text]
            image_dir: Directory containing meme images
            tokenizer: HuggingFace tokenizer for text
            text_column: Column name for text data
            label_column: Column name for labels
            image_size: Size to resize images
            mode: 'text', 'image', or 'multimodal'
            is_training: Whether this is training data
            augment: Whether to apply data augmentation
        """
        self.data = data_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.image_size = image_size
        self.mode = mode
        self.is_training = is_training
        self.augment = augment and is_training
        
        # Setup image transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        
        print(f"[Dataset] Mode: {mode}, Samples: {len(self.data)}, Training: {is_training}")
    
    def _get_train_transform(self):
        """Get training augmentation pipeline."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=AUGMENTATION_CONFIG['horizontal_flip_prob']),
            A.Rotate(limit=AUGMENTATION_CONFIG['rotation_limit'], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=AUGMENTATION_CONFIG['brightness_limit'],
                contrast_limit=AUGMENTATION_CONFIG['contrast_limit'],
                p=0.5
            ),
            A.GaussNoise(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_val_transform(self):
        """Get validation/test transform (no augmentation)."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.data)
    
    def _load_image(self, image_idx):
        """Load and transform an image."""
        image_path = self.image_dir / image_idx
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.augment:
                transformed = self.train_transform(image=image)
            else:
                transformed = self.val_transform(image=image)
            
            return transformed['image']
        
        except Exception as e:
            print(f"[WARNING] Failed to load {image_path}: {e}")
            # Return a blank image
            blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            return self.val_transform(image=blank)['image']
    
    def _tokenize_text(self, text):
        """Tokenize text using the provided tokenizer."""
        if self.tokenizer is None:
            return {}
        
        # Handle missing text
        if pd.isna(text) or text == '':
            text = "[UNK]"
        
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TEXT_LENGTH,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    
    def __getitem__(self, idx):
        """Get a single sample."""
        row = self.data.iloc[idx]
        image_idx = row['index']
        
        sample = {'index': image_idx}
        
        # Get label
        if self.label_column and self.label_column in row:
            sample['label'] = torch.tensor(row[self.label_column], dtype=torch.long)
        
        # Get image
        if self.mode in ['image', 'multimodal']:
            sample['image'] = self._load_image(image_idx)
        
        # Get text
        if self.mode in ['text', 'multimodal']:
            text = row[self.text_column] if self.text_column in row else ''
            text_encoding = self._tokenize_text(text)
            sample.update(text_encoding)
        
        return sample


class TestMemeDataset(MemeDataset):
    """Dataset for test/evaluation data (no labels)."""
    
    def __init__(self, data_df, image_dir, tokenizer=None, text_column='text', **kwargs):
        # Ensure label_column is None for test data
        super().__init__(
            data_df=data_df,
            image_dir=image_dir,
            tokenizer=tokenizer,
            text_column=text_column,
            label_column=None,
            is_training=False,
            augment=False,
            **kwargs
        )


def create_dataloaders(
    train_df,
    val_df,
    image_dir,
    tokenizer=None,
    text_column='text',
    batch_size=BATCH_SIZE,
    mode='multimodal',
    num_workers=NUM_WORKERS
):
    """
    Create training and validation dataloaders.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        image_dir: Directory containing images
        tokenizer: HuggingFace tokenizer
        text_column: Column name for text
        batch_size: Batch size
        mode: 'text', 'image', or 'multimodal'
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = MemeDataset(
        data_df=train_df,
        image_dir=image_dir,
        tokenizer=tokenizer,
        text_column=text_column,
        mode=mode,
        is_training=True,
        augment=True
    )
    
    val_dataset = MemeDataset(
        data_df=val_df,
        image_dir=image_dir,
        tokenizer=tokenizer,
        text_column=text_column,
        mode=mode,
        is_training=False,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def create_test_dataloader(
    test_df,
    image_dir,
    tokenizer=None,
    text_column='text',
    batch_size=BATCH_SIZE,
    mode='multimodal',
    num_workers=NUM_WORKERS
):
    """Create test dataloader (no labels)."""
    test_dataset = TestMemeDataset(
        data_df=test_df,
        image_dir=image_dir,
        tokenizer=tokenizer,
        text_column=text_column,
        mode=mode
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return test_loader


# Test the dataset
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load sample data
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"Loaded {len(df)} samples")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_OPTIONS['xlm_roberta'])
    
    # Create dataset
    dataset = MemeDataset(
        data_df=df.head(100),
        image_dir=TRAIN_IMAGES_DIR,
        tokenizer=tokenizer,
        mode='multimodal',
        is_training=True
    )
    
    # Test a sample
    sample = dataset[0]
    print("\nSample keys:", sample.keys())
    print("Image shape:", sample['image'].shape if 'image' in sample else 'N/A')
    print("Input IDs shape:", sample['input_ids'].shape if 'input_ids' in sample else 'N/A')
    print("Label:", sample['label'].item() if 'label' in sample else 'N/A')
