"""
Dataset module for multimodal classification
Handles image and text data loading with support for different modalities
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold

class MultimodalDataset(Dataset):
    """
    Dataset for multimodal classification
    Supports image-only, text-only, and multimodal configurations
    """
    
    def __init__(self, 
                 dataframe, 
                 image_folder,
                 image_col='Image',
                 text_col='Text',
                 label_col='Label',
                 image_transform=None,
                 text_tokenizer=None,
                 use_modality='both',
                 image_type='original'):
        """
        Args:
            dataframe: pandas DataFrame with image filenames, text, and labels
            image_folder: path to folder containing images
            image_col: column name for image filenames
            text_col: column name for text
            label_col: column name for labels
            image_transform: torchvision transforms for images
            text_tokenizer: tokenizer for text
            use_modality: 'image', 'text', or 'both'
            image_type: 'original' or 'text_removed'
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_folder = image_folder
        self.image_col = image_col
        self.text_col = text_col
        self.label_col = label_col
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
        self.use_modality = use_modality
        self.image_type = image_type
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Get label
        label = int(row[self.label_col])
        
        # Initialize outputs
        image_tensor = None
        text_encoding = None
        
        # Load image if needed
        if self.use_modality in ['image', 'both']:
            image_filename = row[self.image_col]
            
            # Handle text_removed images
            if self.image_type == 'text_removed':
                # Assume text_removed images have '_text_removed' suffix
                base_name, ext = os.path.splitext(image_filename)
                image_filename = f"{base_name}_text_removed{ext}"
            
            image_path = os.path.join(self.image_folder, image_filename)
            
            try:
                image = Image.open(image_path).convert('RGB')
                if self.image_transform:
                    image_tensor = self.image_transform(image)
                else:
                    # Default: convert to tensor
                    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                # Return blank image
                image_tensor = torch.zeros(3, 224, 224)
        
        # Process text if needed
        if self.use_modality in ['text', 'both']:
            text = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
            
            if self.text_tokenizer:
                text_encoding = self.text_tokenizer(text)
            else:
                # Return raw text if no tokenizer
                text_encoding = text
        
        # Return based on modality
        if self.use_modality == 'image':
            return image_tensor, label
        elif self.use_modality == 'text':
            return text_encoding, label
        else:  # both
            return image_tensor, text_encoding, label


def create_dataloaders(csv_path, 
                       image_folder,
                       image_col='Image',
                       text_col='Text',
                       label_col='Label',
                       batch_size=16,
                       image_transform=None,
                       text_tokenizer=None,
                       use_modality='both',
                       image_type='original',
                       k_folds=5,
                       fold_idx=0,
                       seed=42):
    """
    Create train and validation dataloaders using K-fold cross-validation
    
    Args:
        csv_path: path to CSV file
        image_folder: path to image folder
        image_col: column name for image filenames
        text_col: column name for text
        label_col: column name for labels
        batch_size: batch size for dataloaders
        image_transform: image transformations
        text_tokenizer: text tokenizer
        use_modality: 'image', 'text', or 'both'
        image_type: 'original' or 'text_removed'
        k_folds: number of folds for cross-validation
        fold_idx: which fold to use (0 to k_folds-1)
        seed: random seed
        
    Returns:
        train_loader, val_loader, num_classes
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Detect number of classes
    num_classes = df[label_col].nunique()
    print(f"Detected {num_classes} classes in the dataset")
    
    # Create K-fold splits
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    # Get train and validation indices for the current fold
    all_indices = np.arange(len(df))
    splits = list(kfold.split(all_indices))
    train_idx, val_idx = splits[fold_idx]
    
    # Split dataframe
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Fold {fold_idx + 1}/{k_folds}: Train size = {len(train_df)}, Val size = {len(val_df)}")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        train_df,
        image_folder,
        image_col=image_col,
        text_col=text_col,
        label_col=label_col,
        image_transform=image_transform,
        text_tokenizer=text_tokenizer,
        use_modality=use_modality,
        image_type=image_type
    )
    
    val_dataset = MultimodalDataset(
        val_df,
        image_folder,
        image_col=image_col,
        text_col=text_col,
        label_col=label_col,
        image_transform=image_transform,
        text_tokenizer=text_tokenizer,
        use_modality=use_modality,
        image_type=image_type
    )
    
    # Create dataloaders
    # Note: num_workers=0 to avoid multiprocessing issues on Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, num_classes


def get_class_distribution(csv_path, label_col='Label', config=None):
    """
    Get class distribution for computing class weights
    
    Args:
        csv_path: Path to CSV file
        label_col: Column name for labels
        config: Optional config object for CLASS_WEIGHT_EXPONENT
    """
    df = pd.read_csv(csv_path)
    class_counts = df[label_col].value_counts().sort_index()
    print("\nClass distribution:")
    for class_idx, count in class_counts.items():
        print(f"  Class {class_idx}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Compute class weights optimized for macro F1 score
    # Exponent controls aggressiveness:
    #   0.0 = equal weights (no boosting)
    #   0.3 = mild boost (recommended for multimodal)
    #   0.5 = moderate boost (sqrt, default)
    #   1.0 = full inverse frequency (aggressive)
    exponent = 0.5  # default
    if config is not None:
        exponent = getattr(config, 'CLASS_WEIGHT_EXPONENT', 0.5)
    
    total_samples = len(df)
    num_classes = len(class_counts)
    max_count = class_counts.max()
    
    class_weights = []
    print(f"\nClass weights (exponent={exponent}):")
    for i in range(num_classes):
        weight = (max_count / class_counts[i]) ** exponent
        class_weights.append(weight)
        print(f"  Class {i}: weight = {weight:.4f}")
    
    return torch.FloatTensor(class_weights)
