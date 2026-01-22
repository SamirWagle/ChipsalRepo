"""
CHiPSAL 2026 - Subtask A: Quick Start Training Script
=====================================================
Simplified training script for quick experiments
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(str(Path(__file__).parent))
from config import *


# =============================================================================
# SIMPLIFIED DATASET
# =============================================================================

class SimpleMemesDataset(Dataset):
    """Simplified dataset for quick experiments."""
    
    def __init__(self, df, image_dir, tokenizer, is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.is_train = is_train
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5 if is_train else 0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.image_dir / row['index']
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            img = self.transform(image=img)['image']
        except:
            img = torch.zeros(3, 224, 224)
        
        # Tokenize text
        text = str(row.get('text', ''))
        encoding = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        return {
            'image': img,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }


# =============================================================================
# SIMPLE MULTIMODAL MODEL
# =============================================================================

class SimpleMultimodalModel(nn.Module):
    """Simple multimodal model combining text and image."""
    
    def __init__(self, text_model='xlm-roberta-base', image_model='resnet50', num_classes=2):
        super().__init__()
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        
        # Image encoder
        self.image_encoder = timm.create_model(image_model, pretrained=True, num_classes=0)
        image_dim = self.image_encoder.num_features
        
        # Fusion and classification
        self.text_proj = nn.Linear(text_dim, 256)
        self.image_proj = nn.Linear(image_dim, 256)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, image):
        # Text features
        text_out = self.text_encoder(input_ids, attention_mask)
        text_feat = self.text_proj(text_out.last_hidden_state[:, 0, :])
        
        # Image features
        image_feat = self.image_proj(self.image_encoder(image))
        
        # Concatenate and classify
        combined = torch.cat([text_feat, image_feat], dim=-1)
        return self.classifier(combined)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_simple():
    """Simple training loop."""
    print("="*60)
    print("CHiPSAL 2026 - QUICK START TRAINING")
    print("="*60)
    
    # Settings
    EPOCHS = 5
    BATCH = 8
    LR = 2e-5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"Loaded {len(df)} samples")
    
    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=42
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # Datasets
    train_ds = SimpleMemesDataset(train_df, TRAIN_IMAGES_DIR, tokenizer, is_train=True)
    val_ds = SimpleMemesDataset(val_df, TRAIN_IMAGES_DIR, tokenizer, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    
    # Model
    model = SimpleMultimodalModel()
    model = model.to(device)
    
    # Class weights
    class_counts = train_df['label'].value_counts().sort_index()
    weights = torch.tensor([len(train_df)/(2*c) for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0.1*total_steps, total_steps)
    
    # Training loop
    best_f1 = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch in pbar:
            optimizer.zero_grad()
            
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['image'].to(device)
            )
            
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(batch['label'].numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        
        # Validate
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['image'].to(device)
                )
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(batch['label'].numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"\nEpoch {epoch+1}: Train F1={train_f1:.4f}, Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
        
        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'model': model.state_dict(),
                'f1': best_f1
            }, MODEL_DIR / 'best_simple_model.pt')
            print(f"  -> New best model saved! F1={best_f1:.4f}")
    
    print(f"\n[DONE] Best Val F1: {best_f1:.4f}")
    print(f"Model saved to: {MODEL_DIR / 'best_simple_model.pt'}")
    
    # Final evaluation
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))
    
    return model, best_f1


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    model, f1 = train_simple()
