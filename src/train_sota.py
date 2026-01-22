"""
CHiPSAL 2026 - Subtask A: Training with SOTA Models
===================================================
Training pipeline for state-of-the-art Vision-Language Models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from src.models_sota import create_sota_model


# =============================================================================
# DATASET FOR SOTA MODELS
# =============================================================================

class SOTAMemeDataset(Dataset):
    """
    Dataset class compatible with SOTA vision-language models.
    Handles different preprocessing requirements for each model type.
    """
    
    def __init__(
        self,
        df,
        image_dir,
        model_type='siglip',
        processor=None,
        text_column='text',
        is_train=True
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.model_type = model_type
        self.processor = processor
        self.text_column = text_column
        self.is_train = is_train
        
        # Basic augmentation for training
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Rotate(limit=10, p=0.3),
        ])
        
        print(f"[Dataset] Model type: {model_type}, Samples: {len(df)}")
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, image_path):
        """Load and optionally augment image."""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.is_train:
                # Apply augmentations
                image_np = np.array(image)
                augmented = self.train_transform(image=image_np)
                image = Image.fromarray(augmented['image'])
            
            return image
        except Exception as e:
            print(f"[WARNING] Failed to load {image_path}: {e}")
            return Image.new('RGB', (384, 384), color='gray')
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['index']
        
        # Load image
        image = self._load_image(image_path)
        
        # Get text
        text = str(row.get(self.text_column, ''))
        if pd.isna(text) or text == 'nan':
            text = ""
        
        # Get label
        label = row['label']
        
        # Process based on model type
        if self.processor is not None:
            # Use model-specific processor
            try:
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=256
                )
                
                # Squeeze batch dimension
                sample = {k: v.squeeze(0) for k, v in inputs.items()}
                sample['label'] = torch.tensor(label, dtype=torch.long)
                
            except Exception as e:
                # Fallback for models that need separate processing
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
                sample = {k: v.squeeze(0) for k, v in inputs.items()}
                sample['label'] = torch.tensor(label, dtype=torch.long)
        else:
            # Generic processing
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            sample = {
                'pixel_values': transform(image),
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        return sample


def collate_fn(batch):
    """Custom collate function to handle variable-length inputs."""
    result = {}
    
    for key in batch[0].keys():
        if key == 'label':
            result[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    
    return result


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        labels = batch.pop('label').to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                logits = model(**batch)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(**batch)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        labels = batch.pop('label').to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        logits = model(**batch)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, f1, acc, all_preds, all_labels


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def train_sota_model(
    model_type='siglip',
    epochs=10,
    batch_size=8,
    lr=2e-5,
    use_amp=True,
    n_folds=5,
    experiment_name=None
):
    """
    Train a SOTA model with K-Fold cross-validation.
    
    Args:
        model_type: 'siglip', 'blip2', 'instructblip', 'llava', 'evaclip'
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        lr: Learning rate
        use_amp: Use mixed precision
        n_folds: Number of CV folds
        experiment_name: Name for saving
    """
    print("="*60)
    print(f"CHiPSAL 2026 - SOTA MODEL TRAINING")
    print(f"Model: {model_type.upper()}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Experiment directory
    exp_name = experiment_name or f"sota_{model_type}"
    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"Loaded {len(df)} samples")
    
    # Create initial model to get processor
    print("\n[INFO] Loading model and processor...")
    model = create_sota_model(model_type, num_classes=NUM_CLASSES)
    processor = getattr(model, 'processor', None)
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Create datasets
        train_dataset = SOTAMemeDataset(
            df=train_df,
            image_dir=TRAIN_IMAGES_DIR,
            model_type=model_type,
            processor=processor,
            is_train=True
        )
        
        val_dataset = SOTAMemeDataset(
            df=val_df,
            image_dir=TRAIN_IMAGES_DIR,
            model_type=model_type,
            processor=processor,
            is_train=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # Re-create model for each fold
        if fold > 0:
            del model
            torch.cuda.empty_cache()
            model = create_sota_model(model_type, num_classes=NUM_CLASSES)
        
        model = model.to(device)
        
        # Class weights
        class_counts = train_df['label'].value_counts().sort_index()
        weights = torch.tensor([len(train_df)/(2*c) for c in class_counts]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Optimizer (only trainable parameters)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        # Scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Mixed precision
        scaler = GradScaler() if use_amp and torch.cuda.is_available() else None
        
        # Training loop
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            train_loss, train_f1 = train_epoch(
                model, train_loader, optimizer, scheduler, 
                criterion, device, scaler
            )
            
            val_loss, val_f1, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            
            print(f"Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Acc: {val_acc:.4f}")
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold,
                    'epoch': epoch,
                    'best_f1': best_f1
                }, exp_dir / f'best_model_fold_{fold}.pt')
                
                print(f"  -> New best! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"  -> Early stopping at epoch {epoch + 1}")
                    break
        
        fold_results.append({
            'fold': fold,
            'best_f1': best_f1
        })
        
        print(f"\n[Fold {fold + 1}] Best F1: {best_f1:.4f}")
    
    # Summary
    avg_f1 = np.mean([r['best_f1'] for r in fold_results])
    std_f1 = np.std([r['best_f1'] for r in fold_results])
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for r in fold_results:
        print(f"Fold {r['fold'] + 1}: F1 = {r['best_f1']:.4f}")
    print(f"\nAverage F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"\nModels saved to: {exp_dir}")
    
    return fold_results


# =============================================================================
# QUICK START
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SOTA models for CHiPSAL 2026')
    parser.add_argument('--model', type=str, default='siglip',
                        choices=['siglip', 'blip2', 'instructblip', 'llava', 'evaclip'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--folds', type=int, default=5)
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Train
    results = train_sota_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.folds
    )
