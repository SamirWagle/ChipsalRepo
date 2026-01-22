"""
CHiPSAL 2026 - Complete Training Script for Novel Models
========================================================
Run this script to train models that beat MemeNePAL baseline (F1 > 0.6785)

Usage:
    python train_novel.py --model siglip_muril --epochs 10
    python train_novel.py --model mmlora --epochs 15
    python train_novel.py --model ensemble --epochs 10
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
import json
import warnings
import sys

warnings.filterwarnings('ignore')

# Add parent to path
sys.path.append(str(Path(__file__).parent))
from config import *


# =============================================================================
# DATASET
# =============================================================================

class NepaliMemeDataset(Dataset):
    """Dataset for Nepali meme classification."""
    
    def __init__(self, df, image_dir, tokenizer, text_column='text', is_train=True):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.is_train = is_train
        
        # Augmentation
        if is_train:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),
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
            image = np.array(Image.open(img_path).convert('RGB'))
            image = self.transform(image=image)['image']
        except:
            image = torch.zeros(3, 224, 224)
        
        # Get text
        text = str(row.get(self.text_column, ''))
        if pd.isna(text) or text == 'nan':
            text = ""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        return {
            'pixel_values': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(row['label'], dtype=torch.long),
            'text': text  # For keyword detection
        }


def collate_fn(batch):
    """Collate function for dataloader."""
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'texts': [b['text'] for b in batch]
    }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler=None, use_contrastive=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        labels = batch['label'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                if use_contrastive:
                    logits, contrastive_loss = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    ce_loss = criterion(logits, labels)
                    loss = ce_loss + 0.5 * contrastive_loss
                else:
                    logits = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_contrastive:
                logits, contrastive_loss = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                ce_loss = criterion(logits, labels)
                loss = ce_loss + 0.5 * contrastive_loss
            else:
                logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
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
    metrics = {
        'loss': avg_loss,
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'accuracy': accuracy_score(all_labels, all_preds)
    }
    
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_contrastive=False):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        labels = batch['label'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        logits = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    metrics = {
        'loss': avg_loss,
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro')
    }
    
    return metrics, all_preds, all_labels


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    model_type='siglip_muril',
    epochs=10,
    batch_size=8,
    lr=2e-5,
    n_folds=5,
    use_amp=True,
    text_model="google/muril-base-cased"
):
    """
    Main training function.
    
    Args:
        model_type: 'siglip_muril', 'mmlora', 'moe', 'contrastive', 'ensemble'
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        n_folds: Number of CV folds
        use_amp: Use mixed precision
        text_model: Text encoder to use
    """
    print("\n" + "="*70)
    print("CHiPSAL 2026 - NOVEL MODEL TRAINING")
    print(f"Model: {model_type.upper()}")
    print(f"Target: Beat MemeNePAL F1 > 0.6785")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Epochs: {epochs}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Learning rate: {lr}")
    print(f"[INFO] K-Folds: {n_folds}")
    
    # Create output directory
    exp_name = f"novel_{model_type}"
    exp_dir = OUTPUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[INFO] Loading data from {TRAIN_CSV_FILE}...")
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"[INFO] Total samples: {len(df)}")
    print(f"[INFO] Class distribution:")
    print(df['label'].value_counts())
    
    # Load tokenizer
    print(f"\n[INFO] Loading tokenizer: {text_model}")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    
    # Import model creation function
    from src.models_novel import create_novel_model, NovelEnsemble
    
    # K-Fold CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"[Fold {fold+1}] Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Create datasets
        train_dataset = NepaliMemeDataset(
            train_df, TRAIN_IMAGES_DIR, tokenizer, is_train=True
        )
        val_dataset = NepaliMemeDataset(
            val_df, TRAIN_IMAGES_DIR, tokenizer, is_train=False
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_fn
        )
        
        # Create model
        print(f"[INFO] Creating model: {model_type}")
        if model_type == 'ensemble':
            model = NovelEnsemble(["mmlora", "moe", "siglip_muril"])
        else:
            model = create_novel_model(model_type, text_model=text_model)
        
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {total_params:,}")
        print(f"[INFO] Trainable params: {trainable_params:,}")
        
        # Class weights
        class_counts = train_df['label'].value_counts().sort_index()
        weights = torch.tensor([len(train_df)/(2*c) for c in class_counts]).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Optimizer
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01
        )
        
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
        patience = 3
        patience_counter = 0
        use_contrastive = (model_type == 'contrastive')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, device, scaler, use_contrastive
            )
            
            # Evaluate
            val_metrics, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device, use_contrastive
            )
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1_macro']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1_macro']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'fold': fold,
                    'epoch': epoch,
                    'best_f1': best_f1,
                    'val_metrics': val_metrics
                }, exp_dir / f'best_model_fold_{fold}.pt')
                
                print(f"  ✓ New best! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  ✗ Early stopping at epoch {epoch + 1}")
                    break
        
        # Final evaluation
        print(f"\n[Fold {fold+1}] Best F1: {best_f1:.4f}")
        
        fold_results.append({
            'fold': fold,
            'best_f1': best_f1,
            'val_metrics': val_metrics
        })
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    avg_f1 = np.mean([r['best_f1'] for r in fold_results])
    std_f1 = np.std([r['best_f1'] for r in fold_results])
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\nModel: {model_type}")
    print(f"\nPer-Fold Results:")
    for r in fold_results:
        print(f"  Fold {r['fold']+1}: F1 = {r['best_f1']:.4f}")
    
    print(f"\n{'='*50}")
    print(f"Average F1: {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"{'='*50}")
    
    # Compare with baseline
    baseline_f1 = 0.6785
    improvement = ((avg_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"\nMemeNePAL Baseline: {baseline_f1:.4f}")
    print(f"Your Model:         {avg_f1:.4f}")
    
    if avg_f1 > baseline_f1:
        print(f"✓ BEATS BASELINE by {improvement:.1f}%! 🎉")
    else:
        print(f"✗ Below baseline by {abs(improvement):.1f}%")
    
    # Save results
    results = {
        'model_type': model_type,
        'avg_f1': float(avg_f1),
        'std_f1': float(std_f1),
        'baseline_f1': baseline_f1,
        'improvement_pct': float(improvement),
        'fold_results': fold_results,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'n_folds': n_folds,
            'text_model': text_model
        }
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n[INFO] Results saved to: {exp_dir}")
    print(f"[INFO] Models saved to: {exp_dir}/best_model_fold_*.pt")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train novel models for CHiPSAL 2026 Hate Speech Detection'
    )
    parser.add_argument(
        '--model', type=str, default='siglip_muril',
        choices=['siglip_muril', 'mmlora', 'moe', 'contrastive', 'ensemble'],
        help='Model type to train'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Train
    results = train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_folds=args.folds,
        use_amp=not args.no_amp
    )
