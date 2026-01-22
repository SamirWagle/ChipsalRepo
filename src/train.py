"""
CHiPSAL 2026 - Subtask A: Training Pipeline
===========================================
Complete training pipeline with K-Fold CV, metrics tracking, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import time
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.dataset import MemeDataset, create_dataloaders
from src.models import create_model


class Trainer:
    """
    Complete training pipeline for hate speech detection.
    """
    
    def __init__(
        self,
        model_type='multimodal',
        text_model_name=None,
        image_model_name=None,
        experiment_name='baseline',
        use_amp=True
    ):
        """
        Initialize trainer.
        
        Args:
            model_type: 'text', 'image', 'multimodal', or 'clip'
            text_model_name: Name of text model
            image_model_name: Name of image model
            experiment_name: Name for this experiment
            use_amp: Use automatic mixed precision
        """
        self.model_type = model_type
        self.text_model_name = text_model_name or TEXT_MODEL_OPTIONS[DEFAULT_TEXT_MODEL]
        self.image_model_name = image_model_name or VISION_MODEL_OPTIONS[DEFAULT_VISION_MODEL]
        self.experiment_name = experiment_name
        self.use_amp = use_amp and torch.cuda.is_available()
        
        self.device = DEVICE
        self.scaler = GradScaler() if self.use_amp else None
        
        # Experiment directory
        self.exp_dir = OUTPUT_DIR / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tokenizer for text-based models
        if model_type in ['text', 'multimodal']:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        else:
            self.tokenizer = None
        
        print(f"[Trainer] Experiment: {experiment_name}")
        print(f"[Trainer] Model type: {model_type}")
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] AMP: {self.use_amp}")
    
    def compute_metrics(self, predictions, labels):
        """Compute evaluation metrics."""
        preds = np.array(predictions)
        labs = np.array(labels)
        
        metrics = {
            'accuracy': accuracy_score(labs, preds),
            'f1_macro': f1_score(labs, preds, average='macro'),
            'f1_weighted': f1_score(labs, preds, average='weighted'),
            'precision_macro': precision_score(labs, preds, average='macro', zero_division=0),
            'recall_macro': recall_score(labs, preds, average='macro', zero_division=0),
        }
        
        # Per-class F1
        f1_per_class = f1_score(labs, preds, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
        
        return metrics
    
    def train_epoch(self, model, dataloader, optimizer, scheduler, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            labels = batch['label'].to(self.device)
            
            # Prepare inputs based on model type
            inputs = {}
            if self.model_type in ['text', 'multimodal']:
                inputs['input_ids'] = batch['input_ids'].to(self.device)
                inputs['attention_mask'] = batch['attention_mask'].to(self.device)
            if self.model_type in ['image', 'multimodal']:
                inputs['image'] = batch['image'].to(self.device)
            
            optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits = model(**inputs)
                    loss = criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = model(**inputs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion):
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label'].to(self.device)
            
            inputs = {}
            if self.model_type in ['text', 'multimodal']:
                inputs['input_ids'] = batch['input_ids'].to(self.device)
                inputs['attention_mask'] = batch['attention_mask'].to(self.device)
            if self.model_type in ['image', 'multimodal']:
                inputs['image'] = batch['image'].to(self.device)
            
            if self.use_amp:
                with autocast():
                    logits = model(**inputs)
                    loss = criterion(logits, labels)
            else:
                logits = model(**inputs)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_predictions, all_labels)
        metrics['loss'] = avg_loss
        
        return metrics, all_predictions, all_labels
    
    def train_fold(
        self,
        fold,
        train_df,
        val_df,
        image_dir=TRAIN_IMAGES_DIR,
        text_column='text'
    ):
        """Train a single fold."""
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}")
        print(f"{'='*60}")
        
        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            image_dir=image_dir,
            tokenizer=self.tokenizer,
            text_column=text_column,
            batch_size=BATCH_SIZE,
            mode=self.model_type
        )
        
        # Create model
        model = create_model(
            model_type=self.model_type,
            text_model_name=self.text_model_name,
            image_model_name=self.image_model_name
        )
        model = model.to(self.device)
        
        # Class weights for imbalanced data
        class_counts = train_df['label'].value_counts().sort_index()
        class_weights = torch.tensor(
            [len(train_df) / (len(class_counts) * c) for c in class_counts],
            dtype=torch.float32
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Scheduler
        total_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_f1 = 0
        best_epoch = 0
        patience_counter = 0
        history = {'train': [], 'val': []}
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )
            
            # Evaluate
            val_metrics, val_preds, val_labels = self.evaluate(
                model, val_loader, criterion
            )
            
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, F1-macro: {train_metrics['f1_macro']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1-macro: {val_metrics['f1_macro']:.4f}")
            
            # Save best model
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'val_metrics': val_metrics
                }
                torch.save(
                    checkpoint,
                    self.exp_dir / f'best_model_fold_{fold}.pt'
                )
                print(f"[SAVE] New best model! F1: {best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"[EARLY STOP] No improvement for {PATIENCE} epochs")
                    break
        
        print(f"\n[Fold {fold + 1}] Best F1: {best_f1:.4f} at epoch {best_epoch + 1}")
        
        # Load best model and get final predictions
        checkpoint = torch.load(self.exp_dir / f'best_model_fold_{fold}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'fold': fold,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'history': history
        }
    
    def train_kfold(self, df, n_folds=NUM_FOLDS, image_dir=TRAIN_IMAGES_DIR, text_column='text'):
        """
        Train with K-Fold cross-validation.
        
        Args:
            df: Training DataFrame
            n_folds: Number of folds
            image_dir: Image directory
            text_column: Column name for text
            
        Returns:
            results: Dictionary with fold results
        """
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION (K={n_folds})")
        print(f"{'='*60}")
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        
        fold_results = []
        oof_predictions = np.zeros(len(df))
        oof_labels = df['label'].values
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            
            result = self.train_fold(
                fold=fold,
                train_df=train_df,
                val_df=val_df,
                image_dir=image_dir,
                text_column=text_column
            )
            
            fold_results.append(result)
            
            # Clear memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Aggregate results
        avg_f1 = np.mean([r['best_f1'] for r in fold_results])
        std_f1 = np.std([r['best_f1'] for r in fold_results])
        
        print(f"\n{'='*60}")
        print(f"K-FOLD RESULTS")
        print(f"{'='*60}")
        for r in fold_results:
            print(f"Fold {r['fold'] + 1}: F1 = {r['best_f1']:.4f}")
        print(f"\nAverage F1: {avg_f1:.4f} ± {std_f1:.4f}")
        
        # Save results
        results = {
            'experiment_name': self.experiment_name,
            'model_type': self.model_type,
            'n_folds': n_folds,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'fold_results': fold_results
        }
        
        with open(self.exp_dir / 'kfold_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def main():
    """Main training pipeline."""
    print("="*60)
    print("CHiPSAL 2026 - SUBTASK A: TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"[INFO] Loaded {len(df)} samples")
    
    # You can choose different experiment configurations:
    
    # 1. Text-only baseline
    # trainer = Trainer(
    #     model_type='text',
    #     text_model_name='xlm-roberta-base',
    #     experiment_name='text_xlm_roberta'
    # )
    
    # 2. Image-only baseline
    # trainer = Trainer(
    #     model_type='image',
    #     image_model_name='resnet50',
    #     experiment_name='image_resnet50'
    # )
    
    # 3. Multimodal (recommended)
    trainer = Trainer(
        model_type='multimodal',
        text_model_name='xlm-roberta-base',
        image_model_name='resnet50',
        experiment_name='multimodal_baseline'
    )
    
    # Train with K-Fold CV
    results = trainer.train_kfold(
        df=df,
        n_folds=NUM_FOLDS,
        image_dir=TRAIN_IMAGES_DIR,
        text_column='text'
    )
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Results saved to {trainer.exp_dir}")
    
    return results


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    results = main()
