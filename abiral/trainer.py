"""
Training module with K-fold cross-validation and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def get_optimizer(model, config):
    """Create optimizer based on config"""
    if config.OPTIMIZER == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=config.SGD_MOMENTUM,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")


def get_scheduler(optimizer, config, steps_per_epoch=None):
    """Create learning rate scheduler based on config"""
    if not config.USE_SCHEDULER:
        return None
    
    if config.SCHEDULER_TYPE == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS,
            eta_min=config.LEARNING_RATE * 0.01
        )
    elif config.SCHEDULER_TYPE == 'step':
        return StepLR(
            optimizer,
            step_size=config.SCHEDULER_STEP_SIZE,
            gamma=config.SCHEDULER_FACTOR
        )
    elif config.SCHEDULER_TYPE == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.SCHEDULER_TYPE}")


def train_epoch(model, train_loader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    
    # Set encoders to eval mode if frozen
    if config.USE_MODALITY in ['image', 'both'] and config.FREEZE_IMAGE_ENCODER:
        model.image_encoder.eval()
    if config.USE_MODALITY in ['text', 'both'] and config.FREEZE_TEXT_ENCODER:
        model.text_encoder.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        # Parse batch based on modality
        if config.USE_MODALITY == 'image':
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = None
            attention_mask = None
        elif config.USE_MODALITY == 'text':
            text_data, labels = batch
            labels = labels.to(device)
            text_inputs = text_data['input_ids'].squeeze(1).to(device)
            attention_mask = text_data['attention_mask'].squeeze(1).to(device)
            images = None
        else:  # both
            images, text_data, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = text_data['input_ids'].squeeze(1).to(device)
            attention_mask = text_data['attention_mask'].squeeze(1).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(images, text_inputs, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, config):
    """Validate the model"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Parse batch based on modality
            if config.USE_MODALITY == 'image':
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                text_inputs = None
                attention_mask = None
            elif config.USE_MODALITY == 'text':
                text_data, labels = batch
                labels = labels.to(device)
                text_inputs = text_data['input_ids'].squeeze(1).to(device)
                attention_mask = text_data['attention_mask'].squeeze(1).to(device)
                images = None
            else:  # both
                images, text_data, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                text_inputs = text_data['input_ids'].squeeze(1).to(device)
                attention_mask = text_data['attention_mask'].squeeze(1).to(device)
            
            # Forward pass
            logits, _ = model(images, text_inputs, attention_mask)
            loss = criterion(logits, labels)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Compute additional metrics
    if config.NUM_CLASSES == 2:
        f1 = f1_score(all_labels, all_preds, average='binary')
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
    else:
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return metrics


def train_fold(model, train_loader, val_loader, config, fold_idx, save_dir):
    """Train model for one fold"""
    
    device = torch.device(config.DEVICE)
    model = model.to(device)
    
    # Get class weights and label smoothing
    label_smoothing = getattr(config, 'LABEL_SMOOTHING', 0.0)
    
    if hasattr(config, 'CLASS_WEIGHTS') and config.CLASS_WEIGHTS is not None:
        class_weights = config.CLASS_WEIGHTS.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    if label_smoothing > 0:
        print(f"  Using label smoothing: {label_smoothing}")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA,
        mode='min'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            if config.SAVE_BEST_MODEL:
                torch.save(
                    best_model_state,
                    os.path.join(save_dir, f'best_model_fold_{fold_idx}.pt')
                )
        
        # Learning rate scheduling
        if scheduler is not None:
            if config.SCHEDULER_TYPE == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping(val_metrics['loss'], epoch):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best epoch was {early_stopping.best_epoch + 1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final validation
    final_metrics = validate(model, val_loader, criterion, device, config)
    
    return model, history, final_metrics


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
