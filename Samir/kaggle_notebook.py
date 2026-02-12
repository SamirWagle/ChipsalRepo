# %% [markdown]
# # Multimodal Classification - Image + Text
# 
# This notebook implements a multimodal classification system combining:
# - **CLIP** (image encoder)
# - **BGE-M3** (text encoder)
# - **Hybrid Fusion** (attention-based combination)
# 
# ## Dataset Setup
# Upload your dataset to Kaggle with this structure:
# ```
# /kaggle/input/your-dataset-name/
#   ├── train.csv          # Columns: index, text, label
#   └── train_images/      # Folder with images (0.png, 1.png, etc.)
# ```

# %% [code]
# ============================
# 1. INSTALL DEPENDENCIES
# ============================
import sys
!pip install transformers -q
!pip install scikit-learn -q
!pip install matplotlib seaborn -q

# %% [code]
# ============================
# 2. IMPORTS
# ============================
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import random
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    CLIPVisionModel, CLIPProcessor,
    AutoModel, AutoTokenizer
)

# %% [code]
# ============================
# 3. CONFIGURATION
# ============================
class Config:
    # DATASET - UPDATE THESE PATHS FOR YOUR KAGGLE DATASET
    IMAGE_FOLDER = "/kaggle/input/your-dataset-name/train_images"
    CSV_PATH = "/kaggle/input/your-dataset-name/train.csv"
    IMAGE_COL = "index"
    TEXT_COL = "text"
    LABEL_COL = "label"
    
    # Experiment settings
    USE_MODALITY = 'both'  # 'image', 'text', or 'both'
    FUSION_STRATEGY = 'hybrid'  # 'early', 'late_voting', 'hybrid'
    IMAGE_TYPE = 'original'
    
    # Model architecture
    IMAGE_ENCODER = 'clip'
    CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
    FREEZE_IMAGE_ENCODER = True
    
    TEXT_ENCODER = 'bge-m3'
    BGE_MODEL_NAME = 'BAAI/bge-m3'
    FREEZE_TEXT_ENCODER = True
    
    IMAGE_FEATURE_DIM = 512
    TEXT_FEATURE_DIM = 1024
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    HYBRID_ATTENTION_HEADS = 4
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30  # Reduced for Kaggle
    K_FOLDS = 5
    PATIENCE = 10
    MIN_DELTA = 1e-4
    
    # Optimization
    OPTIMIZER = 'adamw'
    USE_SCHEDULER = True
    SCHEDULER_TYPE = 'cosine'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output
    SAVE_DIR = "/kaggle/working/outputs"
    LOG_INTERVAL = 10
    SAVE_BEST_MODEL = True
    
    # Image preprocessing
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Text preprocessing
    MAX_TEXT_LENGTH = 77
    
    # Random seed
    SEED = 42
    NUM_CLASSES = None  # Auto-detected

config = Config()
print(f"Device: {config.DEVICE}")
print(f"Modality: {config.USE_MODALITY}")
print(f"Fusion Strategy: {config.FUSION_STRATEGY}")

# %% [code]
# ============================
# 4. DATASET CLASS
# ============================
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_folder, image_col='index', text_col='text',
                 label_col='label', image_transform=None, text_tokenizer=None,
                 use_modality='both', image_type='original'):
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
        label = int(row[self.label_col])
        
        image_tensor = None
        text_encoding = None
        
        # Load image
        if self.use_modality in ['image', 'both']:
            image_filename = str(row[self.image_col])
            if not image_filename.endswith(('.png', '.jpg', '.jpeg')):
                image_filename = f"{image_filename}.png"
            
            image_path = os.path.join(self.image_folder, image_filename)
            
            try:
                image = Image.open(image_path).convert('RGB')
                if self.image_transform:
                    image_tensor = self.image_transform(image)
                else:
                    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                image_tensor = torch.zeros(3, 224, 224)
        
        # Process text
        if self.use_modality in ['text', 'both']:
            text = str(row[self.text_col]) if pd.notna(row[self.text_col]) else ""
            if self.text_tokenizer:
                text_encoding = self.text_tokenizer(text)
        
        # Return based on modality
        if self.use_modality == 'image':
            return image_tensor, label
        elif self.use_modality == 'text':
            return text_encoding, label
        else:
            return image_tensor, text_encoding, label

# %% [code]
# ============================
# 5. ENCODERS
# ============================
class ImageTransform:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, image):
        processed = self.processor(images=image, return_tensors="pt")
        return processed['pixel_values'].squeeze(0)

class TextTokenizer:
    def __init__(self, tokenizer, max_length=77):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, text):
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class ImageEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', freeze=True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.feature_dim = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output

class TextEncoder(nn.Module):
    def __init__(self, model_name='BAAI/bge-m3', freeze=True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self.feature_dim = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

# %% [code]
# ============================
# 6. FUSION MODELS
# ============================
class UnimodalModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features):
        return self.classifier(features), features

class HybridFusionModel(nn.Module):
    def __init__(self, image_dim, text_dim, num_classes, hidden_dim=256, 
                 dropout=0.3, num_heads=4):
        super().__init__()
        
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, image_features, text_features):
        img_proj = self.image_proj(image_features)
        txt_proj = self.text_proj(text_features)
        
        combined = torch.stack([img_proj, txt_proj], dim=1)
        attended, _ = self.attention(combined, combined, combined)
        
        fused = self.layer_norm(attended.mean(dim=1) + img_proj)
        logits = self.classifier(fused)
        
        return logits, fused

# %% [code]
# ============================
# 7. COMPLETE MODEL
# ============================
class MultimodalClassifier(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.use_modality = config.USE_MODALITY
        
        if self.use_modality in ['image', 'both']:
            self.image_encoder = ImageEncoder(
                config.CLIP_MODEL_NAME,
                freeze=config.FREEZE_IMAGE_ENCODER
            )
        
        if self.use_modality in ['text', 'both']:
            self.text_encoder = TextEncoder(
                config.BGE_MODEL_NAME,
                freeze=config.FREEZE_TEXT_ENCODER
            )
        
        if self.use_modality == 'image':
            self.fusion_model = UnimodalModel(
                config.IMAGE_FEATURE_DIM, num_classes,
                config.HIDDEN_DIM, config.DROPOUT
            )
        elif self.use_modality == 'text':
            self.fusion_model = UnimodalModel(
                config.TEXT_FEATURE_DIM, num_classes,
                config.HIDDEN_DIM, config.DROPOUT
            )
        else:
            self.fusion_model = HybridFusionModel(
                config.IMAGE_FEATURE_DIM, config.TEXT_FEATURE_DIM,
                num_classes, config.HIDDEN_DIM, config.DROPOUT,
                config.HYBRID_ATTENTION_HEADS
            )
    
    def forward(self, images=None, text_inputs=None, attention_mask=None):
        if self.use_modality == 'image':
            img_features = self.image_encoder(images)
            return self.fusion_model(img_features)
        elif self.use_modality == 'text':
            txt_features = self.text_encoder(text_inputs, attention_mask)
            return self.fusion_model(txt_features)
        else:
            img_features = self.image_encoder(images)
            txt_features = self.text_encoder(text_inputs, attention_mask)
            return self.fusion_model(img_features, txt_features)

# %% [code]
# ============================
# 8. TRAINING UTILITIES
# ============================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop

def get_class_weights(csv_path, label_col):
    df = pd.read_csv(csv_path)
    class_counts = df[label_col].value_counts().sort_index()
    total = len(df)
    weights = total / (len(class_counts) * class_counts.values)
    return torch.FloatTensor(weights)

# %% [code]
# ============================
# 9. TRAINING LOOP
# ============================
def train_epoch(model, train_loader, criterion, optimizer, device, config):
    model.train()
    
    if config.USE_MODALITY in ['image', 'both'] and config.FREEZE_IMAGE_ENCODER:
        model.image_encoder.eval()
    if config.USE_MODALITY in ['text', 'both'] and config.FREEZE_TEXT_ENCODER:
        model.text_encoder.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        if config.USE_MODALITY == 'image':
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = attention_mask = None
        elif config.USE_MODALITY == 'text':
            text_data, labels = batch
            labels = labels.to(device)
            text_inputs = text_data['input_ids'].to(device)
            attention_mask = text_data['attention_mask'].to(device)
            images = None
        else:
            images, text_data, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            text_inputs = text_data['input_ids'].to(device)
            attention_mask = text_data['attention_mask'].to(device)
        
        optimizer.zero_grad()
        logits, _ = model(images, text_inputs, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            if config.USE_MODALITY == 'image':
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                text_inputs = attention_mask = None
            elif config.USE_MODALITY == 'text':
                text_data, labels = batch
                labels = labels.to(device)
                text_inputs = text_data['input_ids'].to(device)
                attention_mask = text_data['attention_mask'].to(device)
                images = None
            else:
                images, text_data, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                text_inputs = text_data['input_ids'].to(device)
                attention_mask = text_data['attention_mask'].to(device)
            
            logits, _ = model(images, text_inputs, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'labels': all_labels
    }

# %% [code]
# ============================
# 10. MAIN TRAINING FUNCTION
# ============================
def train_fold(model, train_loader, val_loader, config, fold_idx, save_dir):
    device = torch.device(config.DEVICE)
    model = model.to(device)
    
    class_weights = get_class_weights(config.CSV_PATH, config.LABEL_COL).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    early_stopping = EarlyStopping(config.PATIENCE, config.MIN_DELTA)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config)
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, os.path.join(save_dir, f'best_model_fold_{fold_idx}.pt'))
        
        scheduler.step()
        
        if early_stopping(val_metrics['loss'], epoch):
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    model.load_state_dict(best_model_state)
    final_metrics = validate(model, val_loader, criterion, device, config)
    
    return model, history, final_metrics

# %% [code]
# ============================
# 11. DATA LOADING
# ============================
print("Loading dataset...")
df = pd.read_csv(config.CSV_PATH)
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Detect number of classes
num_classes = df[config.LABEL_COL].nunique()
config.NUM_CLASSES = num_classes
print(f"\nNumber of classes: {num_classes}")

# Class distribution
print("\nClass distribution:")
for cls, count in df[config.LABEL_COL].value_counts().sort_index().items():
    pct = 100 * count / len(df)
    print(f"  Class {cls}: {count} samples ({pct:.2f}%)")

# %% [code]
# ============================
# 12. PREPARE PREPROCESSING
# ============================
print("\nLoading preprocessing models...")

image_transform = None
text_tokenizer = None

if config.USE_MODALITY in ['image', 'both']:
    clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    image_transform = ImageTransform(clip_processor)
    print("✓ Image processor loaded")

if config.USE_MODALITY in ['text', 'both']:
    tokenizer = AutoTokenizer.from_pretrained(config.BGE_MODEL_NAME)
    text_tokenizer = TextTokenizer(tokenizer, config.MAX_TEXT_LENGTH)
    print("✓ Text tokenizer loaded")

# %% [code]
# ============================
# 13. K-FOLD TRAINING
# ============================
set_seed(config.SEED)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(config.SAVE_DIR, f"{config.USE_MODALITY}_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

print(f"\nExperiment directory: {save_dir}")

kfold = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1}/{config.K_FOLDS}")
    print(f"{'='*80}")
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    train_dataset = MultimodalDataset(
        train_df, config.IMAGE_FOLDER, config.IMAGE_COL, config.TEXT_COL,
        config.LABEL_COL, image_transform, text_tokenizer, config.USE_MODALITY
    )
    
    val_dataset = MultimodalDataset(
        val_df, config.IMAGE_FOLDER, config.IMAGE_COL, config.TEXT_COL,
        config.LABEL_COL, image_transform, text_tokenizer, config.USE_MODALITY
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    model = MultimodalClassifier(config, num_classes)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} total parameters")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    trained_model, history, metrics = train_fold(
        model, train_loader, val_loader, config, fold_idx, save_dir
    )
    
    fold_results.append({
        'fold': fold_idx + 1,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall']
    })
    
    # Save fold results
    with open(os.path.join(save_dir, f'fold_{fold_idx}_results.json'), 'w') as f:
        json.dump({
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items() if k not in ['predictions', 'labels']},
            'history': history
        }, f, indent=4)
    
    # Plot confusion matrix
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_fold_{fold_idx}.png'))
    plt.close()

# %% [code]
# ============================
# 14. FINAL RESULTS
# ============================
print("\n" + "="*80)
print("CROSS-VALIDATION SUMMARY")
print("="*80)

results_df = pd.DataFrame(fold_results)
print(results_df)

print("\n" + "="*80)
print("AVERAGE METRICS ACROSS FOLDS")
print("="*80)

for metric in ['accuracy', 'f1', 'precision', 'recall']:
    mean_val = results_df[metric].mean()
    std_val = results_df[metric].std()
    print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")

# Save summary
summary = {
    'config': {
        'use_modality': config.USE_MODALITY,
        'fusion_strategy': config.FUSION_STRATEGY,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'num_epochs': config.NUM_EPOCHS
    },
    'results': {
        'accuracy': {'mean': float(results_df['accuracy'].mean()), 'std': float(results_df['accuracy'].std())},
        'f1': {'mean': float(results_df['f1'].mean()), 'std': float(results_df['f1'].std())},
        'precision': {'mean': float(results_df['precision'].mean()), 'std': float(results_df['precision'].std())},
        'recall': {'mean': float(results_df['recall'].mean()), 'std': float(results_df['recall'].std())}
    },
    'fold_results': fold_results
}

with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ Results saved to {save_dir}")
print("\nTraining complete! 🎉")
