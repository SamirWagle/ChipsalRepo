"""
Main training script for multimodal classification experiments
Supports K-fold cross-validation with all fusion strategies
"""

import os
import json
import sys
import torch
import numpy as np
import random
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import create_dataloaders, get_class_distribution
from encoders import create_encoders, get_image_processor, get_text_tokenizer, ImageTransform
from fusion_models import create_fusion_model
from model import create_complete_model
from trainer import train_fold, plot_training_history, plot_confusion_matrix


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_name(config):
    """Create descriptive experiment name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = []
    parts.append(config.USE_MODALITY)
    
    if config.USE_MODALITY == 'both':
        parts.append(config.FUSION_STRATEGY)
    
    if config.USE_MODALITY in ['image', 'both']:
        parts.append(config.IMAGE_TYPE)
    
    parts.append(timestamp)
    
    return "_".join(parts)


def main(config):
    """
    Main training function
    
    Args:
        config: Configuration object
    """
    # Print configuration
    config.print_config()
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create experiment directory
    exp_name = create_experiment_name(config)
    save_dir = os.path.join(config.SAVE_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nExperiment: {exp_name}")
    print(f"Save directory: {save_dir}\n")
    
    # Save configuration
    config_dict = config.get_config_dict()
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Get class weights for imbalanced data
    class_weights = get_class_distribution(config.CSV_PATH, config.LABEL_COL)
    config.CLASS_WEIGHTS = class_weights
    
    # Create preprocessing functions
    image_transform = None
    text_tokenizer = None
    
    if config.USE_MODALITY in ['image', 'both']:
        image_processor = get_image_processor(config.CLIP_MODEL_NAME)
        image_transform = ImageTransform(image_processor)
    
    if config.USE_MODALITY in ['text', 'both']:
        text_tokenizer = get_text_tokenizer(
            config.TEXT_ENCODER,
            config.BGE_MODEL_NAME if config.TEXT_ENCODER == 'bge-m3' else config.E5_MODEL_NAME,
            config.MAX_TEXT_LENGTH
        )
    
    # K-fold cross-validation
    fold_results = []
    
    for fold_idx in range(config.K_FOLDS):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{config.K_FOLDS}")
        print(f"{'='*80}\n")
        
        # Create data loaders
        train_loader, val_loader, num_classes = create_dataloaders(
            csv_path=config.CSV_PATH,
            image_folder=config.IMAGE_FOLDER,
            image_col=config.IMAGE_COL,
            text_col=config.TEXT_COL,
            label_col=config.LABEL_COL,
            batch_size=config.BATCH_SIZE,
            image_transform=image_transform,
            text_tokenizer=text_tokenizer,
            use_modality=config.USE_MODALITY,
            image_type=config.IMAGE_TYPE,
            k_folds=config.K_FOLDS,
            fold_idx=fold_idx,
            seed=config.SEED
        )
        
        # Update num_classes in config if not set
        if config.NUM_CLASSES is None:
            config.NUM_CLASSES = num_classes
            print(f"Number of classes set to: {num_classes}")
        
        # Create model
        print("\nCreating model...")
        image_encoder, text_encoder = create_encoders(config)
        fusion_model = create_fusion_model(config)
        model = create_complete_model(
            image_encoder, text_encoder, fusion_model, config.USE_MODALITY
        )
        
        # Print model architecture
        print("\nModel Architecture:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train fold
        trained_model, history, final_metrics = train_fold(
            model, train_loader, val_loader, config, fold_idx, save_dir
        )
        
        # Save results
        fold_results.append({
            'fold': fold_idx,
            'accuracy': final_metrics['accuracy'],
            'f1': final_metrics['f1'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'loss': final_metrics['loss']
        })
        
        # Plot training history
        plot_training_history(
            history,
            os.path.join(save_dir, f'training_history_fold_{fold_idx}.png')
        )
        
        # Plot confusion matrix
        class_names = [f"Class {i}" for i in range(config.NUM_CLASSES)]
        plot_confusion_matrix(
            final_metrics['labels'],
            final_metrics['predictions'],
            class_names,
            os.path.join(save_dir, f'confusion_matrix_fold_{fold_idx}.png')
        )
        
        # Save fold results
        with open(os.path.join(save_dir, f'fold_{fold_idx}_results.json'), 'w') as f:
            json.dump({
                'metrics': {
                    'accuracy': float(final_metrics['accuracy']),
                    'f1': float(final_metrics['f1']),
                    'precision': float(final_metrics['precision']),
                    'recall': float(final_metrics['recall']),
                    'loss': float(final_metrics['loss'])
                },
                'history': {k: [float(v) for v in vals] for k, vals in history.items()}
            }, f, indent=4)
    
    # Aggregate results across folds
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*80}\n")
    
    metrics_names = ['accuracy', 'f1', 'precision', 'recall', 'loss']
    summary = {}
    
    for metric in metrics_names:
        values = [fold[metric] for fold in fold_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
        print(f"{metric.capitalize()}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")
    
    # Save summary
    with open(os.path.join(save_dir, 'cross_validation_summary.json'), 'w') as f:
        json.dump({
            'experiment_name': exp_name,
            'configuration': config_dict,
            'summary': {k: {
                'mean': float(v['mean']),
                'std': float(v['std']),
                'values': [float(x) for x in v['values']]
            } for k, v in summary.items()},
            'fold_results': fold_results
        }, f, indent=4)
    
    print(f"\nExperiment completed! Results saved to: {save_dir}")
    
    return summary, save_dir


if __name__ == '__main__':
    # You can update configuration here or create different experiment configs
    
    # Example: Train with hybrid fusion on both modalities
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images/",
        CSV_PATH="../train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='original',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    main(config)
