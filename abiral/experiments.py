"""
Example experiment scripts for ablation studies
Run different modality and fusion combinations
"""

from config import Config
from train import main
import torch


def experiment_image_only():
    """Experiment: Image only with original images"""
    print("\n" + "="*80)
    print("EXPERIMENT: Image Only (Original)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='image',
        IMAGE_TYPE='original',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_text_only():
    """Experiment: Text only"""
    print("\n" + "="*80)
    print("EXPERIMENT: Text Only")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images",
        CSV_PATH="../train.csv",
        USE_MODALITY='text',
        NUM_EPOCHS=40,
        BATCH_SIZE=256,
        LEARNING_RATE=1e-5,
        K_FOLDS=5,
        
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_text_only_improved():
    """Experiment: Text only with improvements (Unfrozen, Weighted, LR adjusted)"""
    print("\n" + "="*80)
    print("EXPERIMENT: Text Only (Improved)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images",
        CSV_PATH="../train/train.csv",
        USE_MODALITY='text',
        NUM_EPOCHS=30,  # Reduced from 50 as fine-tuning converges faster
        BATCH_SIZE=4, # Reduced batch size for fine-tuning memory
        LEARNING_RATE=2e-5, # Standard fine-tuning LR
        K_FOLDS=5,
        FREEZE_TEXT_ENCODER=False, # Unfreeze encoder
        UNFREEZE_TEXT_LAYERS=4,    # Unfreeze last 4/24 layers of BGE-M3 (17%)
        WEIGHT_DECAY=0.01,
        DROPOUT=0.5,
        # Class weight & label smoothing (KEY for fixing over-prediction)
        CLASS_WEIGHT_EXPONENT=0.3,  # Mild boost (was 0.5 = too aggressive)
        LABEL_SMOOTHING=0.1,        # Prevents overconfidence
    )
    
    summary, save_dir = main(config)
    return summary

def experiment_image_only_improved():
    """Experiment: Image only with improvements (Unfrozen CLIP, optimized for macro F1)"""
    print("\n" + "="*80)
    print("EXPERIMENT: Image Only (Improved - Unfrozen CLIP)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images",
        CSV_PATH="../train/train.csv",
        USE_MODALITY='image',
        IMAGE_TYPE='original',
        NUM_EPOCHS=25,  # CLIP converges faster than text models
        BATCH_SIZE=64,  # Optimized for M4 24GB (CLIP can handle larger batches)
        LEARNING_RATE=1e-5,  # CLIP needs lower LR (very sensitive to high LR)
        K_FOLDS=5,
        FREEZE_IMAGE_ENCODER=False,  # Unfreeze CLIP for fine-tuning
        UNFREEZE_IMAGE_LAYERS=3,  # Unfreeze last 3/12 layers of CLIP (25%)
        WEIGHT_DECAY=0.01,  # Strong regularization for small data
        DROPOUT=0.3,  # Moderate dropout (CLIP has built-in regularization)
    )
    
    summary, save_dir = main(config)
    return summary

def experiment_multimodal_early_fusion():
    """Experiment: Multimodal with Early Fusion"""
    print("\n" + "="*80)
    print("EXPERIMENT: Multimodal - Early Fusion")
    print("="*80 + "\n")
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images/",
        CSV_PATH="../train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='early',
        IMAGE_TYPE='original',
        NUM_EPOCHS=10,
        BATCH_SIZE=256,
        LEARNING_RATE=1e-5,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_multimodal_early_fusion_improved():
    """Experiment: Multimodal Early Fusion with PARTIAL Layer Unfreezing (OPTIMAL for small datasets)
    
    This is the RECOMMENDED configuration for meme classification:
    - Unfreezes only last few layers (not all)
    - Mild class weight exponent (0.3) to avoid over-predicting minority class
    - Label smoothing (0.1) for better generalization
    - Lower LR + higher dropout for small dataset stability
    """
    print("\n" + "="*80)
    print("EXPERIMENT: Multimodal - Early Fusion (OPTIMIZED)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="../train/train_images/",
        CSV_PATH="../train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='early',
        IMAGE_TYPE='original',
        
        # Training configuration
        NUM_EPOCHS=30,
        BATCH_SIZE=32,    # Smaller batch = better gradient estimates for small data
        K_FOLDS=5,        # MUST run all 5 folds for ensemble
        
        # Learning rate
        LEARNING_RATE=2e-5,
        
        # PARTIAL UNFREEZING
        FREEZE_IMAGE_ENCODER=True,
        FREEZE_TEXT_ENCODER=True,
        UNFREEZE_IMAGE_LAYERS=2,  # Conservative: last 2/12 CLIP layers
        UNFREEZE_TEXT_LAYERS=4,   # BGE-M3 is multilingual, good for Nepali
        
        # Regularization
        WEIGHT_DECAY=0.01,
        DROPOUT=0.5,       # Higher dropout for small dataset
        HIDDEN_DIM=256,
        
        # Class weight & label smoothing (KEY for fixing over-prediction)
        CLASS_WEIGHT_EXPONENT=0.3,  # Mild boost (was 0.5 = too aggressive)
        LABEL_SMOOTHING=0.1,        # Prevents overconfidence
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_multimodal_late_fusion_voting():
    """Experiment: Multimodal with Late Fusion (Voting)"""
    print("\n" + "="*80)
    print("EXPERIMENT: Multimodal - Late Fusion (Voting)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='late_voting',
        IMAGE_TYPE='original',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_multimodal_late_fusion_bagging():
    """Experiment: Multimodal with Late Fusion (Bagging)"""
    print("\n" + "="*80)
    print("EXPERIMENT: Multimodal - Late Fusion (Bagging)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='late_bagging',
        IMAGE_TYPE='original',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_multimodal_hybrid_fusion():
    """Experiment: Multimodal with Hybrid Fusion (PR,IORITY)"""
    print("\n" + "="*80)
    print("EXPERIMENT: Multimodal - Hybrid Fusion (PRIORITY)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='original',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5,
        HYBRID_ATTENTION_HEADS=4
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_text_removed_image():
    """Experiment: Text-removed images only"""
    print("\n" + "="*80)
    print("EXPERIMENT: Text-Removed Images Only")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='image',
        IMAGE_TYPE='text_removed',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def experiment_text_removed_with_text():
    """Experiment: Text-removed images + extracted text"""
    print("\n" + "="*80)
    print("EXPERIMENT: Text-Removed Images + Extracted Text (Hybrid Fusion)")
    print("="*80 + "\n")
    
    config = Config()
    config.update(
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='hybrid',
        IMAGE_TYPE='text_removed',
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
    )
    
    summary, save_dir = main(config)
    return summary


def run_all_ablation_experiments():
    """
    Run complete ablation study with all combinations
    
    Experiments:
    1. Image only (original)
    2. Text only
    3. Image + Text (Early Fusion)
    4. Image + Text (Late Fusion - Voting)
    5. Image + Text (Late Fusion - Bagging)
    6. Image + Text (Hybrid Fusion) - PRIORITY
    7. Text-removed image only
    8. Text-removed image + Text (Hybrid Fusion)
    """
    results = {}
    
    # Unimodal baselines
    results['image_only'] = experiment_image_only()
    results['text_only'] = experiment_text_only()
    
    # Multimodal experiments
    results['early_fusion'] = experiment_multimodal_early_fusion()
    results['late_voting'] = experiment_multimodal_late_fusion_voting()
    results['late_bagging'] = experiment_multimodal_late_fusion_bagging()
    results['hybrid_fusion'] = experiment_multimodal_hybrid_fusion()  # Priority
    
    # Text-removed experiments
    results['text_removed_image'] = experiment_text_removed_image()
    results['text_removed_multimodal'] = experiment_text_removed_with_text()
    
    # Print summary of all experiments
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80 + "\n")
    
    for exp_name, summary in results.items():
        print(f"{exp_name}:")
        print(f"  Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
        print(f"  F1 Score: {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
        print()
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        
        experiments = {
            'image_only': experiment_image_only,
            'text_only': experiment_text_only,
            'early': experiment_multimodal_early_fusion,
            'early_improved': experiment_multimodal_early_fusion_improved,
            'late_voting': experiment_multimodal_late_fusion_voting,
            'late_bagging': experiment_multimodal_late_fusion_bagging,
            'hybrid': experiment_multimodal_hybrid_fusion,
            'text_removed': experiment_text_removed_image,
            'text_removed_multimodal': experiment_text_removed_with_text,
            'text_only_improved': experiment_text_only_improved,
            'image_only_improved': experiment_image_only_improved,
            'all': run_all_ablation_experiments
        }
        
        if experiment_name in experiments:
            experiments[experiment_name]()
        else:
            print(f"Unknown experiment: {experiment_name}")
            print(f"Available experiments: {list(experiments.keys())}")
    else:
        print("Usage: python experiments.py <experiment_name>")
        print("Available experiments:")
        print("  - image_only")
        print("  - text_only")
        print("  - early (Early Fusion)")
        print("  - early_improved (Early Fusion OPTIMIZED for memes + M4)")
        print("  - late_voting (Late Fusion - Voting)")
        print("  - late_bagging (Late Fusion - Bagging)")
        print("  - hybrid (Hybrid Fusion - PRIORITY)")
        print("  - text_removed")
        print("  - text_removed_multimodal")
        print("  - text_only_improved (NEW: Unfrozen + Weighted)")
        print("  - image_only_improved (NEW: Unfrozen + Weighted)")
        print("  - all (Run all experiments)")
