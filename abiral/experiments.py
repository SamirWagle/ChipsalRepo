"""
Example experiment scripts for ablation studies
Run different modality and fusion combinations
"""

from config import Config
from train import main


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
        NUM_EPOCHS=50,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5,
        
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
        IMAGE_FOLDER="data/images",
        CSV_PATH="data/train.csv",
        USE_MODALITY='both',
        FUSION_STRATEGY='early',
        IMAGE_TYPE='original',
        NUM_EPOCHS=20,
        BATCH_SIZE=16,
        LEARNING_RATE=1e-4,
        K_FOLDS=5
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
            'late_voting': experiment_multimodal_late_fusion_voting,
            'late_bagging': experiment_multimodal_late_fusion_bagging,
            'hybrid': experiment_multimodal_hybrid_fusion,
            'text_removed': experiment_text_removed_image,
            'text_removed_multimodal': experiment_text_removed_with_text,
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
        print("  - late_voting (Late Fusion - Voting)")
        print("  - late_bagging (Late Fusion - Bagging)")
        print("  - hybrid (Hybrid Fusion - PRIORITY)")
        print("  - text_removed")
        print("  - text_removed_multimodal")
        print("  - all (Run all experiments)")
