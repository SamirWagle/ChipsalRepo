"""
Utility functions for analysis and visualization
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_experiment_results(experiment_dir):
    """
    Load results from an experiment directory
    
    Args:
        experiment_dir: Path to experiment output directory
        
    Returns:
        dict with configuration, summary, and fold results
    """
    with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(experiment_dir, 'cross_validation_summary.json'), 'r') as f:
        summary = json.load(f)
    
    return {
        'config': config,
        'summary': summary['summary'],
        'folds': summary['fold_results']
    }


def compare_experiments(experiment_dirs, experiment_names=None):
    """
    Compare results from multiple experiments
    
    Args:
        experiment_dirs: List of experiment directory paths
        experiment_names: Optional list of names for experiments
        
    Returns:
        DataFrame with comparison
    """
    if experiment_names is None:
        experiment_names = [f"Exp_{i+1}" for i in range(len(experiment_dirs))]
    
    results = []
    
    for exp_dir, exp_name in zip(experiment_dirs, experiment_names):
        data = load_experiment_results(exp_dir)
        
        results.append({
            'Experiment': exp_name,
            'Modality': data['config']['USE_MODALITY'],
            'Fusion': data['config'].get('FUSION_STRATEGY', 'N/A'),
            'Image Type': data['config'].get('IMAGE_TYPE', 'N/A'),
            'Accuracy': f"{data['summary']['accuracy']['mean']:.4f} ± {data['summary']['accuracy']['std']:.4f}",
            'F1 Score': f"{data['summary']['f1']['mean']:.4f} ± {data['summary']['f1']['std']:.4f}",
            'Precision': f"{data['summary']['precision']['mean']:.4f} ± {data['summary']['precision']['std']:.4f}",
            'Recall': f"{data['summary']['recall']['mean']:.4f} ± {data['summary']['recall']['std']:.4f}"
        })
    
    return pd.DataFrame(results)


def plot_experiment_comparison(experiment_dirs, experiment_names=None, save_path=None):
    """
    Plot comparison of experiments
    
    Args:
        experiment_dirs: List of experiment directory paths
        experiment_names: Optional list of names
        save_path: Optional path to save plot
    """
    if experiment_names is None:
        experiment_names = [f"Exp_{i+1}" for i in range(len(experiment_dirs))]
    
    # Load data
    metrics = {
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': []
    }
    
    for exp_dir in experiment_dirs:
        data = load_experiment_results(exp_dir)
        for metric in metrics.keys():
            metrics[metric].append(data['summary'][metric]['mean'])
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experiment Comparison', fontsize=16)
    
    metric_names = ['accuracy', 'f1', 'precision', 'recall']
    titles = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(experiment_names))
        bars = ax.bar(x, metrics[metric], color='steelblue', alpha=0.7)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{metrics[metric][i]:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def analyze_fold_variance(experiment_dir):
    """
    Analyze variance across folds
    
    Args:
        experiment_dir: Path to experiment directory
    """
    data = load_experiment_results(experiment_dir)
    
    print(f"\nVariance Analysis for: {data['config']['USE_MODALITY']} - {data['config'].get('FUSION_STRATEGY', 'N/A')}")
    print("="*60)
    
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        values = data['summary'][metric]['values']
        mean = data['summary'][metric]['mean']
        std = data['summary'][metric]['std']
        cv = (std / mean) * 100 if mean > 0 else 0  # Coefficient of variation
        
        print(f"\n{metric.upper()}:")
        print(f"  Mean: {mean:.4f}")
        print(f"  Std:  {std:.4f}")
        print(f"  CV:   {cv:.2f}%")
        print(f"  Min:  {min(values):.4f}")
        print(f"  Max:  {max(values):.4f}")
        print(f"  Values: {[f'{v:.4f}' for v in values]}")


def create_results_table(experiment_dirs, experiment_names=None, save_path=None):
    """
    Create a formatted results table
    
    Args:
        experiment_dirs: List of experiment directory paths
        experiment_names: Optional list of names
        save_path: Optional path to save table (CSV or Excel)
    """
    df = compare_experiments(experiment_dirs, experiment_names)
    
    if save_path:
        if save_path.endswith('.csv'):
            df.to_csv(save_path, index=False)
        elif save_path.endswith('.xlsx'):
            df.to_excel(save_path, index=False)
        print(f"Results table saved to {save_path}")
    
    return df


def plot_training_curves_comparison(experiment_dirs, experiment_names=None, fold_idx=0, save_path=None):
    """
    Compare training curves from multiple experiments
    
    Args:
        experiment_dirs: List of experiment directory paths
        experiment_names: Optional list of names
        fold_idx: Which fold to compare
        save_path: Optional path to save plot
    """
    if experiment_names is None:
        experiment_names = [f"Exp_{i+1}" for i in range(len(experiment_dirs))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves Comparison (Fold {fold_idx})', fontsize=16)
    
    for exp_dir, exp_name in zip(experiment_dirs, experiment_names):
        # Load fold results
        with open(os.path.join(exp_dir, f'fold_{fold_idx}_results.json'), 'r') as f:
            fold_data = json.load(f)
        
        history = fold_data['history']
        
        # Plot loss
        axes[0, 0].plot(history['train_loss'], label=f'{exp_name} (train)', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')
        
        # Plot accuracy
        axes[0, 1].plot(history['train_acc'], label=f'{exp_name} (train)', alpha=0.7)
        axes[0, 1].plot(history['val_acc'], label=f'{exp_name} (val)', alpha=0.7, linestyle='--')
        
        # Plot F1
        axes[1, 0].plot(history['val_f1'], label=exp_name, alpha=0.7)
        
        # Plot learning rate
        axes[1, 1].plot(history['learning_rates'], label=exp_name, alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves comparison saved to {save_path}")
    
    plt.show()


def find_best_experiment(results_dir='outputs'):
    """
    Find the best performing experiment in the results directory
    
    Args:
        results_dir: Directory containing experiment outputs
        
    Returns:
        Path to best experiment and its results
    """
    experiment_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir()]
    
    if not experiment_dirs:
        print(f"No experiments found in {results_dir}")
        return None
    
    best_exp = None
    best_f1 = -1
    
    results = []
    
    for exp_dir in experiment_dirs:
        try:
            data = load_experiment_results(exp_dir)
            f1_mean = data['summary']['f1']['mean']
            
            results.append({
                'path': str(exp_dir),
                'name': exp_dir.name,
                'f1': f1_mean,
                'accuracy': data['summary']['accuracy']['mean']
            })
            
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_exp = exp_dir
        except:
            continue
    
    # Sort by F1 score
    results = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print("\nTop 5 Experiments by F1 Score:")
    print("="*80)
    for i, res in enumerate(results[:5]):
        print(f"{i+1}. {res['name']}")
        print(f"   F1: {res['f1']:.4f} | Accuracy: {res['accuracy']:.4f}")
        print(f"   Path: {res['path']}")
        print()
    
    return str(best_exp), results[0] if results else None


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'find_best':
            best_exp, best_res = find_best_experiment()
            print(f"\nBest experiment: {best_exp}")
        
        elif sys.argv[1] == 'analyze' and len(sys.argv) > 2:
            analyze_fold_variance(sys.argv[2])
        
        elif sys.argv[1] == 'compare' and len(sys.argv) > 2:
            exp_dirs = sys.argv[2:]
            df = compare_experiments(exp_dirs)
            print("\n" + df.to_string())
    else:
        print("Usage:")
        print("  python utils.py find_best")
        print("  python utils.py analyze <experiment_dir>")
        print("  python utils.py compare <exp_dir1> <exp_dir2> ...")
