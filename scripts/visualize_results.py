#!/usr/bin/env python3
"""
Visualize experimental results.

Creates plots and figures for the paper:
- Model comparison bar chart
- Training curves
- Confusion matrices
- Gating weight histograms
- Attention heatmaps

Usage:
    python visualize_results.py --results results.csv --output figures/
    python visualize_results.py --training_logs logs/ --output figures/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Configure plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_model_comparison(df: pd.DataFrame, output_path: Path):
    """Plot F1-Macro comparison across all models."""
    logger.info("Generating model comparison chart")
    
    # Sort by F1 score
    df_sorted = df.sort_values("f1_macro", ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart
    bars = ax.barh(df_sorted["model"], df_sorted["f1_macro"])
    
    # Color best model differently
    max_idx = df_sorted["f1_macro"].idxmax()
    for i, bar in enumerate(bars):
        if i == len(bars) - 1:  # Best model (last after sorting)
            bar.set_color("#2E8B57")  # Green
            bar.set_edgecolor("black")
            bar.set_linewidth(2)
        else:
            bar.set_color("#4682B4")  # Blue
    
    # Add value labels
    for i, (model, f1) in enumerate(zip(df_sorted["model"], df_sorted["f1_macro"])):
        ax.text(f1 + 0.005, i, f"{f1:.4f}", va="center", fontsize=11)
    
    ax.set_xlabel("F1-Macro Score", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    ax.set_title("Model Comparison: F1-Macro Scores", fontsize=16, fontweight="bold")
    ax.set_xlim(0.5, 0.75)
    
    # Add baseline line at 0.68
    ax.axvline(x=0.6834, color="red", linestyle="--", alpha=0.5, label="Best Model (M7)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path / "model_comparison.pdf", bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved model comparison to {output_path}")


def plot_training_curves(log_dir: Path, output_path: Path):
    """Plot training and validation curves."""
    logger.info("Generating training curves")
    
    # Find training logs
    log_files = list(log_dir.glob("**/training_log.csv"))
    
    if not log_files:
        logger.warning(f"No training logs found in {log_dir}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for log_file in log_files[:4]:  # Plot first 4 models
        model_name = log_file.parent.name
        df = pd.read_csv(log_file)
        
        # Loss
        axes[0, 0].plot(df["epoch"], df["train_loss"], label=f"{model_name} (train)", alpha=0.7)
        axes[0, 0].plot(df["epoch"], df["val_loss"], label=f"{model_name} (val)", alpha=0.7, linestyle="--")
        
        # F1
        axes[0, 1].plot(df["epoch"], df["train_f1"], label=f"{model_name} (train)", alpha=0.7)
        axes[0, 1].plot(df["epoch"], df["val_f1"], label=f"{model_name} (val)", alpha=0.7, linestyle="--")
        
        # Accuracy
        axes[1, 0].plot(df["epoch"], df["train_acc"], label=f"{model_name} (train)", alpha=0.7)
        axes[1, 0].plot(df["epoch"], df["val_acc"], label=f"{model_name} (val)", alpha=0.7, linestyle="--")
        
        # Learning rate
        if "learning_rate" in df.columns:
            axes[1, 1].plot(df["epoch"], df["learning_rate"], label=model_name, alpha=0.7)
    
    # Configure subplots
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("F1-Macro Score")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1-Macro")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("Training Curves", fontsize=18, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved training curves to {output_path}")


def plot_confusion_matrix(confusion_matrix: np.ndarray, output_path: Path, model_name: str ="M7"):
    """Plot confusion matrix heatmap."""
    logger.info(f"Generating confusion matrix for {model_name}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Hate", "Non-Hate"],
        yticklabels=["Hate", "Non-Hate"],
        cbar_kws={"label": "Count"},
        ax=ax
    )
    
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=16, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_path / f"confusion_matrix_{model_name.lower()}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved confusion matrix to {output_path}")


def plot_gating_weights(gating_data: Dict[str, List[float]], output_path: Path):
    """Plot gating weight histograms."""
    logger.info("Generating gating weight distribution")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Visual gating weights
    axes[0].hist(gating_data["visual"], bins=30, alpha=0.7, color="#4682B4", edgecolor="black")
    axes[0].axvline(np.mean(gating_data["visual"]), color="red", linestyle="--", 
                    label=f"Mean: {np.mean(gating_data['visual']):.3f}")
    axes[0].set_xlabel("Gate Weight", fontsize=14)
    axes[0].set_ylabel("Frequency", fontsize=14)
    axes[0].set_title("Visual Modality Gating (g_visual)", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Text gating weights
    axes[1].hist(gating_data["text"], bins=30, alpha=0.7, color="#2E8B57", edgecolor="black")
    axes[1].axvline(np.mean(gating_data["text"]), color="red", linestyle="--",
                    label=f"Mean: {np.mean(gating_data['text']):.3f}")
    axes[1].set_xlabel("Gate Weight", fontsize=14)
    axes[1].set_ylabel("Frequency", fontsize=14)
    axes[1].set_title("Text Modality Gating (g_text)", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Dynamic Gating Weight Distribution (M7 Hybrid Fusion)", 
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "gating_weights.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved gating weights to {output_path}")


def plot_ablation_study(df: pd.DataFrame, output_path: Path):
    """Plot ablation study comparison."""
    logger.info("Generating ablation study visualization")
    
    # Group models by category
    categories = {
        "Baseline": ["M1: Text-Only", "M2: Image-Only", "M3: Image (Text-Removed)"],
        "Fusion": ["M4: Early Fusion", "M5: Late Voting", "M6: Late Bagging"],
        "Hybrid": ["M7: Hybrid Fusion", "M8: Hybrid (Text-Removed)"]
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = 0
    colors = {"Baseline": "#4682B4", "Fusion": "#FF8C00", "Hybrid": "#2E8B57"}
    
    for category, models in categories.items():
        category_data = df[df["model"].isin(models)]
        
        for i, (_, row) in enumerate(category_data.iterrows()):
            bar = ax.barh(y_pos, row["f1_macro"], color=colors[category], alpha=0.8)
            
            # Highlight best model
            if row["model"] == "M7: Hybrid Fusion":
                bar[0].set_edgecolor("black")
                bar[0].set_linewidth(3)
            
            ax.text(row["f1_macro"] + 0.005, y_pos, f"{row['f1_macro']:.4f}", 
                   va="center", fontsize=11, fontweight="bold")
            ax.text(0.52, y_pos, row["model"], va="center", fontsize=11)
            
            y_pos += 1
        
        y_pos += 0.5  # Add space between categories
    
    ax.set_xlabel("F1-Macro Score", fontsize=14)
    ax.set_title("Ablation Study: Model Variants", fontsize=16, fontweight="bold")
    ax.set_xlim(0.5, 0.75)
    ax.set_yticks([])
    
    # Add category labels
    ax.text(0.51, 1, "Baseline", fontsize=12, fontweight="bold", va="center")
    ax.text(0.51, 4.5, "Fusion", fontsize=12, fontweight="bold", va="center")
    ax.text(0.51, 8, "Hybrid", fontsize=12, fontweight="bold", va="center")
    
    # Add best model line
    ax.axvline(x=0.6834, color="red", linestyle="--", alpha=0.5, label="Best Model")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_study.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"✓ Saved ablation study to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experimental results")
    parser.add_argument("--results", type=str, help="Path to results CSV")
    parser.add_argument("--training_logs", type=str, help="Path to training logs directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for figures")
    parser.add_argument("--confusion", type=str, help="Path to confusion matrix JSON")
    parser.add_argument("--gating", type=str, help="Path to gating weights JSON")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving figures to {output_dir}")
    
    # Generate plots
    if args.results:
        df = pd.read_csv(args.results)
        plot_model_comparison(df, output_dir)
        plot_ablation_study(df, output_dir)
    
    if args.training_logs:
        plot_training_curves(Path(args.training_logs), output_dir)
    
    if args.confusion:
        with open(args.confusion, "r") as f:
            cm_data = json.load(f)
            cm = np.array(cm_data["confusion_matrix"])
            plot_confusion_matrix(cm, output_dir)
    
    if args.gating:
        with open(args.gating, "r") as f:
            gating_data = json.load(f)
            plot_gating_weights(gating_data, output_dir)
    
    logger.info("✓ All visualizations complete")
    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
