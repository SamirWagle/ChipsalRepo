"""
CHiPSAL 2026 - Subtask A: Data Exploration
==========================================
Explore and visualize the training data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def load_data():
    """Load training data with labels and text."""
    print("[INFO] Loading data...")
    
    # Load the main training CSV
    df = pd.read_csv(TRAIN_CSV_FILE)
    print(f"[INFO] Loaded {len(df)} samples")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    
    return df


def analyze_class_distribution(df):
    """Analyze and visualize class distribution."""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    label_counts = df['label'].value_counts().sort_index()
    print(f"\nClass Distribution:")
    for label, count in label_counts.items():
        class_name = CLASS_NAMES[label]
        percentage = count / len(df) * 100
        print(f"  {label} ({class_name}): {count} samples ({percentage:.2f}%)")
    
    # Calculate class imbalance ratio
    majority_class = label_counts.max()
    minority_class = label_counts.min()
    imbalance_ratio = majority_class / minority_class
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    colors = ['#2ecc71', '#e74c3c']  # Green for Non-Hate, Red for Hate
    bars = axes[0].bar(CLASS_NAMES, label_counts.values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution - Subtask A (Hate Speech Detection)')
    for bar, count in zip(bars, label_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                     str(count), ha='center', fontsize=12)
    
    # Pie chart
    axes[1].pie(label_counts.values, labels=CLASS_NAMES, autopct='%1.1f%%',
                colors=colors, explode=[0, 0.05], shadow=True)
    axes[1].set_title('Class Proportion')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved class distribution plot to {OUTPUT_DIR / 'class_distribution.png'}")
    
    return label_counts


def analyze_text_data(df):
    """Analyze text content from memes."""
    print("\n" + "="*60)
    print("TEXT ANALYSIS")
    print("="*60)
    
    # Text length statistics
    df['text_length'] = df['text'].fillna('').str.len()
    df['word_count'] = df['text'].fillna('').str.split().str.len()
    
    print("\nText Length Statistics:")
    print(df.groupby('label')['text_length'].describe())
    
    print("\nWord Count Statistics:")
    print(df.groupby('label')['word_count'].describe())
    
    # Visualize text length distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for label in [0, 1]:
        subset = df[df['label'] == label]
        axes[0].hist(subset['text_length'], bins=50, alpha=0.6, 
                     label=f'{CLASS_NAMES[label]}', edgecolor='black')
        axes[1].hist(subset['word_count'], bins=30, alpha=0.6,
                     label=f'{CLASS_NAMES[label]}', edgecolor='black')
    
    axes[0].set_xlabel('Text Length (characters)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Text Length Distribution by Class')
    axes[0].legend()
    
    axes[1].set_xlabel('Word Count')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Word Count Distribution by Class')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'text_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Saved text analysis plot to {OUTPUT_DIR / 'text_analysis.png'}")
    
    return df


def check_image_files(df):
    """Verify that all image files exist."""
    print("\n" + "="*60)
    print("IMAGE FILE VERIFICATION")
    print("="*60)
    
    missing_files = []
    existing_files = []
    
    for idx in df['index']:
        img_path = TRAIN_IMAGES_DIR / idx
        if img_path.exists():
            existing_files.append(idx)
        else:
            missing_files.append(idx)
    
    print(f"[INFO] Existing image files: {len(existing_files)}")
    print(f"[INFO] Missing image files: {len(missing_files)}")
    
    if missing_files and len(missing_files) <= 10:
        print(f"[WARNING] Missing files: {missing_files}")
    elif missing_files:
        print(f"[WARNING] First 10 missing files: {missing_files[:10]}")
    
    return existing_files, missing_files


def sample_examples(df, n=5):
    """Display sample examples from each class."""
    print("\n" + "="*60)
    print(f"SAMPLE EXAMPLES (n={n} per class)")
    print("="*60)
    
    for label in [0, 1]:
        print(f"\n--- {CLASS_NAMES[label]} Examples ---")
        samples = df[df['label'] == label].sample(min(n, len(df[df['label'] == label])))
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            text_preview = row['text'][:100] + "..." if len(str(row['text'])) > 100 else row['text']
            print(f"{i}. [{row['index']}]: {text_preview}")


def main():
    """Main exploration pipeline."""
    print("="*60)
    print("CHiPSAL 2026 - SUBTASK A: DATA EXPLORATION")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Analyze class distribution
    label_counts = analyze_class_distribution(df)
    
    # Analyze text data
    df = analyze_text_data(df)
    
    # Check image files
    existing, missing = check_image_files(df)
    
    # Sample examples
    sample_examples(df)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Non-Hate samples: {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
    print(f"Hate samples: {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")
    print(f"Existing images: {len(existing)}")
    print(f"Missing images: {len(missing)}")
    
    # Save processed data info
    info = {
        "total_samples": len(df),
        "non_hate_count": int(label_counts[0]),
        "hate_count": int(label_counts[1]),
        "existing_images": len(existing),
        "missing_images": len(missing),
    }
    
    return df, info


if __name__ == "__main__":
    df, info = main()
