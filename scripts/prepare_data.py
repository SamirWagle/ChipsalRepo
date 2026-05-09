#!/usr/bin/env python3
"""
Data Preparation Script for CHiPSAL 2026

Prepares and validates the dataset structure for training.

Usage:
    python scripts/prepare_data.py --validate
    python scripts/prepare_data.py --split --val_ratio 0.2
    python scripts/prepare_data.py --generate_text_removed
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_preparation.log'),
            logging.StreamHandler()
        ]
    )


def validate_dataset(data_dir: str = "train") -> Tuple[bool, dict]:
    """Validate dataset structure and files."""
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("VALIDATING DATASET")
    logger.info("="*80)
    
    issues = []
    stats = {
        "total_samples": 0,
        "missing_images": 0,
        "corrupted_images": 0,
        "missing_text": 0,
        "class_distribution": {},
    }
    
    # Check directories
    required_dirs = [
        f"{data_dir}/train_images/",
        f"{data_dir}/",
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            issues.append(f"Missing directory: {dir_path}")
    
    # Check CSV files
    csv_path = f"{data_dir}/train.csv"
    if not os.path.exists(csv_path):
        issues.append(f"Missing CSV: {csv_path}")
        return False, {"issues": issues}
    
    # Load CSV
    logger.info(f"\nLoading {csv_path}...")
    df = pd.read_csv(csv_path)
    stats["total_samples"] = len(df)
    
    logger.info(f"Total samples in CSV: {len(df)}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ["index", "text", "label"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
        return False, {"issues": issues}
    
    # Check class distribution
    class_counts = df["label"].value_counts()
    stats["class_distribution"] = class_counts.to_dict()
    
    logger.info("\nClass Distribution:")
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        logger.info(f"  Label {label}: {count} ({pct:.1f}%)")
    
    # Validate images
    logger.info("\nValidating images...")
    image_dir = f"{data_dir}/train_images/"
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_name = row["index"]
        img_path = os.path.join(image_dir, img_name)
        
        # Check if image exists
        if not os.path.exists(img_path):
            stats["missing_images"] += 1
            if stats["missing_images"] <= 5:  # Log first 5
                logger.warning(f"Missing image: {img_name}")
            continue
        
        # Check if image is readable
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            stats["corrupted_images"] += 1
            if stats["corrupted_images"] <= 5:
                logger.warning(f"Corrupted image {img_name}: {e}")
        
        # Check if text is present
        if pd.isna(row["text"]) or len(str(row["text"]).strip()) == 0:
            stats["missing_text"] += 1
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(f"Missing images: {stats['missing_images']}")
    logger.info(f"Corrupted images: {stats['corrupted_images']}")
    logger.info(f"Missing text: {stats['missing_text']}")
    
    if issues:
        logger.error("\nISSUES FOUND:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False, stats
    
    if stats["missing_images"] > 0 or stats["corrupted_images"] > 0:
        logger.warning("\nWarnings found but validation passed.")
        return True, stats
    
    logger.info("\n✓ Dataset validation PASSED!")
    logger.info("="*80 + "\n")
    return True, stats


def create_train_val_split(
    data_dir: str = "train",
    val_ratio: float = 0.2,
    stratified: bool = True,
    seed: int = 42,
):
    """Create train/validation split."""
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("CREATING TRAIN/VAL SPLIT")
    logger.info("="*80)
    logger.info(f"Validation ratio: {val_ratio}")
    logger.info(f"Stratified: {stratified}")
    logger.info(f"Seed: {seed}")
    
    # Load data
    df = pd.read_csv(f"{data_dir}/train.csv")
    
    # Create split
    from sklearn.model_selection import train_test_split
    
    if stratified:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            stratify=df["label"]
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed
        )
    
    logger.info(f"\nTrain samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    
    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    train_df.to_csv("data/splits/train_split.csv", index=False)
    val_df.to_csv("data/splits/val_split.csv", index=False)
    
    logger.info(f"\nSaved to:")
    logger.info(f"  - data/splits/train_split.csv")
    logger.info(f"  - data/splits/val_split.csv")
    logger.info("="*80 + "\n")


def remove_text_from_images(
    input_dir: str = "train/train_images/",
    output_dir: str = "train/train_images_no_text/",
    method: str = "inpainting",
):
    """
    Remove text from images for M3/M8 experiments.
    
    Methods:
        - 'blur': Gaussian blur on detected text regions
        - 'inpainting': CV2 inpainting
        - 'mask': Replace with mean color
    """
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("GENERATING TEXT-REMOVED IMAGES")
    logger.info("="*80)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Method: {method}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a placeholder - actual implementation would use EasyOCR or CRAFT
    # for text detection and then apply the chosen removal method
    
    logger.warning("\nNOTE: Text removal is computationally expensive.")
    logger.warning("For M3/M8 experiments, consider using pre-processed images")
    logger.warning("or implementing EAST/CRAFT text detection + inpainting.")
    
    # Example pseudocode:
    """
    import easyocr
    reader = easyocr.Reader(['ne', 'en'])
    
    for img_file in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        # Detect text
        results = reader.detect(img)
        
        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for bbox in results[0]:
            pts = np.array(bbox, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        # Remove text
        if method == 'inpainting':
            img_no_text = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        elif method == 'blur':
            blurred = cv2.GaussianBlur(img, (21, 21), 0)
            img_no_text = np.where(mask[..., None] == 255, blurred, img)
        elif method == 'mask':
            mean_color = cv2.mean(img)[:3]
            img_no_text = np.where(mask[..., None] == 255, mean_color, img)
        
        # Save
        cv2.imwrite(os.path.join(output_dir, img_file), img_no_text)
    """
    
    logger.info("\n✓ Text removal placeholder created")
    logger.info("  Implement actual text detection for production use")
    logger.info("="*80 + "\n")


def generate_stats_report(data_dir: str = "train"):
    """Generate detailed statistics report."""
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("DATASET STATISTICS")
    logger.info("="*80)
    
    df = pd.read_csv(f"{data_dir}/train.csv")
    
    # Text statistics
    df["text_length"] = df["text"].fillna("").apply(len)
    df["word_count"] = df["text"].fillna("").apply(lambda x: len(x.split()))
    
    logger.info(f"\nText Statistics:")
    logger.info(f"  Mean length: {df['text_length'].mean():.1f} chars")
    logger.info(f"  Median length: {df['text_length'].median():.1f} chars")
    logger.info(f"  Mean word count: {df['word_count'].mean():.1f} words")
    logger.info(f"  Max length: {df['text_length'].max()} chars")
    logger.info(f"  Min length: {df['text_length'].min()} chars")
    
    # Class imbalance
    class_counts = df["label"].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    
    logger.info(f"\nClass Imbalance:")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
    logger.info(f"  Majority class: {class_counts.idxmax()} ({class_counts.max()} samples)")
    logger.info(f"  Minority class: {class_counts.idxmin()} ({class_counts.min()} samples)")
    
    # Image statistics (sample)
    logger.info(f"\nImage Statistics (sampling 100 images):")
    image_dir = f"{data_dir}/train_images/"
    sample_imgs = df.sample(min(100, len(df)), random_state=42)
    
    sizes = []
    for img_name in sample_imgs["index"]:
        img_path = os.path.join(image_dir, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            sizes.append(img.size)
    
    if sizes:
        widths, heights = zip(*sizes)
        logger.info(f"  Mean size: {np.mean(widths):.0f} x {np.mean(heights):.0f}")
        logger.info(f"  Min size: {min(widths)} x {min(heights)}")
        logger.info(f"  Max size: {max(widths)} x {max(heights)}")
    
    logger.info("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare CHiPSAL dataset")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--split", action="store_true", help="Create train/val split")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--generate_text_removed", action="store_true", 
                       help="Generate text-removed images")
    parser.add_argument("--stats", action="store_true", help="Generate statistics report")
    parser.add_argument("--data_dir", type=str, default="train", help="Data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("CHiPSAL 2026 - Data Preparation")
    logger.info("="*80 + "\n")
    
    if args.validate:
        validate_dataset(args.data_dir)
    
    if args.split:
        create_train_val_split(
            data_dir=args.data_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )
    
    if args.generate_text_removed:
        remove_text_from_images()
    
    if args.stats:
        generate_stats_report(args.data_dir)
    
    if not any([args.validate, args.split, args.generate_text_removed, args.stats]):
        logger.info("No action specified. Use --help for options.")
        logger.info("\nQuick start:")
        logger.info("  python scripts/prepare_data.py --validate --stats")
    
    logger.info("\n✓ Data preparation complete!\n")


if __name__ == "__main__":
    main()
