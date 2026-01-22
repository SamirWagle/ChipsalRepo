"""
CHiPSAL 2026 - Subtask A: OCR Text Extraction
=============================================
Extract text from Nepali memes using EasyOCR
"""

import easyocr
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *


class NepaliOCRExtractor:
    """OCR Extractor for Nepali meme images using EasyOCR."""
    
    def __init__(self, languages=['ne', 'en'], use_gpu=True):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of language codes (ne=Nepali, en=English)
            use_gpu: Whether to use GPU for OCR
        """
        print(f"[INFO] Initializing EasyOCR with languages: {languages}")
        print(f"[INFO] GPU: {use_gpu}")
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        self.languages = languages
        
    def extract_text(self, image_path, detail=True):
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file
            detail: If True, return bounding boxes and confidence scores
            
        Returns:
            If detail=True: List of (bbox, text, confidence) tuples
            If detail=False: Combined text string
        """
        try:
            results = self.reader.readtext(str(image_path))
            
            if detail:
                return results
            else:
                # Combine all text
                texts = [item[1] for item in results]
                return ' '.join(texts)
                
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            return [] if detail else ""
    
    def extract_batch(self, image_dir, image_indices, output_file=None, detail=False):
        """
        Extract text from multiple images.
        
        Args:
            image_dir: Directory containing images
            image_indices: List of image filenames
            output_file: Optional path to save results as CSV
            detail: Whether to include detailed OCR info
            
        Returns:
            DataFrame with index and extracted text
        """
        image_dir = Path(image_dir)
        results = []
        
        print(f"[INFO] Extracting text from {len(image_indices)} images...")
        
        for idx in tqdm(image_indices, desc="OCR Extraction"):
            img_path = image_dir / idx
            
            if not img_path.exists():
                print(f"[WARNING] Image not found: {img_path}")
                results.append({
                    'index': idx,
                    'ocr_text': '',
                    'ocr_confidence': 0.0,
                    'num_detections': 0
                })
                continue
            
            ocr_result = self.extract_text(img_path, detail=True)
            
            if ocr_result:
                texts = [item[1] for item in ocr_result]
                confidences = [item[2] for item in ocr_result]
                
                results.append({
                    'index': idx,
                    'ocr_text': ' '.join(texts),
                    'ocr_confidence': np.mean(confidences),
                    'num_detections': len(ocr_result)
                })
            else:
                results.append({
                    'index': idx,
                    'ocr_text': '',
                    'ocr_confidence': 0.0,
                    'num_detections': 0
                })
        
        df = pd.DataFrame(results)
        
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"[INFO] Saved OCR results to {output_file}")
        
        return df
    
    def analyze_ocr_results(self, df):
        """Analyze OCR extraction results."""
        print("\n" + "="*60)
        print("OCR EXTRACTION ANALYSIS")
        print("="*60)
        
        total = len(df)
        empty = len(df[df['ocr_text'] == ''])
        non_empty = total - empty
        
        print(f"Total images: {total}")
        print(f"Images with text: {non_empty} ({non_empty/total*100:.1f}%)")
        print(f"Images without text: {empty} ({empty/total*100:.1f}%)")
        
        # Text length statistics
        df['text_length'] = df['ocr_text'].str.len()
        print(f"\nText Length Statistics:")
        print(df['text_length'].describe())
        
        # Confidence statistics
        valid_conf = df[df['ocr_confidence'] > 0]['ocr_confidence']
        print(f"\nConfidence Statistics:")
        print(valid_conf.describe())
        
        return {
            'total': total,
            'with_text': non_empty,
            'without_text': empty,
            'avg_confidence': valid_conf.mean() if len(valid_conf) > 0 else 0
        }


def main():
    """Main OCR extraction pipeline."""
    print("="*60)
    print("CHiPSAL 2026 - OCR TEXT EXTRACTION")
    print("="*60)
    
    # Load training data
    df_train = pd.read_csv(TRAIN_LABELS_FILE)
    image_indices = df_train['index'].tolist()
    
    print(f"[INFO] Found {len(image_indices)} images to process")
    
    # Initialize OCR extractor
    extractor = NepaliOCRExtractor(
        languages=OCR_LANGUAGES,
        use_gpu=OCR_GPU
    )
    
    # Extract text from all training images
    ocr_output_file = DATA_DIR / "ocr_train_extracted.csv"
    
    df_ocr = extractor.extract_batch(
        image_dir=TRAIN_IMAGES_DIR,
        image_indices=image_indices,
        output_file=ocr_output_file,
        detail=False
    )
    
    # Analyze results
    stats = extractor.analyze_ocr_results(df_ocr)
    
    # Merge with labels
    df_merged = df_train.merge(df_ocr, on='index', how='left')
    df_merged.to_csv(DATA_DIR / "train_with_ocr.csv", index=False)
    print(f"[INFO] Saved merged data to {DATA_DIR / 'train_with_ocr.csv'}")
    
    # Compare OCR text with existing text (if available)
    if TRAIN_CSV_FILE.exists():
        df_existing = pd.read_csv(TRAIN_CSV_FILE)
        comparison = df_merged.merge(df_existing[['index', 'text']], on='index', how='left')
        
        print("\n[INFO] Comparing extracted OCR with existing text...")
        sample = comparison.sample(min(5, len(comparison)))
        for _, row in sample.iterrows():
            print(f"\nImage: {row['index']}")
            print(f"  Existing: {str(row['text'])[:80]}...")
            print(f"  OCR:      {str(row['ocr_text'])[:80]}...")
    
    print("\n[INFO] OCR extraction complete!")
    return df_ocr, stats


if __name__ == "__main__":
    df_ocr, stats = main()
