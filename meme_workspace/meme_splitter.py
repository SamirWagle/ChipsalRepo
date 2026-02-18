#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════╗
# ║             MEME SPLITTER v5.1 — PROJECT RUNNER              ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  1. BATCH PROCESSING (Default):                              ║
# ║     python meme_splitter.py                                  ║
# ║     (Process all images in 'train/train_images')             ║
# ║                                                              ║
# ║  2. SINGLE FILE PROCESSING:                                  ║
# ║     python meme_splitter.py --input ../path/to/image.jpg     ║
# ║                                                              ║
# ║  Outputs to: meme_workspace/meme_output/                     ║
# ╚══════════════════════════════════════════════════════════════╝

import argparse
import os
import sys
import logging
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
# from PIL import Image # Not used explicitly, cv2 is used

# ─────────────────────────────────────────────────────────────────────
# Project Configuration Alignment
# ─────────────────────────────────────────────────────────────────────
# Add 'root' to path to import config (script is in meme_workspace/, config.py is in root)
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

try:
    from config import OCR_LANGUAGES, TRAIN_IMAGES_DIR, OUTPUT_DIR
    PROJECT_CONFIG_LOADED = True
except ImportError:
    # Fallback if config.py missing
    OCR_LANGUAGES = ['ne', 'en', 'hi']
    # Adjust default paths to be relative to workspace or absolute?
    # Better to use absolute if possible or relative to root
    TRAIN_IMAGES_DIR = current_dir.parent / "train" / "train_images"
    OUTPUT_DIR = current_dir / "outputs"  # Output inside workspace
    PROJECT_CONFIG_LOADED = False

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
OCR_CONFIDENCE_THRESHOLD = 0.15  # 15% minimum confidence
INPAINT_RADIUS = 3               # Radius for inpainting (smaller = sharper)
MASK_DILATION_ITERATIONS = 2     # How much to expand mask covers
EDGE_DETECTION_ENABLED = True    # Detect text edges for better masks

# ─────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meme_splitter")

# ─────────────────────────────────────────────────────────────────────
# OCR Engine (Lazy Loading)
# ─────────────────────────────────────────────────────────────────────
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info(f"Loading EasyOCR (languages: {OCR_LANGUAGES})...")
            # gpu=True if available (config alignment)
            _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=True) 
        except ImportError:
            logger.error("EasyOCR not installed. pip install easyocr")
            sys.exit(1)
        except Exception as e:
            logger.warning(f"GPU OCR failed, falling back to CPU: {e}")
            _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader

# ─────────────────────────────────────────────────────────────────────
# CSV Helper
# ─────────────────────────────────────────────────────────────────────
def get_existing_files(csv_path: Path) -> set:
    """Read processed filenames from CSV to avoid redundancy."""
    processed = set()
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                for row in reader:
                    if row:
                        processed.add(row[0]) # filename is first column
        except Exception:
            pass # Ignore errors, just reprocess
    return processed

def append_to_csv(csv_path: Path, filename: str, text: str):
    """
    Append a row to the CSV file.
    Format: index, text
    """
    file_exists = csv_path.exists()
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header if new file matches project format
            if not file_exists:
                writer.writerow(['index', 'text'])
            
            writer.writerow([filename, text])
    except Exception as e:
        logger.error(f"Failed to write to CSV: {e}")

# ─────────────────────────────────────────────────────────────────────
# Image Processing Features
# ─────────────────────────────────────────────────────────────────────

def create_smart_mask(image: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """
    Generate a precise mask for inpainting.
    Uses bounding boxes + edge detection to cover text strokes perfectly
    without removing too much background context.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for bbox, text, conf in detections:
        # Convert bbox points to integer dictionary
        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
        
        # 1. Base Mask: Fill the bounding box
        cv2.fillPoly(mask, [pts], 255)
        
    # 2. Refinement: Use edge detection within the boxes to capture strokes
    if EDGE_DETECTION_ENABLED:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate edges slightly to connect text strokes
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel_edge, iterations=1)
        
        # Combine: Only keep edges that are INSIDE or NEAR the text boxes
        # Dilate the box mask first to assume effective area
        box_expanded = cv2.dilate(mask, kernel_edge, iterations=5) 
        text_edges = cv2.bitwise_and(dilated_edges, box_expanded)
        
        # Add edges to main mask
        mask = cv2.bitwise_or(mask, text_edges)

    # 3. Final Expansion: Ensure comprehensive coverage
    # Adaptive kernel size based on image resolution
    k_size = max(3, min(w, h) // 200) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    mask = cv2.dilate(mask, kernel, iterations=MASK_DILATION_ITERATIONS)
    
    return mask

def inpaint_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove text using bi-directional inpainting.
    Preserves background context (vital for Vision Models).
    """
    # 1. Telea (Fast Marching Method) - Good for small text
    clean_telea = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    
    # 2. Navier-Stokes (Fluid Dynamics) - Good for larger smooth areas
    clean_ns = cv2.inpaint(image, mask, INPAINT_RADIUS, cv2.INPAINT_NS)
    
    # Blend: Use Telea for edges (sharper) and NS for interiors (smoother)
    # Simple average for robustness
    clean = cv2.addWeighted(clean_telea, 0.6, clean_ns, 0.4, 0)
    
    return clean

# ─────────────────────────────────────────────────────────────────────
# Core Pipeline
# ─────────────────────────────────────────────────────────────────────

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Enhance image for better OCR accuracy.
    1. Upscale if small
    2. Convert to Grayscale
    3. Denoise
    """
    h, w = image.shape[:2]
    
    # 1. Upscale if too small (EasyOCR likes ~1000px width)
    if w < 1000:
        scale = 2.0 if w < 500 else 1.5
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale (EasyOCR does this internally but preprocessing allows denoising)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Denoise (Mild)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Convert back to BGR for consistency
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def process_single_file(file_path: Path, output_dir: Path, csv_path: Path, existing_files: set) -> bool:
    """
    Process one meme:
    1. Check duplication -> 2. OCR -> 3. Inpaint -> 4. Save Image -> 5. Append to CSV
    """
    filename = file_path.name
    
    # Check Redundancy
    if filename in existing_files:
        logger.info(f"  ⏭ Skipping {filename} (Already processed)")
        return False

    try:
        # Load Image
        img = cv2.imread(str(file_path))
        if img is None:
            logger.error(f"Failed to load image: {file_path}")
            return False
            
        # 1. OCR Extraction (Optimized)
        reader = get_ocr_reader()
        
        # Preprocess for OCR only (preserve original for inpainting/saving)
        ocr_input = preprocess_for_ocr(img)
        
        results = reader.readtext(
            ocr_input, 
            batch_size=4,
            detail=1,
            paragraph=False, # Line-by-line is safer for memes
            text_threshold=0.6, # Lower threshold to catch faint text
            low_text=0.3,       # Default is 0.4
            link_threshold=0.3, # Default is 0.4
            mag_ratio=1.5,      # Zoom in for better small text
            slope_ths=0.2,      # Tolerate some tilt
            width_ths=0.7       # Merge close words
        )
        
        # Filter confidence
        valid_detections = []
        original_h, original_w = img.shape[:2]
        processed_h, processed_w = ocr_input.shape[:2]
        scale_x = original_w / processed_w
        scale_y = original_h / processed_h
        
        for bbox, text, conf in results:
            if conf >= OCR_CONFIDENCE_THRESHOLD:
                # Map bbox back to original image coordinates
                original_bbox = [
                    [int(p[0] * scale_x), int(p[1] * scale_y)] 
                    for p in bbox
                ]
                valid_detections.append((original_bbox, text, conf))
        
        # Construct full text (reading order)
        valid_detections.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        extracted_text = " ".join([t for _, t, _ in valid_detections])
        
        # 2. Inpainting (Context Preservation)
        if valid_detections:
            mask = create_smart_mask(img, valid_detections)
            clean_img = inpaint_image(img, mask)
        else:
            clean_img = img.copy() # No text to remove
            logger.info(f"  ⚠ No high-confidence text found in {filename}")

        # 3. Save Artifacts
        text_out_dir = output_dir / "text"
        image_out_dir = output_dir / "image"
        text_out_dir.mkdir(parents=True, exist_ok=True)
        image_out_dir.mkdir(parents=True, exist_ok=True)
        
        clean_path = image_out_dir / filename
        text_path = text_out_dir / f"{file_path.stem}.txt"
        
        # Save Image
        cv2.imwrite(str(clean_path), clean_img)
        
        # Save Text File (legacy support)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
            
        # 4. CSV Append
        append_to_csv(csv_path, filename, extracted_text)
        
        preview = extracted_text.replace('\n', ' ')
        if len(preview) > 50: preview = preview[:47] + "..."
        logger.info(f"  ✓ Processed: {filename}")
        logger.info(f"    → Text: '{preview}'")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return False

def discover_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    elif input_path.is_dir():
        # Clean hidden files
        files = sorted([
            p for p in input_path.iterdir() 
            if p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith('.')
        ])
        return files
    return []

# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Meme Splitter v5.1 (CSV Aligned & Optimized)")
    # Default to config TRAIN_IMAGES_DIR if not provided
    default_input = str(TRAIN_IMAGES_DIR) if PROJECT_CONFIG_LOADED else None
    
    parser.add_argument("--input", "-i", type=str, default=default_input, 
                       help=f"Input file or directory (Default: {default_input})")
    parser.add_argument("--output", "-o", type=str, default="meme_output", help="Output directory")
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
        
    input_path = Path(args.input)
    output_dir = Path(args.output)
    csv_path = output_dir / "dataset.csv"
    
    # Ensure output dir exists for CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("MEME SPLITTER v5.1 — CSV ALIGNED & OPTIMIZED")
    print(f"Config Loaded: {PROJECT_CONFIG_LOADED}")
    if PROJECT_CONFIG_LOADED:
        print(f"Project OCR Langs: {OCR_LANGUAGES}")
    print("="*60 + "\n")
    
    files = discover_files(input_path)
    if not files:
        logger.error(f"No valid images found in {input_path}")
        sys.exit(1)
        
    logger.info(f"Found {len(files)} files. Process queue initialized.")
    logger.info(f"Output CSV: {csv_path}")
    
    # Load existing to prevent redundancy
    existing_files = get_existing_files(csv_path)
    if existing_files:
        logger.info(f"Found {len(existing_files)} already processed files in CSV.")
    
    # Pre-load OCR
    get_ocr_reader()
    
    success_count = 0
    skipped_count = 0
    for f in files:
        # Quick check name against set
        if f.name in existing_files:
            skipped_count += 1
            if skipped_count <= 5: # Don't spam logs
                logger.debug(f"Skipping {f.name}")
            continue
            
        if process_single_file(f, output_dir, csv_path, existing_files):
            success_count += 1
            # Update local set to prevent dups in this run
            existing_files.add(f.name)
            
    logger.info("\n" + "="*60)
    logger.info(f"DONE. Processed: {success_count}, Skipped: {skipped_count}, Total: {len(files)}")
    logger.info(f"Clean Images: {output_dir}/image/")
    logger.info(f"Dataset CSV:  {csv_path}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
