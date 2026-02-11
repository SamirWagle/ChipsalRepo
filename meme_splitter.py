# ╔══════════════════════════════════════════════════════════════╗
# ║                 HOW TO RUN MEME SPLITTER                     ║
# ╠══════════════════════════════════════════════════════════════╣
# ║                                                              ║
# ║  Step 1: Activate virtual environment                        ║
# ║    source venv/bin/activate                                  ║
# ║                                                              ║
# ║  Step 2: Run on a folder of memes (bulk processing)          ║
# ║    python meme_splitter.py --input ./your_memes_folder       ║
# ║                                                              ║
# ║  OR: Run on a single meme file                               ║
# ║    python meme_splitter.py --input ./meme.png                ║
# ║                                                              ║
# ║  OR: Specify custom output folder                            ║
# ║    python meme_splitter.py -i ./memes -o ./output            ║
# ║                                                              ║
# ║  Output:                                                     ║
# ║    output/text/   → extracted text as .txt files             ║
# ║    output/image/  → clean images (text removed) as .png      ║
# ║                                                              ║
# ╚══════════════════════════════════════════════════════════════╝

#!/usr/bin/env python3
"""
Meme Splitter v3.0 — Professional text extraction & image cleaning.

For each meme:
  1. Multi-scale OCR extracts all text → saved as structured .txt
  2. Precision mask generation with adaptive dilation
  3. Multi-pass inpainting removes text cleanly → saved as .png

Usage:
    python meme_splitter.py --input <input_folder> [--output <output_folder>]

Output structure:
    <output_folder>/
        image/    ← clean images with text removed (*_img.png)
        text/     ← extracted text files (*_text.txt)
"""

import argparse
import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image, ImageFilter

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}
ALL_SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS

# EasyOCR languages — Nepali + English + Hindi (covers Devanagari well)
OCR_LANGUAGES = ["ne", "en", "hi"]

# OCR confidence threshold — skip detections below this (0.0 to 1.0)
OCR_CONFIDENCE_THRESHOLD = 0.15

# Inpainting settings
INPAINT_RADIUS = 10                 # pixels around text to repair
INPAINT_PASSES = 2                  # number of inpainting passes
INPAINT_METHOD = cv2.INPAINT_NS     # Navier-Stokes (better quality)

# Mask generation settings
MASK_PADDING_PX = 8                 # extra pixels around each text box
DILATION_KERNEL_SIZE = 11           # kernel for mask dilation
DILATION_ITERATIONS = 3             # how many times to dilate

# Edge-aware cleanup: detect and include text shadows/strokes
EDGE_DETECTION_ENABLED = True
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150

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
# OCR Engine (lazy-loaded singleton)
# ─────────────────────────────────────────────────────────────────────
_ocr_reader = None


def get_ocr_reader():
    """Lazy-load EasyOCR reader (downloads models on first use)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
        except ImportError:
            logger.error(
                "easyocr is not installed. Install it with:\n"
                "  pip install easyocr"
            )
            sys.exit(1)

        logger.info(f"Loading OCR engine (languages: {OCR_LANGUAGES})...")
        logger.info("(First run downloads models — this may take a minute)")
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        logger.info("OCR engine loaded ✓")
    return _ocr_reader


# ─────────────────────────────────────────────────────────────────────
# PDF Handling
# ─────────────────────────────────────────────────────────────────────
def load_pdf_as_images(pdf_path: str) -> List[np.ndarray]:
    """Convert PDF pages to OpenCV images."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.error(
            "pdf2image is not installed. Install it with: pip install pdf2image\n"
            "Also requires poppler: brew install poppler (macOS)"
        )
        return []

    try:
        pil_images = convert_from_path(pdf_path, dpi=300)
        cv_images = []
        for pil_img in pil_images:
            rgb = np.array(pil_img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv_images.append(bgr)
        return cv_images
    except Exception as e:
        logger.error(f"Failed to load PDF '{pdf_path}': {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Image Loading
# ─────────────────────────────────────────────────────────────────────
def load_image(file_path: str) -> List[Tuple[np.ndarray, str]]:
    """
    Load an image (or PDF) and return list of (cv2_image, base_name) tuples.
    PDFs may produce multiple pages, hence the list.
    """
    ext = Path(file_path).suffix.lower()
    stem = Path(file_path).stem

    if ext in SUPPORTED_PDF_EXTS:
        pages = load_pdf_as_images(file_path)
        if len(pages) == 1:
            return [(pages[0], stem)]
        return [(page, f"{stem}_page{i+1}") for i, page in enumerate(pages)]

    if ext in SUPPORTED_IMAGE_EXTS:
        try:
            pil_img = Image.open(file_path)
            pil_img = pil_img.convert("RGB")
            img_array = np.array(pil_img)
            cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            return [(cv_img, stem)]
        except Exception as e:
            logger.error(f"Failed to load image '{file_path}': {e}")
            return []

    logger.warning(f"Unsupported file format: {ext} — skipping '{file_path}'")
    return []


# ─────────────────────────────────────────────────────────────────────
# Text Ordering & Structuring
# ─────────────────────────────────────────────────────────────────────
def sort_ocr_results_reading_order(
    results: List[Tuple], image_width: int
) -> List[Tuple]:
    """
    Sort OCR results in natural reading order (top-to-bottom, left-to-right).
    Groups text that appears on the same horizontal line together.
    """
    if not results:
        return results

    # Calculate the center Y of each text box for line grouping
    annotated = []
    for bbox, text, confidence in results:
        pts = np.array(bbox)
        center_y = np.mean(pts[:, 1])
        center_x = np.mean(pts[:, 0])
        min_y = np.min(pts[:, 1])
        height = np.max(pts[:, 1]) - min_y
        annotated.append({
            "bbox": bbox,
            "text": text,
            "confidence": confidence,
            "center_y": center_y,
            "center_x": center_x,
            "min_y": min_y,
            "height": height,
        })

    # Sort by Y position first
    annotated.sort(key=lambda x: x["min_y"])

    # Group items into lines — items within half a text-height are on the same line
    lines = []
    current_line = [annotated[0]]

    for item in annotated[1:]:
        prev = current_line[-1]
        # If this item is within ~60% of the previous item's height, same line
        threshold = max(prev["height"], item["height"]) * 0.6
        if abs(item["center_y"] - prev["center_y"]) < threshold:
            current_line.append(item)
        else:
            # Sort current line left-to-right, then start new line
            current_line.sort(key=lambda x: x["center_x"])
            lines.append(current_line)
            current_line = [item]

    # Don't forget last line
    current_line.sort(key=lambda x: x["center_x"])
    lines.append(current_line)

    # Flatten back to sorted list
    sorted_results = []
    for line in lines:
        for item in line:
            sorted_results.append((item["bbox"], item["text"], item["confidence"]))

    return sorted_results


def format_extracted_text(
    results: List[Tuple], image_width: int
) -> str:
    """
    Format OCR results into clean, readable text.
    Groups text on the same line together, separated by spaces.
    Preserves line breaks between different text lines.
    """
    if not results:
        return ""

    # Sort and calculate annotations
    annotated = []
    for bbox, text, confidence in results:
        pts = np.array(bbox)
        center_y = np.mean(pts[:, 1])
        center_x = np.mean(pts[:, 0])
        min_y = np.min(pts[:, 1])
        height = np.max(pts[:, 1]) - min_y
        annotated.append({
            "text": text.strip(),
            "confidence": confidence,
            "center_y": center_y,
            "center_x": center_x,
            "min_y": min_y,
            "height": height,
        })

    # Sort by Y position
    annotated.sort(key=lambda x: x["min_y"])

    # Group into lines
    lines = []
    current_line = [annotated[0]]

    for item in annotated[1:]:
        prev = current_line[-1]
        threshold = max(prev["height"], item["height"]) * 0.6
        if abs(item["center_y"] - prev["center_y"]) < threshold:
            current_line.append(item)
        else:
            current_line.sort(key=lambda x: x["center_x"])
            lines.append(current_line)
            current_line = [item]

    current_line.sort(key=lambda x: x["center_x"])
    lines.append(current_line)

    # Build text: join items on same line with space, lines with newline
    text_lines = []
    for line_items in lines:
        line_text = " ".join(item["text"] for item in line_items if item["text"])
        if line_text.strip():
            text_lines.append(line_text)

    return "\n".join(text_lines)


# ─────────────────────────────────────────────────────────────────────
# Precision Mask Generation
# ─────────────────────────────────────────────────────────────────────
def create_text_mask(
    image: np.ndarray,
    results: List[Tuple],
) -> np.ndarray:
    """
    Create a high-quality binary mask covering all text regions.

    Uses multiple techniques:
    1. OCR bounding boxes with padding
    2. Adaptive dilation based on text size
    3. Edge-aware expansion to catch strokes/shadows
    4. Morphological closing to fill gaps within text
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for bbox, text, confidence in results:
        pts = np.array(bbox, dtype=np.float32)

        # ─── Add padding around the bounding box ───
        # Calculate center and expand outward
        center = np.mean(pts, axis=0)
        expanded_pts = []
        for pt in pts:
            direction = pt - center
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            expanded_pt = pt + direction * MASK_PADDING_PX
            expanded_pts.append(expanded_pt)

        expanded_pts = np.array(expanded_pts, dtype=np.int32)

        # Clip to image bounds
        expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w - 1)
        expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h - 1)

        cv2.fillPoly(mask, [expanded_pts], 255)

    # ─── Edge-aware expansion ───
    if EDGE_DETECTION_ENABLED:
        edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

        # Only keep edges near existing mask regions
        # Dilate the mask slightly, then AND with edges
        edge_search = cv2.dilate(
            mask, np.ones((25, 25), np.uint8), iterations=1
        )
        nearby_edges = cv2.bitwise_and(edges, edge_search)

        # Add these edges to the mask
        mask = cv2.bitwise_or(mask, nearby_edges)

    # ─── Adaptive dilation ───
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=DILATION_ITERATIONS)

    # ─── Morphological closing to fill small gaps ───
    close_kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # ─── Gaussian blur the mask edges for smoother inpainting ───
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask


# ─────────────────────────────────────────────────────────────────────
# Multi-Pass Inpainting
# ─────────────────────────────────────────────────────────────────────
def inpaint_text_removal(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Remove text using multi-pass inpainting for cleaner results.

    Pass 1: Navier-Stokes inpainting (better for larger regions)
    Pass 2+: Progressive refinement with smaller radius on remaining artifacts
    """
    result = image.copy()

    for pass_num in range(INPAINT_PASSES):
        # Decrease radius on subsequent passes for fine detail
        radius = max(INPAINT_RADIUS - (pass_num * 3), 3)

        if pass_num == 0:
            # First pass: NS method (better quality for text)
            result = cv2.inpaint(result, mask, radius, INPAINT_METHOD)
        else:
            # Subsequent passes: erode mask slightly, re-inpaint edges
            erode_kernel = np.ones((5, 5), np.uint8)
            refined_mask = cv2.erode(mask, erode_kernel, iterations=pass_num)

            # Only re-inpaint the border region of the mask
            border_mask = cv2.subtract(mask, refined_mask)
            if cv2.countNonZero(border_mask) > 0:
                result = cv2.inpaint(result, border_mask, radius, cv2.INPAINT_TELEA)

    return result


# ─────────────────────────────────────────────────────────────────────
# Post-Processing: Clean up inpainted regions
# ─────────────────────────────────────────────────────────────────────
def post_process_image(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Post-process the inpainted image:
    1. Blend edges between inpainted and original regions
    2. Match color temperature in repaired areas
    3. Apply subtle smoothing to inpainted regions
    """
    # Create a feathered (soft) mask for blending
    feather_radius = 15
    soft_mask = cv2.GaussianBlur(
        mask.astype(np.float32), (feather_radius * 2 + 1, feather_radius * 2 + 1), 0
    )
    soft_mask = soft_mask / 255.0

    # Expand dims for 3-channel blending
    soft_mask_3ch = np.stack([soft_mask] * 3, axis=-1)

    # Blend: inpainted where mask is white, original where mask is black
    blended = (
        inpainted.astype(np.float32) * soft_mask_3ch
        + original.astype(np.float32) * (1.0 - soft_mask_3ch)
    ).astype(np.uint8)

    # Apply subtle bilateral filter to inpainted regions for texture consistency
    filtered = cv2.bilateralFilter(blended, d=9, sigmaColor=75, sigmaSpace=75)

    # Only apply filtering within mask region
    binary_mask = (mask > 127).astype(np.uint8)
    binary_mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    result = np.where(binary_mask_3ch == 1, filtered, blended)

    return result


# ─────────────────────────────────────────────────────────────────────
# Core Pipeline: OCR + Mask + Inpaint + Post-process
# ─────────────────────────────────────────────────────────────────────
def extract_text_and_clean_image(
    image: np.ndarray,
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    """
    Professional-grade text extraction and removal.

    Returns:
        (extracted_text, clean_image, metadata)
        metadata contains detection details for reporting
    """
    reader = get_ocr_reader()
    h, w = image.shape[:2]

    # Convert BGR → RGB for EasyOCR
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ─── Run OCR with detail=1 for full bbox info ───
    results = reader.readtext(
        rgb_image,
        detail=1,
        paragraph=False,       # Don't merge — we'll handle grouping ourselves
        min_size=10,           # Minimum text size in pixels
        text_threshold=0.5,    # Text confidence threshold
        link_threshold=0.3,    # Link threshold for merging nearby chars
        low_text=0.3,          # Low text confidence bound
        width_ths=0.7,         # Width threshold for merging boxes
    )

    # ─── Filter by confidence ───
    high_conf = []
    low_conf = []
    for bbox, text, confidence in results:
        if confidence >= OCR_CONFIDENCE_THRESHOLD:
            high_conf.append((bbox, text, confidence))
        else:
            low_conf.append((bbox, text, confidence))

    metadata = {
        "total_detections": len(results),
        "accepted": len(high_conf),
        "rejected_low_conf": len(low_conf),
        "image_size": f"{w}x{h}",
    }

    if not high_conf:
        logger.info("    No text detected above confidence threshold.")
        return "", image.copy(), metadata

    # ─── Sort results in reading order ───
    sorted_results = sort_ocr_results_reading_order(high_conf, w)

    # ─── Extract formatted text ───
    extracted_text = format_extracted_text(sorted_results, w)

    # ─── Generate precision mask ───
    mask = create_text_mask(image, sorted_results)

    # ─── Multi-pass inpainting ───
    inpainted = inpaint_text_removal(image, mask)

    # ─── Post-processing: edge blending + texture smoothing ───
    clean_image = post_process_image(image, inpainted, mask)

    metadata["text_regions"] = len(sorted_results)
    metadata["mask_coverage"] = f"{(cv2.countNonZero(mask) / (h * w)) * 100:.1f}%"

    return extracted_text, clean_image, metadata


# ─────────────────────────────────────────────────────────────────────
# File Discovery
# ─────────────────────────────────────────────────────────────────────
def discover_files(input_dir: str) -> List[str]:
    """Find all supported files in the input directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_dir}")
        sys.exit(1)

    if not input_path.is_dir():
        if input_path.suffix.lower() in ALL_SUPPORTED_EXTS:
            return [str(input_path)]
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            sys.exit(1)

    files = []
    for item in sorted(input_path.iterdir()):
        if item.is_file() and item.suffix.lower() in ALL_SUPPORTED_EXTS:
            files.append(str(item))

    if not files:
        logger.warning(f"No supported files found in: {input_dir}")
        logger.info(f"Supported formats: {', '.join(sorted(ALL_SUPPORTED_EXTS))}")

    return files


# ─────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────
def save_clean_image(image: np.ndarray, output_path: str) -> bool:
    """Save a cleaned image as high-quality PNG."""
    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.save(output_path, "PNG", optimize=True)
        return True
    except Exception as e:
        logger.error(f"Failed to save image '{output_path}': {e}")
        return False


def save_text(text: str, output_path: str) -> bool:
    """Save extracted text to a .txt file (UTF-8)."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        logger.error(f"Failed to save text '{output_path}': {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Main Processing Pipeline
# ─────────────────────────────────────────────────────────────────────
def process_memes(input_dir: str, output_dir: str) -> None:
    """Main pipeline: discover → load → OCR + inpaint → save."""

    text_dir = os.path.join(output_dir, "text")
    image_dir = os.path.join(output_dir, "image")
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    logger.info(f"Input:  {os.path.abspath(input_dir)}")
    logger.info(f"Output: {os.path.abspath(output_dir)}")
    logger.info(f"  ├── text/   (extracted text as .txt)")
    logger.info(f"  └── image/  (clean images, text removed)")
    logger.info(f"  OCR langs:  {', '.join(OCR_LANGUAGES)}")
    logger.info(f"  Confidence: ≥{OCR_CONFIDENCE_THRESHOLD:.0%}")
    logger.info(f"  Inpaint:    {INPAINT_PASSES}-pass, radius={INPAINT_RADIUS}")
    logger.info("─" * 55)

    files = discover_files(input_dir)
    if not files:
        return

    logger.info(f"Found {len(files)} file(s) to process.\n")

    # Pre-load OCR engine
    get_ocr_reader()
    print()

    success_count = 0
    fail_count = 0

    for idx, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        logger.info(f"[{idx}/{len(files)}] Processing: {filename}")

        loaded = load_image(file_path)
        if not loaded:
            fail_count += 1
            continue

        for cv_img, base_name in loaded:
            # ─── Core processing ───
            extracted_text, clean_image, metadata = extract_text_and_clean_image(cv_img)

            # Save clean image
            img_path = os.path.join(image_dir, f"{base_name}_img.png")
            if save_clean_image(clean_image, img_path):
                logger.info(
                    f"  ✓ Image saved: {base_name}_img.png  "
                    f"({clean_image.shape[1]}x{clean_image.shape[0]})  "
                    f"[mask: {metadata.get('mask_coverage', 'N/A')}]"
                )

            # Save extracted text
            txt_path = os.path.join(text_dir, f"{base_name}_text.txt")
            if extracted_text.strip():
                if save_text(extracted_text, txt_path):
                    preview = extracted_text.replace("\n", " | ")
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    logger.info(f"  ✓ Text saved:  {base_name}_text.txt  "
                                f"[{metadata['accepted']} regions, "
                                f"{metadata.get('rejected_low_conf', 0)} rejected]")
                    logger.info(f"    → \"{preview}\"")
                success_count += 1
            else:
                save_text("(no text detected)", txt_path)
                logger.warning(f"  ⚠ No text detected for: {base_name}")
                success_count += 1

        logger.info("")

    # Summary
    logger.info("═" * 55)
    logger.info(f"DONE! Processed {len(files)} file(s)")
    logger.info(f"  ✓ Success: {success_count}")
    if fail_count:
        logger.info(f"  ✗ Failed:  {fail_count}")
    logger.info(f"\nOutput location:")
    logger.info(f"  Text   → {os.path.abspath(text_dir)}/  (.txt files)")
    logger.info(f"  Images → {os.path.abspath(image_dir)}/  (.png files)")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🖼️  Meme Splitter v3 — Professional text extraction & image cleaning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all memes in a folder
  python meme_splitter.py --input ./memes

  # Specify custom output directory
  python meme_splitter.py --input ./memes --output ./split_output

  # Process a single file
  python meme_splitter.py --input ./meme1.png

Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WEBP, PDF
        """,
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input folder containing memes, or a single meme file.",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to output folder (default: ./meme_output next to input).",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        if os.path.isfile(input_path):
            parent = os.path.dirname(input_path)
        else:
            parent = os.path.dirname(input_path.rstrip("/"))
        output_path = os.path.join(parent, "meme_output")

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       🖼️  MEME SPLITTER v3.0 — Professional         ║")
    print("║   Multi-pass OCR + Inpainting + Edge Blending       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    process_memes(input_path, output_path)


if __name__ == "__main__":
    main()
