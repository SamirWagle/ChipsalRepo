"""
Test script for making predictions on new images using trained model

QUICK START EXAMPLES:
====================

1. Test single demo image:
   python quick_test.py

2. Test specific image:
   python quick_test.py --image path/to/image.jpg

3. Generate submission CSV:
   python quick_test.py --mode submission --csv test.csv --images test_images

4. Check trained model configuration:
   python test.py --mode check --model_path outputs/novel_moe

5. Advanced: Generate submission with specific model:
   python test.py --mode submission \
       --model_path outputs/novel_moe/best_model_fold_1.pt \
       --csv_path test.csv \
       --image_folder test_images \
       --output_path submission.csv

AVAILABLE MODELS:
================
Your trained models are in: outputs/novel_moe/
- best_model_fold_0.pt
- best_model_fold_1.pt

Use fold_1 model (usually better performance) or ensemble both for final submission.
"""

import torch
import argparse
import os
import json
import sys
from PIL import Image
from tqdm import tqdm

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference import load_model_for_inference, InferenceEngine


def predict_single_image(model_path, image_path, text=None, config=None):
    """
    Predict class for a single image
    
    Args:
        model_path: Path to trained model weights
        image_path: Path to image file
        text: Optional text associated with image
        config: Configuration object (if None, uses default)
    
    Returns:
        prediction: Predicted class
        probabilities: Class probabilities
    """
    if config is None:
        config = Config()
    
    # Load model
    model, image_transform, text_tokenizer = load_model_for_inference(model_path, config)
    
    # Create inference engine
    engine = InferenceEngine(model, config, image_transform, text_tokenizer)
    
    # Make prediction
    prediction, probabilities, additional_info = engine.predict_single_image(image_path, text)
    
    return prediction, probabilities, additional_info


def predict_from_csv(model_path, csv_path, image_folder, output_path=None, config=None):
    """
    Predict classes for images listed in CSV
    
    Args:
        model_path: Path to trained model weights
        csv_path: Path to CSV file
        image_folder: Folder containing images
        output_path: Path to save predictions
        config: Configuration object
    
    Returns:
        results_df: DataFrame with predictions
    """
    if config is None:
        config = Config()
    
    # Load model
    model, image_transform, text_tokenizer = load_model_for_inference(model_path, config)
    
    # Create inference engine
    engine = InferenceEngine(model, config, image_transform, text_tokenizer)
    
    # Make predictions
    results_df = engine.predict_from_csv(csv_path, image_folder, output_path)
    
    return results_df


def get_prediction_csv(model_path, csv_path, image_folder, output_path, config=None):
    """
    Generate submission-ready CSV with predictions for test data
    
    Args:
        model_path: Path to trained model weights
        csv_path: Path to test CSV file (must have 'index' and optionally 'text' columns)
        image_folder: Path to folder containing test images
        output_path: Path to save submission CSV
        config: Configuration object (if None, uses default)
    
    Returns:
        submission_df: DataFrame with columns ['index', 'label'] ready for submission
    """
    import pandas as pd
    
    if config is None:
        config = Config()
    
    # Load test data
    test_df = pd.read_csv(csv_path)
    
    # Verify required column exists
    if 'index' not in test_df.columns:
        raise ValueError(f"Test CSV must have 'index' column. Found columns: {test_df.columns.tolist()}")
    
    # Load model
    model, image_transform, text_tokenizer = load_model_for_inference(model_path, config)
    
    # Create inference engine
    engine = InferenceEngine(model, config, image_transform, text_tokenizer)
    
    # Prepare predictions list
    predictions = []
    
    print(f"Making predictions on {len(test_df)} test samples...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting'):
        # Get image path
        img_filename = str(row['index'])
        if not img_filename.endswith(('.png', '.jpg', '.jpeg')):
            img_filename = f"{img_filename}.png"
        
        img_path = os.path.join(image_folder, img_filename)
        
        # Get text if available
        text = str(row.get(config.TEXT_COL, "")) if config.TEXT_COL in test_df.columns and pd.notna(row.get(config.TEXT_COL)) else ""
        
        # Make prediction
        try:
            prediction, probabilities, _ = engine.predict_single_image(img_path, text)
            predictions.append(prediction)
        except Exception as e:
            print(f"Warning: Error predicting {img_filename}: {e}")
            predictions.append(0)  # Default prediction on error
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'index': test_df['index'],
        'label': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission CSV saved to: {output_path}")
    print(f"  Total samples: {len(submission_df)}")
    print(f"\nPrediction distribution:")
    for label in sorted(submission_df['label'].unique()):
        count = (submission_df['label'] == label).sum()
        pct = 100 * count / len(submission_df)
        print(f"  Label {label}: {count} samples ({pct:.2f}%)")
    
    return submission_df


def check_config(model_dir):
    """
    Check configuration used to train a model and show recommended inference command
    
    Args:
        model_dir: Directory containing config.json and model weights
    
    Returns:
        config_dict: Dictionary with configuration (or None if not found)
    """
    config_path = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print(f"❌ Error: No config.json found in {model_dir}")
        print(f"   Make sure you're pointing to the experiment output directory")
        return None
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    modality = config_dict.get('USE_MODALITY', 'unknown')
    fusion = config_dict.get('FUSION_STRATEGY', 'unknown')
    
    print("=" * 80)
    print("MODEL CONFIGURATION")
    print("=" * 80)
    print(f"Directory: {model_dir}")
    print("-" * 80)
    
    # Determine model type
    if modality == 'image':
        model_type = "Image-Only"
        modality_arg = "--use_modality image"
        fusion_arg = ""
    elif modality == 'text':
        model_type = "Text-Only"
        modality_arg = "--use_modality text"
        fusion_arg = ""
    else:  # both
        model_type = f"Multimodal ({fusion.replace('_', ' ').title()})"
        modality_arg = "--use_modality both"
        fusion_arg = f"--fusion_strategy {fusion}"
    
    print(f"Model Type: {model_type}")
    print(f"Modality: {modality}")
    if modality == 'both':
        print(f"Fusion Strategy: {fusion}")
    print("-" * 80)
    
    # Show key hyperparameters
    print("\nKey Hyperparameters:")
    print(f"  Batch Size: {config_dict.get('BATCH_SIZE', 'N/A')}")
    print(f"  Learning Rate: {config_dict.get('LEARNING_RATE', 'N/A')}")
    print(f"  Epochs: {config_dict.get('NUM_EPOCHS', 'N/A')}")
    print(f"  Hidden Dim: {config_dict.get('HIDDEN_DIM', 'N/A')}")
    print(f"  Dropout: {config_dict.get('DROPOUT', 'N/A')}")
    
    # Show available models
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if model_files:
        print("\nAvailable Models:")
        for mf in sorted(model_files):
            print(f"  - {mf}")
        model_path = os.path.join(model_dir, model_files[0])
    else:
        model_path = "PATH_TO_MODEL.pt"
    
    # Show recommended command
    print("\n" + "=" * 80)
    print("RECOMMENDED INFERENCE COMMAND")
    print("=" * 80)
    print("For submission generation:\n")
    print(f"python test.py \\")
    print(f"  --mode submission \\")
    print(f"  --model_path {model_path} \\")
    print(f"  --csv_path test.csv \\")
    print(f"  --image_folder test_images \\")
    print(f"  --output_path submission.csv \\")
    print(f"  {modality_arg}", end="")
    if fusion_arg:
        print(f" \\")
        print(f"  {fusion_arg}")
    else:
        print()
    print("=" * 80)
    
    return config_dict


def main():
    parser = argparse.ArgumentParser(description='Multimodal Classification Inference')
    
    # Mode selection (now includes 'check')
    parser.add_argument('--mode', type=str, 
                        choices=['single', 'batch', 'submission', 'check'], 
                        default='single',
                        help='Inference mode: single image, batch from CSV, submission CSV, or check model config')
    
    # Model path (not required for 'check' mode)
    parser.add_argument('--model_path', type=str,
                        help='Path to trained model weights or directory (required except for check mode)')
    
    # Single image mode arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to image file (for single mode)')
    parser.add_argument('--text', type=str, default=None,
                        help='Text associated with image (for single mode)')
    
    # Batch mode arguments
    parser.add_argument('--csv_path', type=str,
                        help='Path to CSV file (for batch mode)')
    parser.add_argument('--image_folder', type=str,
                        help='Folder containing images (for batch/submission mode)')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save predictions (for batch/submission mode)')
    
    # Configuration arguments
    parser.add_argument('--use_modality', type=str, choices=['image', 'text', 'both'],
                        default='both', help='Modality to use')
    parser.add_argument('--fusion_strategy', type=str,
                        choices=['early', 'late_voting', 'late_bagging', 'hybrid'],
                        default='hybrid', help='Fusion strategy (for multimodal)')
    parser.add_argument('--image_type', type=str, choices=['original', 'text_removed'],
                        default='original', help='Image type')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Special handling for 'check' mode
    if args.mode == 'check':
        if args.model_path is None:
            parser.error("--model_path required for check mode")
        check_config(args.model_path)
        return
    
    # Validate model_path for other modes
    if args.model_path is None:
        parser.error("--model_path is required for this mode")
    
    # Create config
    config = Config()
    config.update(
        USE_MODALITY=args.use_modality,
        FUSION_STRATEGY=args.fusion_strategy,
        IMAGE_TYPE=args.image_type,
        NUM_CLASSES=args.num_classes,
        DEVICE=args.device
    )
    
    # Run inference
    if args.mode == 'single':
        if args.image_path is None:
            raise ValueError("--image_path required for single mode")
        
        print(f"Predicting for image: {args.image_path}")
        prediction, probabilities, additional_info = predict_single_image(
            args.model_path, args.image_path, args.text, config
        )
        
        if prediction is None or probabilities is None:
            print("Error: Failed to make prediction")
            return
        
        print(f"\nPrediction: Class {prediction}")
        print(f"Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  Class {i}: {prob:.4f}")
        
        if additional_info and 'modality_weights' in additional_info:
            weights = additional_info['modality_weights'][0]
            print(f"\nModality weights:")
            print(f"  Image: {weights[0]:.4f}")
            print(f"  Text: {weights[1]:.4f}")
    
    elif args.mode == 'batch':
        if args.csv_path is None or args.image_folder is None:
            raise ValueError("--csv_path and --image_folder required for batch mode")
        
        print(f"Predicting for images in CSV: {args.csv_path}")
        results_df = predict_from_csv(
            args.model_path, args.csv_path, args.image_folder, args.output_path, config
        )
        
        print(f"\nPredictions saved to: {args.output_path}")
        print(f"Total images processed: {len(results_df)}")
        
        # Print summary
        print("\nPrediction summary:")
        for class_idx in results_df['Predicted_Label'].unique():
            count = (results_df['Predicted_Label'] == class_idx).sum()
            print(f"  Class {class_idx}: {count} images ({count/len(results_df)*100:.2f}%)")
    
    elif args.mode == 'submission':
        if args.csv_path is None or args.image_folder is None:
            raise ValueError("--csv_path and --image_folder required for submission mode")
        
        print(f"Generating submission CSV from: {args.csv_path}")
        submission_df = get_prediction_csv(
            args.model_path, args.csv_path, args.image_folder, args.output_path, config
        )
        
        print(f"\n✓ Submission ready: {args.output_path}")
        print(f"  Format: index, label")
        print(f"  Rows: {len(submission_df)}")


if __name__ == '__main__':
    main()
