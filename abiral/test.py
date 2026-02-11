"""
Test script for making predictions on new images using trained model
"""

import torch
import argparse
import os
from PIL import Image

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


def main():
    parser = argparse.ArgumentParser(description='Multimodal Classification Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                        help='Inference mode: single image or batch from CSV')
    
    # Single image mode arguments
    parser.add_argument('--image_path', type=str,
                        help='Path to image file (for single mode)')
    parser.add_argument('--text', type=str, default=None,
                        help='Text associated with image (for single mode)')
    
    # Batch mode arguments
    parser.add_argument('--csv_path', type=str,
                        help='Path to CSV file (for batch mode)')
    parser.add_argument('--image_folder', type=str,
                        help='Folder containing images (for batch mode)')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save predictions (for batch mode)')
    
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
        
        print(f"\nPrediction: Class {prediction}")
        print(f"Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"  Class {i}: {prob:.4f}")
        
        if 'modality_weights' in additional_info:
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


if __name__ == '__main__':
    main()
