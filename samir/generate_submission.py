"""
Generate Submission CSV for Test Release Data
Simple script to generate predictions with index, label columns

Usage:
    python generate_submission.py
    python generate_submission.py --single  (use single model instead of ensemble)
"""

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference import load_model_for_inference, InferenceEngine


def main():
    parser = argparse.ArgumentParser(description='Generate submission CSV')
    parser.add_argument('--model_dir', type=str, 
                        default='Output/image_original_20260212_132844',
                        help='Directory containing trained models')
    parser.add_argument('--test_csv', type=str,
                        default='test_release/index_text_test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--test_images', type=str,
                        default='test_release/test_images',
                        help='Path to test images folder')
    parser.add_argument('--output', type=str,
                        default='submission.csv',
                        help='Output CSV file path')
    parser.add_argument('--single', action='store_true',
                        help='Use single best model instead of ensemble')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING SUBMISSION FOR TEST RELEASE")
    print("=" * 80)
    
    # Setup paths
    model_dir = Path(args.model_dir)
    test_csv = Path(args.test_csv)
    test_images = Path(args.test_images)
    
    # Load configuration from saved model
    config = Config()
    config_path = model_dir / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        config.USE_MODALITY = saved_config['USE_MODALITY']
        config.NUM_CLASSES = saved_config['NUM_CLASSES']
        config.IMAGE_TYPE = saved_config.get('IMAGE_TYPE', 'original')
        
        print(f"\nLoaded config: {config_path.name}")
        print(f"  Modality: {config.USE_MODALITY}")
        print(f"  Classes: {config.NUM_CLASSES}")
    
    # Load test data
    print(f"\nLoading: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    
    # Find available models
    model_files = sorted(list(model_dir.glob("best_model_fold_*.pt")))
    print(f"\nFound {len(model_files)} model(s)")
    
    if len(model_files) == 0:
        raise FileNotFoundError(f"No models found in {model_dir}")
    
    # Use ensemble or single model
    use_ensemble = len(model_files) > 1 and not args.single
    models_to_use = model_files if use_ensemble else [model_files[len(model_files)//2]]
    
    print(f"\nStrategy: {'ENSEMBLE' if use_ensemble else 'SINGLE MODEL'}")
    for m in models_to_use:
        print(f"  Using: {m.name}")
    
    # Collect predictions
    all_predictions = []
    
    for model_path in models_to_use:
        print(f"\n{'-'*80}")
        print(f"Processing: {model_path.name}")
        
        # Load model
        model, image_transform, text_tokenizer = load_model_for_inference(
            str(model_path), config
        )
        
        engine = InferenceEngine(model, config, image_transform, text_tokenizer)
        
        # Predict
        predictions = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting'):
            img_filename = str(row['index'])
            
            # Add extension if needed
            if not img_filename.endswith(('.png', '.jpg', '.jpeg')):
                for ext in ['.png', '.jpg', '.jpeg']:
                    if (test_images / (img_filename + ext)).exists():
                        img_filename = img_filename + ext
                        break
            
            img_path = test_images / img_filename
            
            # Get text if model uses it
            text = None
            if config.USE_MODALITY in ['text', 'both'] and 'text' in test_df.columns:
                text_val = row.get('text', '')
                text = str(text_val) if pd.notna(text_val) else ""
            
            try:
                prediction, _, _ = engine.predict_single_image(str(img_path), text)
                if prediction is not None:
                    predictions.append(prediction)
                else:
                    print(f"\nWarning: Got None prediction for {img_filename}, using default 0")
                    predictions.append(0)
            except Exception as e:
                print(f"\nWarning: Error on {img_filename}: {e}")
                predictions.append(0)
        
        all_predictions.append(predictions)
    
    # Ensemble if using multiple models
    if len(all_predictions) > 1:
        import numpy as np
        all_predictions_array = np.array(all_predictions)
        final_predictions = []
        
        for i in range(all_predictions_array.shape[1]):
            votes = all_predictions_array[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            final_predictions.append(int(majority_class))
        
        print(f"\nEnsembled {len(models_to_use)} models (majority voting)")
    else:
        final_predictions = all_predictions[0]
    
    # Create submission
    submission_df = pd.DataFrame({
        'index': test_df['index'],
        'label': final_predictions
    })
    
    # Save
    submission_df.to_csv(args.output, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUBMISSION READY!")
    print(f"{'='*80}")
    print(f"✓ Saved: {args.output}")
    print(f"  Samples: {len(submission_df)}")
    print(f"\nDistribution:")
    for label in sorted(submission_df['label'].unique()):
        count = (submission_df['label'] == label).sum()
        pct = 100 * count / len(submission_df)
        name = 'hate' if label == 1 else 'no-hate'
        print(f"  {name:8s} = {count:4d} ({pct:5.2f}%)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
