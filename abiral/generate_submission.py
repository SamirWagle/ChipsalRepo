"""
Generate Submission CSV for Test Release Data
Enhanced with interactive folder and model selection

Usage:
    # Original mode (backward compatible)
    python generate_submission_enhanced.py
    python generate_submission_enhanced.py --single
    python generate_submission_enhanced.py --model_dir Output/image_original_20260212_132844
    
    # New interactive mode
    python generate_submission_enhanced.py --interactive
    python generate_submission_enhanced.py -i
    
    # Direct selection mode
    python generate_submission_enhanced.py --select-folder --select-model
    python generate_submission_enhanced.py -sf -sm
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


class ModelSelector:
    """Modular component for interactive model/folder selection"""
    
    def __init__(self, base_output_dir='Output'):
        self.base_output_dir = Path(base_output_dir)
    
    def list_output_folders(self):
        """List all available output folders containing models"""
        if not self.base_output_dir.exists():
            print(f"Error: Output directory '{self.base_output_dir}' not found")
            return []
        
        folders = []
        for folder in sorted(self.base_output_dir.iterdir()):
            if folder.is_dir():
                # Check if it has model files
                model_files = list(folder.glob("best_model_fold_*.pt"))
                if len(model_files) > 0:
                    folders.append({
                        'path': folder,
                        'name': folder.name,
                        'model_count': len(model_files)
                    })
        return folders
    
    def select_folder_interactive(self):
        """Interactive folder selection"""
        folders = self.list_output_folders()
        
        if not folders:
            print("\nNo model folders found in Output directory!")
            return None
        
        print("\n" + "="*80)
        print("AVAILABLE MODEL FOLDERS")
        print("="*80)
        
        for i, folder_info in enumerate(folders, 1):
            print(f"{i:2d}. {folder_info['name']}")
            print(f"    └─ {folder_info['model_count']} model(s)")
        
        print("="*80)
        
        while True:
            try:
                choice = input(f"\nSelect folder (1-{len(folders)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(folders):
                    selected = folders[choice_idx]
                    print(f"\n✓ Selected: {selected['name']}")
                    return selected['path']
                else:
                    print(f"Invalid choice. Enter 1-{len(folders)}")
            except ValueError:
                print("Invalid input. Enter a number or 'q'")
    
    def list_models_in_folder(self, folder_path):
        """List all models in a folder"""
        folder_path = Path(folder_path)
        model_files = sorted(list(folder_path.glob("best_model_fold_*.pt")))
        
        models = []
        for model_file in model_files:
            # Extract fold number
            fold_num = model_file.stem.split('_')[-1]
            models.append({
                'path': model_file,
                'name': model_file.name,
                'fold': fold_num
            })
        
        return models
    
    def select_models_interactive(self, folder_path):
        """Interactive model selection (single or multiple)"""
        models = self.list_models_in_folder(folder_path)
        
        if not models:
            print(f"\nNo models found in {folder_path}")
            return None, None
        
        print("\n" + "="*80)
        print("AVAILABLE MODELS")
        print("="*80)
        
        for i, model_info in enumerate(models, 1):
            print(f"{i:2d}. {model_info['name']} (Fold {model_info['fold']})")
        
        print("="*80)
        print("\nOptions:")
        print("  - Single model: Enter one number (e.g., '1')")
        print("  - Multiple models: Enter numbers separated by commas (e.g., '1,2,3')")
        print("  - All models (ensemble): Enter 'all'")
        print("  - Quit: Enter 'q'")
        
        while True:
            choice = input(f"\nYour selection: ").strip()
            
            if choice.lower() == 'q':
                return None, None
            
            if choice.lower() == 'all':
                print(f"\n✓ Selected: ALL {len(models)} models (ensemble)")
                return [m['path'] for m in models], True
            
            try:
                # Parse comma-separated choices
                choices = [int(c.strip()) - 1 for c in choice.split(',')]
                
                # Validate
                if all(0 <= c < len(models) for c in choices):
                    selected_models = [models[i]['path'] for i in choices]
                    
                    use_ensemble = len(selected_models) > 1
                    
                    print(f"\n✓ Selected {len(selected_models)} model(s):")
                    for i in choices:
                        print(f"  - {models[i]['name']}")
                    
                    if use_ensemble:
                        print("  → Will use ENSEMBLE (majority voting)")
                    else:
                        print("  → Will use SINGLE model")
                    
                    return selected_models, use_ensemble
                else:
                    print(f"Invalid choice. Enter numbers between 1-{len(models)}")
            except ValueError:
                print("Invalid input. Enter numbers separated by commas, 'all', or 'q'")


def main():
    parser = argparse.ArgumentParser(
        description='Generate submission CSV with optional interactive selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Original backward-compatible mode
  python generate_submission_enhanced.py
  python generate_submission_enhanced.py --single
  python generate_submission_enhanced.py --model_dir Output/image_original_20260212_132844
  
  # Interactive mode (select folder and models)
  python generate_submission_enhanced.py --interactive
  python generate_submission_enhanced.py -i
  
  # Select folder interactively, but use all models
  python generate_submission_enhanced.py --select-folder
  python generate_submission_enhanced.py -sf
  
  # Select specific models interactively
  python generate_submission_enhanced.py --select-model
  python generate_submission_enhanced.py -sm
        """
    )
    
    # Original arguments (backward compatible)
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
    
    # New interactive selection arguments
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Interactive mode: select both folder and models')
    parser.add_argument('-sf', '--select-folder', action='store_true',
                        help='Interactively select output folder')
    parser.add_argument('-sm', '--select-model', action='store_true',
                        help='Interactively select specific models')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING SUBMISSION FOR TEST RELEASE")
    print("=" * 80)
    
    # Initialize selector
    selector = ModelSelector()
    
    # Determine selection mode
    interactive_mode = args.interactive or args.select_folder or args.select_model
    
    selected_models = None
    use_ensemble = None
    model_dir = Path(args.model_dir)
    
    # Interactive folder selection
    if args.interactive or args.select_folder:
        selected_folder = selector.select_folder_interactive()
        if selected_folder is None:
            print("\nSelection cancelled.")
            return
        model_dir = selected_folder
    
    # Interactive model selection
    if args.interactive or args.select_model:
        selected_models, use_ensemble = selector.select_models_interactive(model_dir)
        if selected_models is None:
            print("\nSelection cancelled.")
            return
    
    # Setup paths
    test_csv = Path(args.test_csv)
    test_images = Path(args.test_images)
    
    # Load configuration from saved model
    config = Config()
    config_path = model_dir / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        
        # Load ALL architecture-critical parameters from saved config
        architecture_params = [
            'USE_MODALITY', 'FUSION_STRATEGY', 'NUM_CLASSES',
            'IMAGE_TYPE', 'IMAGE_ENCODER', 'TEXT_ENCODER',
            'HIDDEN_DIM', 'DROPOUT', 'HYBRID_ATTENTION_HEADS',
            'CLIP_MODEL_NAME', 'BGE_MODEL_NAME', 'E5_MODEL_NAME',
            'IMAGE_FEATURE_DIM', 'TEXT_FEATURE_DIM',
            'FREEZE_IMAGE_ENCODER', 'FREEZE_TEXT_ENCODER',
            'UNFREEZE_IMAGE_LAYERS', 'UNFREEZE_TEXT_LAYERS',
            'MAX_TEXT_LENGTH',
        ]
        
        for param in architecture_params:
            if param in saved_config and saved_config[param] is not None:
                setattr(config, param, saved_config[param])
        
        print(f"\nLoaded config: {config_path.name}")
        print(f"  Modality: {config.USE_MODALITY}")
        print(f"  Fusion: {getattr(config, 'FUSION_STRATEGY', 'N/A')}")
        print(f"  Classes: {config.NUM_CLASSES}")
    
    # Load test data
    print(f"\nLoading: {test_csv}")
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")
    
    # Determine which models to use
    if selected_models is not None:
        # Use interactively selected models
        models_to_use = selected_models
        print(f"\nUsing {len(models_to_use)} interactively selected model(s)")
    else:
        # Original logic: find available models
        model_files = sorted(list(model_dir.glob("best_model_fold_*.pt")))
        print(f"\nFound {len(model_files)} model(s)")
        
        if len(model_files) == 0:
            raise FileNotFoundError(f"No models found in {model_dir}")
        
        # Use ensemble or single model
        if args.single:
            models_to_use = [model_files[len(model_files)//2]]
            use_ensemble = False
        else:
            models_to_use = model_files
            use_ensemble = len(model_files) > 1
    
    print(f"\nStrategy: {'ENSEMBLE' if use_ensemble else 'SINGLE MODEL'}")
    for m in models_to_use:
        print(f"  Using: {Path(m).name}")
    
    # Collect predictions
    all_predictions = []
    
    for model_path in models_to_use:
        print(f"\n{'-'*80}")
        print(f"Processing: {Path(model_path).name}")
        
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
    if len(all_predictions) > 1 and use_ensemble:
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