"""
Quick Test Script - Fast testing with your trained models
Run this to quickly test your trained MoE model!

Usage:
    # Test single demo image
    python quick_test.py
    
    # Test specific image
    python quick_test.py --image demo_input/1005.jpg
    
    # Generate submission CSV (when you have test.csv and test_images folder)
    python quick_test.py --mode submission --csv test.csv --images test_images
"""

import torch
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from config import Config
from inference import InferenceEngine


# =============================================================================
# MODEL LOADER FOR NOVEL MODELS (train_novel.py)
# =============================================================================

def load_novel_model(model_path):
    """
    Load models trained with train_novel.py
    These models are saved as checkpoints with 'model_state_dict' key
    """
    import torch
    from transformers import AutoTokenizer
    from PIL import Image
    import torchvision.transforms as T
    
    print(f"Loading checkpoint from: {model_path}")
    
    # Load checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"Checkpoint info: Fold {checkpoint.get('fold', 'N/A')}, "
              f"Best F1: {checkpoint.get('best_f1', 'N/A'):.4f}")
    else:
        state_dict = checkpoint
    
    # Import model creation function
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.models_novel import create_novel_model
    
    # Create model - default to MOE architecture (change if you used different model)
    # Check the model structure from state_dict to determine model type
    model_type = 'moe'  # Most common from your training
    
    # You can add logic here to detect model type from state_dict keys
    if any('lora' in k for k in state_dict.keys()):
        model_type = 'mmlora'
    elif any('siglip' in k for k in state_dict.keys()):
        model_type = 'siglip_muril'
    
    print(f\"Creating model architecture: {model_type}\")\n    model = create_novel_model(model_type, text_model=\"google/muril-base-cased\")\n    \n    # Load state dict\n    model.load_state_dict(state_dict)\n    model.eval()\n    model = model.to(device)\n    \n    print(\"\u2713 Model loaded successfully!\")\n    \n    # Create image transform\n    image_transform = T.Compose([\n        T.Resize((224, 224)),\n        T.ToTensor(),\n        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n    ])\n    \n    # Create text tokenizer\n    tokenizer = AutoTokenizer.from_pretrained(\"google/muril-base-cased\")\n    \n    def text_tokenizer_fn(text):\n        return tokenizer(\n            text,\n            padding='max_length',\n            truncation=True,\n            max_length=256,\n            return_tensors='pt'\n        )\n    \n    return model, image_transform, text_tokenizer_fn


# =============================================================================
# SIMPLIFIED INFERENCE ENGINE FOR NOVEL MODELS
# =============================================================================

class SimpleInferenceEngine:
    \"\"\"Simplified inference engine for novel models\"\"\"
    \n    def __init__(self, model, image_transform, text_tokenizer):\n        self.model = model\n        self.image_transform = image_transform\n        self.text_tokenizer = text_tokenizer\n        self.device = next(model.parameters()).device\n    \n    def predict_single_image(self, image_path, text=None):\n        \"\"\"Predict class for a single image\"\"\"\n        from PIL import Image\n        import torch\n        \n        # Load and transform image\n        image = Image.open(image_path).convert('RGB')\n        pixel_values = self.image_transform(image).unsqueeze(0).to(self.device)\n        \n        # Tokenize text\n        if text is None or text.strip() == \"\":\n            text = \"\"\n        text_encoding = self.text_tokenizer(text)\n        input_ids = text_encoding['input_ids'].to(self.device)\n        attention_mask = text_encoding['attention_mask'].to(self.device)\n        \n        # Predict\n        self.model.eval()\n        with torch.no_grad():\n            logits = self.model(\n                pixel_values=pixel_values,\n                input_ids=input_ids,\n                attention_mask=attention_mask\n            )\n            \n            # Handle tuple output (some models return logits, loss)\n            if isinstance(logits, tuple):\n                logits = logits[0]\n            \n            probabilities = torch.softmax(logits, dim=-1)\n            prediction = torch.argmax(probabilities, dim=-1)\n        \n        return prediction.item(), probabilities[0].cpu().numpy(), {}\n\n\n# =============================================================================\n# TESTING FUNCTIONS\n# =============================================================================


def quick_test_single_image(model_path, image_path, text=None):
    """Quick test on a single image"""
    print("=" * 80)
    print("QUICK TEST - SINGLE IMAGE PREDICTION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Text: {text if text else 'None'}")
    print("-" * 80)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("\nAvailable models:")
        output_dir = Path(__file__).parent / "Output"
        if output_dir.exists():
            models = list(output_dir.glob("*.pt"))
            if models:
                print(f"\n  Output/")
                for m in models:
                    print(f"    - {m.name}")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    try:
        # Load model using novel model loader
        print("\n🔄 Loading model...")
        model, image_transform, text_tokenizer = load_novel_model(model_path)
        
        # Create inference engine
        engine = SimpleInferenceEngine(model, image_transform, text_tokenizer)
        
        # Make prediction
        print("🔄 Making prediction...")
        prediction, probabilities, additional_info = engine.predict_single_image(image_path, text)
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"✓ Predicted Class: {prediction}")
        print(f"  {'hate' if prediction == 1 else 'no-hate'}")
        print("\nClass Probabilities:")
        class_names = ['no-hate', 'hate']
        for i, prob in enumerate(probabilities):
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_names[i]:10s} [{bar}] {prob:.2%}")
        
        # Show modality weights if available
        if 'modality_weights' in additional_info:
            weights = additional_info['modality_weights'][0]
            print("\nModality Contribution:")
            print(f"  Image: {weights[0]:.2%}")
            print(f"  Text:  {weights[1]:.2%}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


def quick_test_submission(model_path, csv_path, image_folder, output_path="submission.csv"):
    """Generate submission CSV from test data"""
    import pandas as pd
    from tqdm import tqdm
    
    print("=" * 80)
    print("QUICK TEST - SUBMISSION GENERATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test CSV: {csv_path}")
    print(f"Images: {image_folder}")
    print(f"Output: {output_path}")
    print("-" * 80)
    
    # Load configuration
    config = Config()
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"❌ Error: Image folder not found at {image_folder}")
        return
    
    try:
        # Load model
        print("\n🔄 Loading model...")
        model, image_transform, text_tokenizer = load_novel_model(model_path)
        
        # Load test data
        test_df = pd.read_csv(csv_path)
        
        if 'index' not in test_df.columns:
            print(f"❌ Error: CSV must have 'index' column. Found: {test_df.columns.tolist()}")
            return
        
        print(f"\n✓ Loaded {len(test_df)} test samples")
        
        # Create inference engine
        engine = SimpleInferenceEngine(model, image_transform, text_tokenizer)
        
        # Make predictions
        predictions = []
        print("\n🔄 Generating predictions...")
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting'):
            # Get image path
            img_filename = str(row['index'])
            if not img_filename.endswith(('.png', '.jpg', '.jpeg')):
                img_filename = f"{img_filename}.png"
            
            img_path = os.path.join(image_folder, img_filename)
            
            # Get text if available (check for 'text' or 'extracted_text' column)
            text = ""
            for col in ['text', 'extracted_text', 'ocr_text']:
                if col in test_df.columns and pd.notna(row.get(col)):
                    text = str(row.get(col, ""))
                    break
            
            # Make prediction
            try:
                prediction, probabilities, _ = engine.predict_single_image(img_path, text)
                predictions.append(prediction)
            except Exception as e:
                print(f"\n⚠ Warning: Error predicting {img_filename}: {e}")
                predictions.append(0)  # Default to no-hate on error
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'index': test_df['index'],
            'label': predictions
        })
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        
        # Display summary
        print("\n" + "=" * 80)
        print("SUBMISSION READY")
        print("=" * 80)
        print(f"✓ Saved to: {output_path}")
        print(f"  Total samples: {len(submission_df)}")
        print("\nPrediction Distribution:")
        for label in sorted(submission_df['label'].unique()):
            count = (submission_df['label'] == label).sum()
            pct = 100 * count / len(submission_df)
            label_name = 'hate' if label == 1 else 'no-hate'
            print(f"  {label_name:10s} (label={label}): {count:5d} samples ({pct:5.2f}%)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Quick test with trained model')
    
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'submission'],
                        help='Test mode: single image or submission CSV')
    
    # Model path (defaults to best fold 1 model in Output folder)
    default_model = str(Path(__file__).parent / "Output" / "best_model_fold_1.pt")
    parser.add_argument('--model', type=str, default=default_model,
                        help=f'Path to model (default: {default_model})')
    
    # Single image mode
    default_image = str(Path(__file__).parent.parent / "demo_input" / "1005.jpg")
    parser.add_argument('--image', type=str, default=default_image,
                        help=f'Image path (default: {default_image})')
    parser.add_argument('--text', type=str, default=None,
                        help='Optional text for the image')
    
    # Submission mode
    parser.add_argument('--csv', type=str, default='test.csv',
                        help='Test CSV path (default: test.csv)')
    parser.add_argument('--images', type=str, default='test_images',
                        help='Test images folder (default: test_images)')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output CSV path (default: submission.csv)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        quick_test_single_image(args.model, args.image, args.text)
    elif args.mode == 'submission':
        quick_test_submission(args.model, args.csv, args.images, args.output)


if __name__ == '__main__':
    main()
