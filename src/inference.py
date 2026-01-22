"""
CHiPSAL 2026 - Subtask A: Inference and Submission
=================================================
Generate predictions and create submission files
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import zipfile
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.dataset import TestMemeDataset, create_test_dataloader
from src.models import create_model


class Predictor:
    """
    Generate predictions on test/eval data.
    """
    
    def __init__(
        self,
        model_type='multimodal',
        text_model_name=None,
        image_model_name=None,
        checkpoint_dir=None,
        use_ensemble=True
    ):
        """
        Initialize predictor.
        
        Args:
            model_type: Type of model
            text_model_name: Text model name
            image_model_name: Image model name  
            checkpoint_dir: Directory containing model checkpoints
            use_ensemble: Whether to ensemble K-fold models
        """
        self.model_type = model_type
        self.text_model_name = text_model_name or TEXT_MODEL_OPTIONS[DEFAULT_TEXT_MODEL]
        self.image_model_name = image_model_name or VISION_MODEL_OPTIONS[DEFAULT_VISION_MODEL]
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else OUTPUT_DIR / 'multimodal_baseline'
        self.use_ensemble = use_ensemble
        self.device = DEVICE
        
        # Initialize tokenizer
        if model_type in ['text', 'multimodal']:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        else:
            self.tokenizer = None
        
        # Load models
        self.models = self._load_models()
        
        print(f"[Predictor] Loaded {len(self.models)} model(s)")
        print(f"[Predictor] Ensemble: {use_ensemble}")
    
    def _load_models(self):
        """Load trained model(s)."""
        models = []
        
        # Find all fold checkpoints
        checkpoint_files = sorted(self.checkpoint_dir.glob('best_model_fold_*.pt'))
        
        if not checkpoint_files:
            # Try single model
            single_checkpoint = self.checkpoint_dir / 'best_model.pt'
            if single_checkpoint.exists():
                checkpoint_files = [single_checkpoint]
            else:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        
        for ckpt_path in checkpoint_files:
            print(f"[INFO] Loading checkpoint: {ckpt_path.name}")
            
            model = create_model(
                model_type=self.model_type,
                text_model_name=self.text_model_name,
                image_model_name=self.image_model_name
            )
            
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()
            
            if not self.use_ensemble:
                # Just use the first model
                return [model]
            
            models.append(model)
        
        return models
    
    @torch.no_grad()
    def predict(self, dataloader):
        """
        Generate predictions on a dataloader.
        
        Args:
            dataloader: Test dataloader
            
        Returns:
            indices: Image indices
            predictions: Predicted labels
            probabilities: Prediction probabilities
        """
        all_indices = []
        all_probs = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            indices = batch['index']
            all_indices.extend(indices)
            
            inputs = {}
            if self.model_type in ['text', 'multimodal']:
                inputs['input_ids'] = batch['input_ids'].to(self.device)
                inputs['attention_mask'] = batch['attention_mask'].to(self.device)
            if self.model_type in ['image', 'multimodal']:
                inputs['image'] = batch['image'].to(self.device)
            
            # Ensemble predictions
            batch_probs = []
            for model in self.models:
                logits = model(**inputs)
                probs = torch.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu().numpy())
            
            # Average probabilities
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.extend(avg_probs)
        
        all_probs = np.array(all_probs)
        predictions = np.argmax(all_probs, axis=1)
        
        return all_indices, predictions, all_probs
    
    def predict_from_df(self, df, image_dir, text_column='text'):
        """
        Generate predictions from a DataFrame.
        
        Args:
            df: DataFrame with 'index' column
            image_dir: Directory containing images
            text_column: Column name for text (or 'ocr_text')
            
        Returns:
            predictions_df: DataFrame with predictions
        """
        # Create dataloader
        dataloader = create_test_dataloader(
            test_df=df,
            image_dir=image_dir,
            tokenizer=self.tokenizer,
            text_column=text_column,
            batch_size=BATCH_SIZE,
            mode=self.model_type
        )
        
        indices, predictions, probabilities = self.predict(dataloader)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'index': indices,
            'label': predictions,
            'prob_non_hate': probabilities[:, 0],
            'prob_hate': probabilities[:, 1]
        })
        
        return results_df
    
    def create_submission(
        self,
        test_df,
        image_dir,
        output_path,
        text_column='text'
    ):
        """
        Create submission file in competition format.
        
        Args:
            test_df: Test DataFrame with 'index' column
            image_dir: Directory with test images
            output_path: Path for output files
            text_column: Column for text data
            
        Returns:
            submission_df: Submission DataFrame
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[INFO] Generating predictions for {len(test_df)} samples...")
        
        # Get predictions
        results_df = self.predict_from_df(
            df=test_df,
            image_dir=image_dir,
            text_column=text_column
        )
        
        # Sort by index (required by competition)
        results_df = results_df.sort_values('index').reset_index(drop=True)
        
        # Create submission DataFrame (only index and label)
        submission_df = results_df[['index', 'label']].copy()
        
        # Save predictions CSV
        csv_path = output_path / 'predictions.csv'
        submission_df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved predictions to {csv_path}")
        
        # Create ZIP file (required for submission)
        zip_path = output_path / 'submission.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, 'predictions.csv')
        print(f"[INFO] Created submission ZIP: {zip_path}")
        
        # Save full results (with probabilities)
        full_results_path = output_path / 'full_predictions.csv'
        results_df.to_csv(full_results_path, index=False)
        print(f"[INFO] Saved full predictions to {full_results_path}")
        
        # Print statistics
        print(f"\n[SUBMISSION STATISTICS]")
        print(f"Total samples: {len(submission_df)}")
        label_counts = submission_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"  {CLASS_NAMES[label]}: {count} ({count/len(submission_df)*100:.1f}%)")
        
        return submission_df


def predict_test_set(
    checkpoint_dir,
    test_csv_path,
    test_images_dir,
    output_dir,
    model_type='multimodal',
    text_column='text'
):
    """
    Convenience function to predict on test set.
    
    Args:
        checkpoint_dir: Directory with model checkpoints
        test_csv_path: Path to test CSV (with 'index' column)
        test_images_dir: Directory with test images
        output_dir: Output directory for submission
        model_type: Model type used in training
        text_column: Column for text data
    """
    # Load test data
    test_df = pd.read_csv(test_csv_path)
    print(f"[INFO] Loaded {len(test_df)} test samples")
    
    # Initialize predictor
    predictor = Predictor(
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        use_ensemble=True
    )
    
    # Generate submission
    submission_df = predictor.create_submission(
        test_df=test_df,
        image_dir=test_images_dir,
        output_path=output_dir,
        text_column=text_column
    )
    
    return submission_df


def main():
    """Main inference pipeline."""
    print("="*60)
    print("CHiPSAL 2026 - SUBTASK A: INFERENCE")
    print("="*60)
    
    # Example: Predict on evaluation set
    # Modify paths based on your data location
    
    EVAL_CSV = DATA_DIR / "eval_subtask_a.csv"  # Create this or download
    EVAL_IMAGES = BASE_DIR / "eval" / "eval_images"  # Download from competition
    
    if not EVAL_CSV.exists():
        print("[WARNING] Eval CSV not found. Creating dummy for demonstration...")
        # Create dummy eval CSV from train data (for testing)
        train_df = pd.read_csv(TRAIN_CSV_FILE)
        dummy_eval = train_df.sample(min(100, len(train_df)))[['index', 'text']]
        dummy_eval.to_csv(EVAL_CSV, index=False)
        EVAL_IMAGES = TRAIN_IMAGES_DIR  # Use train images for demo
    
    # Load eval data
    eval_df = pd.read_csv(EVAL_CSV)
    print(f"[INFO] Loaded {len(eval_df)} eval samples")
    
    # Initialize predictor
    checkpoint_dir = OUTPUT_DIR / 'multimodal_baseline'
    
    if not checkpoint_dir.exists():
        print("[ERROR] No trained model found!")
        print("[INFO] Please run train.py first to train a model.")
        return None
    
    predictor = Predictor(
        model_type='multimodal',
        checkpoint_dir=checkpoint_dir,
        use_ensemble=True
    )
    
    # Generate submission
    submission_df = predictor.create_submission(
        test_df=eval_df,
        image_dir=EVAL_IMAGES,
        output_path=SUBMISSION_DIR / 'eval',
        text_column='text'
    )
    
    print(f"\n[SUCCESS] Submission created!")
    print(f"[INFO] Submit the ZIP file at: {SUBMISSION_DIR / 'eval' / 'submission.zip'}")
    
    return submission_df


if __name__ == "__main__":
    submission = main()
