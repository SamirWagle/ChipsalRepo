"""
Inference module for making predictions on test images
"""

import torch
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm


class InferenceEngine:
    """
    Inference engine for making predictions on images
    """
    
    def __init__(self, model, config, image_transform=None, text_tokenizer=None):
        """
        Args:
            model: Trained multimodal model
            config: Configuration object
            image_transform: Image preprocessing function
            text_tokenizer: Text tokenizer function
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        self.model.eval()
        
        self.image_transform = image_transform
        self.text_tokenizer = text_tokenizer
    
    def predict_single_image(self, image_path, text=None):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            text: Optional text associated with image (for multimodal)
            
        Returns:
            prediction: Predicted class index
            probabilities: Class probabilities
            additional_info: Additional outputs (e.g., modality weights)
        """
        # Load and preprocess image
        pixel_values = None
        if self.config.USE_MODALITY in ['image', 'both']:
            try:
                image = Image.open(image_path).convert('RGB')
                if self.image_transform is not None:
                    pixel_values = self.image_transform(image).unsqueeze(0).to(self.device)
                else:
                    raise ValueError("Image transform not provided")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None, None, None
        
        # Preprocess text
        input_ids = None
        attention_mask = None
        if self.config.USE_MODALITY in ['text', 'both']:
            if text is None or pd.isna(text):
                text = ""
            
            if self.text_tokenizer is not None:
                text_encoding = self.text_tokenizer(text)
                input_ids = text_encoding['input_ids'].to(self.device)
                attention_mask = text_encoding['attention_mask'].to(self.device)
            else:
                raise ValueError("Text tokenizer not provided")
        
        # Make prediction
        with torch.no_grad():
            predictions, probabilities, additional_outputs = self.model.predict(
                pixel_values, input_ids, attention_mask
            )
        
        prediction = predictions[0].item()
        probabilities = probabilities[0].cpu().numpy()
        
        return prediction, probabilities, additional_outputs
    
    def predict_batch(self, image_folder, image_filenames, texts=None, batch_size=16):
        """
        Predict classes for a batch of images
        
        Args:
            image_folder: Folder containing images
            image_filenames: List of image filenames
            texts: Optional list of texts (for multimodal)
            batch_size: Batch size for inference
            
        Returns:
            predictions: List of predicted class indices
            probabilities: List of class probabilities
        """
        if texts is None:
            texts = [""] * len(image_filenames)
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_filenames), batch_size), desc='Predicting'):
            batch_filenames = image_filenames[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            batch_pixel_values = []
            batch_input_ids = []
            batch_attention_masks = []
            valid_indices = []
            
            # Load batch
            for idx, (filename, text) in enumerate(zip(batch_filenames, batch_texts)):
                image_path = os.path.join(image_folder, filename)
                
                # Load image
                pixel_values = None
                if self.config.USE_MODALITY in ['image', 'both']:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        if self.image_transform is not None:
                            pixel_values = self.image_transform(image)
                            batch_pixel_values.append(pixel_values)
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")
                        continue
                
                # Process text
                if self.config.USE_MODALITY in ['text', 'both']:
                    if pd.isna(text):
                        text = ""
                    if self.text_tokenizer is not None:
                        text_encoding = self.text_tokenizer(text)
                        batch_input_ids.append(text_encoding['input_ids'].squeeze(0))
                        batch_attention_masks.append(text_encoding['attention_mask'].squeeze(0))
                
                valid_indices.append(i + idx)
            
            if len(valid_indices) == 0:
                continue
            
            # Stack batch tensors
            if self.config.USE_MODALITY in ['image', 'both']:
                batch_pixel_values = torch.stack(batch_pixel_values).to(self.device)
            else:
                batch_pixel_values = None
            
            if self.config.USE_MODALITY in ['text', 'both']:
                batch_input_ids = torch.stack(batch_input_ids).to(self.device)
                batch_attention_masks = torch.stack(batch_attention_masks).to(self.device)
            else:
                batch_input_ids = None
                batch_attention_masks = None
            
            # Predict
            with torch.no_grad():
                predictions, probabilities, _ = self.model.predict(
                    batch_pixel_values, batch_input_ids, batch_attention_masks
                )
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_probabilities
    
    def predict_from_csv(self, csv_path, image_folder, output_path=None):
        """
        Predict classes for images listed in CSV
        
        Args:
            csv_path: Path to CSV with image filenames and texts
            image_folder: Folder containing images
            output_path: Optional path to save predictions
            
        Returns:
            results_df: DataFrame with predictions
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Get filenames and texts
        image_filenames = df[self.config.IMAGE_COL].tolist()
        texts = df[self.config.TEXT_COL].tolist() if self.config.TEXT_COL in df.columns else None
        
        # Predict
        predictions, probabilities = self.predict_batch(
            image_folder, image_filenames, texts, batch_size=self.config.BATCH_SIZE
        )
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Predicted_Label'] = predictions
        
        # Add probabilities
        num_classes = probabilities[0].shape[0]
        for i in range(num_classes):
            results_df[f'Prob_Class_{i}'] = [prob[i] for prob in probabilities]
        
        # Save if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return results_df


def load_model_for_inference(model_path, config):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to saved model weights
        config: Configuration object
        
    Returns:
        model: Loaded model ready for inference
        image_transform: Image preprocessing function
        text_tokenizer: Text tokenizer function
    """
    import json
    from encoders import create_encoders, get_image_processor, get_text_tokenizer, ImageTransform
    from fusion_models import create_fusion_model
    from model import create_complete_model
    
    # Try to load config from model directory (if model_path ends with .pt)
    if model_path.endswith('.pt'):
        model_dir = os.path.dirname(model_path)
        config_json_path = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r') as f:
                saved_config = json.load(f)
            
            # Load ALL architecture-critical parameters from saved config
            # These must match exactly how the model was trained
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
            
            print(f"Loaded model config from {config_json_path}")
            print(f"  Modality: {config.USE_MODALITY}")
            print(f"  Fusion strategy: {getattr(config, 'FUSION_STRATEGY', 'N/A')}")
            print(f"  Hidden dim: {getattr(config, 'HIDDEN_DIM', 'N/A')}")
            print(f"  Dropout: {getattr(config, 'DROPOUT', 'N/A')}")
            print(f"  Num classes: {config.NUM_CLASSES}")
    
    # Ensure NUM_CLASSES is set
    if config.NUM_CLASSES is None:
        print("Warning: NUM_CLASSES not found in config. Defaulting to 2")
        config.NUM_CLASSES = 2
    
    # Create encoders
    image_encoder, text_encoder = create_encoders(config)
    
    # Create fusion model
    fusion_model = create_fusion_model(config)
    
    # Create complete model
    model = create_complete_model(
        image_encoder, text_encoder, fusion_model, config.USE_MODALITY
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    
    print(f"Model loaded from {model_path}")
    
    # Create preprocessing functions
    image_transform = None
    text_tokenizer = None
    
    if config.USE_MODALITY in ['image', 'both']:
        image_processor = get_image_processor(config.CLIP_MODEL_NAME)
        image_transform = ImageTransform(image_processor)
    
    if config.USE_MODALITY in ['text', 'both']:
        text_tokenizer = get_text_tokenizer(
            config.TEXT_ENCODER,
            config.BGE_MODEL_NAME if config.TEXT_ENCODER == 'bge-m3' else config.E5_MODEL_NAME,
            config.MAX_TEXT_LENGTH
        )
    
    return model, image_transform, text_tokenizer
