"""
09_inference.py

Inference Script for Best DL Model

This script uses the best DL model (RoBERTa) to perform inference on the demonstration dataset:
1. Loads the best trained RoBERTa model, tokenizer, and label encoder
2. Loads demonstration dataset from datasets/Demonstration dataset.csv
3. Combines Title and Text columns for inference
4. Runs batch inference with confidence scores
5. Saves results as JSON to dl_methods/roberta/logs/

Output: dl_methods/roberta/logs/demonstration_dataset_predictions.json
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from transformers import RobertaForSequenceClassification, RobertaTokenizer

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Configuration
# =========================

# Model paths (RoBERTa - best model)
MODEL_DIR = "dl_methods/roberta"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model", "best")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "model", "tokenizer")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "model", "label_encoder.pkl")
OUTPUT_PATH = os.path.join(MODEL_DIR, "logs", "demonstration_dataset_predictions.json")

# Input dataset
DEMO_DATASET_PATH = os.path.join("datasets", "Demonstration dataset.csv")

# Inference settings (from RoBERTa's best hyperparameters)
MAX_LEN = 128
BATCH_SIZE = 32
NUM_WORKERS = 2

# Label mapping
LABEL_MAPPING = {0: "Great", 1: "Bait"}

# =========================
# Dataset Class for Inference
# =========================

class InferenceDataset(Dataset):
    """PyTorch Dataset for inference (no labels needed)."""
    
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# =========================
# Model Loading
# =========================

def load_model_and_components():
    """Load the best RoBERTa model, tokenizer, and label encoder."""
    print("\n--- Loading Model Components ---")
    
    # Load tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    print("✓ Tokenizer loaded")
    
    # Load label encoder
    print(f"Loading label encoder from {LABEL_ENCODER_PATH}...")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded")
    print(f"  Label classes: {label_encoder.classes_}")
    
    # Load model
    print(f"Loading best model from {BEST_MODEL_PATH}...")
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}")
    
    num_classes = len(label_encoder.classes_)
    model = RobertaForSequenceClassification.from_pretrained(
        BEST_MODEL_PATH,
        num_labels=num_classes,
        local_files_only=True
    )
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("✓ Model loaded and moved to device:", device)
    
    return model, tokenizer, label_encoder

# =========================
# Data Loading
# =========================

def load_demonstration_dataset():
    """Load and preprocess the demonstration dataset."""
    print("\n--- Loading Demonstration Dataset ---")
    
    if not os.path.exists(DEMO_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DEMO_DATASET_PATH}")
    
    df = pd.read_csv(DEMO_DATASET_PATH)
    print(f"Loaded dataset: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check required columns
    if 'Title' not in df.columns or 'Text' not in df.columns:
        raise ValueError("Dataset must contain 'Title' and 'Text' columns")
    
    # Combine Title and Text
    print("Combining Title and Text columns...")
    df['combined_text'] = df.apply(
        lambda row: f"Title: {row['Title']}\n\n{row['Text']}", 
        axis=1
    )
    
    # All samples are labeled as "Bait" (phishing) for reference
    df['true_label'] = 1  # 1 = Bait (phishing)
    df['true_label_name'] = "Bait"
    
    print(f"Total samples: {len(df)}")
    print(f"All samples are labeled as 'Bait' (phishing) for reference")
    
    return df

# =========================
# Inference
# =========================

def run_inference(model, tokenizer, texts: List[str]) -> tuple:
    """
    Run inference on texts and return predictions and probabilities.
    
    Returns:
        Tuple of (predictions, probabilities) where:
        - predictions: List of predicted class indices
        - probabilities: List of probability distributions (softmax scores)
    """
    print("\n--- Running Inference ---")
    
    # Create dataset and dataloader
    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predictions and probabilities
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            _, predictions = torch.max(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    print(f"✓ Inference complete on {len(texts)} samples")
    return all_predictions, all_probabilities

# =========================
# Results Processing
# =========================

def process_results(df: pd.DataFrame, predictions: List[int], probabilities: List[np.ndarray], label_encoder) -> Dict:
    """Process inference results and create output dictionary."""
    print("\n--- Processing Results ---")
    
    # Add predictions to dataframe
    df['predicted_label'] = predictions
    df['predicted_label_name'] = df['predicted_label'].map(LABEL_MAPPING)
    
    # Add confidence scores (probability of predicted class)
    df['confidence_score'] = [prob[pred] for pred, prob in zip(predictions, probabilities)]
    
    # Add probability for each class
    df['prob_great'] = [prob[0] for prob in probabilities]
    df['prob_bait'] = [prob[1] for prob in probabilities]
    
    # Calculate summary statistics
    prediction_counts = Counter(df['predicted_label_name'])
    total_samples = len(df)
    
    # Calculate accuracy (all true labels are "Bait" = 1)
    correct_predictions = (df['predicted_label'] == df['true_label']).sum()
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Create results dictionary
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': 'RoBERTa',
            'model_path': BEST_MODEL_PATH,
            'dataset_path': DEMO_DATASET_PATH,
            'total_samples': int(total_samples),
            'device': str(device),
            'max_len': MAX_LEN,
            'batch_size': BATCH_SIZE,
        },
        'summary': {
            'total_samples': int(total_samples),
            'prediction_distribution': {
                'Great': int(prediction_counts.get('Great', 0)),
                'Bait': int(prediction_counts.get('Bait', 0)),
            },
            'accuracy': float(accuracy),
            'correct_predictions': int(correct_predictions),
            'average_confidence': float(df['confidence_score'].mean()),
        },
        'predictions': []
    }
    
    # Add individual predictions
    for idx, row in df.iterrows():
        results['predictions'].append({
            'sample_id': int(idx),
            'title': str(row['Title']),
            'text': str(row['Text']),
            'combined_text': str(row['combined_text']),
            'true_label': int(row['true_label']),
            'true_label_name': str(row['true_label_name']),
            'predicted_label': int(row['predicted_label']),
            'predicted_label_name': str(row['predicted_label_name']),
            'confidence_score': float(row['confidence_score']),
            'probabilities': {
                'Great': float(row['prob_great']),
                'Bait': float(row['prob_bait']),
            }
        })
    
    return results

# =========================
# Main Execution
# =========================

def main():
    """Main execution function for inference."""
    print("=" * 60)
    print("Inference Script - Best DL Model (RoBERTa)")
    print(f"Running on device: {device}")
    print("=" * 60)
    
    # Load model components
    model, tokenizer, label_encoder = load_model_and_components()
    
    # Load demonstration dataset
    df = load_demonstration_dataset()
    
    # Run inference
    predictions, probabilities = run_inference(
        model, tokenizer, df['combined_text'].tolist()
    )
    
    # Process results
    results = process_results(df, predictions, probabilities, label_encoder)
    
    # Save results
    print("\n--- Saving Results ---")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {OUTPUT_PATH}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"Total samples: {results['summary']['total_samples']}")
    print(f"Prediction distribution:")
    print(f"  Great: {results['summary']['prediction_distribution']['Great']}")
    print(f"  Bait: {results['summary']['prediction_distribution']['Bait']}")
    print(f"Accuracy (all true labels are 'Bait'): {results['summary']['accuracy']:.4f} ({results['summary']['accuracy']*100:.2f}%)")
    print(f"Average confidence: {results['summary']['average_confidence']:.4f}")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()

