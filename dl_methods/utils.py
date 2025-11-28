"""
Shared utilities for Deep Learning pipelines.
Includes data loading, metrics calculation, visualization, and local model path management.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

# =========================
# Configuration
# =========================

PRETRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "pretrained")

# =========================
# Data Loading & Processing
# =========================

class PhishingDataset(Dataset):
    """PyTorch Dataset for phishing email classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dl_dataset(filepath: str) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Basic validation
    required_cols = ['text', 'label']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset must contain columns: {required_cols}")
        
    # Drop NaNs
    initial_len = len(df)
    df = df.dropna(subset=['text', 'label'])
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows with missing values")
        
    # Ensure label is int
    df['label'] = df['label'].astype(int)
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def check_data_leakage(train_df, val_df, test_df):
    """Check for data leakage between splits."""
    print("\nChecking for data leakage...")
    
    train_texts = set(train_df['text'].unique())
    val_texts = set(val_df['text'].unique())
    test_texts = set(test_df['text'].unique())
    
    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)
    
    if train_val_overlap:
        print(f"WARNING: Found {len(train_val_overlap)} overlapping texts between train and validation sets")
    
    if train_test_overlap:
        print(f"WARNING: Found {len(train_test_overlap)} overlapping texts between train and test sets")
        
    if val_test_overlap:
        print(f"WARNING: Found {len(val_test_overlap)} overlapping texts between validation and test sets")
        
    if not (train_val_overlap or train_test_overlap or val_test_overlap):
        print("✅ No data leakage detected between splits")

def calculate_class_weights(labels):
    """Calculate class weights for handling imbalance."""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(class_weights, dtype=torch.float)

def create_data_loaders(
    train_df, val_df, test_df, 
    tokenizer, max_len, batch_size
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = PhishingDataset(
        texts=train_df.text.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = PhishingDataset(
        texts=val_df.text.to_numpy(),
        labels=val_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = PhishingDataset(
        texts=test_df.text.to_numpy(),
        labels=test_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # Create weighted sampler for training to handle class imbalance
    labels = train_df.label.to_numpy()
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

# =========================
# Model Management
# =========================

def get_pretrained_path(model_name: str) -> str:
    """Get the local path for a pretrained model."""
    path = os.path.join(PRETRAINED_MODELS_DIR, model_name)
    if not os.path.exists(path):
        # Check if the directory exists but might be empty or missing files
        # Fallback to model name if local path doesn't exist (will try to download if internet available)
        # But we want to enforce local loading, so we'll warn.
        print(f"WARNING: Local pretrained model path not found: {path}")
        print(f"Please run 'python dl_methods/download_models.py' first.")
        # Return path anyway, the pipeline will fail with a clear error from transformers if empty
    return path

def save_model(model, tokenizer, label_encoder, output_dir, final=False):
    """Save model, tokenizer, and label encoder."""
    save_path = os.path.join(output_dir, "dl_model" if final else "dl_model_best")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Saving {'final' if final else 'best'} model to {save_path}...")
    
    # Save model and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save label encoder
    import pickle
    le_path = os.path.join(output_dir, "dl_label_encoder.pkl")
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {le_path}")

# =========================
# Evaluation & Visualization
# =========================

def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataloader."""
    model = model.eval()
    
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    return predictions, prediction_probs, real_values

def compute_metrics(real_values, predictions):
    """Compute classification metrics."""
    return {
        'accuracy': accuracy_score(real_values, predictions),
        'precision': precision_score(real_values, predictions, zero_division=0),
        'recall': recall_score(real_values, predictions, zero_division=0),
        'f1': f1_score(real_values, predictions, zero_division=0)
    }

def plot_training_history(history, save_path):
    """Plot training and validation accuracy/loss."""
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(real_values, predictions, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(real_values, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

