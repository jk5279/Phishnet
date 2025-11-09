"""
05_dl_pipeline.py

Deep Learning Pipeline

This script trains and evaluates a BERT-based deep learning model for phishing detection:
1. Loads preprocessed DL dataset
2. Performs data leakage checks (duplicate detection)
3. Creates PyTorch datasets and data loaders
4. Sets up BERT model for sequence classification
5. Trains with weighted sampling and class weights
6. Evaluates on test set after each epoch
7. Saves best and final models, tokenizer, and label encoder
8. Saves training curves, evaluation logs, and visualizations

Outputs:
  - models/dl_model_best/ (best model based on test accuracy)
  - models/dl_model/ (final model after all epochs)
  - models/dl_tokenizer/ (BERT tokenizer)
  - models/dl_label_encoder.pkl (label encoder)
  - models/dl_training_history.json (training metrics per epoch)
  - models/dl_training_curve.png (loss and accuracy plots)
  - models/dl_evaluation_log.json (comprehensive evaluation metrics)
  - models/dl_evaluation_report.txt (human-readable evaluation report)
  - models/dl_confusion_matrix.png (confusion matrix visualization)
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import seaborn as sns

from transformers import (
    BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
)

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Configuration
# =========================

# Input/Output paths
INPUT_FILE = os.path.join("cleaned_data", "dl_dataset_final.csv")
MODEL_DIR = "models"
DL_MODEL_PATH = os.path.join(MODEL_DIR, "dl_model")
DL_MODEL_BEST_PATH = os.path.join(MODEL_DIR, "dl_model_best")
DL_TOKENIZER_PATH = os.path.join(MODEL_DIR, "dl_tokenizer")
DL_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "dl_label_encoder.pkl")
DL_TRAINING_CURVE_PATH = os.path.join(MODEL_DIR, "dl_training_curve.png")
DL_EVALUATION_LOG_PATH = os.path.join(MODEL_DIR, "dl_evaluation_log.json")
DL_EVALUATION_REPORT_PATH = os.path.join(MODEL_DIR, "dl_evaluation_report.txt")
DL_CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "dl_confusion_matrix.png")
DL_TRAINING_HISTORY_PATH = os.path.join(MODEL_DIR, "dl_training_history.json")

# Model settings
MODEL_NAME = '/scratch/kimjong/MIE1517-Fall2025-Group11/local_bert_cased'
MAX_LEN = 512
BATCH_SIZE = 8  # Default, will be auto-calculated if AUTO_BATCH_SIZE=True
AUTO_BATCH_SIZE = True  # Set to True to auto-calculate based on GPU memory
BATCH_SIZE_MIN = 64  # Minimum batch size to try
BATCH_SIZE_MAX = 2048  # Maximum batch size to try
BATCH_SIZE_SAFETY_MARGIN = 0.8  # Use 80% of max to avoid OOM during training
N_EPOCHS = 3
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_WORKERS = 2


# =========================
# PyTorch Dataset Class
# =========================

class PhishingDataset(Dataset):
    """Custom PyTorch Dataset for phishing email classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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
            'labels': torch.tensor(label, dtype=torch.long)
        }


# =========================
# Data Loading and Preparation
# =========================

def check_data_leakage(X_train, X_test, y_train, y_test):
    """
    Comprehensive data leakage checks for DL pipeline.
    
    Checks for:
    1. Duplicate texts between train and test
    2. Label leakage (same text with different labels)
    """
    print("\n--- Data Leakage Checks ---")
    
    # Check 1: Exact duplicate texts
    train_texts = set(X_train.astype(str))
    test_texts = set(X_test.astype(str))
    exact_dups = train_texts.intersection(test_texts)
    
    if len(exact_dups) > 0:
        print(f"⚠️  WARNING: {len(exact_dups)} exact duplicate texts between train/test")
        print("   This is data leakage! Checking for label conflicts...")
        
        # Check 2: Check for label leakage (same text, different labels)
        train_dict = dict(zip(X_train.astype(str), y_train))
        test_dict = dict(zip(X_test.astype(str), y_test))
        
        label_conflicts = []
        for text in exact_dups:
            if train_dict[text] != test_dict[text]:
                label_conflicts.append((text[:50], train_dict[text], test_dict[text]))
        
        if label_conflicts:
            print(f"⚠️  WARNING: {len(label_conflicts)} duplicate texts have different labels")
            print("   This indicates data quality issues!")
            for text, train_label, test_label in label_conflicts[:3]:
                print(f"   Example: '{text}...' -> Train: {train_label}, Test: {test_label}")
        else:
            print("✓ No label conflicts in duplicate texts")
        
        # Remove duplicates from test set
        print("   Removing duplicates from test set...")
        mask = ~X_test.astype(str).isin(exact_dups)
        X_test_clean = X_test[mask].copy()
        y_test_clean = y_test[mask].copy()
        print(f"   Removed {len(X_test) - len(X_test_clean)} duplicate samples from test set")
        print("--- Data Leakage Check Complete ---\n")
        return X_test_clean, y_test_clean
    else:
        print("✓ No exact duplicate texts between train/test")
        print("✓ No label conflicts detected (no duplicates to check)")
        print("--- Data Leakage Check Complete ---\n")
        return X_test, y_test


def load_and_prepare_data(filepath: str):
    """
    Load data and prepare for training.
    
    NOTE: Using ONLY raw text - no engineered features.
    The model learns all features from BERT embeddings.
    """
    print("\n--- Section 1: Data Loading & Preprocessing ---")
    
    # Load pre-cleaned data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"ERROR: Could not find file {filepath}")
        print("Please make sure you have run the DL preprocessing pipeline first.")
        raise

    print(f"Loaded pre-cleaned dataset: {df.shape}")
    print(f"Label distribution: {Counter(df['label'])}")

    # Encode labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    print(f"Label mapping: {list(zip(le.classes_, le.transform(le.classes_)))}")
    num_classes = len(le.classes_)

    # Prepare data for splitting
    # NOTE: Using ONLY raw text - no engineered features
    # The model learns all features from BERT embeddings
    X = df['text']
    y = df['label_encoded'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Check for data leakage
    X_test, y_test = check_data_leakage(X_train, X_test, y_train, y_test)
    print(f"Final test set size after leakage removal: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test, le, num_classes


def create_datasets_and_loaders(
    X_train, X_test, y_train, y_test, tokenizer, max_len, batch_size
):
    """Create PyTorch datasets and data loaders."""
    print("\n--- Section 2: Creating Datasets & DataLoaders ---")
    
    # Create datasets
    print("Creating PyTorch Datasets...")
    train_dataset = PhishingDataset(
        texts=X_train.tolist(),
        labels=y_train,
        tokenizer=tokenizer,
        max_len=max_len
    )
    test_dataset = PhishingDataset(
        texts=X_test.tolist(),
        labels=y_test,
        tokenizer=tokenizer,
        max_len=max_len
    )

    # Setup WeightedRandomSampler for class imbalance
    print("Setting up WeightedRandomSampler for training...")
    class_counts = np.bincount(y_train)
    class_weights_per_sample = 1. / class_counts
    sample_weights = np.array([class_weights_per_sample[t] for t in y_train])
    sample_weights_tensor = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # Calculate class weights for loss function
    class_weights_loss = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(
        class_weights_loss, dtype=torch.float
    ).to(device)
    print(f"Weights for loss function: {class_weights_tensor.cpu().numpy()}")

    return train_loader, test_loader, class_weights_tensor


# =========================
# Batch Size Calculation
# =========================

def calculate_optimal_batch_size(
    model_name: str,
    num_classes: int,
    max_len: int,
    tokenizer,
    min_batch: int = BATCH_SIZE_MIN,
    max_batch: int = BATCH_SIZE_MAX,
    safety_margin: float = BATCH_SIZE_SAFETY_MARGIN
):
    """
    Calculate optimal batch size based on available GPU memory.
    
    This function tries progressively larger batch sizes until it finds
    the maximum that fits in GPU memory, then applies a safety margin.
    
    Args:
        model_name: Path to BERT model
        num_classes: Number of classes
        max_len: Maximum sequence length
        tokenizer: BERT tokenizer
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
        safety_margin: Fraction of max batch size to use (0.0-1.0)
        
    Returns:
        int: Optimal batch size
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU with default batch size.")
        return BATCH_SIZE
    
    print("\n--- Calculating Optimal Batch Size ---")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create a dummy model for testing
    test_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )
    test_model.to(device)
    test_model.train()
    
    # Create dummy optimizer
    test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=LEARNING_RATE)
    test_scaler = GradScaler()
    
    # Create dummy data
    dummy_text = "This is a dummy text for batch size calculation. " * 20
    dummy_encoding = tokenizer.encode_plus(
        dummy_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    optimal_batch = min_batch
    batch_size = min_batch
    
    print(f"Testing batch sizes from {min_batch} to {max_batch}...")
    
    while batch_size <= max_batch:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Create batch
            input_ids = dummy_encoding['input_ids'].repeat(batch_size, 1).to(device)
            attention_mask = dummy_encoding['attention_mask'].repeat(batch_size, 1).to(device)
            labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # Forward pass
            with autocast():
                outputs = test_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass (simulates training)
            test_optimizer.zero_grad()
            test_scaler.scale(loss).backward()
            test_scaler.step(test_optimizer)
            test_scaler.update()
            
            # If successful, this batch size works
            optimal_batch = batch_size
            print(f"  ✓ Batch size {batch_size}: OK")
            
            # Try next batch size
            batch_size *= 2  # Double the batch size for next test
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ✗ Batch size {batch_size}: Out of memory")
                torch.cuda.empty_cache()
                break
            else:
                raise e
    
    # Clean up
    del test_model, test_optimizer, test_scaler, input_ids, attention_mask, labels
    torch.cuda.empty_cache()
    
    # Apply safety margin
    final_batch = max(min_batch, int(optimal_batch * safety_margin))
    
    print(f"\nOptimal batch size: {optimal_batch}")
    print(f"Using batch size with {safety_margin*100:.0f}% safety margin: {final_batch}")
    print("--- Batch Size Calculation Complete ---\n")
    
    return final_batch


# =========================
# Model Setup
# =========================

def setup_model(model_name: str, num_classes: int):
    """Setup BERT model for sequence classification."""
    print("\n--- Section 3: Model Setup ---")
    print(f"Loading model ({model_name}) for {num_classes}-class classification...")

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    print(f"Model loaded on device: {device}")

    return model


def setup_training_components(model, train_loader, n_epochs, learning_rate, class_weights_tensor):
    """Setup optimizer, scheduler, loss function, and scaler."""
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

    # Scheduler
    total_steps = len(train_loader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Loss Function (with class weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Mixed Precision Scaler
    scaler = GradScaler()

    return optimizer, scheduler, loss_fn, scaler


# =========================
# Training and Evaluation Functions
# =========================

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", unit="batch")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()

        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)


def eval_model(model, data_loader, device):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


# =========================
# Model Saving
# =========================

def create_test_loader_from_csv(csv_path: str, tokenizer, max_len: int = MAX_LEN, batch_size: int = BATCH_SIZE):
    """
    Create a test DataLoader from a CSV file.
    
    Args:
        csv_path: Full path to the CSV file
        tokenizer: BERT tokenizer
        max_len: Maximum sequence length for tokenization
        batch_size: Batch size for the DataLoader
        
    Returns:
        DataLoader: Test data loader
        
    Note:
        The CSV file must have a 'text' column. If 'label' column is not provided,
        all samples will be labeled as 1.
    """
    print(f"\nCreating test DataLoader from {csv_path}")
    
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find file {csv_path}")
        raise
        
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column")
    
    # Create labels if not provided
    if 'label' not in df.columns:
        print("No 'label' column found. Creating default labels of 1.")
        labels = np.ones(len(df))
    else:
        labels = df['label'].values
        
    # Create dataset
    test_dataset = PhishingDataset(
        texts=df['text'].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    print(f"Created test DataLoader with {len(test_dataset)} samples")
    return test_loader


def save_model_components(model, tokenizer, label_encoder, save_best=False, best_model_path=None):
    """
    Save model, tokenizer, and label encoder.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        label_encoder: Label encoder
        save_best: If True, save as best model; otherwise save as final model
        best_model_path: Path for best model (if save_best=True)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_path = best_model_path if save_best and best_model_path else DL_MODEL_PATH
    
    # Save model
    model.save_pretrained(model_path)
    model_type = "best" if save_best else "final"
    print(f"{model_type.capitalize()} model saved to: {model_path}")
    
    # Save tokenizer (only once, same for all models)
    if not save_best:
        tokenizer.save_pretrained(DL_TOKENIZER_PATH)
        print(f"Tokenizer saved to: {DL_TOKENIZER_PATH}")
        
        # Save label encoder (only once)
        with open(DL_LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to: {DL_LABEL_ENCODER_PATH}")


def save_training_history(history: dict, filepath: str):
    """Save training history (losses, accuracies) to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {filepath}")


def plot_training_curve(history: dict, output_path: str):
    """Plot and save training curve (loss and accuracy over epochs)."""
    epochs = history['epochs']
    train_losses = history['train_loss']
    test_accuracies = history['test_accuracy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    ax1.plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Over Epochs', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    ax2.plot(epochs, test_accuracies, 'o-', label='Test Accuracy', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy Over Epochs', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training curve saved to: {output_path}")


def save_evaluation_logs(
    y_test: list,
    y_pred: list,
    label_encoder: LabelEncoder,
    best_accuracy: float,
    training_history: dict,
    confusion_matrix_array: np.ndarray,
    classification_report_text: str,
    batch_size: int,
    log_path: str,
    report_path: str,
    cm_path: str
):
    """
    Save comprehensive evaluation logs including metrics, reports, and visualizations.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Save JSON log with all metrics
    evaluation_log = {
        'timestamp': datetime.now().isoformat(),
        'best_accuracy': float(best_accuracy),
        'metrics': {
            'test_accuracy': float(best_accuracy),
            'test_precision_weighted': float(precision),
            'test_recall_weighted': float(recall),
            'test_f1_weighted': float(f1)
        },
        'training_history': training_history,
        'confusion_matrix': confusion_matrix_array.tolist(),
        'classification_report': classification_report_text,
        'model_config': {
            'model_name': MODEL_NAME,
            'max_len': MAX_LEN,
            'batch_size': batch_size,
            'n_epochs': N_EPOCHS,
            'learning_rate': LEARNING_RATE
        }
    }
    
    with open(log_path, 'w') as f:
        json.dump(evaluation_log, f, indent=2)
    print(f"Evaluation log saved to: {log_path}")
    
    # Save text report
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DL Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: BERT-based Classifier\n\n")
        f.write("Model Configuration:\n")
        f.write(f"  Base Model: {MODEL_NAME}\n")
        f.write(f"  Max Length: {MAX_LEN}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Epochs: {N_EPOCHS}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n\n")
        f.write(f"Best Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n\n")
        f.write(f"Test Set Metrics:\n")
        f.write(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1 Score (weighted): {f1:.4f}\n\n")
        f.write(f"Training History:\n")
        for epoch, loss, acc in zip(training_history['epochs'], 
                                   training_history['train_loss'],
                                   training_history['test_accuracy']):
            f.write(f"  Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report_text)
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(confusion_matrix_array))
        f.write("\n")
    print(f"Evaluation report saved to: {report_path}")
    
    # Save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_array,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix - BERT Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {cm_path}")


# =========================
# Main Execution
# =========================

def main():
    """Main execution function for DL pipeline."""
    print("=" * 60)
    print("Deep Learning Pipeline (BERT)")
    print(f"Running on device: {device}")
    print("=" * 60)

    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder, num_classes = load_and_prepare_data(INPUT_FILE)

    # Initialize tokenizer
    print(f"\nInitializing tokenizer ({MODEL_NAME})...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Calculate optimal batch size if enabled
    if AUTO_BATCH_SIZE and torch.cuda.is_available():
        optimal_batch_size = calculate_optimal_batch_size(
            MODEL_NAME, num_classes, MAX_LEN, tokenizer,
            min_batch=BATCH_SIZE_MIN,
            max_batch=BATCH_SIZE_MAX,
            safety_margin=BATCH_SIZE_SAFETY_MARGIN
        )
        batch_size = optimal_batch_size
    else:
        batch_size = BATCH_SIZE
        if not AUTO_BATCH_SIZE:
            print(f"\nUsing configured batch size: {batch_size}")
        else:
            print(f"\nUsing default batch size: {batch_size} (CPU mode or AUTO_BATCH_SIZE disabled)")

    # Create datasets and data loaders
    train_loader, test_loader, class_weights_tensor = create_datasets_and_loaders(
        X_train, X_test, y_train, y_test, tokenizer, MAX_LEN, batch_size
    )

    # Setup model
    model = setup_model(MODEL_NAME, num_classes)

    # Setup training components
    optimizer, scheduler, loss_fn, scaler = setup_training_components(
        model, train_loader, N_EPOCHS, LEARNING_RATE, class_weights_tensor
    )

    # Training loop
    print("\n--- Section 5: Training Loop ---")
    best_accuracy = 0.0
    best_model_state = None
    
    # Track training history
    training_history = {
        'epochs': [],
        'train_loss': [],
        'test_accuracy': []
    }

    for epoch in range(N_EPOCHS):
        print(f'\n--- Epoch {epoch + 1} / {N_EPOCHS} ---')

        avg_train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler, scaler
        )
        print(f'Average Training Loss: {avg_train_loss:.4f}')

        # Evaluate on test set
        print("Running evaluation on test set...")
        labels, preds = eval_model(model, test_loader, device)

        accuracy = accuracy_score(labels, preds)
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Classification report
        target_names_str = label_encoder.classes_.astype(str)
        report_text = classification_report(
            labels,
            preds,
            target_names=target_names_str
        )
        print("Classification Report:")
        print(report_text)

        # Track training history
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(float(avg_train_loss))
        training_history['test_accuracy'].append(float(accuracy))

        # Track and save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            print(f"New best accuracy: {best_accuracy:.4f}")
            # Save best model immediately
            model_best = BertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=num_classes
            )
            model_best.load_state_dict(best_model_state)
            model_best.to(device)
            save_model_components(
                model_best, tokenizer, label_encoder,
                save_best=True, best_model_path=DL_MODEL_BEST_PATH
            )

    # Save final model components
    print("\n--- Section 6: Saving Final Model Components ---")
    save_model_components(model, tokenizer, label_encoder)
    
    # Save training history
    print("\n--- Section 7: Saving Training History & Logs ---")
    save_training_history(training_history, DL_TRAINING_HISTORY_PATH)
    
    # Plot and save training curve
    plot_training_curve(training_history, DL_TRAINING_CURVE_PATH)
    
    # Get final evaluation metrics
    labels, preds = eval_model(model, test_loader, device)
    cm = confusion_matrix(labels, preds)
    report_text = classification_report(
        labels, preds, target_names=label_encoder.classes_.astype(str)
    )
    
    # Save comprehensive evaluation logs
    save_evaluation_logs(
        y_test=labels,
        y_pred=preds,
        label_encoder=label_encoder,
        best_accuracy=best_accuracy,
        training_history=training_history,
        confusion_matrix_array=cm,
        classification_report_text=report_text,
        batch_size=batch_size,
        log_path=DL_EVALUATION_LOG_PATH,
        report_path=DL_EVALUATION_REPORT_PATH,
        cm_path=DL_CONFUSION_MATRIX_PATH
    )

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print("\nSaved Model Files:")
    print(f"  - Best model: {DL_MODEL_BEST_PATH}")
    print(f"  - Final model: {DL_MODEL_PATH}")
    print(f"  - Tokenizer: {DL_TOKENIZER_PATH}")
    print(f"  - Label encoder: {DL_LABEL_ENCODER_PATH}")
    print("\nSaved Logs & Visualizations:")
    print(f"  - Training history: {DL_TRAINING_HISTORY_PATH}")
    print(f"  - Training curve: {DL_TRAINING_CURVE_PATH}")
    print(f"  - Evaluation log: {DL_EVALUATION_LOG_PATH}")
    print(f"  - Evaluation report: {DL_EVALUATION_REPORT_PATH}")
    print(f"  - Confusion matrix: {DL_CONFUSION_MATRIX_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

