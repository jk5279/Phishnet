"""
05_dl_pipeline.py

Unified Deep Learning Pipeline with Hyperparameter Tuning

This script trains and evaluates multiple transformer models (BERT, RoBERTa, DistilBERT) for phishing detection:
1. Loads preprocessed DL datasets from cleaned_data/DL/train/, validation/, test/
2. Performs hyperparameter tuning using scikit-optimize (Design of Experiments)
3. Trains multiple transformer models with best hyperparameters
4. Saves all models and logs to dl_methods/ directory structure
5. Reports all models equally, then identifies and reports the best model

Output: dl_methods/{model_name}/model/ and dl_methods/{model_name}/logs/
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    get_linear_schedule_with_warmup
)

# scikit-optimize for Design of Experiments
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Configuration
# =========================

# Input/Output paths
TRAIN_INPUT = os.path.join("cleaned_data", "DL", "train", "train_split.csv")
VAL_INPUT = os.path.join("cleaned_data", "DL", "validation", "validation_split.csv")
TEST_INPUT = os.path.join("cleaned_data", "DL", "test", "test_split.csv")
MODEL_DIR = "dl_methods"

# Pretrained models directory
PRETRAINED_MODELS_DIR = os.path.join("models", "pretrained")

# Default hyperparameter search space (small, discrete, DL-friendly defaults)
HYPERPARAMETER_SPACE = {
    # Common transformer learning rates
    'learning_rate': Categorical([2e-5, 3e-5, 5e-5], name='learning_rate'),
    # Typical batch sizes
    'batch_size': Categorical([8, 16, 32], name='batch_size'),
    # Usual epoch counts
    'epochs': Categorical([3, 4, 5], name='epochs'),
    # Standard sequence lengths
    'max_len': Categorical([128, 256, 512], name='max_len'),
    # Typical warmup step options
    'warmup_steps': Categorical([0, 100, 500], name='warmup_steps'),
}

# Number of hyperparameter search iterations
N_CALLS = 5  # Number of hyperparameter combinations to try (reduced for speed)

# Model configurations
MODEL_CONFIGS = {
    "BERT": {
        "model_class": BertForSequenceClassification,
        "tokenizer_class": BertTokenizer,
        "pretrained_name": "bert-base-cased",
    },
    "RoBERTa": {
        "model_class": RobertaForSequenceClassification,
        "tokenizer_class": RobertaTokenizer,
        "pretrained_name": "roberta-base",
    },
    "DistilBERT": {
        "model_class": DistilBertForSequenceClassification,
        "tokenizer_class": DistilBertTokenizer,
        "pretrained_name": "distilbert-base-cased",
    },
}

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

def get_pretrained_path(model_name: str) -> str:
    """Get the local path for a pretrained model."""
    path = os.path.join(PRETRAINED_MODELS_DIR, model_name)
    if not os.path.exists(path):
        print(f"WARNING: Local pretrained model path not found: {path}")
        print(f"Please run 'python dl_methods/download_models.py' first.")
    return path

def load_data(
    train_path: str, val_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    """Load and prepare data for training from split files."""
    print("\n--- 1. Loading and Preparing Data ---")

    # Load train data
    train_df = pd.read_csv(train_path)
    print(f"Train dataset size: {train_df.shape}")
    print(f"Train label distribution: {Counter(train_df['label'])}")

    # Load validation data
    val_df = pd.read_csv(val_path)
    print(f"Validation dataset size: {val_df.shape}")
    print(f"Validation label distribution: {Counter(val_df['label'])}")

    # Load test data
    test_df = pd.read_csv(test_path)
    print(f"Test dataset size: {test_df.shape}")
    print(f"Test label distribution: {Counter(test_df['label'])}")

    # Check for data leakage
    train_texts = set(train_df['text'].astype(str))
    val_texts = set(val_df['text'].astype(str))
    test_texts = set(test_df['text'].astype(str))
    
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

    # Encode labels
    le = LabelEncoder()
    all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique()
    le.fit(all_labels)
    
    train_df['label_encoded'] = le.transform(train_df['label'])
    val_df['label_encoded'] = le.transform(val_df['label'])
    test_df['label_encoded'] = le.transform(test_df['label'])
    
    print(f"Label mapping: {list(zip(le.classes_, le.transform(le.classes_)))}")

    return train_df, val_df, test_df, le

def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    max_len: int,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = PhishingDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label_encoded'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = PhishingDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label_encoded'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = PhishingDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label_encoded'].tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    # Setup WeightedRandomSampler for class imbalance
    labels = train_df['label_encoded'].values
    class_counts = np.bincount(labels)
    class_weights_per_sample = 1. / class_counts
    sample_weights = np.array([class_weights_per_sample[t] for t in labels])
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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
        'balanced', classes=np.unique(labels), y=labels
    )
    class_weights_tensor = torch.tensor(
        class_weights_loss, dtype=torch.float
    ).to(device)

    return train_loader, val_loader, test_loader, class_weights_tensor

# =========================
# Path Helpers
# =========================

def get_method_dir_name(model_name: str) -> str:
    """Convert model name to directory name."""
    name_mapping = {
        "BERT": "bert",
        "RoBERTa": "roberta",
        "DistilBERT": "distilbert",
    }
    return name_mapping.get(model_name, model_name.lower().replace(" ", "_"))

def get_method_paths(model_name: str) -> Dict[str, str]:
    """Get all paths for a specific method."""
    method_dir = get_method_dir_name(model_name)
    method_base_dir = os.path.join(MODEL_DIR, method_dir)
    model_dir = os.path.join(method_base_dir, "model")
    logs_dir = os.path.join(method_base_dir, "logs")

    return {
        "method_dir": method_base_dir,
        "model_dir": model_dir,
        "logs_dir": logs_dir,
        "best_model_path": os.path.join(model_dir, "best"),
        "final_model_path": os.path.join(model_dir, "final"),
        "tokenizer_path": os.path.join(model_dir, "tokenizer"),
        "label_encoder_path": os.path.join(model_dir, "label_encoder.pkl"),
        "evaluation_log_path": os.path.join(logs_dir, "evaluation_log.json"),
        "evaluation_report_path": os.path.join(logs_dir, "evaluation_report.txt"),
        "training_history_path": os.path.join(logs_dir, "training_history.json"),
        "training_curve_path": os.path.join(logs_dir, "training_curve.png"),
        "confusion_matrix_path": os.path.join(logs_dir, "confusion_matrix.png"),
        "hyperparameters_path": os.path.join(logs_dir, "hyperparameters.json"),
    }

# =========================
# Model Training Functions
# =========================

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", leave=False)

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
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)
        
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_samples
    return avg_loss, accuracy.item()

def eval_epoch(model, data_loader, loss_fn, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_labels, all_preds

def train_model(
    model,
    train_loader,
    val_loader,
    class_weights_tensor,
    epochs: int,
    learning_rate: float,
    warmup_steps: int,
    device
) -> Tuple[Dict, torch.nn.Module]:
    """Train a model and return training history and best model."""
    
    # Setup training components
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scaler = GradScaler()

    # Training history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = eval_epoch(
            model, val_loader, loss_fn, device
        )
        
        # Track history
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, model

# =========================
# Hyperparameter Tuning with DOE
# =========================

def hyperparameter_tuning(
    model_name: str,
    model_config: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_encoder: LabelEncoder,
    n_calls: int = N_CALLS
) -> Tuple[Dict, float]:
    """
    Perform hyperparameter tuning using scikit-optimize (Design of Experiments).
    
    Returns:
        Tuple of (best_hyperparameters, best_validation_accuracy)
    """
    print(f"\n--- Hyperparameter Tuning for {model_name} ---")
    
    # Initialize tokenizer
    tokenizer_class = model_config["tokenizer_class"]
    pretrained_name = model_config["pretrained_name"]
    model_path = get_pretrained_path(pretrained_name)
    tokenizer = tokenizer_class.from_pretrained(model_path, local_files_only=True)
    
    num_classes = len(label_encoder.classes_)
    
    # Define search space
    dimensions = [
        HYPERPARAMETER_SPACE['learning_rate'],
        HYPERPARAMETER_SPACE['batch_size'],
        HYPERPARAMETER_SPACE['epochs'],
        HYPERPARAMETER_SPACE['max_len'],
        HYPERPARAMETER_SPACE['warmup_steps'],
    ]
    
    # Objective function for optimization
    @use_named_args(dimensions=dimensions)
    def objective(learning_rate, batch_size, epochs, max_len, warmup_steps):
        """Objective function to minimize (negative validation accuracy)."""
        try:
            # Ensure correct Python types (skopt may pass numpy/float types)
            learning_rate = float(learning_rate)
            batch_size = int(batch_size)
            epochs = int(epochs)
            max_len = int(max_len)
            warmup_steps = int(warmup_steps)

            # Create data loaders
            train_loader, val_loader, _, class_weights_tensor = create_data_loaders(
                train_df, val_df, test_df, tokenizer, max_len, batch_size
            )
            
            # Initialize model
            model_class = model_config["model_class"]
            model = model_class.from_pretrained(
                model_path,
                num_labels=num_classes,
                local_files_only=True
            )
            model.to(device)
            
            # Train model
            history, _ = train_model(
                model, train_loader, val_loader, class_weights_tensor,
                epochs, learning_rate, warmup_steps, device
            )
            
            # Get best validation accuracy
            best_val_acc = max(history['val_acc'])
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
            # Return negative accuracy (since we're minimizing)
            return -best_val_acc
            
        except Exception as e:
            print(f"Error during hyperparameter search: {e}")
            # Return a poor score if training fails
            return 1.0
    
    # Perform Bayesian optimization
    print(f"Starting hyperparameter search with {n_calls} iterations...")
    # Ensure skopt doesn't require >=10 calls by reducing initial random points
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        n_initial_points=min(5, n_calls),
        random_state=RANDOM_STATE,
        verbose=True
    )
    
    # Extract best hyperparameters
    best_params = {
        'learning_rate': result.x[0],
        'batch_size': int(result.x[1]),
        'epochs': int(result.x[2]),
        'max_len': int(result.x[3]),
        'warmup_steps': int(result.x[4]),
    }
    
    best_val_acc = -result.fun  # Convert back from negative
    
    print(f"\nBest hyperparameters for {model_name}:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_params, best_val_acc

# =========================
# Model Saving
# =========================

def save_model_checkpoint(
    model,
    tokenizer,
    label_encoder,
    model_name: str,
    is_best: bool = False
):
    """Save model checkpoint (best or final)."""
    paths = get_method_paths(model_name)
    
    if is_best:
        save_path = paths["best_model_path"]
    else:
        save_path = paths["final_model_path"]
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")
    
    # Save tokenizer (only once, same for all checkpoints)
    if not is_best:
        tokenizer_path = paths["tokenizer_path"]
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        print(f"Tokenizer saved to: {tokenizer_path}")
        
        # Save label encoder (only once)
        le_path = paths["label_encoder_path"]
        with open(le_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to: {le_path}")

def save_training_history(history: Dict, model_name: str):
    """Save training history to JSON."""
    paths = get_method_paths(model_name)
    os.makedirs(paths["logs_dir"], exist_ok=True)
    
    with open(paths["training_history_path"], 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {paths['training_history_path']}")

def plot_training_curve(history: Dict, model_name: str):
    """Plot and save training curve."""
    paths = get_method_paths(model_name)
    os.makedirs(paths["logs_dir"], exist_ok=True)
    
    epochs = history['epochs']
    train_losses = history['train_loss']
    train_accs = history['train_acc']
    val_losses = history['val_loss']
    val_accs = history['val_acc']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training loss
    ax1.plot(epochs, train_losses, 'o-', label='Train', linewidth=2)
    ax1.plot(epochs, val_losses, 'o-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Over Epochs', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Training accuracy
    ax2.plot(epochs, train_accs, 'o-', label='Train', linewidth=2, color='green')
    ax2.plot(epochs, val_accs, 'o-', label='Validation', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy Over Epochs', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Validation loss only
    ax3.plot(epochs, val_losses, 'o-', label='Validation Loss', linewidth=2, color='red')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('Validation Loss', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Validation accuracy only
    ax4.plot(epochs, val_accs, 'o-', label='Validation Accuracy', linewidth=2, color='blue')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Validation Accuracy', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(paths["training_curve_path"], dpi=150)
    plt.close()
    print(f"Training curve saved to: {paths['training_curve_path']}")

def save_evaluation_logs(
    y_test: List,
    y_pred: List,
    label_encoder: LabelEncoder,
    model_name: str,
    best_hyperparameters: Dict,
    best_val_acc: float,
    test_accuracy: float,
    training_history: Dict,
    confusion_matrix_array: np.ndarray,
    classification_report_text: str,
):
    """Save comprehensive evaluation logs."""
    paths = get_method_paths(model_name)
    os.makedirs(paths["logs_dir"], exist_ok=True)
    
    # Calculate additional metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Save JSON log
    evaluation_log = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'best_hyperparameters': best_hyperparameters,
        'best_validation_accuracy': float(best_val_acc),
        'metrics': {
            'test_accuracy': float(test_accuracy),
            'test_precision_weighted': float(precision),
            'test_recall_weighted': float(recall),
            'test_f1_weighted': float(f1),
        },
        'training_history': training_history,
        'confusion_matrix': confusion_matrix_array.tolist(),
        'classification_report': classification_report_text,
    }
    
    with open(paths["evaluation_log_path"], 'w') as f:
        json.dump(evaluation_log, f, indent=2)
    print(f"Evaluation log saved to: {paths['evaluation_log_path']}")
    
    # Save hyperparameters
    with open(paths["hyperparameters_path"], 'w') as f:
        json.dump(best_hyperparameters, f, indent=2)
    print(f"Hyperparameters saved to: {paths['hyperparameters_path']}")
    
    # Save text report
    with open(paths["evaluation_report_path"], 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DL Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("Best Hyperparameters:\n")
        for param, value in best_hyperparameters.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n")
        f.write(f"\nTest Set Metrics:\n")
        f.write(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1 Score (weighted): {f1:.4f}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report_text)
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(confusion_matrix_array))
        f.write("\n")
    print(f"Evaluation report saved to: {paths['evaluation_report_path']}")
    
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
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(paths["confusion_matrix_path"], dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {paths['confusion_matrix_path']}")

# =========================
# Main Execution
# =========================

def main():
    """Main execution function for DL pipeline."""
    print("=" * 60)
    print("Unified Deep Learning Pipeline")
    print(f"Running on device: {device}")
    print("=" * 60)

    # Load and prepare data
    train_df, val_df, test_df, label_encoder = load_data(
        TRAIN_INPUT, VAL_INPUT, TEST_INPUT
    )
    
    num_classes = len(label_encoder.classes_)
    
    # Train and evaluate all models
    print("\n--- 2. Training and Evaluating All Models ---")
    all_results = {}
    
    for model_name, model_config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {model_name}")
        print(f"{'=' * 60}")
        
        start_time = time()
        
        # Hyperparameter tuning
        print(f"\n--- Hyperparameter Tuning for {model_name} ---")
        best_hyperparams, best_val_acc = hyperparameter_tuning(
            model_name, model_config, train_df, val_df, test_df, label_encoder
        )
        
        # Train final model with best hyperparameters
        print(f"\n--- Training Final Model for {model_name} with Best Hyperparameters ---")
        
        # Initialize tokenizer
        tokenizer_class = model_config["tokenizer_class"]
        pretrained_name = model_config["pretrained_name"]
        model_path = get_pretrained_path(pretrained_name)
        tokenizer = tokenizer_class.from_pretrained(model_path, local_files_only=True)
        
        # Create data loaders with best hyperparameters
        train_loader, val_loader, test_loader, class_weights_tensor = create_data_loaders(
            train_df, val_df, test_df, tokenizer,
            best_hyperparams['max_len'], best_hyperparams['batch_size']
        )
        
        # Initialize model
        model_class = model_config["model_class"]
        model = model_class.from_pretrained(
            model_path,
            num_labels=num_classes,
            local_files_only=True
        )
        model.to(device)
        
        # Train model
        training_history, trained_model = train_model(
            model, train_loader, val_loader, class_weights_tensor,
            best_hyperparams['epochs'], best_hyperparams['learning_rate'],
            best_hyperparams['warmup_steps'], device
        )
        
        # Evaluate on test set
        print(f"\n--- Evaluating {model_name} on Test Set ---")
        test_loss, test_accuracy, y_test, y_pred = eval_epoch(
            trained_model, test_loader, nn.CrossEntropyLoss(weight=class_weights_tensor), device
        )
        
        # Classification report
        target_names = label_encoder.classes_.astype(str)
        report_text = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("Classification Report:")
        print(report_text)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save model and logs
        print(f"\n--- Saving {model_name} ---")
        save_model_checkpoint(trained_model, tokenizer, label_encoder, model_name, is_best=True)
        save_model_checkpoint(trained_model, tokenizer, label_encoder, model_name, is_best=False)
        save_training_history(training_history, model_name)
        plot_training_curve(training_history, model_name)
        save_evaluation_logs(
            y_test=y_test,
            y_pred=y_pred,
            label_encoder=label_encoder,
            model_name=model_name,
            best_hyperparameters=best_hyperparams,
            best_val_acc=best_val_acc,
            test_accuracy=test_accuracy,
            training_history=training_history,
            confusion_matrix_array=cm,
            classification_report_text=report_text,
        )
        
        # Store results
        all_results[model_name] = {
            'best_hyperparameters': best_hyperparams,
            'best_val_acc': best_val_acc,
            'test_accuracy': test_accuracy,
            'training_time': time() - start_time,
        }
        
        # Clean up GPU memory
        del model, trained_model
        torch.cuda.empty_cache()
    
    # Report all models equally
    print("\n" + "=" * 60)
    print("All Models Summary")
    print("=" * 60)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Test Accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"  Best Val Accuracy: {result['best_val_acc']*100:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
        print(f"  Best Hyperparameters: {result['best_hyperparameters']}")
        paths = get_method_paths(name)
        print(f"  Saved to: {paths['method_dir']}")
    
    # Identify and report best model
    best_model_name = max(all_results, key=lambda k: all_results[k]['test_accuracy'])
    best_result = all_results[best_model_name]
    
    print("\n" + "=" * 60)
    print("Best Model")
    print("=" * 60)
    print(f"Model: {best_model_name}")
    print(f"Test Accuracy: {best_result['test_accuracy']*100:.2f}%")
    print(f"Best Validation Accuracy: {best_result['best_val_acc']*100:.2f}%")
    print(f"Best Hyperparameters: {best_result['best_hyperparameters']}")
    best_paths = get_method_paths(best_model_name)
    print(f"Model saved to: {best_paths['method_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
