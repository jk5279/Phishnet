"""
06_inference_pipeline.py

Quick Inference Pipeline using Trained Models

This script provides functions to load and use trained models from both
Machine Learning and Deep Learning pipelines for predictions:
1. Load ML model (sklearn pipeline)
2. Load DL model (BERT), tokenizer, and label encoder
3. Make predictions with either model
4. Optional ensemble predictions combining both models

Usage:
    python 06_inference_pipeline.py
    or import functions for programmatic use
"""

import os
import pickle
import string
import warnings
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Try to import scipy's expit, fallback to numpy implementation
try:
    from scipy.special import expit
except ImportError:
    # Fallback: simple sigmoid implementation
    def expit(x):
        """Sigmoid function: 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

warnings.filterwarnings('ignore')

# =========================
# Configuration
# =========================

# Model paths
MODEL_DIR = "models"
ML_MODEL_PATH = os.path.join(MODEL_DIR, "ml_best_model.pkl")
# Use best model by default (best accuracy on test set)
# Alternative: DL_MODEL_PATH = os.path.join(MODEL_DIR, "dl_model") for final model
DL_MODEL_PATH = os.path.join(MODEL_DIR, "dl_model_best")
DL_TOKENIZER_PATH = os.path.join(MODEL_DIR, "dl_tokenizer")
DL_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "dl_label_encoder.pkl")

# DL settings
MAX_LEN = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================
# ML Model Loading and Prediction
# =========================

def load_ml_model(model_path: str = ML_MODEL_PATH):
    """Load trained ML model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ML model not found at {model_path}. "
            "Please run 04_ml_pipeline.py first to train and save the model."
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"ML model loaded from: {model_path}")
    return model


def predict_ml(text: str, model=None) -> tuple:
    """
    Make prediction using ML model.
    
    Args:
        text: Input text string
        model: Trained ML pipeline (if None, loads from default path)
        
    Returns:
        tuple: (predicted_label, confidence, predicted_class_int)
    """
    if model is None:
        model = load_ml_model()
    
    # Engineer metadata features (same as training)
    df = pd.DataFrame({'text': [text]})
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['punct_count'] = df['text'].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    df['upper_count'] = df['text'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    
    # Predict
    prediction = model.predict(df)[0]
    
    # Handle models with/without predict_proba (e.g., LinearSVC)
    try:
        probabilities = model.predict_proba(df)[0]
    except AttributeError:
        # For models like LinearSVC, use decision_function and convert to probabilities
        decision_scores = model.decision_function(df)
        if len(decision_scores.shape) == 1:
            # Binary classification: convert decision score to probability using sigmoid
            prob_positive = expit(decision_scores[0])
            probabilities = np.array([1 - prob_positive, prob_positive])
        else:
            # Multi-class: use softmax
            probabilities = expit(decision_scores[0])
            probabilities = probabilities / probabilities.sum()  # Normalize
    
    confidence = max(probabilities) * 100
    
    label_map = {0: 'Not Phishing', 1: 'Phishing'}
    predicted_label = label_map[prediction]
    
    return predicted_label, confidence, int(prediction)


def predict_ml_batch(texts: list, model=None) -> list:
    """
    Make batch predictions using ML model.
    
    Args:
        texts: List of input text strings
        model: Trained ML pipeline (if None, loads from default path)
        
    Returns:
        list: List of tuples (predicted_label, confidence, predicted_class_int)
    """
    if model is None:
        model = load_ml_model()
    
    # Engineer metadata features
    df = pd.DataFrame({'text': texts})
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['punct_count'] = df['text'].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation)
    )
    df['upper_count'] = df['text'].apply(
        lambda x: len([w for w in str(x).split() if w.isupper()])
    )
    
    # Predict
    predictions = model.predict(df)
    
    # Handle models with/without predict_proba (e.g., LinearSVC)
    try:
        probabilities = model.predict_proba(df)
    except AttributeError:
        # For models like LinearSVC, use decision_function and convert to probabilities
        decision_scores = model.decision_function(df)
        probabilities = []
        if len(decision_scores.shape) == 1:
            # Binary classification: convert decision scores to probabilities using sigmoid
            for score in decision_scores:
                prob_positive = expit(score)
                probabilities.append(np.array([1 - prob_positive, prob_positive]))
        else:
            # Multi-class: use softmax
            for scores in decision_scores:
                probs = expit(scores)
                probs = probs / probs.sum()  # Normalize
                probabilities.append(probs)
        probabilities = np.array(probabilities)
    
    confidences = [max(probs) * 100 for probs in probabilities]
    
    label_map = {0: 'Not Phishing', 1: 'Phishing'}
    results = [
        (label_map[pred], conf, int(pred))
        for pred, conf in zip(predictions, confidences)
    ]
    
    return results


# =========================
# DL Model Loading and Prediction
# =========================

def load_dl_model(
    model_path: str = DL_MODEL_PATH,
    tokenizer_path: str = DL_TOKENIZER_PATH,
    label_encoder_path: str = DL_LABEL_ENCODER_PATH
):
    """Load trained DL model, tokenizer, and label encoder from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"DL model not found at {model_path}. "
            "Please run 05_dl_pipeline.py first to train and save the model."
        )
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"DL model loaded from: {model_path}")
    print(f"Tokenizer loaded from: {tokenizer_path}")
    print(f"Label encoder loaded from: {label_encoder_path}")
    
    return model, tokenizer, label_encoder


def predict_dl(
    text: str,
    model=None,
    tokenizer=None,
    label_encoder=None,
    max_len: int = MAX_LEN
) -> tuple:
    """
    Make prediction using DL (BERT) model.
    
    Args:
        text: Input text string
        model: Trained BERT model (if None, loads from default path)
        tokenizer: BERT tokenizer (if None, loads from default path)
        label_encoder: Label encoder (if None, loads from default path)
        max_len: Maximum sequence length
        
    Returns:
        tuple: (predicted_label, confidence, predicted_class_int)
    """
    if model is None or tokenizer is None or label_encoder is None:
        model, tokenizer, label_encoder = load_dl_model()
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get prediction and confidence
    probs = torch.softmax(logits, dim=1)
    pred_index = torch.argmax(logits, dim=1).item()
    confidence = probs[0][pred_index].item() * 100
    
    predicted_label = label_encoder.inverse_transform([pred_index])[0]
    
    return predicted_label, confidence, int(pred_index)


def predict_dl_batch(
    texts: list,
    model=None,
    tokenizer=None,
    label_encoder=None,
    max_len: int = MAX_LEN,
    batch_size: int = 8
) -> list:
    """
    Make batch predictions using DL (BERT) model.
    
    Args:
        texts: List of input text strings
        model: Trained BERT model (if None, loads from default path)
        tokenizer: BERT tokenizer (if None, loads from default path)
        label_encoder: Label encoder (if None, loads from default path)
        max_len: Maximum sequence length
        batch_size: Batch size for processing
        
    Returns:
        list: List of tuples (predicted_label, confidence, predicted_class_int)
    """
    if model is None or tokenizer is None or label_encoder is None:
        model, tokenizer, label_encoder = load_dl_model()
    
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encodings = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        # Predict batch
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # Get predictions and confidences
        probs = torch.softmax(logits, dim=1)
        pred_indices = torch.argmax(logits, dim=1).cpu().numpy()
        confidences = [
            probs[j][pred_indices[j]].item() * 100
            for j in range(len(batch_texts))
        ]
        
        # Decode labels
        predicted_labels = label_encoder.inverse_transform(pred_indices)
        
        # Add to results
        for label, conf, idx in zip(predicted_labels, confidences, pred_indices):
            results.append((label, conf, int(idx)))
    
    return results


# =========================
# Ensemble Prediction
# =========================

def predict_ensemble(
    text: str,
    ml_model=None,
    dl_model=None,
    dl_tokenizer=None,
    dl_label_encoder=None,
    weights: dict = None
) -> tuple:
    """
    Make ensemble prediction combining both ML and DL models.
    
    Args:
        text: Input text string
        ml_model: Trained ML pipeline (if None, loads from default path)
        dl_model: Trained BERT model (if None, loads from default path)
        dl_tokenizer: BERT tokenizer (if None, loads from default path)
        dl_label_encoder: Label encoder (if None, loads from default path)
        weights: Dict with 'ml' and 'dl' weights (default: equal weights)
        
    Returns:
        tuple: (predicted_label, confidence, ml_pred, dl_pred)
    """
    if weights is None:
        weights = {'ml': 0.5, 'dl': 0.5}
    
    # Get predictions from both models
    ml_label, ml_conf, ml_pred = predict_ml(text, ml_model)
    dl_label, dl_conf, dl_pred = predict_dl(
        text, dl_model, dl_tokenizer, dl_label_encoder
    )
    
    # Weighted voting
    ml_score = weights['ml'] * (ml_conf / 100.0) if ml_pred == 1 else weights['ml'] * (1 - ml_conf / 100.0)
    dl_score = weights['dl'] * (dl_conf / 100.0) if dl_pred == 1 else weights['dl'] * (1 - dl_conf / 100.0)
    
    ensemble_score = ml_score + dl_score
    ensemble_pred = 1 if ensemble_score > 0.5 else 0
    ensemble_conf = ensemble_score * 100 if ensemble_pred == 1 else (1 - ensemble_score) * 100
    
    label_map = {0: 'Not Phishing', 1: 'Phishing'}
    ensemble_label = label_map[ensemble_pred]
    
    return ensemble_label, ensemble_conf, ml_pred, dl_pred


# =========================
# Example Usage
# =========================

def main():
    """Example usage of inference functions."""
    print("=" * 60)
    print("Inference Pipeline - Example Usage")
    print("=" * 60)
    
    # Example text (phishing email)
    example_text = """Tuition Payment Deadline Missed – Final Grace Period 
    Mohit Malhotra<mohit.malhotra@mail.utoronto.ca> 
    Dear Valued Student, This is your final warning. 
    As of October 6, 2025, our records indicate that your Fall 2025 
    tuition deposit remains unpaid. You are now outside the formal 
    payment deadline of September 30, 2025, and are at immediate risk 
    of deregistration, loss of student status, and deactivation of 
    all university services. Payment must now be submitted immediately 
    using the instructions below: Interac e-Transfer Details 
    Email: Mabintydumbuya_19@hotmail.com"""
    
    print("\n--- Example Text ---")
    print(example_text[:200] + "...")
    
    # Try ML prediction
    try:
        print("\n--- ML Model Prediction ---")
        ml_label, ml_conf, ml_pred = predict_ml(example_text)
        print(f"Prediction: {ml_label}")
        print(f"Confidence: {ml_conf:.2f}%")
        print(f"Class: {ml_pred}")
    except FileNotFoundError as e:
        print(f"ML model not available: {e}")
    
    # Try DL prediction
    try:
        print("\n--- DL Model Prediction ---")
        dl_label, dl_conf, dl_pred = predict_dl(example_text)
        print(f"Prediction: {dl_label}")
        print(f"Confidence: {dl_conf:.2f}%")
        print(f"Class: {dl_pred}")
    except FileNotFoundError as e:
        print(f"DL model not available: {e}")
    
    # Try ensemble prediction
    try:
        print("\n--- Ensemble Prediction ---")
        ensemble_label, ensemble_conf, ml_pred, dl_pred = predict_ensemble(example_text)
        print(f"Prediction: {ensemble_label}")
        print(f"Confidence: {ensemble_conf:.2f}%")
        print(f"ML prediction: {ml_pred}, DL prediction: {dl_pred}")
    except Exception as e:
        print(f"Ensemble prediction not available: {e}")
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

