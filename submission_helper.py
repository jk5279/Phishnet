import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from collections import Counter
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Paths & Config
# =========================
REPO_ROOT = "."
DATA_DIR = os.path.join(REPO_ROOT, "datasets", "common clean dataset - 28112025")
MODEL_DIR = os.path.join(REPO_ROOT, "dl_methods", "roberta")
# Check both best and final locations since gitignore might have affected 'best'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model", "best")
if not os.path.exists(BEST_MODEL_PATH):
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model", "final")
    
TOKENIZER_PATH = os.path.join(MODEL_DIR, "model", "tokenizer")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "model", "label_encoder.pkl")
RESULTS_SUMMARY_PATH = os.path.join(REPO_ROOT, "model_results_summary.csv")
TRAINING_CURVE_PATH = os.path.join(REPO_ROOT, "models", "dl_training_curve.png")

# Constants
MAX_LEN = 128
BATCH_SIZE = 32
LABEL_MAPPING = {0: "Great", 1: "Bait"}  # Adjust if needed based on label encoder

# =========================
# Helper Functions
# =========================

def setup():
    """Perform any necessary setup."""
    print(f"Setup complete. Running on {device}")
    if not os.path.exists(DATA_DIR):
        print(f"Warning: Data directory {DATA_DIR} not found.")

def show_project_overview():
    """Display project description."""
    print("=== Phishnet Project Overview ===")
    print("This project implements a phishing detection pipeline using both Machine Learning")
    print("and Deep Learning approaches. The goal is to classify emails as 'Great' (Benign)")
    print("or 'Bait' (Phishing).")
    print("\nKey Features:")
    print("- Data Aggregation from multiple sources")
    print("- ML Pipeline: Logistic Regression, SVM, Naive Bayes")
    print("- DL Pipeline: BERT, RoBERTa, DistilBERT")
    print("- Model Interpretation: Attention visualization")
    
    img_path = os.path.join(REPO_ROOT, "Phishnet Project Image.jpg")
    if os.path.exists(img_path):
        try:
            img = plt.imread(img_path)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Phishnet Architecture")
            plt.show()
        except Exception as e:
            print(f"Could not load project image: {e}")

def load_data(show_head=True):
    """Load and display a sample of the processed data."""
    print("\n--- Loading Data ---")
    train_path = os.path.join(DATA_DIR, "train_split.csv")
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        print(f"Loaded training data: {df.shape}")
        if show_head:
            print("\nSample Data:")
            display(df.head())
        return df
    else:
        print(f"Error: Training data not found at {train_path}")
        return None

def visualize_data(df):
    """Generate basic visualizations for the dataset."""
    if df is None:
        return
    
    print("\n--- Data Visualization ---")
    
    # Label Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=df)
    plt.title('Label Distribution (0=Benign, 1=Phishing)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()
    
    # Text Length Distribution
    df['text_len'] = df['text'].astype(str).apply(len)
    plt.figure(figsize=(10, 4))
    sns.histplot(df['text_len'], bins=50, log_scale=True)
    plt.title('Text Length Distribution (Log Scale)')
    plt.xlabel('Length (chars)')
    plt.show()

def show_model_architecture():
    """Display model architecture information."""
    print("\n--- Model Architecture ---")
    print("We utilize transformer-based models (RoBERTa) for sequence classification.")
    print("The architecture consists of:")
    print("1. Tokenizer: RoBERTa Tokenizer (Byte-Pair Encoding)")
    print("2. Transformer Encoder: 12-layer RoBERTa base model")
    print("3. Classification Head: Linear layer on top of the [CLS] token output")
    print("\nInput: Tokenized Email Text -> RoBERTa -> Contextual Embeddings -> Classifier -> Probability")

def show_quantitative_results():
    """Display aggregated results and training curves."""
    print("\n--- Quantitative Results ---")
    
    # Table
    if os.path.exists(RESULTS_SUMMARY_PATH):
        print("Model Performance Summary:")
        res_df = pd.read_csv(RESULTS_SUMMARY_PATH)
        display(res_df)
    else:
        print("Results summary file not found.")
        
    # Curve
    if os.path.exists(TRAINING_CURVE_PATH):
        print("\nTraining Curve (Deep Learning):")
        try:
            img = plt.imread(TRAINING_CURVE_PATH)
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except:
            pass

# =========================
# Inference Logic
# =========================

def load_inference_model():
    """Load the trained RoBERTa model."""
    print("\nLoading Model...", end=" ")
    
    # Check tokenizer path - might be in final if not in tokenizer dir
    tok_path = TOKENIZER_PATH
    if not os.path.exists(tok_path):
        tok_path = os.path.join(MODEL_DIR, "model", "final")
        
    if not os.path.exists(tok_path) or not os.path.exists(BEST_MODEL_PATH):
        print(f"Model files not found. Expected at {BEST_MODEL_PATH} and {tok_path}")
        return None, None
    
    try:
        tokenizer = RobertaTokenizer.from_pretrained(tok_path, local_files_only=True)
        # Load label encoder to get num_labels
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
        num_labels = len(le.classes_)
        
        model = RobertaForSequenceClassification.from_pretrained(
            BEST_MODEL_PATH, num_labels=num_labels, local_files_only=True
        )
        model.to(device)
        model.eval()
        print("Done.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_email(model, tokenizer, text):
    """Run prediction on a single text."""
    if model is None:
        return
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    label = LABEL_MAPPING.get(pred_idx, str(pred_idx))
    
    return label, confidence, probs[0].cpu().numpy()

def run_demo_inference(custom_text=None):
    """Run inference on demo dataset or custom text."""
    model, tokenizer = load_inference_model()
    if model is None:
        return

    if custom_text:
        print(f"\n--- Custom Inference ---")
        print(f"Text: {custom_text[:100]}...")
        label, conf, _ = predict_email(model, tokenizer, custom_text)
        print(f"Prediction: {label} (Confidence: {conf:.2%})")
        return

    # Load Demo Dataset
    demo_path = os.path.join(REPO_ROOT, "datasets", "Demonstration dataset.csv")
    if os.path.exists(demo_path):
        print("\n--- Running on Demonstration Dataset ---")
        df = pd.read_csv(demo_path)
        # Ensure Title/Text cols
        if 'Title' in df.columns and 'Text' in df.columns:
            df['combined'] = "Title: " + df['Title'].fillna('') + "\n\n" + df['Text'].fillna('')
        elif 'text' in df.columns:
            df['combined'] = df['text']
        else:
            print("Could not identify text column.")
            return

        # Run on first 3 examples
        for i, row in df.head(3).iterrows():
            text = row['combined']
            label, conf, _ = predict_email(model, tokenizer, text)
            print(f"\nExample {i+1}:")
            print(f"Text: {text[:100]}...")
            print(f"Prediction: {label} (Confidence: {conf:.2%})")
    else:
        print("Demonstration dataset not found.")

def show_discussion():
    print("\n--- Discussion & Insights ---")
    print("1. Performance: The RoBERTa model achieves high accuracy, outperforming traditional ML models.")
    print("2. Challenges: Handling imbalanced data and diverse phishing tactics.")
    print("3. Insights: Attention maps reveal the model focuses on urgency cues and suspicious links.")

def show_interpretation_examples():
    """Display pre-generated attention maps."""
    print("\n--- Model Interpretation (Attention Maps) ---")
    print("These visualizations show which words the model focused on.")
    
    interp_dir = os.path.join(MODEL_DIR, "logs", "interpretations", "attention_correct")
    if os.path.exists(interp_dir):
        # Find some heat_*.png files
        files = [f for f in os.listdir(interp_dir) if f.startswith("heat_") and f.endswith(".png")]
        files = sorted(files)[:2] # Show first 2
        
        for f in files:
            path = os.path.join(interp_dir, f)
            try:
                img = plt.imread(path)
                plt.figure(figsize=(12, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Attention Map: {f}")
                plt.show()
            except:
                pass
    else:
        print("No interpretation images found.")

