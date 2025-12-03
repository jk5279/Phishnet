"""
12_test_attention_visualization.py

Attention Visualization for Test Dataset Samples

This script:
1. Loads the test dataset from cleaned_data/DL/test/test_split.csv
2. Runs inference using RoBERTa model
3. Identifies correct and incorrect predictions
4. Selects 5 samples (mix of correct and incorrect)
5. Generates attention visualizations for the selected samples
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
# Set Matplotlib backend to Agg to prevent display errors on servers
plt.switch_backend('Agg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# Configuration
# =========================

# Model paths
MODEL_DIR = "dl_methods/roberta"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "model", "best")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "model", "tokenizer")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "model", "label_encoder.pkl")

# Input/Output paths
TEST_DATASET_PATH = os.path.join("cleaned_data", "DL", "test", "test_split.csv")
OUTPUT_DIR = os.path.join(MODEL_DIR, "logs", "interpretations", "test_samples")

# Inference settings
MAX_LEN = 256
LABEL_MAPPING = {0: "Great", 1: "Bait"}

# =========================
# Model Loading (Reused from 10_model_interpretation.py)
# =========================

def load_model_and_components():
    """Load the RoBERTa model, tokenizer, and label encoder."""
    print("\n--- Loading Model Components ---")
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        
    num_classes = len(label_encoder.classes_)
    model = RobertaForSequenceClassification.from_pretrained(
        BEST_MODEL_PATH,
        num_labels=num_classes,
        local_files_only=True,
        attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    print(f"✓ Components loaded. Device: {device}")
    return model, tokenizer, label_encoder

# =========================
# Data Loading
# =========================

def load_test_dataset():
    """Load test dataset from CSV file."""
    print("\n--- Loading Test Dataset ---")
    
    if not os.path.exists(TEST_DATASET_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATASET_PATH}")
    
    df = pd.read_csv(TEST_DATASET_PATH)
    print(f"Loaded test dataset: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df

# =========================
# Token Reconstruction (Reused from 10_model_interpretation.py)
# =========================

def reconstruct_tokens_and_weights(tokens: List[str], weights: np.ndarray):
    """
    Robustly merges RoBERTa subwords (starting without Ġ) into whole words
    and averages their attention weights.
    """
    new_tokens = []
    new_weights = []
    
    current_word = ""
    current_weights = []
    
    # Special tokens to explicitly ignore
    special_tokens = {'<s>', '</s>', '<pad>', '<unk>'}
    
    for token, weight in zip(tokens, weights):
        # 1. Skip Special Tokens
        if token in special_tokens:
            continue
            
        # 2. Check logic: RoBERTa uses 'Ġ' to indicate the START of a new word.
        is_start_of_word = token.startswith('Ġ')
        
        # Clean the token representation (remove the Ġ and newlines)
        clean_token = token.replace('Ġ', '').replace('Ċ', '')
        
        if not clean_token.strip(): # Skip if token became empty or just whitespace
            continue

        if is_start_of_word:
            # Save previous word if it exists
            if current_word:
                new_tokens.append(current_word)
                new_weights.append(np.mean(current_weights))
            
            # Start new word
            current_word = clean_token
            current_weights = [weight]
            
        else:
            # It's a suffix/continuation
            if not current_word:
                current_word = clean_token
                current_weights = [weight]
            else:
                current_word += clean_token
                current_weights.append(weight)
    
    # Append the final word
    if current_word:
        new_tokens.append(current_word)
        new_weights.append(np.mean(current_weights))
        
    return new_tokens, np.array(new_weights)

# =========================
# Attention Extraction (Reused from 10_model_interpretation.py)
# =========================

def extract_attention_weights(model, tokenizer, text: str):
    """Extract attention weights from the model for a given text."""
    inputs = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=mask, output_attentions=True)
    
    # Get last layer attention: (batch, num_heads, seq_len, seq_len)
    last_layer_att = outputs.attentions[-1]
    
    # Average across heads -> (seq_len, seq_len)
    avg_att = torch.mean(last_layer_att[0], dim=0).cpu().numpy()
    
    # Get actual tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Get valid length (exclude padding)
    valid_len = mask[0].sum().item()
    
    # Slice to valid length
    tokens = tokens[:valid_len]
    cls_attention = avg_att[0, :valid_len] # Attention from CLS (idx 0) to all others
    
    # Reconstruct
    rec_tokens, rec_weights = reconstruct_tokens_and_weights(tokens, cls_attention)
    
    return rec_tokens, rec_weights

# =========================
# Sample Selection (Before Inference)
# =========================

def select_samples_before_inference(test_df: pd.DataFrame, n_samples: int = 5):
    """Select n_samples from test dataset with diverse labels."""
    print(f"\n--- Selecting {n_samples} Samples from Test Dataset ---")
    
    # Get samples with different labels
    label_0_samples = test_df[test_df['label'] == 0].copy()
    label_1_samples = test_df[test_df['label'] == 1].copy()
    
    # Select mix: 2-3 from each label
    n_label_0 = min(3, len(label_0_samples))
    n_label_1 = min(2, len(label_1_samples))
    
    # Adjust if needed
    if n_label_0 + n_label_1 < n_samples:
        if len(label_0_samples) > len(label_1_samples):
            n_label_0 = min(n_samples - n_label_1, len(label_0_samples))
        else:
            n_label_1 = min(n_samples - n_label_0, len(label_1_samples))
    
    # Select diverse samples
    selected_0 = label_0_samples.sample(n=n_label_0, random_state=42) if len(label_0_samples) > 0 else pd.DataFrame()
    selected_1 = label_1_samples.sample(n=n_label_1, random_state=42) if len(label_1_samples) > 0 else pd.DataFrame()
    
    # Combine and reset index
    selected = pd.concat([selected_0, selected_1]).sample(frac=1, random_state=42).head(n_samples).reset_index(drop=True)
    
    print(f"  Selected {len(selected)} samples:")
    print(f"    Label 0 (Great): {len(selected_0)}, Label 1 (Bait): {len(selected_1)}")
    
    return selected

# =========================
# Inference on Selected Samples
# =========================

def run_inference_on_samples(model, tokenizer, selected_df: pd.DataFrame):
    """Run inference on selected samples and return predictions with metadata."""
    print("\n--- Running Inference on Selected Samples ---")
    
    texts = selected_df['text'].tolist()
    true_labels = selected_df['label'].tolist()
    
    all_predictions = []
    all_probabilities = []
    
    model.eval()
    with torch.no_grad():
        # Tokenize all texts at once
        inputs = tokenizer(
            texts,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get predictions and probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predictions = torch.max(logits, dim=1)
        
        all_predictions = predictions.cpu().numpy()
        all_probabilities = probabilities.cpu().numpy()
    
    # Create results dataframe
    results = []
    for idx, (text, true_label, pred, prob) in enumerate(zip(texts, true_labels, all_predictions, all_probabilities)):
        is_correct = (pred == true_label)
        confidence = prob[pred]
        
        results.append({
            'sample_id': idx,
            'original_index': selected_df.index[idx],  # Original index in test dataset
            'text': text,
            'true_label': int(true_label),
            'predicted_label': int(pred),
            'is_correct': is_correct,
            'confidence': float(confidence),
            'prob_great': float(prob[0]),
            'prob_bait': float(prob[1]),
        })
    
    results_df = pd.DataFrame(results)
    
    correct_count = results_df['is_correct'].sum()
    total_count = len(results_df)
    
    print(f"✓ Inference complete: {total_count} samples")
    print(f"  Correct: {correct_count}, Incorrect: {total_count - correct_count}")
    
    return results_df

# =========================
# Visualization (Reused from 10_model_interpretation.py)
# =========================

def plot_attention_heatmap(tokens, weights, sample_id, true_label, predicted_label, is_correct, output_path):
    """
    Generates a PNG with wrapped text and background colors corresponding to attention.
    Dynamically positions the legend to reduce whitespace.
    """
    # Normalize weights 0-1
    w_min, w_max = weights.min(), weights.max()
    # Avoid division by zero if all weights are the same
    if w_max - w_min < 1e-9:
         norm_weights = weights
    else:
         norm_weights = (weights - w_min) / (w_max - w_min)
    
    # Setup Figure
    fig = plt.figure(figsize=(14, 5), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1]) # Full span axes for manual text placement
    ax.axis('off')
    
    # Constants for Layout (in figure coordinates 0.0 - 1.0)
    start_x, start_y = 0.02, 0.82
    x, y = start_x, start_y
    max_x = 0.96
    line_height = 0.10 # Space between lines
    
    # Box Style
    box_pad = 0.2
    fontsize = 12
    
    # Title with correctness indicator
    correctness = "✓ Correct" if is_correct else "✗ Incorrect"
    title = f"Test Sample {sample_id} | True: {LABEL_MAPPING[true_label]} | Pred: {LABEL_MAPPING[predicted_label]} ({correctness})"
    ax.text(0.5, 0.93, title, ha='center', fontsize=16, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", linewidth=1))

    # --- Text Rendering Loop ---
    # We must draw the canvas once to get a renderer for measuring text sizes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    for token, weight in zip(tokens, norm_weights):
        # Color mapping: YlOrRd colormap
        # Offset slightly (0.15) so lowest attention isn't invisible white/pale yellow
        color_val = 0.15 + 0.85 * weight
        color = plt.cm.YlOrRd(color_val)
        
        # 1. Create dummy text to measure width
        t_obj = ax.text(x, y, token, fontsize=fontsize, 
                        bbox=dict(boxstyle=f"square,pad={box_pad}", fc=color, ec='none', alpha=0.8))
        
        # 2. Measure width in figure coordinates
        bbox = t_obj.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transAxes.inverted())
        width = bbox_data.width
        
        # 3. Check for Line Wrap
        if x + width > max_x:
            t_obj.remove() # Remove terminal text
            x = start_x # Reset X
            y -= line_height # Move Y down to next line
            
            # Re-draw at new position
            t_obj = ax.text(x, y, token, fontsize=fontsize, 
                            bbox=dict(boxstyle=f"square,pad={box_pad}", fc=color, ec='none', alpha=0.8))
            
            # Re-measure width just in case
            bbox = t_obj.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transAxes.inverted())
            width = bbox_data.width

        # 4. Advance X position. Add small margin between words.
        x += width + 0.005 

    # --- Dynamic Legend Placement ---
    
    legend_gap = 0.12  # Gap between text and legend
    legend_height = 0.04
    
    # Calculate bottom position based on where text ended
    legend_bottom = y - legend_gap - legend_height
    
    # Safety check: Ensure legend doesn't fall off bottom
    min_bottom_margin = 0.02
    if legend_bottom < min_bottom_margin:
         legend_bottom = min_bottom_margin

    # Create Colorbar Axes at dynamic position
    cbar_ax = fig.add_axes([0.3, legend_bottom, 0.4, legend_height])
    
    # Create Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Attention Score (from [CLS])', fontweight='bold', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Save using bbox_inches='tight'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', pad_inches=0.05, dpi=600)
    plt.close()
    print(f"  ✓ Saved: {output_path}")

# =========================
# Main Execution
# =========================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Test Dataset Attention Visualization")
    print("=" * 70)
    
    # Load model components
    model, tokenizer, label_encoder = load_model_and_components()
    
    # Load test dataset
    test_df = load_test_dataset()
    
    # Select 5 samples first (more efficient)
    selected_df = select_samples_before_inference(test_df, n_samples=20)
    
    # Run inference only on selected samples
    results_df = run_inference_on_samples(model, tokenizer, selected_df)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate visualizations
    print(f"\n--- Generating Attention Visualizations ---")
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'test_dataset_path': TEST_DATASET_PATH,
        'total_test_samples': len(test_df),
        'selected_samples': []
    }
    
    for idx, row in results_df.iterrows():
        sample_id = row['sample_id']
        text = row['text']
        true_label = row['true_label']
        predicted_label = row['predicted_label']
        is_correct = row['is_correct']
        confidence = row['confidence']
        
        print(f"\nProcessing sample {sample_id}...")
        
        try:
            # Extract attention weights
            rec_tokens, rec_weights = extract_attention_weights(model, tokenizer, text)
            
            # Only plot if we have tokens left after cleaning
            if len(rec_tokens) > 0:
                output_path = os.path.join(OUTPUT_DIR, f"test_sample_{sample_id}.png")
                plot_attention_heatmap(
                    rec_tokens, rec_weights, 
                    sample_id, true_label, predicted_label, is_correct,
                    output_path
                )
                
                # Add to metadata
                original_idx = row.get('original_index', sample_id)
                metadata['selected_samples'].append({
                    'sample_id': int(sample_id),
                    'original_index_in_test': int(original_idx),
                    'true_label': int(true_label),
                    'true_label_name': LABEL_MAPPING[true_label],
                    'predicted_label': int(predicted_label),
                    'predicted_label_name': LABEL_MAPPING[predicted_label],
                    'is_correct': bool(is_correct),
                    'confidence': float(confidence),
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                })
            else:
                print(f"  ⚠ Skipping sample {sample_id} (no valid tokens after cleaning)")
        
        except Exception as e:
            print(f"  ✗ Error processing sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {metadata_path}")
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()

