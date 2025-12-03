"""
10_model_interpretation.py

Model Interpretation Script for RoBERTa

Analyzes predictions using attention visualization on the CLS token.
Fixes: Overlaps, subword artifacts, spacing, and EXCESSIVE WHITESPACE below content.
"""

import os
import json
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from typing import List, Dict, Tuple
from transformers import RobertaForSequenceClassification, RobertaTokenizer

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
PREDICTIONS_JSON = os.path.join(MODEL_DIR, "logs", "demonstration_dataset_predictions.json")
OUTPUT_DIR = os.path.join(MODEL_DIR, "logs", "interpretations")

# Inference settings
MAX_LEN = 128
LABEL_MAPPING = {0: "Great", 1: "Bait"}

# =========================
# Model Loading
# =========================

def load_model_and_components():
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

def load_predictions():
    if not os.path.exists(PREDICTIONS_JSON):
        raise FileNotFoundError(f"Predictions file not found at {PREDICTIONS_JSON}")
    
    with open(PREDICTIONS_JSON, 'r') as f:
        data = json.load(f)
    
    predictions = data['predictions']
    correct = [p for p in predictions if p['predicted_label'] == p['true_label']]
    incorrect = [p for p in predictions if p['predicted_label'] != p['true_label']]
    
    print(f"Total: {len(predictions)} | Correct: {len(correct)} | Incorrect: {len(incorrect)}")
    return correct, incorrect

# =========================
# Core Logic: Reconstruction
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
# Attention Extraction
# =========================

def extract_attention_weights(model, tokenizer, text: str):
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
# Visualization (Fixed Whitespace)
# =========================

def plot_attention_heatmap(tokens, weights, sample_id, predicted_label, output_path):
    """
    Generates a PNG with wrapped text and background colors corresponding to attention.
    Dynamically positions the legend to reduce whitespace.
    """
    # Normalize weights 0-1
    w_min, w_max = weights.min(), weights.max()
    # Avoid division by zero if all weights are the same
    if w_max - w_min < 1e-9:
         norm_weights = weights # Should be all zeros assuming min is 0
    else:
         norm_weights = (weights - w_min) / (w_max - w_min)
    
    # Setup Figure
    # Reduced height slightly as we expect less whitespace now
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
    
    # Title
    title = f"Sample {sample_id} | Pred: {LABEL_MAPPING[predicted_label]}"
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

    # --- Dynamic Legend Placement (THE FIX) ---
    
    # 'y' now holds the vertical coordinate of the baseline of the last text line.
    # We want the legend placed dynamically below this final 'y'.
    
    legend_gap = 0.12  # Gap between text and legend (approx a line height + bit more)
    legend_height = 0.04
    
    # Calculate bottom position based on where text ended
    legend_bottom = y - legend_gap - legend_height
    
    # Safety check: Ensure legend doesn't fall off bottom if text is very long.
    # Pin to a minimum bottom margin of 0.02 if needed.
    min_bottom_margin = 0.02
    if legend_bottom < min_bottom_margin:
         legend_bottom = min_bottom_margin

    # Create Colorbar Axes at dynamic position
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.3, legend_bottom, 0.4, legend_height])
    
    # Create Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Attention Score (from [CLS])', fontweight='bold', fontsize=10)
    cbar.ax.tick_params(labelsize=9) # Smaller ticks
    
    # Save using bbox_inches='tight' which will now crop snugly around our content
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', pad_inches=0.05, dpi=600)
    plt.close()
    print(f"  ✓ Saved: {output_path}")

def plot_attention_summary(tokens, weights, sample_id, predicted_label, output_path):
    """Bar chart of top tokens."""
    top_n = min(15, len(tokens))
    # Get indices of top weights
    indices = np.argsort(weights)[-top_n:]
    
    top_tokens = [tokens[i] for i in indices]
    top_weights = weights[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    # Use viridis for bar charts as it's clearer for comparisons than YlOrRd
    bars = ax.barh(range(top_n), top_weights, color=plt.cm.viridis(top_weights/top_weights.max()))
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_tokens, fontsize=11)
    ax.set_xlabel("Attention Score")
    ax.set_title(f"Top Focus Words - Sample {sample_id}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

# =========================
# Main Execution
# =========================

def process_sample(model, tokenizer, sample, output_dir, category):
    sample_id = sample['sample_id']
    text = sample['combined_text']
    pred = sample['predicted_label']
    
    path_heatmap = os.path.join(output_dir, f"attention_{category}", f"heat_{sample_id}.png")
    path_summary = os.path.join(output_dir, f"attention_{category}", f"bar_{sample_id}.png")
    os.makedirs(os.path.dirname(path_heatmap), exist_ok=True)
    
    try:
        rec_tokens, rec_weights = extract_attention_weights(model, tokenizer, text)
        # Only plot if we have tokens left after cleaning
        if len(rec_tokens) > 0:
            plot_attention_heatmap(rec_tokens, rec_weights, sample_id, pred, path_heatmap)
            plot_attention_summary(rec_tokens, rec_weights, sample_id, pred, path_summary)
        else:
             print(f"  ⚠ Skipping sample {sample_id} (no valid tokens after cleaning)")
    except Exception as e:
        print(f"Error sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=== RoBERTa Interpretation Tool ===")
    
    model, tokenizer, _ = load_model_and_components()
    correct, incorrect = load_predictions()
    
    print(f"\nProcessing {len(correct)} Correct samples...")
    for s in correct:
        process_sample(model, tokenizer, s, OUTPUT_DIR, "correct")
        
    print(f"\nProcessing {len(incorrect)} Incorrect samples...")
    for s in incorrect:
        process_sample(model, tokenizer, s, OUTPUT_DIR, "incorrect")
        
    print("\nDone.")

if __name__ == "__main__":
    main()