"""
03_dl_preprocessing_eda.py

Preprocessing for Deep Learning Pipeline and Exploratory Data Analysis

This script performs DL-specific preprocessing (gentle cleaning to preserve case/punctuation)
and generates exploratory data analysis visualizations:
1. Gentle text cleaning (preserves case and punctuation for transformers)
2. Deduplication
3. Length filtering (5-2000 tokens)
4. EDA visualizations (distributions, correlations, n-grams, word clouds)

Output: 
- cleaned_data/DL/train/train_split.csv (processed train set)
- cleaned_data/DL/validation/validation_split.csv (processed validation set)
- cleaned_data/DL/test/test_split.csv (processed test set)
- eda_outputs/dl_cleaning/train/dl_*.png (EDA visualizations for train split)
- eda_outputs/dl_cleaning/validation/dl_*.png (EDA visualizations for validation split)
- eda_outputs/dl_cleaning/test/dl_*.png (EDA visualizations for test split)

Input: 
- cleaned_data/train_split.csv (from 01_data_aggregation.py)
- cleaned_data/validation_split.csv (from 01_data_aggregation.py)
- cleaned_data/test_split.csv (from 01_data_aggregation.py)
"""

import os
import re
import quopri
import multiprocessing
import warnings
import string
import nltk
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm.contrib.concurrent import process_map
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# =========================
# Configuration
# =========================

# Input/Output paths
TRAIN_INPUT = os.path.join("cleaned_data", "train_split.csv")
VAL_INPUT = os.path.join("cleaned_data", "validation_split.csv")
TEST_INPUT = os.path.join("cleaned_data", "test_split.csv")
OUTPUT_DIR = os.path.join("cleaned_data", "DL")
EDA_OUTPUT_DIR = "eda_outputs/dl_cleaning"

# Preprocessing settings
MIN_TOKEN_LENGTH = 5
MAX_TOKEN_LENGTH = 2000
NUM_WORKERS = multiprocessing.cpu_count()

# EDA settings
sns.set(style="whitegrid", context="notebook")


# =========================
# Step B1: Text Cleaning (Gentle for DL/Transformers)
# =========================

def clean_email_text_dl(text: object) -> str:
    """
    Gentle cleaning function for the Transformer (DL) pipeline.
    
    CRITICAL: Does NOT lowercase or remove punctuation.
    Preserves case and punctuation for transformer tokenizers.
    """
    if not isinstance(text, str):
        return ""

    # 1. Fix Encoding Artifacts (Quoted-Printable)
    try:
        text_bytes = text.encode("latin-1", errors="ignore")
        decoded_bytes = quopri.decodestring(text_bytes)
        text = decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        pass

    # 2. Strip HTML Tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # 3. Replace URLs with [URL]
    # We add spaces around tokens to ensure they are tokenized correctly
    text = re.sub(r"(https?://\S+|www\.\S+)", " [URL] ", text)

    # 4. Replace Emails with [EMAIL]
    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", " [EMAIL] ", text)

    # 5. Normalize Whitespace (replace \n, \t, etc. with a single space)
    text = re.sub(r"\s+", " ", text).strip()

    # --- NO lowercasing, NO punctuation removal ---

    return text


def clean_dataset_dl(
    input_filename: str,
    output_filename: str,
    workers: int,
) -> pd.DataFrame:
    """
    Loads the master dataset and applies the gentle DL cleaning.
    """
    print(f"\n--- Step B1: DL Text Cleaning ---")
    print(f"Loading: {input_filename}")
    try:
        df = pd.read_csv(input_filename)
    except Exception as e:
        print(f"Error loading {input_filename}. Make sure it exists. Error: {e}")
        return pd.DataFrame()

    df = df.dropna(subset=["text"])
    df = df.copy()

    print(f"Cleaning text using {workers} workers...")
    df["text"] = df["text"].fillna("")
    results = process_map(
        clean_email_text_dl,
        df["text"],
        max_workers=workers,
        chunksize=500,
        desc="DL Cleaning",
    )
    df["text"] = results

    # Text Integrity Validation
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    dropped = before - len(df)
    if dropped > 0:
        print(f" - Dropped {dropped} rows that became empty after cleaning.")

    # Reorder columns
    first_cols = ["text", "label"]
    for col in ["source_file", "source_name", "record_id"]:
        if col in df.columns:
            first_cols.append(col)

    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    print(f" - Saved DL cleaned: {output_filename} ({len(df)} rows)")
    print("--- Step B1 Complete ---")
    return df


# =========================
# Step B2: Deduplication
# =========================

def deduplicate_dataset_dl(
    df: pd.DataFrame,
    output_filename: str,
) -> pd.DataFrame:
    """
    Deduplicates the cleaned DL dataset.
    """
    print(f"\n--- Step B2: Deduplication ---")
    df = df.dropna(subset=["text"])
    before = len(df)

    df = df.drop_duplicates(subset=["text"], keep="first")
    after = len(df)

    print("\n==================================")
    print("Deduplication Report")
    print(f"  Rows before: {before}")
    print(f"  Rows after:  {after}")
    print(f"  Removed:     {before - after}")
    print("==================================")

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    print(f" - Saved DL deduplicated: {output_filename} ({after} unique rows)")
    print("--- Step B2 Complete ---")
    return df


# =========================
# Step B3: Length Filtering
# =========================

def filter_by_length_dl(
    df: pd.DataFrame,
    output_filename: str,
    min_len: int,
    max_len: int,
) -> pd.DataFrame:
    """
    Filters the dataset based on token count.
    Uses 2000 as a loose upper bound; tokenizer will handle final truncation to 512.
    """
    print(f"\n--- Step B3: Length Filtering ---")
    print(f"Filtering rows with token count < {min_len} or > {max_len}...")

    # Calculate token count (simple split on space)
    df["token_count"] = df["text"].str.split().str.len()
    before = len(df)

    df = df[(df["token_count"] >= min_len) & (df["token_count"] <= max_len)]

    after = len(df)
    dropped = before - after

    print("\n==================================")
    print("Length Filtering Report")
    print(f"  Rows before: {before}")
    print(f"  Rows after:  {after}")
    print(f"  Removed:     {dropped}")
    print("==================================")

    df = df.drop(columns=["token_count"])

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    print(f" - Saved DL final: {output_filename} ({after} rows)")
    print("--- Step B3 Complete ---")
    return df


# =========================
# EDA Functions (adapted for DL)
# =========================

def setup_nltk():
    """Download NLTK stopwords if not already available."""
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception as e:
        print(f"Warning: Could not download NLTK stopwords: {e}")
        return set()


def plot_target_distribution(df: pd.DataFrame, output_dir: str):
    """Plot target label distribution."""
    print("\n--- EDA: Target Distribution ---")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x="label")
    ax.set_title("Distribution of Labels (0=Not Phishing, 1=Phishing) - DL Dataset")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    # Ensure axis labels and title are not cut off in saved figure
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "dl_target_label_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(" - Saved: dl_target_label_distribution.png")


def plot_combined_target_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
):
    """Plot combined target label distribution for train, validation, and test in 3 subplots."""
    print("\n--- EDA: Combined Target Distribution (All Splits) ---")
    fig, axes = plt.subplots(1, 3, figsize=(5.85, 4.60))
    
    splits = [
        (train_df, "Train", axes[0]),
        (val_df, "Validation", axes[1]),
        (test_df, "Test", axes[2]),
    ]
    
    for df, split_name, ax in splits:
        if df is not None and not df.empty:
            sns.countplot(data=df, x="label", ax=ax)
            ax.set_title(f"{split_name} Split")
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
        else:
            ax.text(0.5, 0.5, f"{split_name} data not available", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{split_name} Split")
    
    plt.suptitle("Label Distribution Across Splits (0=Not Phishing, 1=Phishing)", 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "dl_combined_target_label_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(" - Saved: dl_combined_target_label_distribution.png")


def engineer_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer metadata features from text.
    
    NOTE: These features are ONLY for EDA visualization purposes.
    They are NOT used in the DL training pipeline, which relies
    solely on learned features from BERT embeddings.
    """
    print("\n--- EDA: Engineering Metadata Features (for visualization only) ---")
    df = df.copy()
    df['text_length'] = df['text'].apply(lambda x: len(str(x)))
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['punct_count'] = df['text'].apply(
        lambda x: sum([1 for char in str(x) if char in string.punctuation])
    )
    df['upper_count'] = df['text'].apply(
        lambda x: len([word for word in str(x).split() if word.isupper()])
    )
    print(" - Metadata features created.")
    return df


def plot_metadata_distributions(df: pd.DataFrame, output_dir: str):
    """Plot distributions of metadata features."""
    print("\n--- EDA: Metadata Feature Distributions ---")
    df['label_name'] = df['label'].map({0: 'Not Phishing (0)', 1: 'Phishing (1)'})

    meta_features = ['text_length', 'word_count', 'punct_count', 'upper_count']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, feature in enumerate(meta_features):
        sns.histplot(
            data=df, x=feature, hue='label_name', kde=False,
            ax=axes[i], bins=50, element="step"
        )
        # Truncate x-axis at 99th percentile for better readability
        q99 = df[feature].quantile(0.99)
        if q99 > 0:
            axes[i].set_xlim(0, q99)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature.replace('_', ' ').title())

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dl_metadata_histograms.png"), dpi=150)
    plt.close()
    print(" - Saved: dl_metadata_histograms.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot correlation heatmap of engineered features."""
    print("\n--- EDA: Correlation Heatmap ---")
    numeric_features = ['label', 'text_length', 'word_count', 'punct_count', 'upper_count']
    corr_matrix = df[numeric_features].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title('Correlation of Engineered Features and Label - DL Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dl_engineered_features_heatmap.png"), dpi=150)
    plt.close()
    print(" - Saved: dl_engineered_features_heatmap.png")


def plot_source_file_analysis(df: pd.DataFrame, output_dir: str):
    """Plot label distribution by source file."""
    print("\n--- EDA: Source File Analysis ---")
    if 'source_file' not in df.columns:
        print(" - Skipping: source_file column not found")
        return

    top_sources = df['source_file'].value_counts().head(10).index
    df_top_sources = df[df['source_file'].isin(top_sources)]

    plt.figure(figsize=(10, 8))
    sns.countplot(data=df_top_sources, y='source_file', hue='label', order=top_sources)
    plt.title('Label Distribution by Top 10 Source Files - DL Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dl_source_file_by_label.png"), dpi=150)
    plt.close()
    print(" - Saved: dl_source_file_by_label.png")


def get_top_ngrams(corpus, n_gram_range=(1, 1), top_k=20):
    """Get top n-grams from corpus."""
    vec = CountVectorizer(ngram_range=n_gram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_k]


def plot_top_ngrams(ngram_list, title, ax):
    """Plot top n-grams."""
    ngrams = [item[0] for item in ngram_list]
    frequencies = [item[1] for item in ngram_list]
    sns.barplot(x=frequencies, y=ngrams, ax=ax, palette='viridis')
    ax.set_title(title)


def plot_ngram_analysis(df: pd.DataFrame, output_dir: str):
    """Plot n-gram analysis for phishing vs non-phishing."""
    print("\n--- EDA: N-Gram Analysis ---")
    spam_corpus = df[df['label'] == 1]['text'].astype(str)
    ham_corpus = df[df['label'] == 0]['text'].astype(str)

    # Get top n-grams
    top_spam_unigrams = get_top_ngrams(spam_corpus, n_gram_range=(1, 1), top_k=20)
    top_ham_unigrams = get_top_ngrams(ham_corpus, n_gram_range=(1, 1), top_k=20)
    top_spam_bigrams = get_top_ngrams(spam_corpus, n_gram_range=(2, 2), top_k=20)
    top_ham_bigrams = get_top_ngrams(ham_corpus, n_gram_range=(2, 2), top_k=20)

    # Plot n-grams
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    plot_top_ngrams(top_spam_unigrams, 'Top 20 Phishing Unigrams (Label 1)', axes[0, 0])
    plot_top_ngrams(top_ham_unigrams, 'Top 20 Not Phishing Unigrams (Label 0)', axes[0, 1])
    plot_top_ngrams(top_spam_bigrams, 'Top 20 Phishing Bigrams (Label 1)', axes[1, 0])
    plot_top_ngrams(top_ham_bigrams, 'Top 20 Not Phishing Bigrams (Label 0)', axes[1, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dl_ngram_analysis.png"), dpi=150)
    plt.close()
    print(" - Saved: dl_ngram_analysis.png")


def plot_word_clouds(df: pd.DataFrame, output_dir: str, stop_words_set: set):
    """Generate word clouds for phishing vs non-phishing."""
    print("\n--- EDA: Word Clouds ---")
    spam_corpus = df[df['label'] == 1]['text'].astype(str)
    ham_corpus = df[df['label'] == 0]['text'].astype(str)

    spam_text = " ".join(text for text in spam_corpus)
    ham_text = " ".join(text for text in ham_corpus)

    if spam_text and ham_text:
        try:
            wordcloud_spam = WordCloud(
                stopwords=stop_words_set, background_color="white",
                max_words=100, width=800, height=400
            ).generate(spam_text)
            wordcloud_ham = WordCloud(
                stopwords=stop_words_set, background_color="white",
                max_words=100, width=800, height=400
            ).generate(ham_text)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(wordcloud_spam, interpolation='bilinear')
            ax1.set_title('Phishing (Label 1)', fontsize=20)
            ax1.axis("off")

            ax2.imshow(wordcloud_ham, interpolation='bilinear')
            ax2.set_title('Not Phishing (Label 0)', fontsize=20)
            ax2.axis("off")

            plt.savefig(os.path.join(output_dir, "dl_word_clouds.png"), dpi=150)
            plt.close()
            print(" - Saved: dl_word_clouds.png")
        except Exception as e:
            print(f" - Skipping word clouds due to error: {e}")
    else:
        print(" - Skipping word clouds (empty corpus)")


def plot_combined_word_clouds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    stop_words_set: set,
):
    """Generate combined word clouds for train, validation, and test with separate labels (6 subplots)."""
    print("\n--- EDA: Combined Word Clouds (All Splits, Separated by Label) ---")
    # 3 rows (splits) × 2 columns (labels) = 6 subplots
    fig, axes = plt.subplots(3, 2, figsize=(5.85, 4.60))
    
    splits = [
        (train_df, "Train", axes[0, :]),
        (val_df, "Validation", axes[1, :]),
        (test_df, "Test", axes[2, :]),
    ]
    
    for df, split_name, (ax0, ax1) in splits:
        if df is not None and not df.empty:
            # Separate text by label
            label_0_text = " ".join(df[df['label'] == 0]['text'].astype(str))
            label_1_text = " ".join(df[df['label'] == 1]['text'].astype(str))
            
            # Label 0 (Not Phishing) word cloud
            if label_0_text:
                try:
                    wordcloud_0 = WordCloud(
                        stopwords=stop_words_set,
                        background_color="white",
                        max_words=100,
                        width=400,
                        height=300,
                    ).generate(label_0_text)
                    
                    ax0.imshow(wordcloud_0, interpolation='bilinear')
                    ax0.set_title(f"{split_name} - Not Phishing (Label 0)", fontsize=9)
                    ax0.axis("off")
                except Exception as e:
                    ax0.text(0.5, 0.5, f"Error:\n{e}", 
                           ha="center", va="center", transform=ax0.transAxes, fontsize=7)
                    ax0.set_title(f"{split_name} - Not Phishing (Label 0)", fontsize=9)
                    ax0.axis("off")
            else:
                ax0.text(0.5, 0.5, f"{split_name}\nLabel 0\nempty", 
                       ha="center", va="center", transform=ax0.transAxes, fontsize=7)
                ax0.set_title(f"{split_name} - Not Phishing (Label 0)", fontsize=9)
                ax0.axis("off")
            
            # Label 1 (Phishing) word cloud
            if label_1_text:
                try:
                    wordcloud_1 = WordCloud(
                        stopwords=stop_words_set,
                        background_color="white",
                        max_words=100,
                        width=400,
                        height=300,
                    ).generate(label_1_text)
                    
                    ax1.imshow(wordcloud_1, interpolation='bilinear')
                    ax1.set_title(f"{split_name} - Phishing (Label 1)", fontsize=9)
                    ax1.axis("off")
                except Exception as e:
                    ax1.text(0.5, 0.5, f"Error:\n{e}", 
                           ha="center", va="center", transform=ax1.transAxes, fontsize=7)
                    ax1.set_title(f"{split_name} - Phishing (Label 1)", fontsize=9)
                    ax1.axis("off")
            else:
                ax1.text(0.5, 0.5, f"{split_name}\nLabel 1\nempty", 
                       ha="center", va="center", transform=ax1.transAxes, fontsize=7)
                ax1.set_title(f"{split_name} - Phishing (Label 1)", fontsize=9)
                ax1.axis("off")
        else:
            # Both subplots show "not available" message
            ax0.text(0.5, 0.5, f"{split_name}\ndata\nnot available", 
                   ha="center", va="center", transform=ax0.transAxes, fontsize=7)
            ax0.set_title(f"{split_name} - Not Phishing (Label 0)", fontsize=9)
            ax0.axis("off")
            
            ax1.text(0.5, 0.5, f"{split_name}\ndata\nnot available", 
                   ha="center", va="center", transform=ax1.transAxes, fontsize=7)
            ax1.set_title(f"{split_name} - Phishing (Label 1)", fontsize=9)
            ax1.axis("off")
    
    plt.suptitle("Word Clouds Across Splits (Separated by Label)", 
                 fontsize=12, y=0.995)
    plt.tight_layout()
    # Increased DPI for higher resolution (300 instead of 150)
    plt.savefig(
        os.path.join(output_dir, "dl_combined_word_clouds.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(" - Saved: dl_combined_word_clouds.png")


def run_eda(df: pd.DataFrame, output_dir: str):
    """Run all EDA visualizations."""
    print("\n" + "=" * 60)
    print("Starting Exploratory Data Analysis (DL Dataset)")
    print("=" * 60)

    # Setup
    os.makedirs(output_dir, exist_ok=True)
    stop_words_set = setup_nltk()

    # Load data
    print(f"\n--- Loading Data ---")
    print(f"Dataset shape: {df.shape}")

    # Run EDA steps
    plot_target_distribution(df, output_dir)
    df = engineer_metadata_features(df)
    plot_metadata_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_source_file_analysis(df, output_dir)
    plot_ngram_analysis(df, output_dir)
    plot_word_clouds(df, output_dir, stop_words_set)

    print("\n" + "=" * 60)
    print("EDA Complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


# ===================================================================
# Main Execution
# ===================================================================

def process_split(input_file, split_name, output_dir):
    """Process a single split (train/validation/test) through the DL preprocessing pipeline."""
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name.upper()} Split")
    print(f"{'=' * 60}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths for this split
    cleaned_output = os.path.join(output_dir, f"{split_name}_cleaned.csv")
    deduped_output = os.path.join(output_dir, f"{split_name}_deduped.csv")
    # Final output goes to organized subdirectory
    final_output = os.path.join(output_dir, f"{split_name}_split.csv")
    
    # Step B1: Clean
    cleaned_df = clean_dataset_dl(
        input_filename=input_file,
        output_filename=cleaned_output,
        workers=NUM_WORKERS
    )
    
    if cleaned_df.empty:
        print(f"\n{split_name.capitalize()} split processing stopped: Could not load or clean input file.")
        return None
    
    # Step B2: Deduplicate
    deduped_df = deduplicate_dataset_dl(
        cleaned_df,
        output_filename=deduped_output
    )
    
    # Step B3: Filter by Length
    final_df = filter_by_length_dl(
        deduped_df,
        output_filename=final_output,
        min_len=MIN_TOKEN_LENGTH,
        max_len=MAX_TOKEN_LENGTH
    )
    
    print(f"\n{split_name.capitalize()} split processing complete: {len(final_df)} rows")
    return final_df


def main():
    """Main execution function for DL preprocessing and EDA."""
    print("=" * 60)
    print("DL Preprocessing and EDA Pipeline")
    print("=" * 60)

    multiprocessing.freeze_support()

    # Process each split - save to DL subdirectories
    train_output_dir = os.path.join(OUTPUT_DIR, "train")
    val_output_dir = os.path.join(OUTPUT_DIR, "validation")
    test_output_dir = os.path.join(OUTPUT_DIR, "test")
    
    train_df = process_split(TRAIN_INPUT, "train", train_output_dir)
    val_df = process_split(VAL_INPUT, "validation", val_output_dir)
    test_df = process_split(TEST_INPUT, "test", test_output_dir)
    
    # Run EDA on each split separately
    print("\n" + "=" * 60)
    print("DL Preprocessing Pipeline (B1-B3) Complete")
    
    if train_df is not None:
        print(f"Train rows: {len(train_df)}")
        train_eda_dir = os.path.join(EDA_OUTPUT_DIR, "train")
        print(f"\nRunning EDA on train split...")
        run_eda(train_df, train_eda_dir)
    
    if val_df is not None:
        print(f"Validation rows: {len(val_df)}")
        val_eda_dir = os.path.join(EDA_OUTPUT_DIR, "validation")
        print(f"\nRunning EDA on validation split...")
        run_eda(val_df, val_eda_dir)
    
    if test_df is not None:
        print(f"Test rows: {len(test_df)}")
        test_eda_dir = os.path.join(EDA_OUTPUT_DIR, "test")
        print(f"\nRunning EDA on test split...")
        run_eda(test_df, test_eda_dir)
    
    if train_df is not None and val_df is not None and test_df is not None:
        total_rows = len(train_df) + len(val_df) + len(test_df)
        print(f"\nTotal rows: {total_rows}")
        
        # Create combined plots for all splits
        print("\n" + "=" * 60)
        print("Creating Combined EDA Visualizations")
        print("=" * 60)
        
        # Use the main EDA output directory for combined plots
        combined_output_dir = EDA_OUTPUT_DIR
        os.makedirs(combined_output_dir, exist_ok=True)
        stop_words_set = setup_nltk()
        
        # Combined target distribution
        plot_combined_target_distribution(train_df, val_df, test_df, combined_output_dir)
        
        # Combined word clouds
        plot_combined_word_clouds(train_df, val_df, test_df, combined_output_dir, stop_words_set)
        
        print("\nCombined EDA visualizations complete!")
    else:
        print("\nDL Pipeline stopped: One or more splits failed to process.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

