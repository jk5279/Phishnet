"""
03_dl_preprocessing_eda.py

Preprocessing for Deep Learning Pipeline and Exploratory Data Analysis

This script performs DL-specific preprocessing (gentle cleaning to preserve case/punctuation)
and generates exploratory data analysis visualizations:
1. Gentle text cleaning (preserves case and punctuation for transformers)
2. Deduplication
3. Length filtering (5-2000 tokens)
4. EDA visualizations (distributions, correlations, n-grams, word clouds)

Output: cleaned_data/dl_dataset_final.csv and eda_outputs/dl_*.png
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
INPUT_FILE = os.path.join("cleaned_data", "master_email_dataset_final.csv")
OUTPUT_DIR = "cleaned_data"
EDA_OUTPUT_DIR = "eda_outputs"
CLEANED_OUTPUT = os.path.join(OUTPUT_DIR, "dl_dataset_cleaned.csv")
DEDUPED_OUTPUT = os.path.join(OUTPUT_DIR, "dl_dataset_deduped.csv")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "dl_dataset_final.csv")

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
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label')
    plt.title('Distribution of Labels (0=Not Phishing, 1=Phishing) - DL Dataset')
    plt.savefig(os.path.join(output_dir, "dl_target_label_distribution.png"), dpi=150)
    plt.close()
    print(" - Saved: dl_target_label_distribution.png")


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

def main():
    """Main execution function for DL preprocessing and EDA."""
    print("=" * 60)
    print("DL Preprocessing and EDA Pipeline")
    print("=" * 60)

    multiprocessing.freeze_support()

    # Step B1: Clean
    cleaned_dl_df = clean_dataset_dl(
        input_filename=INPUT_FILE,
        output_filename=CLEANED_OUTPUT,
        workers=NUM_WORKERS
    )

    if cleaned_dl_df.empty:
        print("\nDL Pipeline stopped: Could not load or clean input file.")
        return

    # Step B2: Deduplicate
    deduped_dl_df = deduplicate_dataset_dl(
        cleaned_dl_df,
        output_filename=DEDUPED_OUTPUT
    )

    # Step B3: Filter by Length
    final_dl_df = filter_by_length_dl(
        deduped_dl_df,
        output_filename=FINAL_OUTPUT,
        min_len=MIN_TOKEN_LENGTH,
        max_len=MAX_TOKEN_LENGTH
    )

    print("\n" + "=" * 60)
    print("DL Preprocessing Pipeline (B1-B3) Complete")
    print(f"Final dataset: {FINAL_OUTPUT}")
    print(f"Total rows: {len(final_dl_df)}")
    print("=" * 60)

    # Run EDA
    run_eda(final_dl_df, EDA_OUTPUT_DIR)


if __name__ == "__main__":
    main()

