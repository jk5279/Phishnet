"""
01_data_aggregation.py

Data Aggregation and Preliminary Preprocessing Pipeline

This script performs the initial data aggregation and preprocessing steps:
1. Combines raw CSV files from multiple directories
2. Consolidates text/label columns and standardizes labels
3. Applies initial text cleaning (HTML removal, URL/email replacement)
4. Deduplicates the dataset

Output: master_email_dataset_final.csv
"""

import os
import glob
import csv
import sys
import re
import quopri
import multiprocessing
from typing import List, Dict, Optional

import pandas as pd
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
from tqdm.contrib.concurrent import process_map
import warnings

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# =========================
# Configuration
# =========================

# Input/Output paths
RAW_DATA_DIR = "Dataset/raw - DO NOT OVERWRITE"
OUTPUT_DIR = "cleaned_data"
COMBINED_OUTPUT = os.path.join(OUTPUT_DIR, "combined_emails.csv")
MASTER_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset.csv")
CLEANED_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset_cleaned.csv")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset_final.csv")

# Column mapping configuration
COLUMN_CONFIG = {
    "text_columns": [
        "Email Text",
        "Text",
        "body",
    ],
    "label_columns": [
        "Email Type",
        "Class",
        "label",
    ],
}

# Label standardization mapping
LABEL_MAP = {
    "Phishing Email": 1,
    "1.0": 1,
    "1": 1,
    "spam": 1,
    "Safe Email": 0,
    "0.0": 0,
    "0": 0,
    "ham": 0,
}

# Processing settings
NUM_WORKERS = multiprocessing.cpu_count()


# =========================
# Helper Functions
# =========================

def set_max_csv_field_size_limit() -> None:
    """Set the CSV field size limit as high as possible."""
    print("Setting maximum CSV field size limit...")
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
    print(f" - CSV field size limit set to {max_int}")


def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Read a CSV using robust fallbacks for different encodings and formats."""
    try:
        df = pd.read_csv(path, engine="c")
        print(f"   - Read OK (C): {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"   - C engine failed for {os.path.basename(path)}: {e}")

    try:
        print("     - Retrying C + latin1 + on_bad_lines='skip'...")
        df = pd.read_csv(path, engine="c", encoding="latin1", on_bad_lines="skip")
        print(f"     - Read OK (C+latin1+skip): {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"     - Fallback 1 failed: {e}")

    try:
        print("     - Retrying PYTHON engine...")
        df = pd.read_csv(path, engine="python")
        print(f"     - Read OK (python): {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"     - Fallback 2 failed: {e}")

    try:
        print("     - Retrying PYTHON + latin1...")
        df = pd.read_csv(path, engine="python", encoding="latin1")
        print(f"     - Read OK (python+latin1): {os.path.basename(path)}")
        return df
    except Exception as e:
        print(f"     - All fallbacks failed for {os.path.basename(path)}: {e}")
        return None


def is_git_lfs_pointer(df: pd.DataFrame) -> bool:
    """Check if DataFrame appears to be a Git LFS pointer file."""
    if df is None or df.empty:
        return False
    # Git LFS pointer files have a column starting with "version https://git-lfs.github.com/spec/v1"
    columns = list(df.columns)
    return any('version https://git-lfs.github.com/spec/v1' in str(col) for col in columns)


def consolidate_column(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """Overlay multiple columns left-to-right to produce one series."""
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        print(f"   - WARNING: None of {candidates} found. Returning empty column.")
        return pd.Series(index=df.index, dtype=object)

    print(f"   - Consolidating columns: {existing}")
    col = df[existing[0]].copy()
    for c in existing[1:]:
        col = col.fillna(df[c])
    return col


# =========================
# Step 1: Combine raw CSVs
# =========================

def combine_csvs_from_directory(root_directory: str, output_filename: str) -> pd.DataFrame:
    """
    Combine all CSV files from a directory tree into a single DataFrame.
    
    Args:
        root_directory: Root directory containing CSV files
        output_filename: Output CSV file path
        
    Returns:
        Combined DataFrame
    """
    if not os.path.isdir(root_directory):
        raise FileNotFoundError(f"Directory not found: {root_directory}")

    print(f"\n--- Step 1: Combining CSV Files ---")
    print(f"Scanning for CSVs under: {root_directory}")
    pattern = os.path.join(root_directory, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {root_directory}")

    print(f"Found {len(files)} CSV files. Reading...")
    dfs: List[pd.DataFrame] = []
    git_lfs_files: List[str] = []

    for f in files:
        df = safe_read_csv(f)
        if df is not None:
            # Check if this is a Git LFS pointer file
            if is_git_lfs_pointer(df):
                git_lfs_files.append(os.path.basename(f))
                print(f"   - WARNING: {os.path.basename(f)} appears to be a Git LFS pointer file (not actual data)")
                continue
            
            # Add provenance columns
            df["source_file"] = os.path.basename(f)
            df["record_id"] = df.index

            # Get relative path of the file's directory from the root
            relative_dir_path = os.path.relpath(os.path.dirname(f), root_directory)

            if relative_dir_path == ".":
                # File is in the root, use root directory's name as source_name
                df["source_name"] = os.path.basename(os.path.normpath(root_directory))
            else:
                # File is in a subdirectory, use the first-level subdirectory name
                df["source_name"] = relative_dir_path.split(os.sep)[0]

            dfs.append(df)
    
    # Warn about Git LFS files
    if git_lfs_files:
        print(f"\n⚠️  WARNING: Found {len(git_lfs_files)} Git LFS pointer file(s):")
        for fname in git_lfs_files[:10]:  # Show first 10
            print(f"   - {fname}")
        if len(git_lfs_files) > 10:
            print(f"   ... and {len(git_lfs_files) - 10} more")
        print("\nThese files are Git LFS pointers, not actual data files.")
        print("To download the actual files, run: git lfs pull")
        print("Or configure Git LFS and fetch the files from your repository.")

    if not dfs:
        if git_lfs_files:
            error_msg = (
                "\n" + "=" * 60 + "\n"
                "ERROR: No valid CSV data files could be read.\n"
                "All CSV files appear to be Git LFS pointer files.\n\n"
                "SOLUTION: Download the actual data files using Git LFS:\n"
                "  1. Make sure Git LFS is installed: git lfs install\n"
                "  2. Pull the actual files: git lfs pull\n"
                "  3. Run this script again\n"
                + "=" * 60 + "\n"
            )
            raise RuntimeError(error_msg)
        raise RuntimeError("No data could be read from any CSVs.")

    print("Combining dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    
    # Show available columns for debugging
    print(f"\n--- Available columns in combined dataset ---")
    print(f"Columns: {list(combined.columns)}")
    print(f"Total columns: {len(combined.columns)}")
    
    # Check if we have expected data columns
    expected_text_cols = ['Email Text', 'Text', 'body']
    expected_label_cols = ['Email Type', 'Class', 'label']
    has_text_cols = any(col in combined.columns for col in expected_text_cols)
    has_label_cols = any(col in combined.columns for col in expected_label_cols)
    
    if not has_text_cols or not has_label_cols:
        print("\n⚠️  WARNING: Expected columns not found!")
        print(f"   Looking for TEXT columns: {expected_text_cols}")
        print(f"   Looking for LABEL columns: {expected_label_cols}")
        print("\n   If your CSV files use different column names, update COLUMN_CONFIG in the script.")
        print("   Or check if the files are Git LFS pointer files (run 'git lfs pull' to download actual data).")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    combined.to_csv(output_filename, index=False)
    print(f" - Combined rows: {len(combined)}")
    print(f" - Saved: {output_filename}")
    return combined


# ============================================
# Step 2: Consolidate text/labels and standardize
# ============================================

def process_to_master(
    df: pd.DataFrame,
    output_filename: str,
    column_config: Dict[str, List[str]],
    label_map: Dict[str, int],
) -> pd.DataFrame:
    """
    Consolidate text/label columns and standardize labels.
    
    Args:
        df: Input DataFrame
        output_filename: Output CSV file path
        column_config: Configuration for text and label columns
        label_map: Mapping for label standardization
        
    Returns:
        Processed DataFrame
    """
    print(f"\n--- Step 2: Processing to Master Dataset ---")
    print("Processing to master dataset...")
    
    # Show available columns for debugging
    print(f"\nAvailable columns in dataframe: {list(df.columns)}")
    print(f"Looking for TEXT columns: {column_config['text_columns']}")
    print(f"Looking for LABEL columns: {column_config['label_columns']}")
    
    print("[STEP 1] Consolidating TEXT columns...")
    master_text = consolidate_column(df, column_config["text_columns"])

    print("[STEP 2] Consolidating LABEL columns...")
    master_label_src = consolidate_column(df, column_config["label_columns"])

    print("[STEP 3] Standardizing labels...")
    normalized_map = {str(k).lower().strip(): v for k, v in label_map.items()}
    normalized_labels = master_label_src.astype(str).str.lower().str.strip()
    master_label = normalized_labels.map(normalized_map)
    print("   - Label mapping done.")

    print("[STEP 4] Creating final frame and cleaning...")
    final_df = pd.DataFrame({"text": master_text, "label": master_label})

    # Propagate provenance columns
    for col in ["source_file", "source_name", "record_id"]:
        if col in df.columns:
            final_df[col] = df[col]

    total = len(final_df)
    na_text = final_df["text"].isna().sum()
    na_label = final_df["label"].isna().sum()
    print("\n--- Processing Report ---")
    print(f"Total rows read: {total}")
    print(f"Rows with missing text: {na_text}")
    print(f"Rows with unmapped labels: {na_label}")

    # Show unmapped original labels (help extend LABEL_MAP)
    if na_label > 0:
        unmapped_vals = master_label_src[final_df["label"].isna()].dropna().unique()
        print("\n  > Unmapped label values (add to LABEL_MAP if needed):")
        for v in unmapped_vals[:20]:
            print(f"    - '{v}' (Type: {type(v)})")
        if len(unmapped_vals) > 20:
            print(f"    ... and {len(unmapped_vals) - 20} more")

    print("\n[STEP 5] Dropping rows with missing text or labels...")
    before = len(final_df)
    final_df = final_df.dropna(subset=["text", "label"])
    after = len(final_df)
    print(f"   - Dropped {before - after} rows.")
    if after == 0:
        print("\n" + "=" * 60)
        print("ERROR: Master dataset empty after cleaning.")
        print("=" * 60)
        print("\nPossible issues:")
        print("1. Column names don't match expected values")
        print("2. All text/label columns are empty")
        print("\nTo fix this:")
        print("- Check the available columns shown above")
        print("- Update COLUMN_CONFIG in the script to match your CSV column names")
        print("- Check if the CSV files have the expected data")
        print("=" * 60)
        raise RuntimeError("Master dataset empty after cleaning. Check column names and data.")

    final_df["label"] = final_df["label"].astype(int)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    final_df.to_csv(output_filename, index=False)
    print(f" - Saved master: {output_filename} ({len(final_df)} rows)")
    return final_df


# =========================
# Step 3: Text cleaning
# =========================

def clean_email_text(text: object) -> str:
    """
    Clean email text by removing HTML, normalizing URLs/emails, and standardizing format.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Decode quoted-printable artifacts
    try:
        text_bytes = text.encode("latin-1", errors="ignore")
        decoded_bytes = quopri.decodestring(text_bytes)
        text = decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        pass

    # Strip HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # Lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(r"(https?://\S+|www\.\S+)", "[url]", text)

    # Replace emails
    text = re.sub(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", "[email]", text)

    # Keep alnum, space, and []
    text = re.sub(r"[^a-z0-9\s\[\]]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataset(
    df: pd.DataFrame,
    output_filename: str,
    workers: int,
) -> pd.DataFrame:
    """
    Apply text cleaning to the entire dataset using multiprocessing.
    
    Args:
        df: Input DataFrame
        output_filename: Output CSV file path
        workers: Number of worker processes
        
    Returns:
        Cleaned DataFrame
    """
    print(f"\n--- Step 3: Text Cleaning ---")
    print(f"Cleaning text using {workers} workers...")
    df = df.copy()
    df["text"] = df["text"].fillna("")
    results = process_map(
        clean_email_text,
        df["text"],
        max_workers=workers,
        chunksize=500,
        desc="Cleaning Emails",
    )
    df["text"] = results

    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    dropped = before - len(df)
    if dropped > 0:
        print(f" - Dropped {dropped} rows that became empty after cleaning.")

    # Reorder columns to include provenance columns
    first_cols = ["text", "label"]
    for col in ["source_file", "source_name", "record_id"]:
        if col in df.columns:
            first_cols.append(col)

    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    print(f" - Saved cleaned: {output_filename} ({len(df)} rows)")
    return df


# =========================
# Step 4: Deduplicate
# =========================

def deduplicate_dataset(
    df: pd.DataFrame,
    output_filename: str,
) -> pd.DataFrame:
    """
    Remove duplicate texts from the dataset.
    
    Args:
        df: Input DataFrame
        output_filename: Output CSV file path
        
    Returns:
        Deduplicated DataFrame
    """
    print(f"\n--- Step 4: Deduplication ---")
    print("Deduplicating by 'text' column...")
    df = df.dropna(subset=["text"])
    before = len(df)
    # Keeps the first occurrence and its metadata (source_file, etc.)
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
    print(f" - Saved final: {output_filename} ({after} unique rows)")
    return df


# ===================================================================
# Main Execution
# ===================================================================

def main():
    """Main execution function for the data aggregation pipeline."""
    print("=" * 60)
    print("Data Aggregation and Preliminary Preprocessing Pipeline")
    print("=" * 60)
    
    # Set up multiprocessing
    multiprocessing.freeze_support()
    
    # Set CSV field size limit
    set_max_csv_field_size_limit()

    # Step 1: Combine CSV files
    combined_df = combine_csvs_from_directory(RAW_DATA_DIR, COMBINED_OUTPUT)

    # Step 2: Consolidate + map labels
    master_df = process_to_master(
        combined_df, MASTER_OUTPUT, COLUMN_CONFIG, LABEL_MAP
    )

    # Step 3: Clean text
    cleaned_df = clean_dataset(master_df, CLEANED_OUTPUT, NUM_WORKERS)

    # Step 4: Deduplicate
    final_df = deduplicate_dataset(cleaned_df, FINAL_OUTPUT)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Final dataset: {FINAL_OUTPUT}")
    print(f"Total rows: {len(final_df)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

