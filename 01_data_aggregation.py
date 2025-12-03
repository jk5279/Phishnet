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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# =========================
# Configuration
# =========================

# Input/Output paths
RAW_DATA_DIR = "datasets/raw - DO NOT OVERWRITE"
TEST_DATASETS_DIR = os.path.join(RAW_DATA_DIR, "test_datasets")
OUTPUT_DIR = "cleaned_data"

# Intermediate outputs (optional/debug)
COMBINED_OUTPUT = os.path.join(OUTPUT_DIR, "combined_emails.csv")
MASTER_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset.csv")
CLEANED_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset_cleaned.csv")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "master_email_dataset_final.csv")

# New pipeline outputs
DATASET1_OUTPUT = os.path.join(OUTPUT_DIR, "dataset1_processed.csv")
DATASET2_OUTPUT = os.path.join(OUTPUT_DIR, "dataset2_processed.csv")
DATASET3_OUTPUT = os.path.join(OUTPUT_DIR, "dataset3_combined.csv")
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, "train", "train_split.csv")
VAL_OUTPUT = os.path.join(OUTPUT_DIR, "validation", "validation_split.csv")
TEST_OUTPUT = os.path.join(OUTPUT_DIR, "test", "test_split.csv")

# Processing thresholds
MIN_TEXT_LENGTH = 50

# Dataset size configuration
TARGET_DATASET_SIZE = 100000

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/Validation/Test split ratios
TRAIN_SPLIT_RATIO = 0.70
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
# Note: VAL_SPLIT_RATIO + TEST_SPLIT_RATIO = 0.30 (used for first split)
TEMP_SPLIT_RATIO = VAL_SPLIT_RATIO + TEST_SPLIT_RATIO  # 0.30
VAL_TEST_SPLIT_RATIO = 0.50  # Split temp into equal val/test (15% each)

# Multiprocessing configuration
CHUNKSIZE = 500  # Chunk size for parallel processing

# CSV field size limit configuration
CSV_FIELD_SIZE_DIVISOR = 10  # Divisor for reducing field size limit on overflow

# Display limits
MAX_GIT_LFS_FILES_TO_SHOW = 10  # Number of Git LFS files to display
MAX_UNMAPPED_LABELS_TO_SHOW = 20  # Number of unmapped labels to display

# Display formatting
SEPARATOR_LENGTH_LONG = 60  # Length of separator lines for major sections
SEPARATOR_LENGTH_SHORT = 40  # Length of separator lines for minor sections

# Shuffle configuration
SHUFFLE_FRACTION = 1.0  # Fraction to shuffle (1.0 = 100%)

# Column mapping configuration
COLUMN_CONFIG = {
    "text_columns": [
        "text",  # Lowercase (used in test_datasets)
        "Email Text",
        "Text",
        "body",
    ],
    "label_columns": [
        "label",  # Lowercase (used in test_datasets)
        "Email Type",
        "Class",
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
    "Not Phishing": 0,
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
            max_int = int(max_int / CSV_FIELD_SIZE_DIVISOR)
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
    return any(
        "version https://git-lfs.github.com/spec/v1" in str(col) for col in columns
    )


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


def combine_csvs_from_directory(
    root_directory: str, output_filename: str, exclude_dirs: List[str] = None
) -> pd.DataFrame:
    """
    Combine all CSV files from a directory tree into a single DataFrame.

    Args:
        root_directory: Root directory containing CSV files
        output_filename: Output CSV file path
        exclude_dirs: Optional list of directory paths to exclude

    Returns:
        Combined DataFrame
    """
    if not os.path.isdir(root_directory):
        raise FileNotFoundError(f"Directory not found: {root_directory}")

    print(f"\n--- Step 1: Combining CSV Files ---")
    print(f"Scanning for CSVs under: {root_directory}")

    if exclude_dirs:
        # Normalize exclude paths for comparison
        exclude_dirs = [os.path.normpath(d) for d in exclude_dirs]
        print(f"Excluding directories: {exclude_dirs}")

    pattern = os.path.join(root_directory, "**", "*.csv")
    files = glob.glob(pattern, recursive=True)

    # Filter out files in excluded directories
    if exclude_dirs:
        filtered_files = []
        for f in files:
            file_dir = os.path.normpath(os.path.dirname(f))
            # Check if file_dir starts with any of the excluded paths
            is_excluded = any(
                file_dir == exc or file_dir.startswith(exc + os.sep)
                for exc in exclude_dirs
            )
            if not is_excluded:
                filtered_files.append(f)
        files = filtered_files

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
                print(
                    f"   - WARNING: {os.path.basename(f)} appears to be a Git LFS pointer file (not actual data)"
                )
                continue

            # Add provenance columns
            # Replace commas with semicolons to avoid CSV parsing issues
            df["source_file"] = os.path.basename(f).replace(",", ";")
            df["record_id"] = df.index

            # Get relative path of the file's directory from the root
            relative_dir_path = os.path.relpath(os.path.dirname(f), root_directory)

            if relative_dir_path == ".":
                # File is in the root, use root directory's name as source_name
                df["source_name"] = os.path.basename(os.path.normpath(root_directory)).replace(",", ";")
            else:
                # File is in a subdirectory, use the first-level subdirectory name
                df["source_name"] = relative_dir_path.split(os.sep)[0].replace(",", ";")

            dfs.append(df)

    # Warn about Git LFS files
    if git_lfs_files:
        print(f"\nWARNING: Found {len(git_lfs_files)} Git LFS pointer file(s):")
        for fname in git_lfs_files[:MAX_GIT_LFS_FILES_TO_SHOW]:
            print(f"   - {fname}")
        if len(git_lfs_files) > MAX_GIT_LFS_FILES_TO_SHOW:
            print(f"   ... and {len(git_lfs_files) - MAX_GIT_LFS_FILES_TO_SHOW} more")
        print("\nThese files are Git LFS pointers, not actual data files.")
        print("To download the actual files, run: git lfs pull")
        print("Or configure Git LFS and fetch the files from your repository.")

    if not dfs:
        if git_lfs_files:
            error_msg = (
                "\n" + "=" * SEPARATOR_LENGTH_LONG + "\n"
                "ERROR: No valid CSV data files could be read.\n"
                "All CSV files appear to be Git LFS pointer files.\n\n"
                "SOLUTION: Download the actual data files using Git LFS:\n"
                "  1. Make sure Git LFS is installed: git lfs install\n"
                "  2. Pull the actual files: git lfs pull\n"
                "  3. Run this script again\n" + "=" * SEPARATOR_LENGTH_LONG + "\n"
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
    expected_text_cols = ["Email Text", "Text", "body"]
    expected_label_cols = ["Email Type", "Class", "label"]
    has_text_cols = any(col in combined.columns for col in expected_text_cols)
    has_label_cols = any(col in combined.columns for col in expected_label_cols)

    if not has_text_cols or not has_label_cols:
        print("\nWARNING: Expected columns not found!")
        print(f"   Looking for TEXT columns: {expected_text_cols}")
        print(f"   Looking for LABEL columns: {expected_label_cols}")
        print(
            "\n   If your CSV files use different column names, update COLUMN_CONFIG in the script."
        )
        print(
            "   Or check if the files are Git LFS pointer files (run 'git lfs pull' to download actual data)."
        )

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    combined.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
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
        for v in unmapped_vals[:MAX_UNMAPPED_LABELS_TO_SHOW]:
            print(f"    - '{v}' (Type: {type(v)})")
        if len(unmapped_vals) > MAX_UNMAPPED_LABELS_TO_SHOW:
            print(
                f"    ... and {len(unmapped_vals) - MAX_UNMAPPED_LABELS_TO_SHOW} more"
            )

    print("\n[STEP 5] Dropping rows with missing text or labels...")
    before = len(final_df)
    final_df = final_df.dropna(subset=["text", "label"])
    after = len(final_df)
    print(f"   - Dropped {before - after} rows.")
    if after == 0:
        print("\n" + "=" * SEPARATOR_LENGTH_LONG)
        print("ERROR: Master dataset empty after cleaning.")
        print("=" * SEPARATOR_LENGTH_LONG)
        print("\nPossible issues:")
        print("1. Column names don't match expected values")
        print("2. All text/label columns are empty")
        print("\nTo fix this:")
        print("- Check the available columns shown above")
        print("- Update COLUMN_CONFIG in the script to match your CSV column names")
        print("- Check if the CSV files have the expected data")
        print("=" * SEPARATOR_LENGTH_LONG)
        raise RuntimeError(
            "Master dataset empty after cleaning. Check column names and data."
        )

    final_df["label"] = final_df["label"].astype(int)

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    final_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
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
        chunksize=CHUNKSIZE,
        desc="Cleaning Emails",
    )
    df["text"] = results

    before = len(df)
    # Filter by minimum length
    df = df[df["text"].str.strip().str.len() >= MIN_TEXT_LENGTH]
    dropped = before - len(df)
    if dropped > 0:
        print(
            f" - Dropped {dropped} rows with text length < {MIN_TEXT_LENGTH} characters."
        )

    # Reorder columns to include provenance columns
    first_cols = ["text", "label"]
    for col in ["source_file", "source_name", "record_id"]:
        if col in df.columns:
            first_cols.append(col)

    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
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
    df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f" - Saved final: {output_filename} ({after} unique rows)")
    return df


# ===================================================================
# New Orchestration Functions
# ===================================================================


def create_processed_dataset(
    input_directory: str, output_filename: str, exclude_dirs: List[str] = None
) -> pd.DataFrame:
    """
    Full pipeline to create a processed dataset from a directory.
    Combines -> Consolidates -> Cleans -> Deduplicates
    """
    print(f"\n" + "=" * SEPARATOR_LENGTH_SHORT)
    print(f"Creating Processed Dataset")
    print(f"Source: {input_directory}")
    print(f"Output: {output_filename}")
    if exclude_dirs:
        print(f"Excluding: {exclude_dirs}")
    print(f"=" * SEPARATOR_LENGTH_SHORT)

    # 1. Combine
    # Create temporary filename for combined step
    base_name = os.path.basename(output_filename).replace(".csv", "")
    temp_combined = os.path.join(
        os.path.dirname(output_filename), f"temp_{base_name}_combined.csv"
    )

    combined_df = combine_csvs_from_directory(
        input_directory, temp_combined, exclude_dirs
    )

    # 2. Process to master (labels/columns)
    temp_master = os.path.join(
        os.path.dirname(output_filename), f"temp_{base_name}_master.csv"
    )
    master_df = process_to_master(combined_df, temp_master, COLUMN_CONFIG, LABEL_MAP)

    # 3. Clean text
    temp_cleaned = os.path.join(
        os.path.dirname(output_filename), f"temp_{base_name}_cleaned.csv"
    )
    cleaned_df = clean_dataset(master_df, temp_cleaned, NUM_WORKERS)

    # 4. Deduplicate
    final_df = deduplicate_dataset(cleaned_df, output_filename)

    # Cleanup temps (optional, keeping them might be useful for debugging)
    for f in [temp_combined, temp_master, temp_cleaned]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    return final_df


def sample_and_combine_datasets(
    dataset1: pd.DataFrame, dataset2: pd.DataFrame, output_filename: str
) -> pd.DataFrame:
    """
    Sample from dataset1 and combine with dataset2 to create dataset3.
    """
    print(f"\n--- Sampling and Combining Datasets ---")
    print(f"Dataset 1 (Training Source): {len(dataset1)} rows")
    print(f"Dataset 2 (Test Source): {len(dataset2)} rows")

    d2_len = len(dataset2)

    if d2_len >= TARGET_DATASET_SIZE:
        print(
            f"Dataset 2 has >= {TARGET_DATASET_SIZE:,} rows. Skipping sampling from Dataset 1."
        )
        combined_df = dataset2.copy()
    else:
        rows_needed = TARGET_DATASET_SIZE - d2_len
        rows_available = len(dataset1)

        # Sample what we can
        sample_size = min(rows_needed, rows_available)

        if sample_size < rows_needed:
            print(
                f"WARNING: Dataset 1 has fewer rows ({rows_available}) than needed ({rows_needed}). Using all available."
            )

        if sample_size > 0:
            print(f"Sampling {sample_size} rows from Dataset 1...")
            sampled_d1 = dataset1.sample(n=sample_size, random_state=RANDOM_SEED)
            combined_df = pd.concat([sampled_d1, dataset2], ignore_index=True)
        else:
            combined_df = dataset2.copy()

    # Shuffle
    combined_df = combined_df.sample(
        frac=SHUFFLE_FRACTION, random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print(f"Combined Dataset (Dataset 3): {len(combined_df)} rows")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    combined_df.to_csv(output_filename, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f" - Saved: {output_filename}")

    return combined_df


def split_dataset(
    df: pd.DataFrame, train_out: str, val_out: str, test_out: str
) -> tuple:
    """
    Split dataset into Train (70%), Validation (15%), Test (15%) with stratification.
    """
    print(f"\n--- Splitting Dataset (Stratified) ---")

    if "label" not in df.columns:
        raise ValueError("Cannot split dataset: 'label' column missing.")

    # First split: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        df, test_size=TEMP_SPLIT_RATIO, stratify=df["label"], random_state=RANDOM_SEED
    )

    # Second split: Validation (15% of total -> 50% of temp) vs Test (15% of total -> 50% of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=VAL_TEST_SPLIT_RATIO,
        stratify=temp_df["label"],
        random_state=RANDOM_SEED,
    )

    print(f"Train set:      {len(train_df)} rows ({(len(train_df)/len(df))*100:.1f}%)")
    print(f"Validation set: {len(val_df)} rows ({(len(val_df)/len(df))*100:.1f}%)")
    print(f"Test set:       {len(test_df)} rows ({(len(test_df)/len(df))*100:.1f}%)")

    # Save files
    for d, path in [(train_df, train_out), (val_df, val_out), (test_df, test_out)]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f" - Saved: {path}")

    return train_df, val_df, test_df


# ===================================================================
# Main Execution
# ===================================================================


def main():
    """Main execution function for the data aggregation pipeline."""
    print("=" * SEPARATOR_LENGTH_LONG)
    print("Data Aggregation and Preliminary Preprocessing Pipeline")
    print("=" * SEPARATOR_LENGTH_LONG)

    # Set up multiprocessing
    multiprocessing.freeze_support()

    # Set CSV field size limit
    set_max_csv_field_size_limit()

    # --- Stage 1: Create Dataset 1 (Raw excluding Test Datasets) ---
    print("\n\n>>> STAGE 1: Creating Dataset 1 (Training Source) <<<")
    try:
        dataset1 = create_processed_dataset(
            input_directory=RAW_DATA_DIR,
            output_filename=DATASET1_OUTPUT,
            exclude_dirs=[TEST_DATASETS_DIR],
        )
    except Exception as e:
        print(f"Error creating Dataset 1: {e}")
        # Continue? Or exit? Assuming critical failure if dataset1 fails.
        sys.exit(1)

    # --- Stage 2: Create Dataset 2 (Test Datasets only) ---
    print("\n\n>>> STAGE 2: Creating Dataset 2 (Test Source) <<<")
    if os.path.exists(TEST_DATASETS_DIR):
        try:
            dataset2 = create_processed_dataset(
                input_directory=TEST_DATASETS_DIR,
                output_filename=DATASET2_OUTPUT,
                exclude_dirs=None,
            )
        except Exception as e:
            print(f"Error creating Dataset 2: {e}")
            dataset2 = pd.DataFrame(columns=["text", "label"])  # Empty fallback
    else:
        print(f"WARNING: Test datasets directory not found: {TEST_DATASETS_DIR}")
        dataset2 = pd.DataFrame(columns=["text", "label"])

    # --- Stage 3: Sample and Combine ---
    print("\n\n>>> STAGE 3: Sampling and Combining <<<")
    if dataset1.empty and dataset2.empty:
        print("ERROR: Both datasets are empty. Cannot proceed.")
        sys.exit(1)

    dataset3 = sample_and_combine_datasets(dataset1, dataset2, DATASET3_OUTPUT)

    # --- Stage 4: Split ---
    print("\n\n>>> STAGE 4: Splitting Train/Val/Test <<<")
    if not dataset3.empty:
        split_dataset(dataset3, TRAIN_OUTPUT, VAL_OUTPUT, TEST_OUTPUT)
    else:
        print("Dataset 3 is empty, skipping split.")

    print("\n" + "=" * SEPARATOR_LENGTH_LONG)
    print("Pipeline complete!")
    print(f"Final outputs in: {OUTPUT_DIR}")
    print("=" * SEPARATOR_LENGTH_LONG)


if __name__ == "__main__":
    main()
